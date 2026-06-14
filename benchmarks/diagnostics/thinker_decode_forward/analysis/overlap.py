#!/usr/bin/env python3
"""Sweep-line critical-path breakdown for decode CUDA graph steps.

Usage:
  python analysis/overlap.py REPORT.sqlite [--graph-id 2] [--device 0]
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict


def cat(name: str | None) -> str:
    if name is None:
        return "other"
    if name.startswith("ncclDevKernel_AllReduce"):
        return "allreduce"
    if name.startswith("ncclDevKernel_AllGather"):
        return "allgather"
    if name.startswith("ncclDevKernel"):
        return "nccl_other"
    if "fused_moe_kernel" in name:
        return "moe_gemm"
    if any(k in name for k in ("moe_sum_reduce", "moe_align", "count_and_sort_expert", "topkGatingSoftmax")):
        return "moe_route"
    if "act_and_mul" in name:
        return "moe_act"
    if "BatchDecodeWithPagedKV" in name or "BatchPrefillWithPagedKV" in name or "pytorch_flash::flash_fwd" in name:
        return "attn_core"
    if any(k in name for k in ("MergeStates", "flashinfer_kv_indices", "kv_indices")):
        return "attn_misc"
    if any(k in name.lower() for k in ("gemm", "cutlass")) or "splitKreduce" in name:
        return "gemm_dense"
    if any(k in name for k in ("RMSNorm", "layer_norm", "fused_qknorm")):
        return "norm"
    if "mrope" in name:
        return "rope"
    return "other"


COMPUTE = {"moe_gemm", "gemm_dense", "attn_core", "moe_route", "moe_act", "norm", "rope"}


def pick_graph_id(cur: sqlite3.Cursor, device: int, graph_id: int | None) -> int:
    if graph_id is not None:
        return graph_id
    row = cur.execute(
        """
        SELECT graphId FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=? AND graphNodeId IS NOT NULL
        GROUP BY graphId ORDER BY count(*) DESC LIMIT 1
        """,
        (device,),
    ).fetchone()
    if not row:
        raise SystemExit("No decode graphs in trace")
    return int(row[0])


def sweep(cur: sqlite3.Cursor, sid: dict[int, str], gid: int, device: int, label: str) -> None:
    rows = list(
        cur.execute(
            """
            SELECT start, end, demangledName, shortName, streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId=? AND graphNodeId IS NOT NULL AND graphId=?
            ORDER BY start
            """,
            (device, gid),
        )
    )
    nodes = len(
        cur.execute(
            """
            SELECT DISTINCT graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId=? AND graphId=? AND graphNodeId IS NOT NULL
            """,
            (device, gid),
        ).fetchall()
    )
    nl = len(rows) // nodes
    excl: dict[str, float] = defaultdict(float)
    gap = 0.0
    overlap2 = 0.0
    wall = 0.0
    union = 0.0
    ar_exposed = 0.0
    ar_union = 0.0
    ar_under_moe = 0.0
    conc_time: dict[int, float] = defaultdict(float)
    streams: set[int] = set()

    for li in range(nl):
        chunk = rows[li * nodes : (li + 1) * nodes]
        ev: list[tuple[int, int, str]] = []
        for s, e, dm, sn, st in chunk:
            c = cat(sid.get(dm) or sid.get(sn))
            streams.add(st)
            ev.append((s, 1, c))
            ev.append((e, -1, c))
        ev.sort()
        active: dict[str, int] = defaultdict(int)
        tot = 0
        prev = ev[0][0]
        for t, d, c in ev:
            dt = t - prev
            if dt > 0:
                ncat = sum(1 for v in active.values() if v > 0)
                conc_time[min(tot, 4)] += dt
                if tot == 0:
                    gap += dt
                elif ncat == 1:
                    for k, v in active.items():
                        if v > 0:
                            excl[k] += dt
                else:
                    overlap2 += dt
                if active["allreduce"] > 0:
                    ar_union += dt
                    comp = sum(active[k] for k in COMPUTE)
                    if comp == 0:
                        ar_exposed += dt
                    else:
                        ar_under_moe += dt
                if tot > 0:
                    union += dt
            active[c] += d
            tot += d
            prev = t
        wall += max(e for _, e, _, _, _ in chunk) - min(s for s, _, _, _, _ in chunk)

    print(f"\n===== {label} graphId={gid} launches={nl} unique_streams={len(streams)} =====")
    print(
        f"  wall/step={wall / nl / 1e6:.2f}ms  union-busy/step={union / nl / 1e6:.2f}ms  "
        f"gap/step={gap / nl / 1e6:.2f}ms"
    )
    parts = " ".join(f"{k}k={100 * v / wall:.0f}%" for k, v in sorted(conc_time.items()))
    print(f"  concurrency (share of wall): {parts}")
    print(
        f"  ALL-REDUCE: union={ar_union / nl / 1e6:.2f}ms/step  "
        f"exposed={ar_exposed / nl / 1e6:.2f}ms  hidden-under-compute={ar_under_moe / nl / 1e6:.2f}ms"
    )
    print("  exclusive (only-this-cat) ms/step:")
    for k, v in sorted(excl.items(), key=lambda x: -x[1]):
        print(f"     {k:12s} {v / nl / 1e6:6.2f}ms  ({100 * v / wall:.0f}% of wall)")
    print(f"  overlap(>=2 cats) ms/step = {overlap2 / nl / 1e6:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite")
    ap.add_argument("--graph-id", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    sid = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
    gid = pick_graph_id(cur, args.device, args.graph_id)
    sweep(cur, sid, gid, args.device, "STEADY DECODE")


if __name__ == "__main__":
    main()
