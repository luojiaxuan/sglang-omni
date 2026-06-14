#!/usr/bin/env python3
"""Decode-only nsys kernel split by CUDA graph bucket (graphId).

Usage:
  python analysis/decode_split.py REPORT.sqlite [--graph-id 2] [--device 0]

Decode kernels = rows with non-null graphNodeId. Dominant steady bucket is usually
the graphId with the most launches (bs~32 on A6000 was graphId=2).
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
    if "BatchDecodeWithPagedKV" in name or "BatchPrefillWithPagedKV" in name:
        return "attn_core"
    if "pytorch_flash::flash_fwd" in name:
        return "attn_core"
    if any(k in name for k in ("MergeStates", "flashinfer_kv_indices", "kv_indices")):
        return "attn_misc"
    if any(k in name.lower() for k in ("gemm", "cutlass")) or "splitKreduce" in name:
        return "gemm_dense"
    if any(k in name for k in ("RMSNorm", "layer_norm", "fused_qknorm")):
        return "norm"
    if "mrope" in name:
        return "rope"
    if "memcpy" in name or "Memset" in name or "memset" in name:
        return "memcpy"
    return "other"


def merge_union(ivs: list[tuple[int, int]]) -> int:
    ivs = sorted(ivs)
    tot = 0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    tot += ce - cs
    return tot


def list_graph_ids(cur: sqlite3.Cursor, device: int) -> list[tuple[int, int, float]]:
    rows = cur.execute(
        """
        SELECT graphId, count(*) AS k,
               sum(end-start)*1.0/count(*) AS ns_per_launch
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=? AND graphNodeId IS NOT NULL
        GROUP BY graphId ORDER BY k DESC
        """,
        (device,),
    ).fetchall()
    return [(int(g), int(k), ns / 1e6) for g, k, ns in rows]


def analyze(cur: sqlite3.Cursor, sid: dict[int, str], gid: int, device: int, label: str) -> None:
    rows = list(
        cur.execute(
            """
            SELECT start, end, demangledName, shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId=? AND graphNodeId IS NOT NULL AND graphId=?
            ORDER BY start
            """,
            (device, gid),
        )
    )
    node_rows = cur.execute(
        """
        SELECT DISTINCT graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=? AND graphId=? AND graphNodeId IS NOT NULL
        """,
        (device, gid),
    ).fetchall()
    nodes = len(node_rows)
    if nodes == 0 or len(rows) % nodes != 0:
        print(f"  skip graphId={gid}: nodes={nodes} rows={len(rows)}")
        return
    nl = len(rows) // nodes
    cat_busy: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, int] = defaultdict(int)
    walls: list[int] = []
    unions: list[int] = []
    ar_per: list[int] = []
    for li in range(nl):
        chunk = rows[li * nodes : (li + 1) * nodes]
        ivs = [(s, e) for s, e, _, _ in chunk]
        wall = max(e for _, e in ivs) - min(s for s, _ in ivs)
        union = merge_union(ivs)
        walls.append(wall)
        unions.append(union)
        arc = 0
        for s, e, dm, sn in chunk:
            nm = sid.get(dm) or sid.get(sn)
            c = cat(nm)
            cat_busy[c] += e - s
            cat_cnt[c] += 1
            if c == "allreduce":
                arc += 1
        ar_per.append(arc)
    walls_ms = sum(walls) / nl / 1e6
    union_ms = sum(unions) / nl / 1e6
    print(f"\n===== {label}  graphId={gid}  launches={nl}  nodes/step={nodes} =====")
    print(
        f"  per-step WALL = {walls_ms:.2f} ms   GPU-busy(union) = {union_ms:.2f} ms   "
        f"idle/gap = {walls_ms - union_ms:.2f} ms ({100 * (walls_ms - union_ms) / walls_ms:.1f}% of wall)"
    )
    print(f"  all-reduce kernels/step = {sum(ar_per) / nl:.0f}")
    print(f"  {'category':12s} {'ms/step':>8s} {'%wall':>7s} {'%busy':>7s}  inst/step")
    sb = sum(cat_busy.values())
    for c in sorted(cat_busy, key=lambda x: -cat_busy[x]):
        ms = cat_busy[c] / nl / 1e6
        print(
            f"  {c:12s} {ms:8.2f} {100 * ms / walls_ms:6.1f}% "
            f"{100 * cat_busy[c] / sb:6.1f}%  {cat_cnt[c] / nl:.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite", help="nsys-exported .sqlite file")
    ap.add_argument("--graph-id", type=int, action="append", dest="graph_ids")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--top", type=int, default=2, help="analyze top-N graph buckets by launch count")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    sid = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    buckets = list_graph_ids(cur, args.device)
    if not buckets:
        print("No decode CUDA graphs found (graphNodeId IS NOT NULL).")
        return
    print("Decode graph buckets (device %d):" % args.device)
    for g, k, ms in buckets:
        print(f"  graphId={g}  launches={k}  ~{ms:.2f} ms/launch")

    gids = args.graph_ids
    if not gids:
        gids = [g for g, _, _ in buckets[: args.top]]
    for i, gid in enumerate(gids):
        label = "STEADY DECODE (dominant)" if i == 0 else f"DECODE bucket #{i + 1}"
        analyze(cur, sid, gid, args.device, label)


if __name__ == "__main__":
    main()
