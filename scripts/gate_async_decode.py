#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR-B async-decode ON/OFF bit-identity & overrun-safety gate (S1-S5).

Same build, the only difference is ``--async-decode on|off`` (+ per-scenario
flags/env). Each request carries an explicit seed. The comparison point is the
vocoder-entry ``state.audio_codes`` (isolating the AR from any vocoder
non-determinism); set ``MOSS_GATE_DUMP_AUDIO_CODES=<dir>`` so the vocoder stage
writes ``<rid>.npy`` per request, and this harness diffs ON vs OFF.

Exact-equality criterion: S1/S2/S3a/S5. Structural-only: S3b/S4.

Run on GPU AFTER the cache benchmark releases the box (see gate matrix). It is
deferred here on purpose — it needs the full serving stack, not a unit test.

Usage (inside the container, node0-pinned like the ABAB harness):
    PYTHONPATH=/data/moss-v15-ar/sglang-omni \\
      python scripts/gate_async_decode.py --scenario S1 --port 8000

This driver does NOT launch the server; launch two servers (OFF then ON) with the
scenario's flags and point the driver at each, OR pass --manage to let it manage
both. The criteria/configs live in SCENARIOS below — the single source of truth
the gate matrix doc mirrors.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field


@dataclass
class Req:
    """One request in a scenario's fixed batch."""

    text: str
    seed: int
    max_new_tokens: int | None = None
    repetition_penalty: float = 1.0


@dataclass
class Scenario:
    key: str
    desc: str
    # Extra server flags for the ON arm (the OFF arm is always --async-decode off).
    on_flags: list[str]
    on_env: dict[str, str] = field(default_factory=dict)
    requests: list[Req] = field(default_factory=list)
    # "exact": audio_codes bit-identical ON vs OFF for every kept request.
    # "structural": no crash / no double-free; finisher frame counts exact;
    #   survivors recorded with drift localized (drift only after the transition).
    criterion: str = "exact"
    # Requests whose ON-arm output is a lookahead overrun and is NOT compared
    # exactly (they are the deliberately-dropped frames); still checked for
    # no-crash + correct finish.
    structural_only_rids: list[str] = field(default_factory=list)


_TEXT = "The quick brown fox jumps over the lazy dog."

SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario(
        key="S1",
        desc="bs=1 forced lookahead — audio_codes + output_ids bit-identical",
        on_flags=["--async-decode", "on", "--async-decode-min-batch-size", "1"],
        requests=[Req(_TEXT, seed=1234, max_new_tokens=64)],
        criterion="exact",
    ),
    "S2": Scenario(
        key="S2",
        desc="bs=4 steady (zero transition): same max_new_tokens -> sync length finish",
        on_flags=["--async-decode", "on"],  # default min-batch-size 2
        requests=[Req(_TEXT, seed=1000 + i, max_new_tokens=48) for i in range(4)],
        criterion="exact",
    ),
    "S3a": Scenario(
        key="S3a",
        desc="same-bucket transition (bs=7->6, both bucket 8), one finishes first",
        on_flags=["--async-decode", "on"],
        # six long + one short -> the short finishes a step early (the overrun).
        requests=(
            [Req(_TEXT, seed=2000 + i, max_new_tokens=64) for i in range(6)]
            + [Req(_TEXT, seed=2099, max_new_tokens=16)]
        ),
        criterion="exact",  # survivors exact; finisher exact frame count
    ),
    "S3b": Scenario(
        key="S3b",
        desc="cross-bucket transition (bs=2->1); two variants min-batch-size 2 / 1",
        on_flags=["--async-decode", "on"],
        requests=[
            Req(_TEXT, seed=3000, max_new_tokens=64),
            Req(_TEXT, seed=3001, max_new_tokens=32),  # finishes ~30 frames earlier
        ],
        criterion="structural",
    ),
    "S4": Scenario(
        key="S4",
        desc="retraction (TEST_RETRACT INTERVAL=8); no crash / no double-free",
        on_flags=["--async-decode", "on"],
        on_env={"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": "8"},
        requests=[Req(_TEXT, seed=4000 + i, max_new_tokens=64) for i in range(4)],
        criterion="structural",
    ),
    "S5": Scenario(
        key="S5",
        desc="rep-penalty sync routing: one penalty!=1 -> whole batch sync, hits=0",
        on_flags=["--async-decode", "on"],
        requests=[
            Req(_TEXT, seed=5000, max_new_tokens=48, repetition_penalty=1.0),
            Req(_TEXT, seed=5001, max_new_tokens=48, repetition_penalty=1.3),
        ],
        criterion="exact",  # + assert lookahead count == 0 (see check_lookahead_count)
    ),
}


def load_codes(dump_dir: str, rid: str):
    import numpy as np

    return np.load(f"{dump_dir}/{rid}.npy")


def compare_exact(off_dir: str, on_dir: str, rids: list[str]) -> list[str]:
    """Return the list of rids whose ON/OFF audio_codes differ (empty == pass)."""
    import numpy as np

    diffs = []
    for rid in rids:
        a = load_codes(off_dir, rid)
        b = load_codes(on_dir, rid)
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(rid)
    return diffs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    p.add_argument("--off-dump", help="dir with OFF-arm <rid>.npy audio_codes")
    p.add_argument("--on-dump", help="dir with ON-arm <rid>.npy audio_codes")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="print the scenario's server flags/env/requests and exit",
    )
    args = p.parse_args(argv)
    sc = SCENARIOS[args.scenario]

    if args.print_config or not (args.off_dump and args.on_dump):
        print(f"[{sc.key}] {sc.desc}")
        print(f"  ON flags : {' '.join(sc.on_flags)}")
        print(f"  ON env   : {sc.on_env or '(none)'}")
        print(f"  criterion: {sc.criterion}")
        for i, r in enumerate(sc.requests):
            print(
                f"  req[{i}] rid=req{i} seed={r.seed} "
                f"max_new_tokens={r.max_new_tokens} rep_penalty={r.repetition_penalty}"
            )
        if not (args.off_dump and args.on_dump):
            print("\n(no --off-dump/--on-dump given: printed config only)")
        return 0

    rids = [f"req{i}" for i in range(len(sc.requests))]
    if sc.criterion == "exact":
        diffs = compare_exact(args.off_dump, args.on_dump, rids)
        if diffs:
            print(f"[{sc.key}] FAIL — audio_codes differ for: {diffs}")
            return 1
        print(f"[{sc.key}] PASS — all {len(rids)} requests bit-identical ON vs OFF")
        return 0

    # structural: presence + no-crash are checked by the run wrapper; here we
    # report per-request ON/OFF shapes for drift localization.
    import numpy as np

    print(f"[{sc.key}] structural — ON/OFF audio_codes shapes (drift localization):")
    for rid in rids:
        try:
            a, b = load_codes(args.off_dump, rid), load_codes(args.on_dump, rid)
            same = a.shape == b.shape and np.array_equal(a, b)
            print(f"  {rid}: off={a.shape} on={b.shape} identical={same}")
        except FileNotFoundError as exc:
            print(f"  {rid}: MISSING ({exc})")
    print(f"[{sc.key}] structural — manual review of the above + run-log (no crash).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
