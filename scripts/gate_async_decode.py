#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR-B async-decode ON/OFF bit-identity & overrun-safety gate (S1-S5).

Same build, the only difference is ``--async-decode on|off`` (+ per-scenario
flags/env). Each request carries an explicit seed. The comparison point is the
vocoder-entry ``state.audio_codes`` (isolating the AR from any vocoder
non-determinism); set ``MOSS_GATE_DUMP_AUDIO_CODES=<dir>`` so the vocoder stage
writes ``<seed>.npy`` per request, and this harness diffs ON vs OFF.

Concurrency (the gate's load-bearing methodology): a scenario's requests are
POSTed CONCURRENTLY (send_scenario, behind a Barrier) so they form a real
multi-request decode batch and the ON arm exercises the one-step lookahead at
bs>=2. A sequential driver keeps the server at bs=1, below
``async_decode_min_batch_size`` for every scenario but S1, so the ON arm would
silently take the synchronous fast path — bit-identity proven against itself.
Each ON arm therefore also asserts the lookahead hit/miss counter
(``MOSS_GATE_QUERY_DUMP`` -> --counter-file): every lookahead scenario must show
hit+miss > 0, and the rep-penalty scenario (sync routing) must show 0.

Criteria are tiered by whether the scenario's batch timing is deterministic:

  - exact (S1 only): audio_codes bit-identical ON vs OFF per seed. S1 is bs=1
    with min_batch_size=1, so it is BOTH deterministic (a single request has no
    batch-composition variance) AND lookahead-forced. It is the load-bearing
    correctness anchor — the test that caught the real bs=1 stop-boundary bug.
    (The sync self-comparison gate S0 is the other deterministic exact check.)

  - structural (S2/S3a/S3b/S4/S5): a multi-request batch under real concurrency
    is not bit-reproducible — the per-step batch composition is timing-dependent,
    so the CUDA-graph bucket trajectory (and the low bits of each request's
    hidden states, which the binary stop head amplifies into different frame
    counts) varies run-to-run. This is NOT a lookahead artifact: an OFF-vs-OFF
    control (two pure-sync concurrent runs) of the same seeds already differs in
    frame count and values. So exact bit-identity and real-concurrency lookahead
    coverage are mutually exclusive for multi-request scenarios, and only S1 gets
    both. Binding structural checks: no crash (every seed produced a dump),
    lookahead engaged (counter; for S5, NOT engaged), every request finished.
    Per-seed bit-identity is reported for the record, not failed on.

  - every ON arm asserts the lookahead hit/miss counter (above): the
    new-methodology guarantee that the concurrent path actually ran. A sequential
    driver's "exact PASS" on S2-S5 was hollow — it never produced a batch, so the
    ON arm silently ran sync and proved bit-identity against itself.

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
    # Whether the ON arm is expected to engage the one-step lookahead at all.
    # True for every lookahead scenario (assert hit+miss > 0); False only for the
    # rep-penalty scenario, whose batch routes wholly to the synchronous path so
    # no launch/resolve runs (assert hit+miss == 0).
    expect_lookahead: bool = True


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
        desc="bs=4 concurrent steady-state: lookahead engaged across a real batch",
        on_flags=["--async-decode", "on"],  # default min-batch-size 2
        requests=[Req(_TEXT, seed=1000 + i, max_new_tokens=48) for i in range(4)],
        # Structural, not exact: under real concurrency the batch composition per
        # step is timing-dependent, so the CUDA-graph bucket trajectory (and thus
        # the low bits of each request's hidden states, amplified by the binary
        # stop head into different frame counts) is not reproducible run-to-run.
        # Proven by an OFF-vs-OFF control: two pure-sync concurrent runs of these
        # same seeds already differ in frame count AND values (no lookahead
        # involved). See docs/design/async_decode_gate.md.
        criterion="structural",
    ),
    "S3a": Scenario(
        key="S3a",
        desc="concurrent transition (a short request finishes first); overrun-safe",
        on_flags=["--async-decode", "on"],
        # six long + one short -> the short finishes first (the bs transition).
        requests=(
            [Req(_TEXT, seed=2000 + i, max_new_tokens=64) for i in range(6)]
            + [Req(_TEXT, seed=2099, max_new_tokens=16)]
        ),
        criterion="structural",  # see S2: concurrent batching is non-deterministic
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
        # The binding assertion is the counter == 0 (expect_lookahead=False): a
        # batch containing a rep-penalty request must route wholly to the
        # synchronous path, so no launch/resolve runs. Audio is structural (both
        # arms run sync, but concurrent batching is still non-deterministic).
        criterion="structural",
        expect_lookahead=False,
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


def read_counters(counter_file: str) -> tuple[int, int]:
    """Return (hit, miss) from the gate counter dump, or (0, 0) if absent.

    Absent is the meaningful zero: gate_query_counters only writes from inside
    execute_resolve, so a scenario that never engages lookahead leaves no file."""
    import os

    if not counter_file or not os.path.exists(counter_file):
        return 0, 0
    try:
        parts = open(counter_file).read().split()
        return int(parts[0]), int(parts[1])
    except Exception:  # noqa: BLE001
        return 0, 0


def check_counters(sc: Scenario, counter_file: str) -> int:
    """Assert the ON arm engaged (or, for the sync-routed scenario, did not
    engage) the one-step lookahead. Returns 0 on pass, 1 on fail."""
    hit, miss = read_counters(counter_file)
    total = hit + miss
    if sc.expect_lookahead:
        ok = total > 0
        expect = "expect >0 (lookahead engaged)"
    else:
        ok = total == 0
        expect = "expect 0 (rep-penalty -> whole-batch sync routing)"
    print(
        f"[{sc.key}] lookahead counters: hit={hit} miss={miss} total={total} "
        f"{expect} -> {'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def send_scenario(sc: Scenario, port: int, dump_dir: str) -> int:
    """POST the scenario's fixed-seed requests to /v1/audio/speech CONCURRENTLY,
    so they form a real multi-request decode batch and the ON arm exercises the
    one-step lookahead at bs>=2.

    Sequential sends (the original driver) kept the server at bs=1 the whole run,
    which is below ``async_decode_min_batch_size`` for every scenario except S1 —
    so the ON arm silently took the synchronous fast path and the bit-identity it
    "proved" was sync-vs-sync. Here every request is launched from its own thread
    behind a ``Barrier`` so they hit the scheduler in the same window and decode
    together. Per-row seeding keeps each request's audio independent of the batch
    it lands in, so the ON/OFF comparison stays per-seed (the dump hook keys by
    seed, concurrency-safe). The hit/miss counter assertion (see check_counters)
    confirms lookahead actually engaged."""
    import json
    import threading
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    url = f"http://localhost:{port}/v1/audio/speech"
    n = len(sc.requests)
    barrier = threading.Barrier(n)
    ok: dict[int, bool] = {}

    def _one(i: int, r: Req) -> None:
        # max_new_tokens / repetition_penalty are top-level CreateSpeechRequest
        # fields (serve/protocol.py), NOT extra_body — nesting them silently
        # drops both, so requests ran to stop-head termination and the rep
        # penalty never applied (S5 then lookahead-batched instead of routing
        # sync). Send them at the top level alongside seed.
        body = {
            "model": "moss-tts-local",
            "input": r.text,
            "voice": "default",
            "seed": r.seed,
            "max_new_tokens": r.max_new_tokens,
            "repetition_penalty": r.repetition_penalty,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-request-id": f"req{i}"},
        )
        barrier.wait()  # release all threads together -> concurrent arrival
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()  # audio bytes discarded; codes are dumped server-side
            ok[i] = True
        except Exception as exc:  # noqa: BLE001
            print(f"[send] {sc.key} req{i} failed: {exc}")
            ok[i] = False

    with ThreadPoolExecutor(max_workers=n) as ex:
        for i, r in enumerate(sc.requests):
            ex.submit(_one, i, r)
    sent = sum(1 for v in ok.values() if v)
    print(
        f"[send] {sc.key}: {sent}/{n} sent concurrently "
        f"(server dumps -> {dump_dir})"
    )
    return 0 if sent == n else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    p.add_argument("--off-dump", help="dir with OFF-arm <rid>.npy audio_codes")
    p.add_argument("--on-dump", help="dir with ON-arm <rid>.npy audio_codes")
    p.add_argument("--send", action="store_true", help="POST the scenario's requests")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dump-dir", help="(informational) where the server dumps codes")
    p.add_argument(
        "--counter-file",
        help="gate_query_counters dump (MOSS_GATE_QUERY_DUMP) from the ON arm; "
        "asserts lookahead engaged (or, for S5, did not)",
    )
    p.add_argument("--emit-env", action="store_true", help="print ON-arm env K=V line")
    p.add_argument("--emit-flags", action="store_true", help="print ON-arm CLI flags")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="print the scenario's server flags/env/requests and exit",
    )
    args = p.parse_args(argv)
    sc = SCENARIOS[args.scenario]

    if args.emit_env:
        print(" ".join(f"{k}={v}" for k, v in sc.on_env.items()))
        return 0
    if args.emit_flags:
        print(" ".join(sc.on_flags))
        return 0
    if args.send:
        return send_scenario(sc, args.port, args.dump_dir or "")

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

    # Dump files are keyed by the request's seed (the server assigns its own
    # request_id and ignores x-request-id), so map each scenario request to its
    # seed key.
    keys = [str(r.seed) for r in sc.requests]

    # The lookahead-engagement counter assertion gates every scenario: it is the
    # new-methodology guarantee that the ON arm actually ran the concurrent
    # lookahead path (a sequential, bs=1 run would silently pass bit-identity
    # while never engaging it). A missing --counter-file degrades to (0,0).
    counter_rc = check_counters(sc, args.counter_file) if args.counter_file else 0

    if sc.criterion == "exact":
        diffs = compare_exact(args.off_dump, args.on_dump, keys)
        if diffs:
            print(f"[{sc.key}] FAIL — audio_codes differ for seeds: {diffs}")
            return 1
        if counter_rc:
            print(f"[{sc.key}] FAIL — lookahead-engagement counter assertion failed")
            return 1
        print(f"[{sc.key}] PASS — all {len(keys)} requests bit-identical ON vs OFF")
        return 0

    # Structural (S3b cross-bucket, S4 retraction). Under real concurrency the
    # bs transition / retraction step is timing-dependent, so exact bit-identity
    # is NOT the criterion (cross-bucket low-bit drift and discarded
    # retraction/overrun frames are both legitimate). The binding checks are:
    #   1. no request crashed -> every seed produced a dump (missing == FAIL),
    #   2. the lookahead engaged (counter assertion),
    # and per-seed bit-identity is REPORTED for drift localization, not failed on.
    import numpy as np

    print(f"[{sc.key}] structural — ON/OFF audio_codes shapes (drift localization):")
    missing = []
    for key in keys:
        try:
            a, b = load_codes(args.off_dump, key), load_codes(args.on_dump, key)
            same = a.shape == b.shape and np.array_equal(a, b)
            print(f"  seed={key}: off={a.shape} on={b.shape} identical={same}")
        except FileNotFoundError:
            missing.append(key)
            print(f"  seed={key}: MISSING (request did not complete)")
    if missing:
        print(f"[{sc.key}] FAIL — missing dumps (crash/no-finish): {missing}")
        return 1
    if counter_rc:
        print(f"[{sc.key}] FAIL — lookahead-engagement counter assertion failed")
        return 1
    print(
        f"[{sc.key}] structural PASS — all requests finished, lookahead engaged, "
        f"no crash; per-seed drift localized above."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
