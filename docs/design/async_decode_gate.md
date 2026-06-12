# MOSS-TTS-Local async-decode gate (S1-S5)

Bit-identity and overrun-safety gate for the PR-B one-step lookahead. Same
build, the only difference between the two arms is `--async-decode on|off` plus
each scenario's flags/env. The comparison point is the vocoder-entry
`state.audio_codes` (isolating the AR from vocoder non-determinism), dumped per
request keyed by seed.

The GPU integration harness (the server launcher, the dump/counter hooks, the
concurrent driver, the ABAB runner) is operational tooling that needs a live
serving stack and GPUs, so it lives outside the product tree rather than in this
PR's diff. Its CPU-testable contract, the scenario table and the comparison and
counter logic, is locked by
`tests/unit_test/moss_tts_local/test_gate_scenarios.py`; this doc explains the
methodology the harness implements.

## Concurrency (load-bearing)

A scenario's requests are POSTed **concurrently** (one thread each, released
together behind a `Barrier`) so they form a real multi-request decode batch and
the ON arm actually exercises the lookahead at `bs >= 2`.

This matters because a **sequential** driver keeps the server at `bs = 1` for the
whole run, which is below `async_decode_min_batch_size` for every scenario except
S1. The ON arm then silently takes the synchronous fast path, and the
"bit-identity" it proves is sync-vs-sync — it never touches the launch/resolve
machinery the gate exists to validate. (This is exactly how an earlier
sequential run passed S2-S5 while the bs=1 stop-boundary bug sat in the lookahead
path; only S1, forced to `min_batch_size = 1`, engaged it.)

Per-row seeding keeps each request's sampling independent of the batch it lands
in, so the ON/OFF comparison stays per-seed regardless of how requests batch.

## Lookahead-engagement counter

Concurrency alone is necessary but not self-evident, so every ON arm also asserts
the lookahead hit/miss counter. The ON server sets `MOSS_GATE_QUERY_DUMP=<file>`;
`gate_query_counters` writes the cumulative `_async_query_hit _async_query_miss`
from inside `execute_resolve` on every resolve, and the comparison reads it via
`--counter-file`:

| scenario | expectation | meaning |
| --- | --- | --- |
| S1-S4 | `hit + miss > 0` | the lookahead launch/resolve path actually ran |
| S5 | `hit + miss == 0` (file absent) | rep-penalty routed the whole batch to sync, so no launch/resolve ran |

A scenario that fails its counter assertion fails the gate, even if the
audio_codes happen to match — a match without engagement is not evidence.

## Why concurrency precludes multi-request bit-identity (the OFF-vs-OFF control)

Exact `audio_codes` bit-identity and real-concurrency lookahead coverage are
**mutually exclusive** for any multi-request scenario. A sequential driver gives
deterministic batching (always `bs = 1`) but never engages the lookahead;
concurrent sending engages it but makes the per-step batch composition depend on
millisecond-level arrival timing. That composition selects the CUDA-graph bucket
each step replays, which perturbs the low bits of every request's hidden states,
which the **binary stop head** amplifies into a different stop frame — so the
same seed finishes at a different frame count from one concurrent run to the
next.

This was confirmed with an **OFF-vs-OFF control**: the same seeds sent
concurrently to two pure-synchronous (`--async-decode off`, no lookahead at all)
servers already diverge.

| seed | OFF run A | OFF run B |
| --- | --- | --- |
| S2/1000 | 36 frames | 36 frames, **values differ** |
| S2/1002 | 49 frames | 36 frames |
| S2/1001 | 43 frames | 42 frames |
| S3a/2099 | 43 frames | 34 frames |

Two sync runs differ in both frame count and values (seed 1000: identical frame
count, different values), with zero lookahead involved. So a multi-request ON-vs-OFF
difference is this same batching non-determinism, **not** a lookahead defect (a
lookahead defect would show ON != OFF while OFF-A == OFF-B). Only `bs = 1` (S1)
escapes it, which is exactly why S1 is the bit-identity anchor — and why it is
the test that caught the real bs=1 stop-boundary bug.

The raw OFF-A/OFF-B drift dumps are retained for review.

## Criteria (tiered by batch-timing determinism)

- **exact — S1 (and the sync self-comparison S0)**: `audio_codes` bit-identical
  ON vs OFF per seed. S1 is `bs = 1`, `min_batch_size = 1`: both deterministic
  (no batch-composition variance) and lookahead-forced. The load-bearing
  correctness anchor.

- **structural — S2, S3a, S3b, S4, S5**: a multi-request concurrent batch is not
  bit-reproducible (above), so bit-identity is not asserted. Binding checks:
  1. **no crash / no double-free** — every seed produced a dump (a missing dump
     fails the scenario),
  2. **lookahead engaged** — the counter (`> 0`; for S5, `== 0`),
  3. every request finished (no 4096-frame runaway).

  Per-seed bit-identity is **reported** for the record, not failed on. The
  scenario intents still hold structurally: S3b exercises a cross-bucket
  `bs 2 -> 1` transition, S4 exercises retraction (`TEST_RETRACT INTERVAL=8`,
  discarded in-flight frames re-sampled on resume), S5 exercises rep-penalty sync
  routing (asserted by `counter == 0`).

## Methodology change vs the original (sequential) gate

The original gate matrix marked S2/S3a/S5 "exact" and sent requests one at a
time. That made the **ON arm's exact PASS hollow**: at `bs = 1` it stayed below
`async_decode_min_batch_size` for every scenario but S1, so it silently took the
synchronous fast path — it proved bit-identity of sync against sync and never
touched the lookahead. The concurrent rewrite makes three changes, recorded here:

1. Requests are sent concurrently, so S2-S5 exercise real batches (counter > 0).
2. Every ON arm asserts the engagement counter; S5 asserts `== 0`.
3. Multi-request scenarios move from exact to structural, because the OFF-vs-OFF
   control proves concurrent batching is non-deterministic by construction (not
   by lookahead bug). Bit-identity stays an anchor only where batch timing is
   deterministic (S0 sync self-compare, S1 `bs = 1`).

## Running

The operational harness (kept as cluster tooling, outside this PR) drives both
arms per scenario on the same cards: launch OFF, send concurrently, kill; launch
ON, send, kill; compare. It is run on the GPU box before merge and the run log is
pasted into the PR.

CPU coverage in-tree: the launch/resolve contract is tested by
`tests/unit_test/moss_tts_local/test_pipeline.py` and
`tests/unit_test/pipeline/test_async_decode.py`; the gate scenario table and the
comparison/counter logic by `test_gate_scenarios.py`.
