# sglang-omni vs vLLM — pure Qwen3-Omni-30B-A3B-Instruct (no-RAG), TP=2

Engine A/B on the **same 2 GPUs** (host 2,3), same model, same SimulEval streaming
protocol. Per-segment sweep over the ACL6060 en->zh dev set (468 segments), each
segment an independent streaming instance; concurrency N = number of parallel
SimulEval workers. Scored with FBK `stream_laal_term.py` (no glossary):
BLEU (sacrebleu zh), StreamLAAL, StreamLAAL_CA (computation-aware), char unit.

Engines: vLLM `serve` and sglang-omni (stock scheduler, **no M1 coalesce knobs**),
both TP=2, max concurrency 32, mem_fraction_static 0.75, seg=1920ms, max_new=40.

## Raw results

| engine | N  | wall_s | seg/s  | BLEU  | StreamLAAL | StreamLAAL_CA |
|--------|----|--------|--------|-------|------------|---------------|
| vllm   | 1  | 570.18 | 0.821  | 33.22 | 1340.50    | 1659.00       |
| vllm   | 8  | 90.98  | 5.144  | 33.36 | 1347.86    | 1728.62       |
| vllm   | 16 | 55.15  | 8.485  | 32.99 | 1344.37    | 1777.69       |
| vllm   | 32 | 36.07  | 12.973 | 33.49 | 1328.17    | 1891.71       |
| sglang | 1  | 425.10 | 1.101  | 32.94 | 1325.81    | 1546.14       |
| sglang | 8  | 95.23  | 4.915  | 32.34 | 1341.86    | 1722.72       |
| sglang | 16 | 64.65  | 7.239  | 32.37 | 1339.72    | 1851.79       |
| sglang | 32 | 49.10  | 9.531  | 32.86 | 1338.11    | 2143.66       |

## Gap (sglang relative to vLLM)

| N  | throughput sglang/vllm | StreamLAAL_CA delta (sglang - vllm) |
|----|------------------------|-------------------------------------|
| 1  | 1.34x (sglang faster)  | -113 ms (sglang better)             |
| 8  | 0.96x                  | -6 ms (tie)                         |
| 16 | 0.85x                  | +74 ms (sglang worse)               |
| 32 | 0.73x (vllm +36%)      | +252 ms (sglang worse, +13%)        |

## Findings

1. **Quality parity.** BLEU ~33 both engines (sglang ~0.6 lower, within decode
   noise). StreamLAAL (theoretical) ~1330-1348 both, flat — confirms identical
   policy and a fair A/B. The gap is purely a **systems/scheduling** gap.
2. **sglang wins single-stream.** N=1: sglang 1.10 vs vLLM 0.82 seg/s (+34%),
   StreamLAAL_CA 113 ms lower. Lighter per-request path / faster single decode.
3. **vLLM scales much better.** Scaling efficiency (seg/s at N=32 / N=1):
   vLLM 15.8x vs sglang 8.7x over 32x concurrency. Crossover ~N=8.
4. **The gap widens monotonically with N** (the #760 prefill-fragmentation story):
   at N=32 vLLM delivers +36% throughput and 252 ms (-13%) lower computation-aware
   latency. sglang's StreamLAAL_CA inflates +598 ms (N=1->32) vs vLLM's +233 ms.

**Optimization target:** close the N>=16 throughput + StreamLAAL_CA gap (concurrent
prefill batching/coalescing under load), without regressing the N=1 advantage.

## M1 A/B: prefill-coalesce (issue #760 PR) vs stock sglang

sglang-omni branch `pr/qwen3-omni-prefill-760` (isolated worktree), knobs:
`SGLANG_OMNI_PREFILL_COALESCE_MS=25`, `POLL_MS=1`, `TP_IDLE_BROADCAST_SKIP=1`,
`LOG_PREFILL_STATS=1` (MIN=2, TARGET=32, MAX_TOKENS=default). Same harness/data.

| N  | seg/s stock | seg/s M1 | CA stock | CA M1  | BLEU M1 |
|----|-------------|----------|----------|--------|---------|
| 1  | 1.101       | 1.086    | 1546.1   | 1548.8 | 32.94   |
| 8  | 4.915       | 5.040    | 1722.7   | 1707.0 | 32.39   |
| 16 | 7.239       | 7.364    | 1851.8   | 1828.2 | 32.80   |
| 32 | 9.531       | 9.407    | 2143.7   | 2145.9 | 32.46   |

**Coalescing demonstrably engages:** prefill `avg_batch` rose from a stock 1.00 to
~3.0-4.3 (peak 4.33, max_batch 25) at N=16/32 -- fragmentation is really reduced.
**But it does not close the gap:** +1.7-2.5% throughput and ~1% StreamLAAL_CA at
N=8/16, ~0 at N=32; N=1 unaffected (MIN gate works as designed). vs vLLM at N=32:
still +38% throughput / +254 ms CA in vLLM's favor.

### Conclusion / pivot for M2

Prefill de-fragmentation is **necessary but not sufficient**. Two reasons it can't
be the main lever here:
1. With streaming KV reuse each prefill is tiny (`#new-token ~30-50`), so prefill
   compute is a small fraction of a step -- batching cheap prefills saves little.
2. The coalesce window only engages on the idle->prefill transition
   (`running_batch==0`); under heavy concurrency the scheduler is rarely fully
   idle, so it can't fire often enough to matter.

The dominant gap is in the **decode phase / prefill-decode interleaving**: vLLM
sustains higher decode throughput under 32 concurrent streams and mixes prefill
into decode steps (continuous batching) instead of stalling decode. M2 should
target that, validated against this same StreamLAAL/BLEU harness.

## N=32 profile (sglang scheduler telemetry, fresh full-468 run, 51.2s / 9.13 seg/s)

Parsed `scheduler_metrics_mixin` Decode/Prefill lines:

| metric                | value                          |
|-----------------------|--------------------------------|
| decode occupancy      | mean **17.6 / 32 = 55%** (p10=2, p90=30) |
| gen throughput        | mean **~204 tok/s** (p90 266)  |
| KV token usage        | **~0.01** (GPU ~idle)          |
| prefill steps         | **~498 in 51s (~10/s)**, #new-token p50=3 |
| prefill `#running-req`| mean **18.5** (prefills fire WHILE 18-30 decodes run) |

**Root cause:** the streaming protocol re-prefills every 1.92s audio chunk for
every session => ~10 tiny prefills/s, each a separate scheduler step that stalls
the running decode batch. Decode occupancy sits at ~55% and gen throughput at
~200 tok/s while the GPU is idle (KV ~0%). It is a **step-structure / continuous-
batching** problem, not prefill batch size or memory.

**Why M1 cannot fix it:** `_coalesce_prefill_window` only engages when
`running_batch == 0`. At N=32 the decode batch is almost never empty, so M1 never
fires in steady state (its avg_batch=4.3 was drain-window only).

## M2 candidates (in impact order)

1. **Mixed prefill+decode batches (chunked-prefill piggyback), vLLM-style** — run
   the small chunk prefill in the same forward step as decode so it never stalls
   the batch. Biggest lever; matches vLLM's mechanism directly.
2. **Relax M1's gate** — coalesce concurrent prefills that arrive *during* decode
   (drop the `running_batch==0` requirement, keep the token-budget OOM guard).
   Smaller change to existing code; cuts the ~10 prefill-steps/s.
3. **Per-step overhead** — decode already uses CUDA graph; prefills do not. Reduce
   Python scheduler + TP-broadcast cost per prefill step.

## M2 experiment results: prefill-side levers do NOT close the gap

Both prefill-side optimizations were tested with the same N=32 profile (full 468):

| config                 | seg/s | decode occupancy | gen tok/s |
|------------------------|-------|------------------|-----------|
| stock                  | 9.13  | 55% (17.6/32)    | ~204      |
| + M1 prefill-coalesce  | 9.41  | (drain-only)     | -         |
| + mixed-chunk (cps2048)| 9.26  | 37% (11.9/32)    | ~104      |
| **vLLM**               | 12.97 | (full batch)     | -         |

Neither helps. Conclusion: **the gap is not prefill handling.**

## Refined root cause: decode-batch under-occupancy from the multi-stage pipeline

Signature across the sweep:
- sglang is FASTER single-stream (N=1: 1.10 vs 0.82 seg/s) but SLOWER at N=32
  (9.13 vs 12.97) -> a pure scaling problem, not a per-request-speed problem.
- At N=32 the thinker decode batch is only ~50% occupied while the GPU is idle
  (KV usage ~0%, gen ~200 tok/s). The thinker is starved, not compute-bound.

sglang-omni runs a **multi-stage pipeline** (preprocessing -> audio_encoder ->
mm_aggregate -> thinker -> detok) with shm/nccl relay between stages, each in its
own process. Under concurrency, at any instant ~half the in-flight requests are in
the non-thinker stages or in relay, so the expensive thinker stage never sees a
full batch -> ~50% occupancy -> ~half the achievable throughput. vLLM is
monolithic (encoder + LLM in one process, one continuous batch), so it keeps the
decode batch full. This matches every observation (fast N=1, ~50% thinker
occupancy, idle GPU, prefill-side fixes ineffective).

## Real M2 levers (deeper; need the per-turn critical-path profile to confirm)

1. **Overlap pipeline stages with thinker decode** so requests don't leave the
   thinker batch while their next chunk is pre/encoded (the ~50% occupancy loss).
2. **Cut per-turn relay latency** (shm vs nccl vs nixl relay-backend; stage
   colocation) on the streaming critical path.
3. Confirm with a stage-level timing breakdown (client per-turn latency +
   per-stage residency) vs vLLM's monolithic path before investing.

## STAGE-LEVEL PROFILE (confirmed) — per-turn critical path, N=1 vs N=32

Env-gated per-turn round-trip capture in `remote_omni_agent._post` (one `.lat`
file/worker, `REMOTE_OMNI_LAT_DIR`), stock shm sglang, same model/GPUs/protocol.

| concurrency | per-turn RTT mean | p50 | p90 | p99 | seg/s | scaling |
|-------------|-------------------|-----|------|------|-------|---------|
| N=1  (no contention) | **232 ms** | 220 | 268  | 418  | 1.046 | 1.0x |
| N=32 (under load)    | **822 ms** | 760 | 1207 | 1857 | 8.705 | 8.3x |

**The per-turn latency inflates 3.54x (232 -> 822 ms); that inflation IS the
entire scaling gap.** Closed-loop Little's law: throughput = concurrency /
per-turn-latency. If latency held flat at 232 ms, N=32 -> 32x scaling. It rises
to 822 ms, so scaling -> 32 / 3.54 = **9.0x predicted == 8.3x measured.** No
other term is needed: the gap is 100% per-turn latency blowup under load.

Server-side telemetry for the same N=32 window corroborates *where* the time goes:
- decode batch **14.8 / 32 = 46% occupied**; prefill `#running-req` mean **15.5 /
  32 (~48%)**. Both thinker phases run ~half-full even with 32 clients offered.
- i.e. at any instant **~53% of in-flight requests are NOT in the thinker** —
  they sit in preprocessing / audio_encoder / mm_aggregate / detok / shm relay.

Decompose the 822 ms: the uncontended path (encode+thinker+relay) costs ~232 ms
(the N=1 number); the extra **~590 ms is pure queue/stage-residency** waiting to
re-enter the half-full thinker batch. That matches 46% occupancy exactly.

**This confirms the multi-stage-pipeline root cause** over any prefill story:
the thinker is starved because half the offered concurrency is always parked in
upstream stages + relay, so (a) the batch never fills (46-48%) and (b) each
turn's round-trip triples. vLLM (monolithic, one continuous batch) has nowhere
for requests to park, so its per-turn latency stays far flatter and it scales
~16x. Prefill-side levers (M1, mixed-chunk) cannot touch this — confirmed null.

### relay-backend A/B: not testable on this topology
Tried `RELAY_BACKEND=nccl` to cut relay latency: server **hangs at startup**.
Cause: heterogeneous GPU map (audio_encoder on 1 GPU, thinker TP=2 across both)
makes NCCL relay deadlock during init (ranks never all join one comm). `nixl`
needs extra runtime not in the image. **`shm` is the only viable relay** for the
encoder-1GPU + thinker-TP2 layout, so relay backend can't be swapped to shrink
the ~590 ms here. The lever must instead be **stage overlap** (keep a request in
the thinker batch while its next chunk pre/encodes) or **encoder colocation**, so
thinker occupancy rises from ~47% toward vLLM-like full batches.

### Net for M2
The actionable target is occupancy, not prefill: **overlap the encoder/relay
stages with thinker decode** (and/or colocate stages to kill the relay hop) so
the thinker sees ~32-way batches and per-turn RTT stays near its 232 ms floor.
Expected ceiling: ~3.5x less per-turn latency at N=32 -> close most of the
vLLM throughput gap. This is an architectural change to the pipeline relay/
scheduling, scoped as the real M2.

## M2 attempt 1 — within-thinker CPU/GPU overlap: REGRESSES (rules out the cheap lever)

Deployment confirmed from the launcher: thinker is its **own process**, TP=2 on
cuda:0,1; audio_encoder pinned to `--encoder-gpu 1` => **shares cuda:1 with
thinker-rank-1**. The thinker bootstrap (`qwen3_omni/bootstrap.py`) builds
`OmniScheduler` with neither `enable_overlap` nor `enable_async_decode`, so it
runs `_event_loop_normal` (no CPU/GPU overlap). Anomaly motivating the test:
~200 tok/s over ~15 running reqs = ~13 tok/s/req, very slow for 30B-A3B (3B
active) -> suspected dead GPU time between forwards.

Wired an env-gated opt-in (`SGLANG_OMNI_THINKER_OVERLAP=overlap|async`, default
off) and A/B'd at N=32 (full 468), same server/GPUs/protocol:

| config           | seg/s | per-turn RTT mean | p50 | p99  |
|------------------|-------|-------------------|-----|------|
| stock (normal)   | **8.71** | **822 ms**     | 760 | 1857 |
| + overlap loop   | 7.76 (**-11%**) | 916 ms (**+11%**) | 819 | 3398 |

**Overlap makes it worse, both throughput and latency.** This is the smoking gun
for the real cause: enabling overlap keeps the thinker GPU (cuda:0+1) busier,
which **steals cuda:1 cycles from the audio_encoder** on the critical feeding
path. Because the thinker is TP=2, every encoder kernel on cuda:1 stalls BOTH
thinker ranks -> the pipeline feeds slower -> net regression. So the ~13
tok/s/req and 46% occupancy are NOT within-thinker host overhead; they are
**encoder<->thinker GPU contention on the shared cuda:1** inherent to the
split-process pipeline. (Within-thinker overlap is the wrong lever — confirmed
null/negative, same class as M1 and mixed-chunk.)

### Real M2 = encoder colocation / decontention (the only thing left)
vLLM is monolithic: the audio encoder forward is part of the same engine's
prefill on the same CUDA stream/scheduler, so it never competes as a separate
process for the GPU. sglang-omni splits the encoder into its own process pinned
to a thinker GPU -> cross-process kernel serialization on cuda:1. With only 2
GPUs (both needed for the 30B TP=2 weights) the encoder cannot get a free GPU,
so the fix must be **fold the audio_encoder into the thinker engine** (same
process + stream, vLLM-style) or otherwise decontend cuda:1 (MPS / stream
priority). That is a real architectural change, scoped as M2 proper.

## M2 attempt 2 — DEDICATED encoder GPU (3-GPU test): refutes GPU contention

Node has GPUs 4-7 free. Ran thinker TP=2 on cuda:0,1 (host 4,5) + audio_encoder
on its OWN cuda:2 (host 6) -> zero encoder<->thinker GPU contention. Full 468,
N=32, vs the 2-GPU shared baseline:

| config (N=32)                       | seg/s | RTT mean | decode occ | gen tok/s |
|-------------------------------------|-------|----------|------------|-----------|
| baseline (2-GPU, enc shares cuda:1) | 8.71  | 822 ms   | 46%        | 199       |
| 2-GPU + within-thinker overlap      | 7.76  | 916 ms   | -          | -         |
| **3-GPU, dedicated enc, normal**    | 8.29  | 861 ms   | **62%**    | 199       |
| **3-GPU, dedicated enc, +overlap**  | 8.40  | 860 ms   | **64%**    | 222       |
| **vLLM (2-GPU)**                    | 12.97 | ~552 ms  | full       | -         |

**Dedicating a GPU to the encoder raised decode occupancy 46% -> 62-64% but
throughput and per-turn RTT did NOT move.** Occupancy up, throughput flat =>
the thinker is NOT the constraint, and encoder GPU contention was NOT the cause.
On the dedicated GPU, overlap is neutral (8.40 vs 8.29) instead of regressing,
confirming the earlier regression was purely encoder-cycle theft on the shared
cuda:1.

### THE root cause (localized): host-side per-turn critical path, not GPU
Under N=32 load (3-GPU dedicated run), measured live:
- thinker GPUs: ~57-75% util (NOT saturated); encoder GPU ~4% (trivial).
- the 3 server procs (1 "pipeline" + 2 thinker TP ranks) each sit at ~100% CPU.
- throughput is rigidly `32 workers / per-turn-RTT`; RTT ~820-860 ms across
  EVERY config -> ~39 turns/s -> ~8.3-8.7 seg/s, invariant.

Every GPU-side lever tried is null because none touch the host path:
M1 prefill-coalesce, mixed-chunk, within-thinker overlap, dedicated encoder GPU.
The ~590 ms per-turn inflation (232 ms N=1 -> 822 ms N=32) is **host-side
serialization in the multi-process pipeline**: each turn traverses preprocess ->
audio_encoder dispatch -> mm_aggregate -> cross-process shm relay -> thinker ->
relay back -> detok, and the single GIL-bound "pipeline" process (preprocess +
encoders + aggregate + detok + HTTP ingress/egress for all 32 streams) plus the
per-rank thinker scheduler CPU cap the turn rate at ~39/s. vLLM is monolithic
(one process, continuous batch, no cross-process relay), so its RTT stays ~552 ms
and it scales to 12.97 seg/s.

### Conclusion: the vLLM gap is architectural host-path latency, not scheduling
No prefill/decode/GPU scheduling knob can close it. The only levers that can:
1. **Cut relay hops per turn** (colocate preprocess+encoder+aggregate+detok with,
   or into, the thinker engine; fewer process boundaries on the streaming path).
2. **Remove the GIL bottleneck** in the "pipeline" process (parallelize per-stream
   host work; move preprocessing/detok off the single event loop).
3. **Reduce cross-process shm relay latency** on the per-turn path.
All are substantial structural changes to sglang-omni's pipeline runtime, not
scheduler tweaks. (M1 remains a legit small de-fragmentation PR but is not the
vLLM-gap fix.)

## PER-STAGE RESIDENCY (built-in request-event profiler) — localizes the host path

Used sglang-omni's own request-level profiler (`POST /start_request_profile`,
merge with `python -m sglang_omni.profiler`). Stock 2-GPU config, N=32 load
(profiling overhead drops throughput to ~6.7 seg/s, so ABSOLUTE ms are inflated;
the RELATIVE split is the signal). 636 requests.

### Stage breakdown (avg residency per request)
| stage (input_received -> complete) | avg ms | share | notes |
|------------------------------------|--------|-------|-------|
| **thinker**                        | **701**| **64%**| ingest + prefill + decode |
| audio_encoder                      | 170    | 15%   | encode compute only 86 ms; rest queue |
| mm_aggregate                       | 170    | 15%   | identity pass-through => ~all queue |
| preprocessing                      | 34     | 3%    | compute 9.6 ms |
| decode (detok)                     | 3      | <1%   | |

### Hop (relay) breakdown — NEGLIGIBLE
| hop | avg ms |
|-----|--------|
| mm_aggregate -> thinker (shm relay) | **22** |
| thinker -> decode | 2.7 |
| preprocessing -> mm_aggregate | 1.8 |
| others | <0.1 |
| **total relay per turn** | **~26 ms (~2%)** |

**=> "cut relay hops" is REFUTED: the shm relay is only ~2% of the per-turn path.**

### Thinker residency split (queue vs compute)
| interval | avg ms |
|----------|--------|
| input_received -> queue_enter (request INGEST by busy loop) | 139 |
| queue_enter -> prefill_start (admission QUEUE WAIT)         | **10** |
| prefill_start -> complete (prefill+decode, CPU-stretched)   | ~550 |

Admission queue is tiny (10 ms) -> NOT a prefill backlog. The cost is (a) ~139 ms
for the busy thinker loop to even ingest the incoming request (it finishes a
batch before returning to recv_requests) and (b) decode stretched by per-step
Python overhead (thinker pegged at 100% CPU while its GPU sits at 60-75%).

### Final localization
The per-turn path is host/CPU-bound at two places, NOT the relay:
1. **Thinker scheduler loop (64%)** — 100% CPU, GPU 60-75%; per-step Python
   overhead + slow request ingestion. Lever: cut per-step host cost (CUDA-graph
   decode coverage, leaner scheduler step, faster recv/build), NOT relay.
2. **GIL-bound "pipeline" process (~30%)** — audio_encoder + mm_aggregate carry
   ~170 ms residency each that is mostly queue (aggregate is identity yet 170 ms),
   because preprocess+encoders+aggregate+detok+HTTP for all 32 streams share one
   process/event loop/GIL. Obvious lever = split into separate processes — but that
   was TESTED and REGRESSES (see "de-GIL A/B" below): cross-process serialization
   of the multimodal payloads costs more than the GIL. This ~30% is NOT relievable
   by re-partitioning.

vLLM avoids BOTH (monolithic, one optimized C++/CUDA engine, no per-stage Python
hops), which is the entire ~552 ms vs ~822 ms per-turn-RTT difference.

---

## De-GIL A/B (per-stage processes) — aries, within-node, N=32  [NEGATIVE]

Tested the "split the GIL-bound pipeline process" lever proposed above.
`PER_STAGE_PROCESSES=1` gives preprocessing / image_encoder / audio_encoder /
mm_aggregate / decode each its own OS process (the speech-pipeline topology),
vs stock (all five in one "pipeline" process). Same idle node (`aries`, RTX
A6000), back-to-back, thinker TP=2 on the same 2 GPUs, 468 segments, 1850 turns.

| topology | wall s | seg/s | BLEU | StreamLAAL | StreamLAAL_CA | RTT mean | RTT p50 | RTT p90 |
|----------|--------|-------|------|-----------|---------------|----------|---------|---------|
| stock (1 pipeline proc)    | 62.7  | **7.46** | 32.0 | 1316 | 2421 | 946 ms  | 809 ms  | 1392 ms |
| per-stage procs (de-GIL)   | 115.4 | 4.06     | 32.4 | 1328 | 3332 | 1824 ms | 1702 ms | 2766 ms |

**Result: de-GIL REGRESSES −46% throughput / +2.1x per-turn RTT, quality flat.**
On-node stock 809 ms p50 RTT matches the taurus 822 ms baseline (node comparable;
stock 7.46 seg/s here vs 8.71 on taurus is client-CPU/measurement variance — the
de-GIL delta is a clean within-node A/B).

Why it regresses: splitting converts in-process stage handoffs into cross-process
shm relays that serialize large multimodal payloads (audio features, encoder and
merged thinker-input embeddings) and adds more serial single-threaded stage
processes. That cost dominates the GIL serialization it removes. The monolithic
"pipeline" process is already the better partition for speech->text => the
host-side cost is structural; the direction that could help is FEWER stage
boundaries (more colocation / more monolithic), not more.

Caveat: client (32 SimulEval procs) and server shared the node in both arms, so
de-GIL's extra stage processes add some host-CPU contention — the exact −46% may
be amplified by that. The direction (de-GIL does not help; it regresses) is
robust (same client setup both arms; stock RTT matches the taurus baseline).

Repro:

    PER_STAGE_PROCESSES=1 GPUS=0,1 PORT=8132 bash eval/servers/serve_sglang_qwen3omni.sh
    REMOTE_OMNI_LAT_DIR=/tmp/lat ENGINE=sglang BASE_URL=http://127.0.0.1:8132 \
      OUT_ROOT=/tmp/degil NLIST=32 bash eval/run_sweep.sh
