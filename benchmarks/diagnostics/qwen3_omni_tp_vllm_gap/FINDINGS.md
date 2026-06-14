<!-- Evidence companion to docs/sglang_omni_tp_concurrency_issue.md (#760) -->
<!-- Measured follow-up: corrects the prefill-fragmentation hypothesis with data. -->

# Qwen3-Omni TP=2 concurrency: measured vLLM gap and root cause (update to #760)

## TL;DR

I built a SimulEval-native StreamLAAL/BLEU harness and A/B'd **stock sglang-omni
vs vLLM** for *pure* `Qwen3-Omni-30B-A3B-Instruct` (no-RAG, en->zh) on the **same
2 GPUs**, TP=2, max 32 concurrent streaming sessions. Findings:

1. **Quality is at parity** (BLEU ~33, StreamLAAL ~1.33 s both engines). The gap
   is purely systems/throughput.
2. **The gap is real and grows with concurrency:** at N=32 vLLM does **12.97
   seg/s vs sglang 8.71** (vLLM +49%), with lower computation-aware latency.
3. **The gap is NOT prefill fragmentation, and NOT any GPU-side knob.** M1
   prefill-coalesce, mixed-chunk, thinker CPU/GPU overlap, and even giving the
   audio encoder its **own dedicated GPU** are all null on throughput; and
   **splitting the pipeline into one process per stage actively regresses it
   (−46%)** — see §2/§4.
4. **Root cause = host-side per-turn latency** in the multi-process pipeline.
   Per-turn round-trip inflates 232 ms (N=1) -> 822 ms (N=32) and that 3.5x is
   100% of the scaling gap. Per-stage profiling: **thinker stage 64%, encoder+
   aggregate queue 30%, cross-process relay only ~2%.** The shared "pipeline"
   process is GIL-serialized; the thinker presents as 100% CPU but a step-phase
   micro-profile (§5) shows that is a GPU-sync wait, not host compute -- the
   thinker is **GPU-forward-bound**. vLLM's monolithic engine avoids both the
   relay and the serialization.

So #760's prefill-coalescing work is a legitimate small de-fragmentation PR, but
it is **not** the lever that closes the vLLM gap. The gap is architectural.

## Methodology

- **Harness:** SimulEval streaming agent -> raw OpenAI engine (vLLM `serve` vs
  sglang-omni server), one agent, identical streaming policy. Scored with FBK
  `stream_laal_term.py` (no glossary): BLEU (sacrebleu zh, char), StreamLAAL,
  StreamLAAL_CA. Code: `eval/streaming_sst/` in rasst-demo.
- **Workload:** ACL6060 dev, en->zh, 468 segments, each an independent streaming
  instance; concurrency N = number of parallel SimulEval workers.
- **Model:** pure `Qwen3-Omni-30B-A3B-Instruct` (no-RAG), text output.
- **Hardware:** 2x 48 GB GPUs, TP=2, `mem_fraction_static=0.75`,
  `max_running_requests=32`, `seg=1920 ms`, `max_new_tokens=40`.

## 1. Quantified gap (stock sglang vs vLLM)

| engine | N | seg/s | BLEU | StreamLAAL | StreamLAAL_CA |
|--------|---|-------|------|-----------|---------------|
| vllm   | 1 | 0.82  | 33.2 | 1340 | 1659 |
| vllm   | 32| **12.97** | 33.5 | 1328 | 1892 |
| sglang | 1 | **1.10** | 32.9 | 1326 | 1546 |
| sglang | 32| 8.71  | 32.9 | 1338 | 2144 |

- sglang is **faster single-stream** (N=1: 1.10 vs 0.82 seg/s) but **scales worse**
  (8.7x vs 15.8x over 32x concurrency); crossover ~N=8.
- Quality parity confirms a fair A/B; the gap is systems-only.

## 2. Systematic negative results (what does NOT close the gap)

All at N=32, full 468, same harness:

| change | seg/s | per-turn RTT | verdict |
|--------|-------|--------------|---------|
| stock sglang | 8.71 | 822 ms | baseline |
| M1 prefill-coalesce (#760 PR) | 9.41 | — | +marginal, gap intact |
| mixed-chunk (chunked_prefill) | 9.26 | — | null |
| thinker CPU/GPU overlap (`_event_loop_overlap`) | 7.76 | 916 ms | **regresses** |
| **dedicated encoder GPU** (3-GPU) | 8.29 | 861 ms | occ 46%->62%, **tput flat** |
| dedicated encoder GPU + overlap | 8.40 | 860 ms | null |
| **per-stage processes (de-GIL split)** | 4.06† | 1702 ms† | **regresses −46% / +2× RTT** |
| **vLLM** | **12.97** | ~552 ms | target |

Key tell: a dedicated encoder GPU raised thinker decode-batch occupancy from 46%
to 62% **yet throughput and latency did not move** — proving the thinker/GPU is
not the constraint.

† Within-node A/B on the idle `aries` node (RTX A6000, same GPU model). On-node
stock baseline is **7.46 seg/s / 809 ms p50 RTT** (the 809 ms matches the 822 ms
taurus baseline, confirming the node is comparable); the de-GIL split gives
**4.06 seg/s / 1702 ms p50 RTT** at quality parity (BLEU 32.0→32.4, 1850 turns
each). Splitting every non-thinker stage into its own process — the speech-pipeline
topology — **doubled per-turn RTT**. See §4 for why.

## 3. Root cause: host-side per-turn latency

Throughput is rigidly `32 workers / per-turn-RTT`, and RTT is stuck at ~820-860
ms across every config above. Decomposition with sglang-omni's own request-event
profiler (`/start_request_profile` + `python -m sglang_omni.profiler`), N=32:

**Per-stage residency (relative; profiling inflates absolute ms):**

| stage | avg residency | share |
|-------|---------------|-------|
| thinker (ingest + prefill + decode) | 701 ms | **64%** |
| audio_encoder (compute 86 ms + queue) | 170 ms | 15% |
| mm_aggregate (identity -> ~all queue) | 170 ms | 15% |
| preprocessing | 34 ms | 3% |
| **all cross-process relay hops** | **~26 ms** | **~2%** |

**Thinker split:** admission queue is only 10 ms (not a prefill backlog); ~139 ms
is the busy scheduler loop just *ingesting* the request, and the rest is decode
stretched by per-step Python overhead. During load the thinker process sits at
**100% CPU with its GPUs at 60-75%**, and the shared "pipeline" process
(preprocess+encoders+aggregate+detok+HTTP for all 32 streams) is GIL-serialized
(an identity aggregate stage accrues 170 ms of queue). **But that queue is not
relievable by de-GIL'ing into separate processes — splitting regresses (§4).**

## 4. Implication / corrected direction

- **Relay-cutting is refuted** (~2% of the path). shm/nccl/nixl relay backend is
  not the lever (nccl/nixl also won't init on the encoder-1GPU + thinker-TP2 map).
- **De-GIL by process-splitting is refuted (measured).** Giving every non-thinker
  stage its own process — the obvious "remove the GIL" fix, and the topology the
  speech pipeline already uses — **regressed N=32 throughput 46% (7.46→4.06 seg/s)
  and doubled per-turn RTT (809→1702 ms p50)** at quality parity. The monolithic
  "pipeline" process is already the better partition for this workload: splitting
  turns cheap in-process stage handoffs into cross-process relays that must
  serialize large multimodal payloads (audio features, encoder/merged embeddings)
  and inserts more serial single-threaded stage processes — costing far more than
  the GIL serialization it removes. So the host-side queue in §3 is **not**
  relievable by re-partitioning the pipeline; if anything the fix points the other
  way (fewer stage boundaries / more colocation, i.e. *more* monolithic).
- A candidate host-side lever was the **thinker scheduler step cost** (broaden
  CUDA-graph decode coverage, leaner per-step Python, faster request ingest).
  **A step-phase micro-profile refutes this (§5):** per decode step the host is
  only ~5 ms (graph hit 100%, backends already flashinfer); the ~30 ms/step is the
  GPU forward. The one cheap GPU-side win found is **custom all-reduce on NVLink
  (+6.5%, BLEU parity)**; async-decode is the wrong lever (~3% ceiling).
- Net: the gap looks **structural**. vLLM is monolithic (one engine, continuous
  batch, no per-stage Python hops, no payload serialization between stages) — that
  is the whole ~552 ms vs ~810 ms per-turn-RTT difference. sglang-omni's
  multi-stage design buys modularity and the full omni talker/code2wav path at a
  host-path cost that, for speech→text, is hard to claw back without making the
  text path more monolithic.

## 5. Update: the thinker decode is GPU-forward-bound, not host-bound (micro-profile)

The "thinker at 100% CPU" in §3 motivated an async-decode plan (overlap host
post-processing with the next GPU forward). Before building it I added env-gated
**per-step phase timing** to the scheduler/runner (`SGLANG_OMNI_PHASE_PROFILE`,
and `SGLANG_OMNI_PHASE_SYNC` to force a `cuda.synchronize()` after the forward so
GPU time lands in `forward` instead of the later `.tolist()`), N=32:

| decode step (mean ms) | forward | finalize | build | recv+sched+stream+proc | step total |
|-----------------------|---------|----------|-------|------------------------|------------|
| no-sync (default)     | 1.1     | **25.1** | 1.9   | ~2.3                   | 30.2       |
| **sync (true GPU)**   | **30.2**| 0.2      | 2.1   | ~2.6                   | 35.0       |

Forcing the sync moves **25 ms from `finalize` into `forward`**: the 25 ms read as
host post-processing is `.tolist()` **blocking on the in-flight decode forward**.
True split per decode step: **~30 ms GPU forward + ~5 ms host**, of which only
**~1 ms (`host_post`) is overlappable** by async-decode. Prefill is the same
story: forward **72-78 ms GPU**, host ~11-17 ms (recv-dominated ingest). Config is
already optimal -- `max_running_requests=32` lands, **100% CUDA-graph hit**
(capture_bs=[1,2,4,8,12,16,24,32]), attention+sampling already flashinfer; mean
decode batch ~18-20/32 (never >32, so cuda_graph_max_bs=48 is useless).

**Consequences**
- **async-decode is the wrong lever** -- ceiling ~3% (overlaps ~1 ms of a 35 ms
  step), not the hoped ~20%. The earlier "host-side bottleneck" was the GPU-sync
  wait misread as CPU. Cancelled.
- A 30 ms decode forward for a 3B-active MoE is dominated by **memory-bound expert
  weight loading** at bs~18; the only cheap GPU-side knob is the TP all-reduce.

### Custom all-reduce (the one cheap GPU-side win) -- confirmed

The launcher defaulted `disable_custom_all_reduce=True`, forcing NCCL for the
per-layer all-reduces every step. Topology here is **GPU0-GPU1 = NV4 (NVLink)**,
so custom (P2P/NVLink) all-reduce is safe. Within-node A/B on `aries`, 3 N=32
sweeps/mode (first run cold, server warms across runs):

| mode (N=32)            | seg/s runs            | mean             | BLEU  |
|------------------------|-----------------------|------------------|-------|
| stock (NCCL)           | 7.43 / 7.98 / 8.10    | 7.835 +/-0.293   | ~32.7 |
| **custom-AR (NVLink)** | 7.73 / 8.72 / 8.58    | **8.343 +/-0.440** | ~32.5 |

**+6.5% mean** (warm-vs-warm ~+7%, no overlap), **BLEU parity**, p50 RTT
867->830 ms. The decode GPU forward itself barely moves (memory-bound; all-reduce
is a small fraction) -- the gain is lower host all-reduce launch cost + faster
prefill all-reduce. Small but real and free (flip the flag; env-gated via
`SGLANG_OMNI_DISABLE_CUSTOM_AR`). It narrows ~11% of the aries gap to vLLM --
**not** a gap-closer, consistent with the structural conclusion in §4.

## 6. P1: thinker forward attribution -- decode is memory-bound, prefill token-bound

Following §5 (the cost is the forward, not host), I bucketed **true-GPU forward
time by batch size** (`SGLANG_OMNI_PHASE_SYNC=1`, new `[fwd-by-bs]` log),
sweeping N=8/16/24/32 to cover the decode-bs range in one server lifetime.

**Decode forward vs bs** (at CUDA-graph-captured sizes; true GPU ms):

| bs        | 1    | 2    | 4    | 8    | 16   | 24   | 32   |
|-----------|------|------|------|------|------|------|------|
| forward ms| 10.1 | 12.7 | 16.3 | 18.0 | 22.5 | 24.5 | 25.9 |

Forward grows **10 -> 26 ms while tokens grow 1 -> 32** -- strongly **sublinear**.
That is the signature of a **memory-bound** decode: a ~10 ms fixed per-step floor
(30B-MoE expert-weight load + 2 all-reduce/layer + launch) with a small marginal
token cost. Per-token efficiency is **~12x better at bs=32 than bs=1** (0.81 vs
10.1 ms/token). So **marginal tokens in the decode batch are nearly free.**

**Prefill forward is flat ~60-65 ms** regardless of request count (token-bound;
chunked at 4096), mean 63.8 ms. Over the sweep prefill is the **larger GPU
consumer** (2117 steps x 63.8 ms = 135 s GPU vs decode 5621 x 20.0 ms = 112 s),
because speech-context prefill is heavy -- inherent to streaming SST.

**The decode batch is under-filled (P2).** Clean N=32-only baseline (no sync,
realistic seg/s ~7.2-7.9): mean decode bs **17.7/32** over the run, **~23/32 in
steady state** (delta of a mid-run window) -- i.e. **~55-72% occupancy**. The
cause is **prefill stealing scheduler iterations**: the step split is **decode
1256 : prefill 896**, so **42% of iterations are prefill** (each ~88 ms wall:
62-70 ms GPU forward + ingest, vs ~33 ms decode), and decode/prefill run as
*separate* iterations -- during each prefill step no decode advances, so the
decode batch drains. Throughput scales **4.07 (N8) -> 6.40 (N16) -> 7.76 (N24)
-> 7.97 (N32) seg/s** -- diminishing returns well before the floor is amortized.
Since P1 makes marginal decode tokens nearly free, raising occupancy toward 32
(decode-priority scheduling or true prefill/decode fusion) is the live lever;
note `--enable-mixed-chunk` was null in §3, so fusion needs deeper wiring, not
just the flag.

**Consequences (sets P1/P2 levers):**
- CUDA graph is **already optimal** (100% hit, capture to bs=32) -- not a lever.
- all-reduce is a **small fraction** of the floor (§5 A/B) -- already shipped (P0).
- The memory-bound floor means the cheap throughput win is **filling the decode
  batch (P2)**, not faster compute: P1 *proves* the batching lever. Lowering the
  floor itself needs **FP8/weight-traffic** reduction -- a large, quality-risky
  change, deferred.

## Reproduction

```bash
# servers (TP=2, pure Qwen3-Omni-30B-A3B, no-RAG)
GPUS=2,3 PORT=8101 bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh
bash eval/streaming_sst/servers/serve_vllm_qwen3omni.sh

# concurrency sweep + StreamLAAL/BLEU scoring
bash eval/streaming_sst/run_sweep.sh

# per-stage residency (built-in profiler)
curl -X POST :8101/start_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"r","event_dir":"/path/events"}'
#   ... run an N=32 load ...
curl -X POST :8101/stop_request_profile -H 'Content-Type: application/json' -d '{}'
python -m sglang_omni.profiler /path/events --format table

# §5 step-phase micro-profile (env-gated, prints [step phases] per decode/prefill)
SGLANG_OMNI_PHASE_PROFILE=1 GPUS=0,1 PORT=8101 \
  bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh   # add SGLANG_OMNI_PHASE_SYNC=1 for true-GPU split

# §6 forward-vs-batch-size curve ([fwd-by-bs]) + decode bs histogram ([decode stats])
SGLANG_OMNI_PHASE_PROFILE=1 SGLANG_OMNI_PHASE_SYNC=1 SGLANG_OMNI_DECODE_STATS=1 \
  GPUS=0,1 PORT=8101 bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh
#   then sweep NLIST="8 16 24 32" to cover the decode-bs range

# §5 custom all-reduce A/B (NVLink): 1=NCCL (stock), 0=custom
SGLANG_OMNI_DISABLE_CUSTOM_AR=0 GPUS=0,1 PORT=8101 \
  bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh
```

Full data, all A/B runs, and analysis: `rasst_eval/runs/COMPARISON.md`.
