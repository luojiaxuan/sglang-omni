# H100 findings - thinker decode, general online-FP8 (W8A8) vs BF16

H100 decode-forward A/B for the **general online-FP8 (W8A8)** MoE path vs the
BF16 baseline. This run used the same branch/engine path as the B200 result, and
the same text steady-decode harness (`decode_load_text.py`, concurrency 32,
`max_tokens=256`, PHASE_SYNC enabled).

> **Headline (H100): online FP8 W8A8 halves thinker weight memory, but does not
> produce the expected decode-forward speedup.** With a clean matched memory
> contract (`thinker_memory_fraction=mem_fraction_static=0.55`), FP8 improves
> bs=32 forward from about 5.8 ms to 5.6-5.7 ms (roughly 2-3%), and the total
> decode step stays essentially flat at about 7.4 ms. This misses the planned
> >=10% H100 success bar, so the simple "H100 is memory-bound enough for generic
> W8A8 FP8 to win" hypothesis is not supported by this run.

## Setup

| item | value |
|------|-------|
| repo | `fork=https://github.com/luojiaxuan/sglang-omni.git` |
| branch | `jaxanluo/h100-decode-forward` worktree from `perf/b200-moe-fp8` |
| benchmark commit | `61d1dd102d418697d3f7da3f78b742af3eafbcde` |
| host | `host-85-234-79-62` |
| hardware | 2x NVIDIA H100 80GB HBM3, driver 580.126.20 |
| image/runtime | `sglang-omni:0.5.12.post1`, CUDA 13.0 |
| Python stack | torch 2.11.0+cu130, sglang 0.5.12.post1, transformers 5.6.0, flashinfer 0.6.11.post1, sgl_kernel 0.4.2.post2 |
| model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` thinker, TP=2 |
| load | 32 concurrent text generations, `max_tokens=256`, about 45 s steady |
| profiler | `SGLANG_OMNI_PHASE_PROFILE=1 SGLANG_OMNI_PHASE_SYNC=1` |
| common flags | mixed-chunk not passed, CUDA graph on, `max_running_requests=32`, `max_prefill_tokens=16384` |
| artifact root | `/data/sglang-omni-h100-runs/20260616-h100-fp8` |

## Operational caveat - FP8 default memory contract OOM

The runbook default BF16 contract (`mem_fraction_static=0.75`) works for BF16.
The first FP8 prewarm attempt with the same `0.75` contract failed during KV cache
allocation:

| attempt | result |
|---------|--------|
| BF16, `mem_fraction_static=0.75` | starts and benchmarks cleanly |
| FP8, `mem_fraction_static=0.75` | OOM in `token_to_kv_pool` allocation after FP8 policy is selected |
| BF16/FP8, `thinker_memory_fraction=mem_fraction_static=0.55` | both start and benchmark cleanly |

The successful FP8 run confirms the expected resident thinker weight footprint
(`14.74 GB` per rank). The OOM looks like a KV-cache sizing/memory-contract issue
for online FP8 at the default benchmark budget, not a failure to enable FP8.

Key failed log:
`/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/prewarm_fp8_oom_mem075.log`

## Decode forward GPU time per step (bs=32, PHASE_SYNC true-GPU)

The clean A/B below uses the matched `0.55` memory contract for both BF16 and
FP8. A BF16 `0.75` baseline is also shown to confirm the smaller KV budget did
not materially change forward timing.

| metric | BF16 0.75 | BF16 0.55 | FP8 0.55 | FP8 vs BF16 0.55 |
|--------|-----------|-----------|----------|------------------|
| `[fwd-by-bs]` decode bs=32, rank0 | 5.8 ms | 5.8 ms | 5.7 ms | -0.1 ms (-1.7%) |
| `[fwd-by-bs]` decode bs=32, rank1 | 5.8 ms | 5.8 ms | 5.6 ms | -0.2 ms (-3.4%) |
| `[step phases]` mean GPU forward | 5.85-5.86 ms | 5.85 ms | 5.72-5.74 ms | about -2% |
| decode step total | 7.37 ms | 7.43 ms | 7.42 ms | flat |
| decode throughput (PHASE_SYNC on) | 3890.9 tok/s | 3864.3 tok/s | 3954.1 tok/s | +2.3% |
| CUDA graph hit rate | 1.000 | 1.000 | 1.000 | - |
| thinker weight mem / GPU | 28.59 GB | 28.59 GB | 14.74 GB | -48% |
| smoke correctness | `Paris` | `Paris` | `Paris` | - |

FP8 is therefore functioning and saving memory, but the measured decode-forward
improvement is small enough that it does not move end-to-end step time.

## Decode step breakdown - GPU forward vs host/scheduler

At the matched `0.55` contract:

| per decode step (bs=32-ish steady) | BF16 0.55 | FP8 0.55 |
|------------------------------------|-----------|----------|
| step total | 7.43 ms | 7.42 ms |
| GPU forward | 5.85 ms | 5.72-5.74 ms |
| host_pre (recv+sched+build) | 1.23 ms | 1.31-1.33 ms |
| host_post (finalize+stream+proc) | 0.35 ms | 0.37 ms |
| host total | 1.58 ms (21%) | 1.68-1.70 ms (23%) |

H100 host/scheduler tax is lower than the B200 number recorded in
`FINDINGS_B200.md` (~2.5 ms, ~30%), but it is still a meaningful ceiling: the
small FP8 forward gain is swallowed by host/build variance.

## Kernel/config notes

FP8 selected the intended online policy:

`effective_quantization=fp8 server_quantization=fp8 native_fp8_block_quant=False moe_runner_backend=triton fp8_gemm_backend=auto`

The H100 FP8 MoE path is still untuned:

`Config file not found at .../E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json`

BF16 also falls back from a Triton 3.6 H100 config to an older Triton config, but
it has a usable H100 BF16 config. FP8 uses the generic default `fp8_w8a8` config,
so the most likely forward-side next step is to tune the H100 `fp8_w8a8` fused
MoE kernel and rerun the same A/B.

## Decision

This H100 run does **not** validate a large memory-bound FP8 win. The current
generic online-FP8 path is a memory saver and maybe a tiny forward win, but not
a compelling H100 decode optimization yet.

Recommended next steps:

1. Tune H100 `fp8_w8a8` fused-MoE config (`E=128,N=384`) and rerun BF16/FP8.
2. Fix or document the FP8 `0.75` startup OOM / KV sizing issue; the successful
   run required `thinker_memory_fraction=mem_fraction_static=0.55`.
3. Continue scheduler/host profiling: H100 host tax is about 1.6-1.7 ms/step
   (~21-23%), enough to hide small forward-only gains.

## Artifacts

| file | contents |
|------|----------|
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/bf16.log` | BF16 default `0.75` server log |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/results/bf16_load.txt` | BF16 default `0.75` client load result |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/prewarm_fp8_oom_mem075.log` | failed FP8 prewarm at `0.75` |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/prewarm_fp8_mem055.log` | successful FP8 prewarm at `0.55` |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/fp8_mem055.log` | FP8 measured server log |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/results/fp8_load_mem055.txt` | FP8 measured client load result |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/logs/bf16_mem055.log` | BF16 matched `0.55` server log |
| `/data/sglang-omni-h100-runs/20260616-h100-fp8/results/bf16_load_mem055.txt` | BF16 matched `0.55` client load result |
