# H100 findings - thinker decode, BF16 baseline and FP8 decision

H100 decode-forward A/B for the **general online-FP8 (W8A8)** MoE path vs the
BF16 baseline. This run used the same branch/engine path as the B200 result, and
the same text steady-decode harness (`decode_load_text.py`, concurrency 32,
`max_tokens=256`, PHASE_SYNC enabled).

> **Current H100 decision: drop generic online-FP8 as the active optimization
> path and return to BF16.** FP8 halves thinker weight memory, but with a clean
> matched memory contract (`thinker_memory_fraction=mem_fraction_static=0.55`)
> it only improves bs=32 forward from about 5.8 ms to 5.6-5.7 ms (roughly 2-3%),
> while total decode step stays essentially flat. It is useful as a memory-saving
> mode, not as the next H100 latency/throughput lever.

## BF16 rebaseline after dropping FP8 - 2026-06-16

After the FP8 A/B did not clear the >=10% H100 success bar, I reran the H100
baseline in BF16 and swept text decode concurrency at the same stable `0.55`
memory contract. Artifacts are under
`/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline`.

| load | client throughput | steady `[fwd-by-bs]` decode | steady `[step phases]` decode | host/scheduler tax | CUDA graph |
|------|-------------------|-----------------------------|-------------------------------|--------------------|------------|
| c16 | 2274.0 tok/s | bs=16: 5.3 ms | 6.71 ms step, 5.34-5.35 ms forward | 1.36-1.38 ms (20%) | 1.000, max_bs=16 |
| c24 | 3257.0 tok/s | bs=24: 5.4 ms | 7.06 ms step, 5.39-5.40 ms forward | 1.65-1.66 ms (23%) | 1.000, max_bs=24 |
| c32 | 3775.2 tok/s | bs=32: 5.8 ms | 7.75 ms step, 5.91-5.92 ms forward | 1.82-1.86 ms (24%) | 1.000, max_bs=32 |

Notes:

1. BF16 forward is already much faster on H100 than on A6000. The A6000 baseline
   was 18.6 ms/step with MoE expert GEMM at 59% of the critical path; on H100 the
   whole BF16 forward is only about 5.3-5.9 ms.
2. Forward does not scale linearly with batch: going from c16 to c32 doubles the
   steady decode batch but only raises forward from about 5.3 ms to 5.9 ms. That
   weakens the old "pure MoE weight-bandwidth bottleneck" hypothesis.
3. Step time and host/scheduler work still grow with load. At c32, about 1.8 ms
   of the 7.75 ms step is outside GPU forward (`recv+sched+build+finalize+
   stream+proc`), enough to hide any small forward-only win like the observed FP8
   2-3%.
4. The H100 BF16 MoE path also logs a Triton 3.6 config miss and falls back to the
   older Triton 3.2 H100 config, including missing `down_moe` tuned config. This
   is a more plausible next kernel-side target than generic FP8, but it needs an
   nsys split to confirm MoE is still the dominant part inside the 5-6 ms forward.

Revised bottleneck read: H100 is no longer the A6000 regime. The largest single
component is still GPU forward, but the optimization ceiling is now shared by
BF16 kernel quality and host/scheduler overhead. Generic FP8 is not the right
first lever because it reduces weight footprint without materially moving step
time.

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
a compelling H100 decode optimization.

Use the BF16 c32 rebaseline as the active H100 reference and spend the next pass
on:

1. BF16 H100 forward split: run/fix nsys decode split to break the 5-6 ms forward
   into MoE, attention, dense GEMM, and TP/all-reduce.
2. Scheduler/build overhead: c32 spends about 1.8 ms/step outside GPU forward;
   this is now large enough to matter on H100.
3. H100 BF16 MoE config quality: the logs fall back from Triton 3.6 to older H100
   configs and miss the `down_moe` tuned config.
4. Revisit FP8 only after an H100-specific tuned `fp8_w8a8` MoE config exists, or
   when memory capacity is the goal rather than decode latency.

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
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/logs/bf16_c16_mem055.log` | BF16 c16 rebaseline server log |
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/results/bf16_c16_mem055.txt` | BF16 c16 client load result |
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/logs/bf16_c24_mem055.log` | BF16 c24 rebaseline server log |
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/results/bf16_c24_mem055.txt` | BF16 c24 client load result |
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/logs/bf16_c32_mem055.log` | BF16 c32 rebaseline server log |
| `/data/sglang-omni-h100-runs/20260616-h100-bf16-rebaseline/results/bf16_c32_mem055.txt` | BF16 c32 client load result |
