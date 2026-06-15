# B200 findings — thinker decode, general online-FP8 (W8A8) vs BF16

B200 decode-forward A/B for the **general online-FP8 (W8A8)** MoE path vs the BF16
baseline. Online FP8 quantizes the standard BF16 checkpoint at load time
(per-tensor `float8_e4m3fn`) and the Omni backend policy pins the portable triton
fused-MoE FP8 runner (works on B200/H100/H200; no native FP8 checkpoint needed).

> **Headline (B200): online FP8 W8A8 does _not_ speed up thinker decode — it is
> ~7–10% _slower_ — even though it halves expert-weight memory.** The decode-forward
> MoE kernel runs an **untuned** `fp8_w8a8` triton config on B200, and W8A8 adds
> per-step activation-quant overhead; on B200's ~8 TB/s HBM3e the halved weight
> traffic does not pay for that. The enablement is correct and general (likely a
> win on the more bandwidth-limited H100/H200); the B200 kernel needs tuning.

## Setup

| item | value |
|------|-------|
| model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` thinker, TP=2 |
| hardware | 2× NVIDIA B200 (sm_100, 183 GB), CUDA 13.0, torch 2.11, sglang 0.5.12.post1 |
| load | 32 concurrent text generations, `max_tokens=256`, ~45 s steady (decode_load_text.py) |
| profiler | `SGLANG_OMNI_PHASE_PROFILE=1 SGLANG_OMNI_PHASE_SYNC=1` → `[fwd-by-bs]` true GPU forward ms |
| flags | mixed-chunk off, CUDA graph on, `moe_runner_backend` auto→**triton** (FP8) |

> **Methodology note.** The headline A/B uses the engine's built-in PHASE_SYNC
> `[fwd-by-bs]` true-GPU forward timer (a direct, decode-isolated measurement),
> not the nsys decode-graph split. The nsys harness was blocked twice on this
> stack: (a) its SST load driver (`run_concurrency.py`) needs the `simuleval` CLI,
> which is not installed; (b) with a text-driven load, nsys 2025.6.3 captured no
> CUDA kernel data and renamed the sqlite kernel table, so `analysis/decode_split.py`
> needs updating for the new schema. The nsys kernel-level split (MoE % of decode)
> is therefore still open — see next steps.

## Decode forward GPU time per step (bs=32, PHASE_SYNC true-GPU)

| metric | BF16 | FP8 online W8A8 | Δ |
|--------|------|------------------|---|
| `[fwd-by-bs]` decode bs=32, rank0 | **5.9 ms** | **6.2 ms** | +5% |
| `[fwd-by-bs]` decode bs=32, rank1 | **5.8 ms** | **6.5 ms** | +12% |
| `[step phases]` mean GPU forward | 6.2–6.5 ms | 6.2–6.5 ms | ~flat |
| decode throughput (32 conc, text) | **2628 tok/s** | **2503 tok/s** | **−4.8%** |
| CUDA graph hit rate | 1.000 | 1.000 | — |
| thinker weight mem / GPU | **28.59 GB** | **14.74 GB** | **−48% (≈2×)** ✓ |

Both TP ranks and the independent throughput metric agree FP8 is slightly slower —
this is a real regression, not noise. Expert-weight memory is halved as expected,
confirming the FP8 weights are resident; the saving just doesn't convert to decode
speed on B200.

## Decode step breakdown — GPU forward vs host/scheduler (`[step phases]`)

The bigger structural finding on B200: **only ~70% of each decode step is GPU
forward; ~30% (~2.5 ms) is host/scheduler overhead** — roughly constant across
BF16/FP8, so it caps any forward-only (FP8/kernel) win.

| per decode step (bs=32) | BF16 | FP8 |
|--------------------------|------|-----|
| step total | 8.44 ms | 8.79 ms |
| GPU forward | 5.9 ms | 6.2–6.4 ms |
| host_pre (recv+sched+build) | ~1.9 ms | ~1.8–2.0 ms |
| host_post (finalize+stream+proc) | ~0.6 ms | ~0.6 ms |
| **host total** | **~2.5 ms (~30%)** | **~2.5 ms (~30%)** |

`recv` (~0.7–1.1 ms, the IPC/relay receive) and `sched` (~0.57 ms) dominate host
overhead. On A6000 the GPU forward was 18.6 ms (host overhead a smaller fraction);
on B200 the forward shrank to ~5.9 ms while host overhead stayed ~2.5 ms, so the
**relative** scheduling cost roughly tripled. Reducing GPU forward alone (FP8,
kernel tuning) is bounded by this ~30% host tax.

## Correctness (greedy, FP8 vs BF16)

Identical, coherent outputs — FP8 W8A8 does not regress quality on these probes:

| prompt | BF16 | FP8 |
|--------|------|-----|
| capital of France (one word) | `Paris` | `Paris` |
| translate 你好，世界。 | `Hello, world.` | `Hello, world.` |

## Why FP8 doesn't help on B200 (analysis)

1. **Untuned MoE kernel.** Server log: `Using default MoE kernel config ... Config
   file not found at .../E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`.
   The fp8_w8a8 fused-MoE runs a generic config; it likely does not realize the
   2× weight-BW saving and is slower than the BF16 MoE path.
2. **W8A8 activation overhead.** Online FP8 also quantizes activations per-tensor
   each step (`per_tensor_quant_fp8` kernels) — extra passes that, for the small
   decode batch, are not amortized.
3. **B200 is bandwidth-rich.** A6000 (~0.77 TB/s) decode was 18.6 ms/step with MoE
   = 59% of the critical path. B200 (~8 TB/s HBM3e) decode forward is only ~5.9 ms,
   so the weight-read fraction is far smaller and halving it yields little — while
   the FP8 kernel/quant overhead is fixed.

## Operational note — first-run JIT race (fixed)

Online FP8 lazily JIT-compiles its quant kernels on the first prefill; under TP
that compile races the NCCL collective and corrupts the CUDA context
(`Failed to CUDA calloc async ...`), killing the thinker. **Workaround:** warm the
JIT cache once (persistent in `~/.cache/tvm-ffi`) via
[`scripts/prewarm_fp8_jit.sh`](./scripts/prewarm_fp8_jit.sh) (a one-shot launch
under `CUDA_LAUNCH_BLOCKING=1`); afterwards normal FP8 launches start cleanly.
Also requires `ninja` on PATH (the JIT build tool).

## Recommended next steps (locate the bottleneck first, then customize per chip)

Direction: **don't rush to H100 or to kernel rewrites — first establish where the
B200 decode step actually spends time, then optimize per chip.**

1. **Quantify the B200 decode bottleneck.**
   - GPU forward split: fix `analysis/decode_split.py` for nsys 2025.6.3 (renamed
     sqlite schema; current text-driven capture also yielded no kernel data — needs
     investigation) to get MoE / attention / all-reduce / dense-GEMM % of the
     ~5.9 ms forward. This sets the ceiling for any forward-side win on B200.
   - Host/scheduler split: the ~2.5 ms/step (~30%) host overhead (`recv` IPC-relay
     ~1 ms, `sched` ~0.57 ms) is the largest *relative* lever on B200. Profile the
     omni scheduler/relay decode loop; this was deemed out-of-scope for the A6000
     forward project but is now first-order on a fast GPU.
2. **Per-chip customization** based on (1): the optimal lever differs by GPU
   (B200 forward is bandwidth-rich → host/scheduler matters more; A6000/H100/H200
   are more forward/memory-bound → FP8/weight-traffic matters more).
3. _(Deferred until (1) justifies it)_ tune the `fp8_w8a8` MoE kernel for B200
   (`E=128,N=384`, `sglang/benchmark/kernels/fused_moe_triton`); re-test FP8 on
   H100/H200 (more memory-bound — A6000 predicts a win there); weight-only
   W8A16/W4A16 to drop the W8A8 activation-quant overhead.
