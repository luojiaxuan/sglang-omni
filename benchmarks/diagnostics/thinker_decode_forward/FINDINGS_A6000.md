# A6000 baseline — thinker decode-only nsys (#760 P1)

Reference numbers from RTX A6000 (NVLink pair), job 46210, branch
`perf/thinker-decode-opt`. Use as the BF16 baseline when comparing B200 runs.

## Setup

| item | value |
|------|-------|
| model | `Qwen3-Omni-30B-A3B-Instruct` thinker, TP=2 |
| load | N=32 steady decode, 30 s ramp + 30 s nsys |
| flags | mixed-chunk **off**, CUDA graph **on** |
| isolation | nsys sqlite `graphId IS NOT NULL` = decode CUDA graph |

## Per decode step (bs≈32, graphId=2)

| metric | value |
|--------|-------|
| wall / step | **18.6 ms** |
| GPU-busy (union) | 18.2 ms (98%) |
| inter-kernel gap | 0.40 ms (2.1%) |
| concurrency | 88% of wall has ≥2 kernels active |

## GPU time (% of union-busy)

| category | % GPU-busy |
|----------|-----------|
| MoE expert GEMM | **69%** |
| all-reduce (NCCL) | 15% |
| dense GEMMs | 8% |
| attention + KV | 2% |
| misc | 5% |

## Critical path (% of wall, sweep-line)

| category | ms/step | % wall |
|----------|---------|--------|
| MoE expert GEMM | 11.0 | **59%** |
| all-reduce (exposed) | 2.4 | 13% |
| dense GEMMs | 1.2 | 6% |
| attention + KV | 0.2 | 1% |
| misc | 0.6 | 3% |
| gaps | 0.4 | 2% |
| overlap (≥2 cats) | 2.8 | 15% |

## Batch scaling

| decode bs | wall / step |
|-----------|-------------|
| ~24 | 16.9 ms |
| ~32 | 18.6 ms |

+33% tokens → +9% wall → **memory-bound on expert weight traffic**.

## Decision

Next lever: **MoE FP8/int8 expert quantization** (not TP comm). custom-AR ~3%
realistic upside.

Full write-up:
[#760#issuecomment-4703214587](https://github.com/sgl-project/sglang-omni/issues/760#issuecomment-4703214587)
