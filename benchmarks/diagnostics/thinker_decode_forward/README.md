# Thinker decode GPU forward — profiling & B200 FP8 experiments

Self-contained package for [#760](https://github.com/sgl-project/sglang-omni/issues/760)
P1 decode-forward breakdown and **B200 MoE FP8** follow-up. Everything runs from a
**git clone** — no dependency on any particular cluster filesystem layout.

**A6000 baseline numbers:** [`FINDINGS_A6000.md`](./FINDINGS_A6000.md)

## Quick start (B200 or any node)

```bash
git clone https://github.com/luojiaxuan/sglang-omni.git
cd sglang-omni
git checkout perf/b200-moe-fp8   # or perf/thinker-decode-opt

# Install sglang-omni + deps (see repo README). Then:

export MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct   # or local checkpoint dir
export HF_HOME=$HOME/.cache/huggingface
export GPUS=0,1
export TP_SIZE=2

# Smoke: start thinker server
bash benchmarks/diagnostics/thinker_decode_forward/scripts/serve_thinker.sh

# Decode-only nsys (mixed-chunk OFF, N=32 load)
bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_nsys_decode.sh
```

Artifacts land under `benchmarks/diagnostics/thinker_decode_forward/artifacts/` (gitignored).

## Layout

```
thinker_decode_forward/
├── README.md                 # this file
├── FINDINGS_A6000.md         # BF16 baseline reference
├── scripts/
│   ├── _common.sh            # env defaults (no host-specific paths)
│   ├── serve_thinker.sh      # local (default) or RUN_MODE=docker
│   ├── p1_nsys_decode.sh     # steady decode nsys + auto analysis
│   ├── p1_fwd_by_bs.sh       # forward vs batch-size curve
│   └── car_bench.py          # custom-AR vs NCCL microbench
├── analysis/
│   ├── decode_split.py       # per-category ms/step by graphId
│   └── overlap.py            # critical-path / AR exposure
└── slurm/
    └── p1_nsys_decode.slurm  # optional SLURM wrapper (edit partition)
```

**Sibling dependency:** server launcher and N=32 eval load reuse
`benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/` (same repo, no external rasst-demo).

## Environment variables

| variable | default | notes |
|----------|---------|-------|
| `MODEL_PATH` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | HF hub id or local dir |
| `HF_HOME` | `$HOME/.cache/huggingface` | model cache |
| `ARTIFACT_ROOT` | `.../artifacts` | logs, nsys, runs |
| `DATA_DIR` | `$ARTIFACT_ROOT/data/acl6060_zh_segments` | streaming load wavs |
| `GPUS` | `0,1` | physical GPU ids (docker) or used as-is (local) |
| `TP_SIZE` | `2` | thinker tensor parallel |
| `RUN_MODE` | `local` | `local` or `docker` (+ set `IMAGE=`) |
| `ENABLE_MIXED_CHUNK` | `1` | empty for nsys decode-only |
| `QUANTIZATION` | (empty) | `fp8` = online per-tensor W8A8 from the BF16 checkpoint (general: triton fused-MoE FP8 on B200/H100/H200). Empty = BF16 baseline |
| `MOE_RUNNER_BACKEND` | (empty) | optional MoE runner override (`triton` / `cutlass`); leave empty to let the Omni policy choose |
| `NSYS_PREFIX` | (empty) | e.g. `nsys launch --session-new=...` |

**Diagnostic env (engine built-in, pass through at launch):**

- `SGLANG_OMNI_DECODE_STATS=1` — decode bs histogram + CUDA graph hit rate
- `SGLANG_OMNI_PHASE_PROFILE=1` + `SGLANG_OMNI_PHASE_SYNC=1` — true-GPU `[fwd-by-bs]`

## Prepare eval data (optional, for N=32 load)

```bash
# From repo root; needs ACL6060 dev set + simuleval deps
python benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/eval/prepare_acl6060_segments.py \
  --out-dir benchmarks/diagnostics/thinker_decode_forward/artifacts/data/acl6060_zh_segments
export DATA_DIR=benchmarks/diagnostics/thinker_decode_forward/artifacts/data/acl6060_zh_segments
```

Without `DATA_DIR`, `p1_nsys_decode.sh` still captures server idle/decode if you
drive load another way.

## Workflows

### 1. Decode-only nsys split

```bash
ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  SGLANG_OMNI_DECODE_STATS=1 \
  bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_nsys_decode.sh
```

Post-process manually:

```bash
REP=benchmarks/diagnostics/thinker_decode_forward/artifacts/nsys/decode_tp2_manual
nsys export --type sqlite "${REP}.nsys-rep" -o "${REP}.sqlite"
python benchmarks/diagnostics/thinker_decode_forward/analysis/decode_split.py "${REP}.sqlite"
python benchmarks/diagnostics/thinker_decode_forward/analysis/overlap.py "${REP}.sqlite"
```

### 2. Forward vs batch size (memory-bound check)

```bash
bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_fwd_by_bs.sh
# Read [fwd-by-bs] decode lines in server log — NOT seg/s (PHASE_SYNC perturbs tput).
```

### 3. custom-AR vs NCCL microbench

```bash
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 \
  benchmarks/diagnostics/thinker_decode_forward/scripts/car_bench.py
```

### 4. FP8 MoE experiment (general: B200 / H100 / H200)

**Online FP8 (recommended, no special checkpoint).** Quantize the standard BF16
checkpoint to per-tensor W8A8 at load time — halves expert weight traffic, the
dominant cost of memory-bound decode. The Omni backend policy pins the portable
triton fused-MoE FP8 runner. Compare against the BF16 baseline with the same nsys
methodology:

```bash
# BF16 baseline
ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  bash scripts/p1_nsys_decode.sh

# Online FP8 (same checkpoint)
QUANTIZATION=fp8 ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  bash scripts/p1_nsys_decode.sh
```

Compare MoE ms/step and decode wall vs [`FINDINGS_A6000.md`](./FINDINGS_A6000.md)
and record results in `FINDINGS_B200.md`.

**Native FP8 checkpoint (B200-optimized, optional).** Use
`marksverdhei/Qwen3-Omni-30B-A3B-FP8` (`MODEL_PATH=...FP8 QUANTIZATION=fp8`) — the
policy then selects CUTLASS block-FP8 MoE where supported. See
`examples/configs/qwen3_omni_fp8_colocated.yaml` and
`docs/basic_usage/qwen3_omni.md` § FP8.

**TP=1 (B200 192 GB).** Try `TP_SIZE=1 GPUS=0` to remove the all-reduce entirely.

## SLURM

Edit partition/resources, then:

```bash
sbatch benchmarks/diagnostics/thinker_decode_forward/slurm/p1_nsys_decode.slurm
```

## Related

- Full vLLM gap diagnosis: `../qwen3_omni_tp_vllm_gap/`
- Issue #760 P1 comment: https://github.com/sgl-project/sglang-omni/issues/760#issuecomment-4703214587
