# Shared paths for thinker_decode_forward (#760 P1 / B200 GPU forward).
# Source from other scripts:  source "$(dirname "$0")/_common.sh"
#
# All paths are env-overridable. No host-specific defaults — clone the repo,
# set MODEL_PATH / HF_HOME / DATA_DIR, and run.

_pkg_root() {
  cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")/.." && pwd
}

_repo_root() {
  cd "$(_pkg_root)/../../.." && pwd
}

PKG_ROOT="${PKG_ROOT:-$(_pkg_root)}"
REPO_ROOT="${REPO_ROOT:-$(_repo_root)}"
DIAG_GAP="${DIAG_GAP:-${REPO_ROOT}/benchmarks/diagnostics/qwen3_omni_tp_vllm_gap}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PKG_ROOT}/artifacts}"
mkdir -p "${ARTIFACT_ROOT}"/{logs,runs,nsys,data}

SERVER_PY="${SERVER_PY:-${DIAG_GAP}/scripts/sglang_omni_qwen3_text_tp_server.py}"
LOAD_PY="${LOAD_PY:-${DIAG_GAP}/eval/run_concurrency.py}"
SWEEP_SH="${SWEEP_SH:-${DIAG_GAP}/eval/run_sweep.sh}"

# HuggingFace checkpoint (dir or hub id). No default snapshot path — set explicitly.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen3-omni}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"

PORT="${PORT:-8100}"
GPUS="${GPUS:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
THINKER_GPUS="${THINKER_GPUS:-0,1}"
ENCODER_GPU="${ENCODER_GPU:-1}"
THINKER_MAX_SEQ_LEN="${THINKER_MAX_SEQ_LEN:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
RELAY_BACKEND="${RELAY_BACKEND:-shm}"
SEG_TMP_DIR="${SEG_TMP_DIR:-/dev/shm/thinker_decode_forward}"
CONTAINER_NAME="${CONTAINER_NAME:-thinker_decode_forward}"
IMAGE="${IMAGE:-}"   # set when RUN_MODE=docker

# P2c mixed-chunk: default ON for end-to-end; empty ENABLE_MIXED_CHUNK for nsys decode-only.
ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
PER_STAGE_PROCESSES="${PER_STAGE_PROCESSES:-}"

NSYS_PREFIX="${NSYS_PREFIX:-}"

# Eval data: per-segment ACL6060 wavs (see README to generate).
DATA_DIR="${DATA_DIR:-${ARTIFACT_ROOT}/data/acl6060_zh_segments}"
PYTHON="${PYTHON:-python3}"

RUN_MODE="${RUN_MODE:-local}"   # local | docker
