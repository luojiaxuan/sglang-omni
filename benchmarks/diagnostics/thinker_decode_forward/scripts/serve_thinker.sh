#!/usr/bin/env bash
# Launch Qwen3-Omni thinker (TP) for decode-forward profiling / FP8 experiments.
#
# Local (default):
#   MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct GPUS=0,1 PORT=8100 \
#     bash scripts/serve_thinker.sh
#
# Docker (optional):
#   RUN_MODE=docker IMAGE=your-sglang-omni:latest GPUS=0,1 bash scripts/serve_thinker.sh
#
# nsys session launch (mixed-chunk OFF for clean decode graphs):
#   NSYS_PREFIX='nsys launch --session-new=sstprof --trace=cuda,nvtx,nccl --cuda-graph-trace=node' \
#     ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 bash scripts/serve_thinker.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"

mkdir -p "${SEG_TMP_DIR}"

if [ ! -f "${SERVER_PY}" ]; then
  echo "[serve] ERROR: server script not found: ${SERVER_PY}" >&2
  exit 1
fi

_build_server_cmd() {
  local -a cmd=("${PYTHON}" "${SERVER_PY}"
    --model-path "${MODEL_PATH}"
    --model-name "${SERVED_NAME}"
    --pipeline-name qwen3-omni
    --relay-backend "${RELAY_BACKEND}"
    --host 127.0.0.1 --port "${PORT}"
    --ipc-base-path /tmp/thinker_decode_forward
    --thinker-tp-size "${TP_SIZE}"
    --thinker-gpus "${THINKER_GPUS}"
    --encoder-gpu "${ENCODER_GPU}"
    --thinker-max-seq-len "${THINKER_MAX_SEQ_LEN}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  )
  [ -n "${MEM_FRACTION_STATIC}" ] && cmd+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
  [ -n "${ENABLE_MIXED_CHUNK}" ] && cmd+=(--enable-mixed-chunk)
  [ "${CHUNKED_PREFILL_SIZE:-0}" -gt 0 ] 2>/dev/null && \
    cmd+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
  [ -n "${PER_STAGE_PROCESSES}" ] && cmd+=(--per-stage-processes)
  printf '%s\0' "${cmd[@]}"
}

echo "[serve] mode=${RUN_MODE} repo=${REPO_ROOT} model=${MODEL_PATH} gpus=${GPUS} tp=${TP_SIZE} port=${PORT}"

if [ "${RUN_MODE}" = "docker" ]; then
  if [ -z "${IMAGE}" ]; then
    echo "[serve] ERROR: set IMAGE= when RUN_MODE=docker" >&2
    exit 1
  fi
  _NGPU="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"
  CUDA_VIS="$(seq -s, 0 $((_NGPU - 1)))"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  EXTRA_ENV=()
  while IFS='=' read -r _k _v; do
    case "${_k}" in SGLANG_OMNI_*) EXTRA_ENV+=(-e "${_k}") ;; esac
  done < <(env)
  # Build inner command as a single quoted string for docker bash -lc.
  INNER="cd ${REPO_ROOT} && export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-} && "
  INNER+="${PYTHON} -c \"import torch; print('[serve]', torch.__version__, torch.cuda.device_count())\" && "
  INNER+="exec ${NSYS_PREFIX} ${PYTHON} ${SERVER_PY}"
  INNER+=" --model-path ${MODEL_PATH} --model-name ${SERVED_NAME} --pipeline-name qwen3-omni"
  INNER+=" --relay-backend ${RELAY_BACKEND} --host 127.0.0.1 --port ${PORT}"
  INNER+=" --ipc-base-path /tmp/thinker_decode_forward --thinker-tp-size ${TP_SIZE}"
  INNER+=" --thinker-gpus ${THINKER_GPUS} --encoder-gpu ${ENCODER_GPU}"
  INNER+=" --thinker-max-seq-len ${THINKER_MAX_SEQ_LEN}"
  INNER+=" --max-running-requests ${MAX_RUNNING_REQUESTS} --max-prefill-tokens ${MAX_PREFILL_TOKENS}"
  INNER+=" --mem-fraction-static ${MEM_FRACTION_STATIC}"
  [ -n "${ENABLE_MIXED_CHUNK}" ] && INNER+=" --enable-mixed-chunk"
  [ "${CHUNKED_PREFILL_SIZE:-0}" -gt 0 ] 2>/dev/null && INNER+=" --chunked-prefill-size ${CHUNKED_PREFILL_SIZE}"
  [ -n "${PER_STAGE_PROCESSES}" ] && INNER+=" --per-stage-processes"
  exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "\"device=${GPUS}\"" \
    --ipc host --network host --shm-size 64g \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${HF_HOME}:${HF_HOME}" \
    -v "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}" \
    -w "${REPO_ROOT}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VIS}" \
    -e HF_HOME="${HF_HOME}" \
    -e XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    -e TORCH_HOME="${TORCH_HOME}" \
    -e TOKENIZERS_PARALLELISM=false \
    -e SGLANG_OMNI_STARTUP_TIMEOUT="${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}" \
    ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
    "${IMAGE}" bash -lc "${INNER}"
else
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export HF_HOME XDG_CACHE_HOME TORCH_HOME
  mapfile -d '' -t SERVER_CMD < <(_build_server_cmd)
  "${PYTHON}" -c "import torch; print('[serve] torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
  if [ -n "${NSYS_PREFIX}" ]; then
    # shellcheck disable=SC2086
    exec ${NSYS_PREFIX} "${SERVER_CMD[@]}"
  else
    exec "${SERVER_CMD[@]}"
  fi
fi
