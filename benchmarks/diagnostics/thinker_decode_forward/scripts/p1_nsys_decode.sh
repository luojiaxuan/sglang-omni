#!/usr/bin/env bash
# P1 (#760): steady N=32 decode nsys capture + kernel split prep.
#
# Uses nsys session control (launch -> start -> stop) so collection begins only
# after the server is healthy and decode is steady. Requires nsys >= 2024 with
# session commands. mixed-chunk OFF => clean decode CUDA graphs.
#
# Usage (from repo root, after setting MODEL_PATH and preparing DATA_DIR):
#   bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_nsys_decode.sh
#
# Env overrides: GPUS, PORT, RAMP, COLLECT, CONCURRENCY, ARTIFACT_ROOT, RUN_MODE.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

JOB="${SLURM_JOB_ID:-manual}"
NSYS_DIR="${ARTIFACT_ROOT}/nsys"
REPORT="${NSYS_DIR}/decode_tp${TP_SIZE}_${JOB}"
RUNROOT="${ARTIFACT_ROOT}/runs"
SLOG="${RUNROOT}/p1nsys_${JOB}.server.log"
SESS="${NSYS_SESSION:-sstprof}"
RAMP="${RAMP:-30}"
COLLECT="${COLLECT:-30}"
CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-8172}"

mkdir -p "${NSYS_DIR}" "${RUNROOT}" "${ARTIFACT_ROOT}/logs" "${TMPDIR:-/dev/shm/p1nsys_${JOB}}"
export TMPDIR="${TMPDIR:-/dev/shm/p1nsys_${JOB}}"
STOPFLAG="${TMPDIR}/stop"
GPUS_LOCAL="${GPUS}"

echo "[p1nsys] node=$(hostname) job=${JOB} gpus=${GPUS_LOCAL} mode=${RUN_MODE} $(date)"
echo "[p1nsys] repo=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[p1nsys] model=${MODEL_PATH} report=${REPORT}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "[p1nsys] WARN: DATA_DIR missing (${DATA_DIR}). Load loop will fail; prepare data per README." >&2
fi

rm -f "${STOPFLAG}"
if [ "${RUN_MODE}" = "docker" ]; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

NSYS_PREFIX="nsys launch --session-new=${SESS} --trace=cuda,nvtx,nccl --cuda-graph-trace=node"
echo "[p1nsys] NSYS_PREFIX=${NSYS_PREFIX}"

SGLANG_OMNI_DECODE_STATS=1 SGLANG_OMNI_DECODE_STATS_INTERVAL=2 \
  ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  NSYS_PREFIX="${NSYS_PREFIX}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  PORT="${PORT}" GPUS="${GPUS_LOCAL}" \
  SEG_TMP_DIR="${SEG_TMP_DIR}" \
  bash "${SCRIPT_DIR}/serve_thinker.sh" > "${SLOG}" 2>&1 &
SRVPID=$!

ok=0
for i in $(seq 1 700); do
  curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q healthy && {
    ok=1; echo "[p1nsys] healthy after $((i*2))s ($(date +%T))"; break
  }
  [ $((i % 30)) -eq 0 ] && echo "[p1nsys] loading $((i*2))s ..."
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "[p1nsys] NOT healthy; tail:"; tail -60 "${SLOG}"
  kill "${SRVPID}" 2>/dev/null || true
  [ "${RUN_MODE}" = "docker" ] && docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 1
fi

if [ -f "${LOAD_PY}" ] && [ -d "${DATA_DIR}" ]; then
  ( r=0; while [ ! -f "${STOPFLAG}" ]; do r=$((r+1));
      "${PYTHON}" "${LOAD_PY}" \
        --engine sglang --base-url "http://127.0.0.1:${PORT}" --model-name "${SERVED_NAME}" \
        --concurrency "${CONCURRENCY}" --data-dir "${DATA_DIR}" \
        --out-dir "${TMPDIR}/load_r${r}" \
        --source-segment-size 1920 --max-new-tokens 40 --limit 480 \
        < /dev/null > "${TMPDIR}/load_r${r}.log" 2>&1 || true
    done ) &
  LOADPID=$!
  echo "[p1nsys] load pid=${LOADPID}; ramp ${RAMP}s ..."
else
  LOADPID=""
  echo "[p1nsys] skipping load (no DATA_DIR or LOAD_PY)"
fi
sleep "${RAMP}"

_nsys() {
  if [ "${RUN_MODE}" = "docker" ]; then
    docker exec "${CONTAINER_NAME}" nsys "$@"
  else
    nsys "$@"
  fi
}

echo "[p1nsys] nsys START -> ${COLLECT}s steady decode"
_nsys start --session="${SESS}" --sample=none --cpuctxsw=none --backtrace=none -o "${REPORT}" 2>&1 | tail -3
sleep "${COLLECT}"
echo "[p1nsys] nsys STOP"
_nsys stop --session="${SESS}" 2>&1 | tail -3

touch "${STOPFLAG}"
[ -n "${LOADPID}" ] && kill "${LOADPID}" 2>/dev/null || true
kill "${SRVPID}" 2>/dev/null || true

echo "[p1nsys] waiting for ${REPORT}.nsys-rep ..."
for i in $(seq 1 90); do
  if [ -f "${REPORT}.nsys-rep" ]; then
    s1=$(stat -c %s "${REPORT}.nsys-rep" 2>/dev/null || echo 0)
    sleep 3
    s2=$(stat -c %s "${REPORT}.nsys-rep" 2>/dev/null || echo 0)
    if [ "$s1" = "$s2" ] && [ "$s1" != 0 ]; then
      echo "[p1nsys] report stable: $((s2/1024/1024)) MiB"; break
    fi
  fi
  sleep 2
done

[ "${RUN_MODE}" = "docker" ] && { docker stop -t 20 "${CONTAINER_NAME}" >/dev/null 2>&1 || true; docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }

echo "[p1nsys] [decode stats] cross-check:"
grep -hE '\[decode stats\]|\[thinker args\]' "${SLOG}" 2>/dev/null | tail -6 || true

if [ ! -f "${REPORT}.nsys-rep" ]; then
  echo "[p1nsys] ERROR: no report"; tail -40 "${SLOG}"; exit 2
fi

echo "[p1nsys] ===== cuda_gpu_kern_sum (top kernels) ====="
nsys stats --force-export=true --report cuda_gpu_kern_sum --format table "${REPORT}.nsys-rep" 2>/dev/null | sed -n '1,46p'

SQLITE="${REPORT}.sqlite"
if [ ! -f "${SQLITE}" ]; then
  nsys export --type sqlite "${REPORT}.nsys-rep" -o "${SQLITE}" 2>/dev/null || true
fi

if [ -f "${SQLITE}" ]; then
  echo "[p1nsys] decode-only split (auto graph buckets):"
  "${PYTHON}" "${PKG_ROOT}/analysis/decode_split.py" "${SQLITE}" || true
  "${PYTHON}" "${PKG_ROOT}/analysis/overlap.py" "${SQLITE}" || true
fi

echo "[p1nsys] report: ${REPORT}.nsys-rep"
echo "[p1nsys] DONE $(date)"
