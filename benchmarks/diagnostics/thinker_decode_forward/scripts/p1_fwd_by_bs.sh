#!/usr/bin/env bash
# P1 (#760): thinker forward time vs decode batch size ([fwd-by-bs] curve).
#
# SGLANG_OMNI_PHASE_SYNC=1 => forward bucket is true GPU time. Do NOT read seg/s
# (sync perturbs throughput); read per-bs forward split only.
#
# Usage:
#   bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_fwd_by_bs.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

JOB="${SLURM_JOB_ID:-manual}"
RUNROOT="${ARTIFACT_ROOT}/runs"
SLOG="${RUNROOT}/p1fwdbs_${JOB}.server.log"
OUTROOT="${RUNROOT}/p1fwdbs_${JOB}"
LATDIR="${TMPDIR:-/dev/shm/p1fwdbs_${JOB}}/lat"
PORT="${PORT:-8147}"

mkdir -p "${OUTROOT}" "${LATDIR}" "${ARTIFACT_ROOT}/logs"
export TMPDIR="${TMPDIR:-/dev/shm/p1fwdbs_${JOB}}"

echo "[p1fwdbs] node=$(hostname) job=${JOB} gpus=${GPUS} $(date)"

export SGLANG_OMNI_DECODE_STATS=1
export SGLANG_OMNI_DECODE_STATS_INTERVAL=5
export SGLANG_OMNI_PHASE_PROFILE=1
export SGLANG_OMNI_PHASE_PROFILE_INTERVAL=5
export SGLANG_OMNI_PHASE_SYNC=1

PORT="${PORT}" GPUS="${GPUS}" CONTAINER_NAME="${CONTAINER_NAME}" \
  bash "${SCRIPT_DIR}/serve_thinker.sh" > "${SLOG}" 2>&1 &
SRVPID=$!

ok=0
for i in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q healthy && {
    ok=1; echo "[p1fwdbs] healthy after $((i*2))s"; break
  }
  [ $((i % 30)) -eq 0 ] && echo "[p1fwdbs] loading $((i*2))s ..."
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "[p1fwdbs] NOT healthy"; tail -150 "${SLOG}"
  kill "${SRVPID}" 2>/dev/null || true
  exit 1
fi

grep -h '\[thinker args\]' "${SLOG}" | tail -1 || true

if [ -f "${SWEEP_SH}" ] && [ -d "${DATA_DIR}" ]; then
  REMOTE_OMNI_LAT_DIR="${LATDIR}" ENGINE=sglang BASE_URL="http://127.0.0.1:${PORT}" \
    OUT_ROOT="${OUTROOT}" NLIST="8 16 24 32" \
    bash "${SWEEP_SH}" 2>&1 | tail -10
  cat "${OUTROOT}/results.tsv" 2>/dev/null || true
else
  echo "[p1fwdbs] skip sweep (need DATA_DIR + ${SWEEP_SH})"
fi

echo "[p1fwdbs] ===== [fwd-by-bs] decode ====="
grep -h '\[fwd-by-bs\] decode' "${SLOG}" | tail -1 || echo "(none)"
echo "[p1fwdbs] ===== [fwd-by-bs] prefill ====="
grep -h '\[fwd-by-bs\] prefill' "${SLOG}" | tail -1 || echo "(none)"
echo "[p1fwdbs] ===== [decode stats] ====="
grep -h '\[decode stats\]' "${SLOG}" | tail -1 || echo "(none)"

kill "${SRVPID}" 2>/dev/null || true
[ "${RUN_MODE}" = "docker" ] && docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "[p1fwdbs] log: ${SLOG}"
echo "[p1fwdbs] DONE $(date)"
