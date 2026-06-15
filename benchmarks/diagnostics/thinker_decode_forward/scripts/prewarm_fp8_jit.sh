#!/usr/bin/env bash
# Pre-warm the online-FP8 JIT kernel cache for the Qwen3-Omni thinker.
#
# WHY: online FP8 (`QUANTIZATION=fp8` on a BF16 checkpoint) lazily JIT-compiles
# the per-tensor / per-token-group FP8 quant kernels on the FIRST prefill. Under
# TP that first compile races with the live CUDA stream / NCCL collective and can
# corrupt the context ("Failed to CUDA calloc async ..."), killing the thinker.
# Compiling once under CUDA_LAUNCH_BLOCKING=1 serializes it safely and populates
# the persistent tvm-ffi cache (~/.cache/tvm-ffi), after which normal (non-blocking)
# FP8 launches start cleanly. Run this ONCE per machine/JIT-cache before FP8 runs.
#
# Usage (same env as serve_thinker.sh):
#   MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct GPUS=1,2 TP_SIZE=2 \
#     bash scripts/prewarm_fp8_jit.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

PORT="${PORT:-8133}"
echo "[prewarm] FP8 JIT warmup: model=${MODEL_PATH} gpus=${GPUS} tp=${TP_SIZE} port=${PORT}"

# Serialize all CUDA launches so the first FP8 quant JIT compile cannot race the
# NCCL collective; this is the whole point of the warmup.
QUANTIZATION=fp8 CUDA_LAUNCH_BLOCKING=1 \
  ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  PORT="${PORT}" \
  bash "${SCRIPT_DIR}/serve_thinker.sh" > "${ARTIFACT_ROOT}/logs/prewarm_fp8.log" 2>&1 &
SRVPID=$!

ok=0
for i in $(seq 1 240); do
  curl -fsS -m 4 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -qi healthy && { ok=1; break; }
  grep -qE "died during startup|RuntimeError: Process" "${ARTIFACT_ROOT}/logs/prewarm_fp8.log" 2>/dev/null && break
  sleep 3
done

if [ "${ok}" = 1 ]; then
  echo "[prewarm] healthy; sending one warmup request to trigger FP8 prefill JIT ..."
  curl -sS -m 180 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${SERVED_NAME}"'","messages":[{"role":"user","content":"Hello"}],"modalities":["text"],"temperature":0,"max_tokens":8}' \
    2>&1 | head -c 200; echo
  echo "[prewarm] done; JIT cache populated under ~/.cache/tvm-ffi"
else
  echo "[prewarm] ERROR: server not healthy; tail:" >&2
  tail -40 "${ARTIFACT_ROOT}/logs/prewarm_fp8.log" >&2
fi

# Thorough teardown: kill the server tree (compute workers + coordinator/HTTP).
"${PYTHON}" - "${SRVPID}" <<'PYKILL'
import os, signal, glob, sys
keep = os.getpid()
sig = ("thinker_decode_forward", "sglang_omni_qwen3_text_tp_server", "spawn_main",
       "launch_server", "stage_workers")
for d in glob.glob("/proc/[0-9]*"):
    try:
        pid = int(d.rsplit("/", 1)[-1])
        if pid == keep:
            continue
        cl = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        continue
    if "PYKILL" in cl:
        continue
    if ("python" in cl or "/bin/bash" in cl) and any(s in cl for s in sig):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
PYKILL
echo "[prewarm] server torn down."
