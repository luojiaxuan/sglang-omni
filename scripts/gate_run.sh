#!/usr/bin/env bash
# PR-B async-decode S1-S5 gate — ON/OFF orchestration (GPU, run after cache frees).
#
# For each scenario: serve OFF (sync) and ON (async + scenario flags/env) on the
# SAME cards, dump vocoder-entry audio_codes per request via the gate-only hook
# (scripts/gate_serve.py installs it), send the scenario's fixed-seed requests,
# then diff ON vs OFF with scripts/gate_async_decode.py.
#
# Carries the full ABAB foolproofing (the spawn-worker leak antidote cache hit
# tonight): BUILD_SHA + actual sglang_omni import path logged per arm; setsid
# process-group teardown; inter-arm VRAM rescan that kills any non-claim
# compute-app left on our cards.
#
# Card/core binding follow the run-time occupancy map (co-tenant on 6+7 tonight):
#   GATE_CARDS=3,4  GATE_SRV_CORES=0-23,64-87  GATE_CLI_CORES=24-31,88-95  (node0)
# Override via env before invoking. Pin claim_gpu on the held cards across arms
# (ownership rule corollary ②) — this driver leaves the server resident, so the
# gap is only the brief inter-arm teardown.
set -uo pipefail

REPO=${REPO:-/data/moss-v15-ar/sglang-omni}
BENCH=${BENCH:-/data/moss-v15-ar/bench/gate}
LOGS=${LOGS:-/data/moss-v15-ar/logs}
PORT=${GATE_PORT:-8000}
# Bit-identity gate yields the clean node to a co-located statistical benchmark:
# default to node1 cores so a cache/ABAB run can own node0 (override via env).
CARDS=${GATE_CARDS:-3,4}
SRV_CORES=${GATE_SRV_CORES:-32-55,96-119}
CLI_CORES=${GATE_CLI_CORES:-56-63,120-127}
MEMBIND=${GATE_MEMBIND:-1}
CFG=examples/configs/moss_tts_local.yaml
SCENARIOS=${1:-S1 S2 S3a S3b S4 S5}
LAST_PGID=""

mkdir -p "$BENCH" "$LOGS"
# All relative paths (scripts/, examples/) resolve from the build repo; the emit
# calls below run before any launch_server cd, so anchor cwd here.
cd "$REPO" || { echo "REPO not found: $REPO" >&2; exit 1; }

wait_ready() {  # $1=logfile
  for _ in $(seq 1 150); do
    grep -q "Application startup complete" "$1" 2>/dev/null && return 0
    grep -qE "Traceback|address already in use|CUDA error|RuntimeError" "$1" 2>/dev/null && return 1
    sleep 2
  done
  return 1
}

# Sets LAST_PGID (global, not $(...) — daemon would inherit the substitution
# pipe and block forever). $extra_flags/$extra_env are the scenario's ON args.
launch_server() {  # $1=logfile $2=dump_dir $3=extra_env $4...=extra_flags
  local log="$1" dump="$2" env_kv="$3"; shift 3
  cd "$REPO"
  {
    echo "=== BUILD CHECK ==="
    echo "BUILD_SHA=$(git rev-parse HEAD)"
    PYTHONPATH=$REPO python -c "import sglang_omni; print('IMPORT_FILE='+sglang_omni.__file__)"
    echo "=== GATE SERVER START (dump=$dump env=$env_kv flags=$*) ==="
  } > "$log" 2>&1
  setsid bash -c "PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$CARDS MOSS_GATE_DUMP_AUDIO_CODES='$dump' $env_kv numactl --membind=$MEMBIND -C $SRV_CORES python scripts/gate_serve.py serve --config $CFG --port $PORT $* >> '$log' 2>&1" </dev/null >/dev/null 2>&1 &
  LAST_PGID=$!
}

kill_server() {  # $1=pgid ; group-kill then belt: clean non-claim apps on our cards
  kill -9 -"$1" 2>/dev/null
  # The uvicorn API process can escape the setsid group (multiprocessing spawn),
  # leaving the port bound so the next arm falls back to a random port; reap it.
  pkill -9 -f "scripts/gate_serve.py" 2>/dev/null
  sleep 4
  local claimpids gpu worker
  claimpids=" $(pgrep -f claim_gpu | tr '\n' ' ') "
  for gpu in ${CARDS//,/ }; do
    for worker in $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      case "$claimpids" in *" $worker "*) continue ;; esac
      kill -9 "$worker" 2>/dev/null
    done
  done
  sleep 3
}

import_ok() { grep -q "IMPORT_FILE=$REPO/" "$1" && echo 1 || echo 0; }

# Send the scenario's fixed-seed requests (per-request seed/len/penalty); the
# server's gate hook dumps audio_codes keyed by request_id into $1.
send_requests() {  # $1=dump_dir $2=scenario $3=actual_port
  PYTHONPATH=$REPO numactl --membind=$MEMBIND -C "$CLI_CORES" \
    python scripts/gate_async_decode.py --scenario "$2" --send --port "$3" \
    --dump-dir "$1" >> "$LOGS/gate_${2}_send.log" 2>&1
}

run_arm() {  # $1=scenario $2=arm(off|on) $3=dump_dir $4=env $5...=flags
  local sc="$1" arm="$2" dump="$3" env_kv="$4"; shift 4
  local log="$LOGS/gate_${sc}_${arm}.log"
  rm -rf "$dump"; mkdir -p "$dump"
  echo "  [$sc/$arm] launch ($*)"
  launch_server "$log" "$dump" "$env_kv" "$@"
  local pgid=$LAST_PGID
  if ! wait_ready "$log"; then echo "  [$sc/$arm] FAIL: server not ready"; kill_server "$pgid"; return 1; fi
  if [ "$(import_ok "$log")" != "1" ]; then echo "  [$sc/$arm] ABORT: wrong import path"; kill_server "$pgid"; exit 3; fi
  # Use the ACTUAL bound port (the server falls back to a random port if the
  # requested one is busy), not the requested one.
  local aport
  aport=$(grep -oE "0\.0\.0\.0:[0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
  aport=${aport:-$PORT}
  echo "  [$sc/$arm] ready on port $aport; sending"
  send_requests "$dump" "$sc" "$aport"
  kill_server "$pgid"
}

for sc in $SCENARIOS; do
  echo ">>> scenario $sc"
  ENV_KV=$(python scripts/gate_async_decode.py --scenario "$sc" --emit-env)
  FLAGS=$(python scripts/gate_async_decode.py --scenario "$sc" --emit-flags)
  off_dir="$BENCH/${sc}_off"; on_dir="$BENCH/${sc}_on"
  # OFF arm is always sync; ON arm carries the scenario flags/env.
  run_arm "$sc" off "$off_dir" "" --async-decode off || continue
  run_arm "$sc" on "$on_dir" "$ENV_KV" $FLAGS || continue
  python scripts/gate_async_decode.py --scenario "$sc" --off-dump "$off_dir" --on-dump "$on_dir"
done
echo "=== gate done; per-scenario verdicts above ==="
