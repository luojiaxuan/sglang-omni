#!/usr/bin/env bash
# PR-B speedup benchmark: SAME binary, --async-decode off (sync) vs on (async
# lookahead), ABAB-interleaved. Clean measurement (NO query-hit instrumentation
# on the measured runs). Captures rtf / output-throughput / mean+p99 latency /
# qps from the client's speed_results.json.
#
#   perf_abab.sh <label> <concurrency> <samples> <rounds>
# e.g. perf_abab.sh c16_full 16 0 3   (samples 0 = full SeedTTS-EN set)
#
# Card/core: GPU 5(AR)+4(codec), node1 (yield node0 to any co-tenant). Inherits
# the cache discipline: token-count auto, %idle gate, setsid teardown, port parse.
set -uo pipefail
REPO=${REPO:-/data/moss-v15-ar/sglang-omni-prb}
BENCH=${BENCH:-/data/moss-v15-ar/bench/perf}; LOGS=${LOGS:-/data/moss-v15-ar/logs}
CARDS=${PERF_CARDS:-5,4}; PORT=${PERF_PORT:-8030}
SRV="numactl --membind=1 -C 32-55,96-119"; CLI="numactl --membind=1 -C 56-63,120-127"
MODEL=OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
META=zhaochenyang20/seed-tts-eval-arrow; CFG=examples/configs/moss_tts_local.yaml
LABEL=${1:?label}; CONC=${2:?conc}; SAMPLES=${3:?samples}; ROUNDS=${4:-3}
RESULTS=$BENCH/perf_${LABEL}.csv; LAST_PGID=""
mkdir -p "$BENCH" "$LOGS"
echo "arm,round,rtf_mean,rtf_p99,out_throughput,latency_mean,latency_p99,qps,node1_busy,load1" > "$RESULTS"

# node1 (my region) %idle gate — only my cores matter (cache, if any, owns node0).
node1_busy() {
  python - <<'PY'
import time
c=list(range(32,64))+list(range(96,128))
def snap():
    o={}
    for l in open("/proc/stat"):
        if l.startswith("cpu") and not l.startswith("cpu "):
            p=l.split(); i=int(p[0][3:]); v=list(map(int,p[1:])); o[i]=(v[3]+v[4],sum(v))
    return o
a=snap(); time.sleep(2); b=snap()
di=sum(b[x][0]-a[x][0] for x in c if x in a); dt=sum(b[x][1]-a[x][1] for x in c if x in a)
print(round(100*(1-di/dt),1) if dt>0 else 0.0)
PY
}

wait_ready() { for _ in $(seq 1 150); do grep -q "Application startup complete" "$1" && return 0; grep -qE "Traceback|CUDA error" "$1" && return 1; sleep 2; done; return 1; }

launch() {  # $1=flags... ; serves PR-B, sets LAST_PGID, echoes log path
  local log="$LOGS/perf_${LABEL}_$RANDOM.log"; cd "$REPO"
  echo "BUILD_SHA=$(git rev-parse HEAD)" > "$log"
  setsid bash -c "PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$CARDS $SRV python scripts/gate_serve.py serve --config $CFG --port $PORT $* >> '$log' 2>&1" </dev/null >/dev/null 2>&1 &
  LAST_PGID=$!; echo "$log"
}
kill_srv() { kill -9 -"$1" 2>/dev/null; pkill -9 -f scripts/gate_serve.py 2>/dev/null; sleep 4
  for w in $(nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader; nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader); do case " $(pgrep -f claim_gpu|tr '\n' ' ') " in *" $w "*) continue;; esac; kill -9 "$w" 2>/dev/null; done; sleep 3; }

run_arm() {  # $1=arm(off|on) $2=round $3=async_flag
  local arm="$1" rnd="$2" flag="$3"
  local log; log=$(launch $flag)
  if ! wait_ready "$log"; then echo "$arm r$rnd FAIL not-ready"; kill_srv "$LAST_PGID"; return 1; fi
  local pgid=$LAST_PGID
  local aport; aport=$(grep -oE "0\.0\.0\.0:[0-9]+" "$log"|grep -oE "[0-9]+$"|tail -1); aport=${aport:-$PORT}
  # %idle gate: wait until my node1 region is quiet (busy<25%), up to ~1 min
  local nb; for _ in $(seq 1 20); do nb=$(node1_busy); awk "BEGIN{exit !($nb<25)}" && break; sleep 3; done
  local odir="$BENCH/${LABEL}_${arm}_r${rnd}"; rm -rf "$odir"
  local sflag=""; [ "$SAMPLES" -gt 0 ] && sflag="--max-samples $SAMPLES"
  # warmup (discarded)
  PYTHONPATH=$REPO $CLI python -m benchmarks.eval.benchmark_tts_seedtts --meta "$META" --model "$MODEL" \
    --port "$aport" --use-existing-server --ref-format references --token-count auto --lang en \
    --max-concurrency "$CONC" --max-samples 8 --generate-only --output-dir "${odir}_warm" >/dev/null 2>&1 || true
  PYTHONPATH=$REPO $CLI python -m benchmarks.eval.benchmark_tts_seedtts --meta "$META" --model "$MODEL" \
    --port "$aport" --use-existing-server --ref-format references --token-count auto --lang en \
    --max-concurrency "$CONC" $sflag --generate-only --output-dir "$odir" >> "$LOGS/perf_${LABEL}_client.log" 2>&1
  local load1; load1=$(cut -d' ' -f1 /proc/loadavg)
  python - "$odir" "$arm" "$rnd" "$nb" "$load1" >> "$RESULTS" <<'PY'
import json,sys
o,arm,rnd,nb,load1=sys.argv[1:6]
d=json.load(open(f"{o}/speed_results.json"))["summary"]
print(f"{arm},{rnd},{d['rtf_mean']},{d['rtf_p99']},{d['output_throughput']},{d['latency_mean_s']},{d['latency_p99_s']},{d['throughput_qps']},{nb},{load1}")
PY
  echo "  $arm r$rnd: rtf=$(tail -1 "$RESULTS"|cut -d, -f3) out_tps=$(tail -1 "$RESULTS"|cut -d, -f5) node1_busy=$nb"
  kill_srv "$pgid"
}

echo ">>> perf [$LABEL] conc=$CONC samples=${SAMPLES:-full} rounds=$ROUNDS (ABAB OFF/ON)"
for r in $(seq 1 "$ROUNDS"); do
  run_arm off "$r" "--async-decode off" || true
  run_arm on  "$r" "--async-decode on"  || true
done
cd "$REPO"
echo "=== [$LABEL] done; aggregate ==="
python "$REPO/scripts/perf_aggregate.py" "$RESULTS"
