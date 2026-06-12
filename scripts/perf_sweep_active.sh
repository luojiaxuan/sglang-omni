#!/usr/bin/env bash
# Active-mode PR-B async-decode perf sweep. Concurrency c1/c2/c4/c8/c16 x ABAB
# OFF/ON, n=5, mean +/- 95% CI (scripts/perf_aggregate.py). Independent archive;
# does NOT touch #758 (which goes ready on the correctness gate). The numbers are
# the post-merge flag-flip evidence.
#
# Active acquisition (cluster ownership rule): a card with util ~0 across several
# samples but resident memory is stale -> kill its non-claim PIDs and reclaim it.
# A card with util > 0 (or util that jumps across samples) is an active job ->
# never touched. Prefer 2 same-node cards (AR+codec split, examples config); fall
# back to 1 card colocate (annotated). Self-protect: hold a claim on the cards
# between rounds (killed during each timed run; see perf_abab.sh). Resumable: each
# concurrency CSV is append-only and re-skips completed (arm,round) rows.
set -uo pipefail
REPO=/data/moss-v15-ar/sglang-omni-prb
ARCHIVE=/data/moss-v15-ar/bench/perf-sweep
LOGS=/data/moss-v15-ar/logs/perf-sweep
CLAIM=/data/claim_gpu.py
CFG2=examples/configs/moss_tts_local.yaml
CFGCO=/data/moss-v15-ar/moss_colocate.yaml
CONCS=${CONCS:-1 2 4 8 16}; ROUNDS=${ROUNDS:-5}; SAMPLES=${SAMPLES:-50}; PORT=${PORT:-8040}
mkdir -p "$ARCHIVE" "$LOGS"
STATUS=$ARCHIVE/status.txt
say() { echo "[$(date +%H:%M:%S)] $*"; echo "$*" > "$STATUS"; }

umax() {  # $1=card -> max util over 4 samples (~3s); catches intermittent jobs
  local c=$1 mx=0 u
  for _ in 1 2 3 4; do
    u=$(nvidia-smi -i "$c" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    [ "${u:-0}" -gt "$mx" ] && mx=${u:-0}; sleep 0.7
  done
  echo "$mx"
}
mem_of() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; }

try_reclaim() {  # $1=card -> 0 if free/reclaimed
  local c=$1 m u pids cp p killed=0
  m=$(mem_of "$c"); [ "${m:-0}" -lt 2000 ] && return 0          # already free
  u=$(umax "$c"); [ "${u:-100}" -ge 5 ] && return 1            # active (or jumpy) -> skip
  pids=$(nvidia-smi -i "$c" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [ -z "$pids" ] && { m=$(mem_of "$c"); [ "${m:-9999}" -lt 2000 ] && return 0 || return 1; }
  cp=" $(pgrep -f claim_gpu | tr '\n' ' ') "
  for p in $pids; do case "$cp" in *" $p "*) continue;; esac; kill -9 "$p" 2>/dev/null; killed=1; done
  [ "$killed" = 1 ] && sleep 3
  m=$(mem_of "$c"); [ "${m:-9999}" -lt 2000 ]
}

acquire() {  # $1=min cards. echo same-node cards (2 preferred); empty if < min
  local min=${1:-1} got=()
  for grp in "4 5 6 7" "0 1 2 3"; do      # try node1 then node0, keep same-node
    local g=()
    for c in $grp; do [ ${#g[@]} -ge 2 ] && break; try_reclaim "$c" && g+=("$c"); done
    if [ ${#g[@]} -ge 2 ]; then echo "${g[0]} ${g[1]}"; return; fi
    [ ${#g[@]} -gt ${#got[@]} ] && got=("${g[@]}")
  done
  if [ ${#got[@]} -ge "$min" ]; then echo "${got[0]}"; else echo ""; fi
}

# --- wait for cards, then pin config/cores by node. Prefer 2 (AR+codec split)
# for the first ~30 min; only then settle for a 1-card colocate run. ---
cards=""
for attempt in $(seq 1 240); do          # up to ~4h, 60s cadence
  min=2; [ "$attempt" -gt 30 ] && min=1
  cards=$(acquire "$min"); [ -n "$cards" ] && break
  say "waiting for cards (attempt $attempt, want>=$min; box busy)"; sleep 60
done
[ -z "$cards" ] && { say "timeout-no-cards"; exit 0; }
set -- $cards
if [ "$#" -ge 2 ]; then
  PERF_CARDS="$1,$2"; PERF_CFG=$CFG2; MODE="2card-split($1=AR,$2=codec)"
else
  PERF_CARDS="$1"; PERF_CFG=$CFGCO; MODE="1card-colocate($1)"
fi
case "$1" in 4|5|6|7) MB=1; SRVC="32-55,96-119"; CLIC="56-63,120-127"; BUSY="32-63,96-127";;
            *)        MB=0; SRVC="0-23,64-87";   CLIC="24-31,88-95";   BUSY="0-31,64-95";; esac
say "acquired cards=$PERF_CARDS mode=$MODE node-membind=$MB"
echo "config: cards=$PERF_CARDS mode=$MODE cfg=$PERF_CFG membind=$MB samples=$SAMPLES rounds=$ROUNDS" > "$ARCHIVE/CONFIG.txt"

export REPO BENCH="$ARCHIVE" LOGS PERF_CARDS PERF_CFG PERF_CLAIM="$CLAIM" PERF_PORT="$PORT"
export PERF_SRV_CORES="$SRVC" PERF_CLI_CORES="$CLIC" PERF_MEMBIND="$MB" PERF_BUSY_CORES="$BUSY"

for conc in $CONCS; do
  say "sweep c=$conc (cards=$PERF_CARDS mode=$MODE)"
  # inter-concurrency VRAM rescan: no server runs between blocks, so any
  # non-claim compute-app resident on our held cards is an intruder -> clear it.
  for c in ${PERF_CARDS//,/ }; do
    cp=" $(pgrep -f claim_gpu | tr '\n' ' ') "
    for w in $(nvidia-smi -i "$c" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      case "$cp" in *" $w "*) continue;; esac
      kill -9 "$w" 2>/dev/null
    done
  done
  bash "$REPO/scripts/perf_abab.sh" "c${conc}" "$conc" "$SAMPLES" "$ROUNDS" >> "$LOGS/sweep_c${conc}.log" 2>&1 || say "c=$conc had failures (see log)"
done

# combined summary across all concurrencies
say "aggregating"
{
  echo "# PR-B async-decode perf sweep ($MODE, samples=$SAMPLES, n=$ROUNDS, 95% CI)"
  echo "# $(cat $ARCHIVE/CONFIG.txt)"
  for conc in $CONCS; do
    echo; echo "##### c=$conc #####"
    python "$REPO/scripts/perf_aggregate.py" "$ARCHIVE/perf_c${conc}.csv" 2>&1
  done
} > "$ARCHIVE/SUMMARY.txt" 2>&1
pkill -9 -f "label perfsweepclaim" 2>/dev/null || true
say "done mode=$MODE"
