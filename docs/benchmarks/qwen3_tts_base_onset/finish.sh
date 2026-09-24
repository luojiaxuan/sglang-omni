#!/usr/bin/env bash
# note (luojiaxuan): finishes a sweep whose similarity phase ran out of memory on runaway clips, and runs the
# empirical-set reconnaissance alongside.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUN_DIR/venv/bin/activate"
export SEEDTTS_SIM_CACHE_DIR=/data/cache/seedtts-sim OMP_NUM_THREADS=32
IFS=',' read -ra GPU_UUIDS <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$RUN_DIR/logs"

similarity() {
  local slot="$1" tag="$2" model="$3" out="$RUN_DIR/out/$2"
  export CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$slot]}" PYTHONPATH="$RUN_DIR/pyshim:$RUN_DIR/code"
  cd "$RUN_DIR/code"
  python "$RUN_DIR/similarity_one_by_one.py" --model "$model" --similarity-only --no-ref-text \
    --ref-format references --output-dir "$out/seedtts" > "$RUN_DIR/logs/seedtts-sim-$tag.log" 2>&1
  python "$RUN_DIR/seedtts_onset.py" "$out/seedtts" > "$out/seedtts_onset.jsonl"
  touch "$out/DONE"
}

similarity 0 Qwen3-TTS-12Hz-0.6B-Base-N1-v2 Qwen/Qwen3-TTS-12Hz-0.6B-Base &
first=$!
similarity 1 Qwen3-TTS-12Hz-0.6B-Base-N2-v2 Qwen/Qwen3-TTS-12Hz-0.6B-Base &
second=$!
bash "$RUN_DIR/emp_experiment.sh" > "$RUN_DIR/logs/emp.log" 2>&1 &
third=$!
status=0
for pid in "$first" "$second" "$third"; do wait "$pid" || status=1; done
if [ "$status" = 0 ]; then touch "$RUN_DIR/out/FINISH_DONE"; fi
exit "$status"
