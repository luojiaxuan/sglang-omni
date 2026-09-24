#!/usr/bin/env bash
# note (luojiaxuan): scores similarity one clip at a time for the listed sweep jobs, one GPU each, and marks them DONE.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUN_DIR/venv/bin/activate"
export SEEDTTS_SIM_CACHE_DIR=/data/cache/seedtts-sim OMP_NUM_THREADS=32 PYTHONPATH="$RUN_DIR/pyshim:$RUN_DIR/code"
IFS=',' read -ra GPU_UUIDS <<< "$CUDA_VISIBLE_DEVICES"
cd "$RUN_DIR/code"
pids=()
slot=0
for tag in "$@"; do
  model="Qwen/Qwen3-TTS-12Hz-${tag#Qwen3-TTS-12Hz-}"
  model="${model%%-N*}"
  out="$RUN_DIR/out/$tag"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$slot]}"
    python "$RUN_DIR/similarity_one_by_one.py" --model "$model" --similarity-only --no-ref-text \
      --ref-format references --output-dir "$out/seedtts" > "$RUN_DIR/logs/seedtts-sim-$tag.log" 2>&1
    python "$RUN_DIR/seedtts_onset.py" "$out/seedtts" > "$out/seedtts_onset.jsonl"
    touch "$out/DONE"
  ) &
  pids+=($!)
  slot=$((slot + 1))
done
for pid in "${pids[@]}"; do wait "$pid"; done
