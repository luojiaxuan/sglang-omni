#!/usr/bin/env bash
# note (luojiaxuan): container entry for the mask-length sweep. Jobs are (checkpoint, N) pairs dealt
# round-robin to one worker per granted GPU; each job serves, measures, then scores off the TTS server.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$RUN_DIR/venv/.ready" ]; then
  rm -rf "$RUN_DIR/venv"
  uv venv --system-site-packages "$RUN_DIR/venv" -p /usr/bin/python3.12
  echo 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' \
    > "$RUN_DIR/venv/lib/python3.12/site-packages/sglang-image.pth"
  source "$RUN_DIR/venv/bin/activate"
  deps="$RUN_DIR/code/.github/scripts/omni_missing_dependencies.py"
  mapfile -t missing < <(python "$deps" "$RUN_DIR/code/pyproject.toml" | sed "/^$/d")
  for requirement in "${missing[@]}"; do python -m pip install "$requirement"; done
  mapfile -t overrides < <(python "$deps" --overrides "$RUN_DIR/code/pyproject.toml" | sed "/^$/d")
  if [ "${#overrides[@]}" -gt 0 ]; then python -m pip install --no-deps "${overrides[@]}"; fi
  touch "$RUN_DIR/venv/.ready"
fi
source "$RUN_DIR/venv/bin/activate"
export PYTHONPATH="$RUN_DIR/pyshim:$RUN_DIR/code"
export SEEDTTS_SIM_CACHE_DIR=/data/cache/seedtts-sim
# note (luojiaxuan): one OpenMP pool per core per process oversubscribes the CPUs once several servers share the node.
export OMP_NUM_THREADS=32
read -ra MASK_FRAMES <<< "${MASK_FRAMES_LIST:-0 1 2 3 4 6}"
read -ra MODELS <<< "${MODELS:-Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
IFS=',' read -ra GPU_UUIDS <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/out"
cd "$RUN_DIR/code"

python -m benchmarks.metrics.speaker_similarity_assets --warm-cache > "$RUN_DIR/logs/assets.log" 2>&1

JOBS=()
for model in "${MODELS[@]}"; do
  for frames in "${MASK_FRAMES[@]}"; do JOBS+=("$model $frames"); done
done

worker() {
  local slot="$1"
  local port=$((18800 + 10 * slot))
  export CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$slot]}"
  for ((job = slot; job < ${#JOBS[@]}; job += ${#GPU_UUIDS[@]})); do
    local model frames
    read -r model frames <<< "${JOBS[$job]}"
    local tag="${model##*/}-N$frames" out="$RUN_DIR/out/${model##*/}-N$frames"
    if [ -f "$out/DONE" ]; then continue; fi
    mkdir -p "$out"
    local config="examples/configs/qwen3_tts_1_7b.yaml"
    case "$model" in *0.6B*) config="examples/configs/qwen3_tts_0_6b.yaml" ;; esac
    python -m sglang_omni.cli serve --model-path "$model" --config "$config" \
      --host 127.0.0.1 --port "$port" --allowed-local-media-path "$RUN_DIR/refs" \
      --tts_engine.factory.leading_silence_mask_frames "$frames" \
      > "$RUN_DIR/logs/server-$tag.log" 2>&1 &
    local server=$!
    for _ in $(seq 1 360); do
      if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then break; fi
      if ! kill -0 "$server" 2> /dev/null; then echo "server exited: $tag" >&2; return 1; fi
      sleep 5
    done
    curl -sf "http://127.0.0.1:$port/health" > /dev/null
    python "$RUN_DIR/onset_client.py" --model "$model" --refs "$RUN_DIR/refs/meta.lst" \
      --out "$out/grid" --base-url "http://127.0.0.1:$port" --modes xvec \
      > "$RUN_DIR/logs/grid-$tag.log" 2>&1
    python -m benchmarks.eval.benchmark_tts_seedtts --model "$model" --port "$port" \
      --use-existing-server --generate-only --no-ref-text --ref-format references \
      --seed 0 --concurrency 16 --skip-gpu-cleanup --output-dir "$out/seedtts" \
      > "$RUN_DIR/logs/seedtts-generate-$tag.log" 2>&1
    kill "$server"
    wait "$server" || true
    python -m benchmarks.eval.benchmark_tts_seedtts --model "$model" --port "$port" \
      --transcribe-only --no-ref-text --ref-format references --skip-gpu-cleanup \
      --output-dir "$out/seedtts" > "$RUN_DIR/logs/seedtts-wer-$tag.log" 2>&1
    python -m benchmarks.eval.benchmark_tts_seedtts --model "$model" \
      --similarity-only --no-ref-text --ref-format references \
      --output-dir "$out/seedtts" > "$RUN_DIR/logs/seedtts-sim-$tag.log" 2>&1
    python "$RUN_DIR/seedtts_onset.py" "$out/seedtts" > "$out/seedtts_onset.jsonl"
    touch "$out/DONE"
  done
}

pids=()
for slot in "${!GPU_UUIDS[@]}"; do
  worker "$slot" > "$RUN_DIR/logs/worker-$slot.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" = 0 ]; then touch "$RUN_DIR/out/ALL_DONE"; fi
exit "$status"
