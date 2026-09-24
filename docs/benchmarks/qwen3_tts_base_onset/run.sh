#!/usr/bin/env bash
# note (luojiaxuan): container entry. Serves each Base checkpoint in turn on one GPU and runs the onset grid against it.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
# note (luojiaxuan): same layering as the CI venv, image site-packages plus the pinned deps the image lacks.
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
PORT=18731
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/out"

for model in ${MODELS:-Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-0.6B-Base}; do
  tag="${model##*/}"
  if [ -f "$RUN_DIR/out/$tag/DONE" ]; then continue; fi
  config="$RUN_DIR/code/examples/configs/qwen3_tts_1_7b.yaml"
  case "$tag" in *0.6B*) config="$RUN_DIR/code/examples/configs/qwen3_tts_0_6b.yaml" ;; esac
  python3 -m sglang_omni.cli serve --model-path "$model" --config "$config" \
    --host 127.0.0.1 --port "$PORT" --allowed-local-media-path "$RUN_DIR/refs" \
    > "$RUN_DIR/logs/server-$tag.log" 2>&1 &
  server=$!
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
    if ! kill -0 "$server" 2> /dev/null; then echo "server exited: $tag" >&2; exit 1; fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null
  python3 "$RUN_DIR/onset_client.py" --model "$model" --refs "$RUN_DIR/refs/meta.lst" \
    --out "$RUN_DIR/out/$tag" --base-url "http://127.0.0.1:$PORT" \
    > "$RUN_DIR/logs/client-$tag.log" 2>&1
  kill "$server"
  wait "$server" || true
  touch "$RUN_DIR/out/$tag/DONE"
done
touch "$RUN_DIR/out/ALL_DONE"
