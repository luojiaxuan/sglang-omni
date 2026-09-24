#!/usr/bin/env bash
# note (luojiaxuan): reconnaissance only. Logs the first 12 generated codebook-0 ids of every request from a
# patched code copy, with the grid sent one request at a time so log order matches record order.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUN_DIR/venv/bin/activate"
export OMP_NUM_THREADS=32
IFS=',' read -ra GPU_UUIDS <<< "$CUDA_VISIBLE_DEVICES"
rm -rf "$RUN_DIR/code_dbg" && cp -r "$RUN_DIR/code" "$RUN_DIR/code_dbg"
python - <<PY
from pathlib import Path
p = Path("$RUN_DIR/code_dbg/sglang_omni/models/qwen3_tts/request_builders.py"); s = p.read_text()
anchor = "    code_parts: list[torch.Tensor] = []\n    if data.ref_code is not None and data.ref_code_len:"
assert anchor in s
s = s.replace(anchor, "    logging.getLogger('onsetdbg').warning('ONSETDBG ' + json.dumps([int(c[0]) for c in data.output_codes[:12]]))\n" + anchor, 1)
if "\nimport logging\n" not in s:
    s = s.replace("\nimport json\n", "\nimport json\nimport logging\n", 1)
p.write_text(s)
PY
run() {
  local slot="$1" frames="$2" port=$((18950 + 10 * $1)) tag="Qwen3-TTS-12Hz-1.7B-Base-N$2-dbg"
  export CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$slot]}" PYTHONPATH="$RUN_DIR/pyshim:$RUN_DIR/code_dbg"
  cd "$RUN_DIR/code_dbg"
  python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config examples/configs/qwen3_tts_1_7b.yaml \
    --host 127.0.0.1 --port "$port" --allowed-local-media-path /data \
    --tts_engine.factory.leading_silence_mask_frames "$frames" > "$RUN_DIR/logs/dbg-server-$tag.log" 2>&1 &
  local server=$!
  for _ in $(seq 1 360); do curl -sf "http://127.0.0.1:$port/health" > /dev/null && break; kill -0 "$server" || return 1; sleep 5; done
  python "$RUN_DIR/onset_client.py" --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --refs "$RUN_DIR/refs/meta.lst" \
    --out "$RUN_DIR/out/$tag/grid" --base-url "http://127.0.0.1:$port" --modes xvec --concurrency 1 \
    > "$RUN_DIR/logs/dbg-grid-$tag.log" 2>&1
  kill "$server"; wait "$server" || true
  grep -o 'ONSETDBG .*' "$RUN_DIR/logs/dbg-server-$tag.log" > "$RUN_DIR/out/$tag/codes0.txt"
  touch "$RUN_DIR/out/$tag/DONE_DBG"
}
run 0 0 &
first=$!
run 1 1 &
second=$!
wait "$first"; wait "$second"
touch "$RUN_DIR/out/DBG_DONE"
