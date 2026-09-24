#!/usr/bin/env bash
# note (luojiaxuan): reconnaissance only. Serves 1.7B with the silence set taken from re-encoded baseline leading silence, N=1 and N=2, onset grid only.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUN_DIR/venv/bin/activate"
export OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES="$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f3)"
export PYTHONPATH="$RUN_DIR:$RUN_DIR/pyshim:$RUN_DIR/code"
python - <<'PY'
import collections, json, numpy as np, soundfile
from pathlib import Path
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
run = Path(__import__("os").environ["PYTHONPATH"].split(":")[0])
grid = run / "out/Qwen3-TTS-12Hz-1.7B-Base-N0/grid"
tokenizer = load_qwen3_tts_tokenizer("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda:0", dtype="bfloat16", attn_implementation=None)
silence = collections.Counter()
for line in (grid / "records.jsonl").read_text().splitlines():
    record = json.loads(line)
    audio, rate = soundfile.read(grid / "wav" / record["wav"], dtype="float32")
    codes = tokenizer.encode([audio], sr=rate).audio_codes[0][:, 0].tolist()
    silence.update(codes[: int(record["onset_peak_ms"] // 80)])
ids = sorted(t for t, c in silence.items() if c >= 3)
(run / "empirical_ids_1.7B.json").write_text(json.dumps(ids))
print("empirical ids", len(ids), ids)
PY
rm -rf "$RUN_DIR/code_emp" && cp -r "$RUN_DIR/code" "$RUN_DIR/code_emp"
python - <<PY
from pathlib import Path
p = Path("$RUN_DIR/code_emp/sglang_omni/models/qwen3_tts/engine_builder.py"); s = p.read_text()
s = s.replace("def derive_silence_codec_ids(speech_tokenizer: Any, device: str) -> torch.Tensor:\n", "def derive_silence_codec_ids(speech_tokenizer: Any, device: str) -> torch.Tensor:\n    import json\n    return torch.tensor(json.loads(open('$RUN_DIR/empirical_ids_1.7B.json').read()), dtype=torch.long).to(device)\n", 1)
p.write_text(s)
PY
export PYTHONPATH="$RUN_DIR/pyshim:$RUN_DIR/code_emp"
cd "$RUN_DIR/code_emp"
for frames in 1 2; do
  tag="Qwen3-TTS-12Hz-1.7B-Base-N$frames-emp"
  python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config examples/configs/qwen3_tts_1_7b.yaml \
    --host 127.0.0.1 --port 18900 --allowed-local-media-path /data \
    --tts_engine.factory.leading_silence_mask_frames "$frames" > "$RUN_DIR/logs/emp-server-$tag.log" 2>&1 &
  server=$!
  for _ in $(seq 1 360); do curl -sf http://127.0.0.1:18900/health > /dev/null && break; kill -0 "$server" || exit 1; sleep 5; done
  python "$RUN_DIR/onset_client.py" --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --refs "$RUN_DIR/refs/meta.lst" \
    --out "$RUN_DIR/out/$tag/grid" --base-url http://127.0.0.1:18900 --modes xvec > "$RUN_DIR/logs/emp-grid-$tag.log" 2>&1
  kill "$server"; wait "$server" || true
  for _ in $(seq 1 120); do [ "$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2048 ] && break; sleep 2; done
  touch "$RUN_DIR/out/$tag/DONE_GRID"
done
echo EMP_DONE
