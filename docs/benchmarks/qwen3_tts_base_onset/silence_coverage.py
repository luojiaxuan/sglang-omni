"""Codebook-0 ids the codec assigns to the leading silence and to the speech of generated clips.

Re-encodes onset-grid WAVs with the checkpoint's speech tokenizer and reports how much of the
leading silence the synthetic-probe silence set covers, and how often speech frames fall in it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import soundfile

from sglang_omni.models.qwen3_tts.engine_builder import derive_silence_codec_ids
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

FRAME_MS = 80
SPEECH_RMS_DBFS = -35.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("grid_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    tokenizer = load_qwen3_tts_tokenizer(
        args.model, device="cuda:0", dtype="bfloat16", attn_implementation=None
    )
    probe_ids = set(derive_silence_codec_ids(tokenizer, "cpu").tolist())
    print(json.dumps({"probe_silence_ids": sorted(probe_ids)}))

    for grid_dir in args.grid_dirs:
        silence = collections.Counter()
        speech = collections.Counter()
        silence_by_ref: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for line in (grid_dir / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            audio, sample_rate = soundfile.read(grid_dir / "wav" / record["wav"], dtype="float32")
            codes = tokenizer.encode([audio], sr=sample_rate).audio_codes[0][:, 0].tolist()
            frame = sample_rate * FRAME_MS // 1000
            rms_dbfs = [
                20 * np.log10(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) + 1e-9)
                for i in range(len(codes))
            ]
            onset_frame = int(record["onset_peak_ms"] // FRAME_MS)
            for i, token in enumerate(codes):
                if i < onset_frame:
                    silence[token] += 1
                    silence_by_ref[record["ref"]][token] += 1
                elif rms_dbfs[i] > SPEECH_RMS_DBFS:
                    speech[token] += 1

        def covered(counter: collections.Counter, ids: set[int]) -> float:
            total = sum(counter.values())
            return sum(count for token, count in counter.items() if token in ids) / max(total, 1)

        empirical_ids = {token for token, count in silence.items() if count >= 3}
        print(json.dumps({
            "grid": str(grid_dir),
            "leading_silence_frames": sum(silence.values()),
            "speech_frames": sum(speech.values()),
            "probe_covers_silence": round(covered(silence, probe_ids), 4),
            "probe_share_of_speech": round(covered(speech, probe_ids), 4),
            "probe_covers_silence_by_ref": {
                ref: round(covered(counter, probe_ids), 4) for ref, counter in sorted(silence_by_ref.items())
            },
            "silence_top_ids": silence.most_common(25),
            "empirical_ids_count": len(empirical_ids),
            "empirical_covers_silence": round(covered(silence, empirical_ids), 4),
            "empirical_share_of_speech": round(covered(speech, empirical_ids), 4),
        }))


if __name__ == "__main__":
    main()
