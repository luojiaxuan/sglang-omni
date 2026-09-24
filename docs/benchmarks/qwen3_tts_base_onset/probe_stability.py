"""Seed stability of the shipped silence-probe set, and its coverage of generated silence and speech."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

from sglang_omni.models.qwen3_tts import engine_builder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

FRAME_MS = 80
SPEECH_RMS_DBFS = -35.0


def probe_ids(tokenizer, seed: int) -> set[int]:
    sample_rate = tokenizer.get_input_sample_rate()
    generator = np.random.default_rng(seed)
    shapes = [
        engine_builder.colored_noise(e, int(engine_builder.SILENCE_PROBE_SECONDS * sample_rate), generator)
        for e in engine_builder.SILENCE_PROBE_SPECTRAL_EXPONENTS
    ]
    levels = (float("-inf"), *range(engine_builder.SILENCE_PROBE_FLOOR_DBFS, engine_builder.SILENCE_PROBE_CEILING_DBFS + 1, engine_builder.SILENCE_PROBE_STEP_DB))
    waveforms = [np.float32(10 ** (level / 20)) * shape for level in levels for shape in shapes]
    codes = tokenizer.encode(waveforms, sr=sample_rate).audio_codes
    return set(torch.unique(torch.cat([code[:, 0] for code in codes])).tolist())


def main() -> None:
    model = sys.argv[1]
    tokenizer = load_qwen3_tts_tokenizer(model, device="cuda:0", dtype="bfloat16", attn_implementation=None)
    shipped = set(engine_builder.derive_silence_codec_ids(tokenizer, "cpu").tolist())
    sets = [probe_ids(tokenizer, seed) for seed in range(5)]
    assert sets[0] == shipped
    print(json.dumps({"model": model, "shipped_ids": sorted(shipped), "sizes_by_seed": [len(s) for s in sets],
                      "intersection": len(set.intersection(*sets)), "union": len(set.union(*sets))}))
    for grid_dir in map(Path, sys.argv[2:]):
        silence, speech = collections.Counter(), collections.Counter()
        for line in (grid_dir / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            audio, rate = soundfile.read(grid_dir / "wav" / record["wav"], dtype="float32")
            codes = tokenizer.encode([audio], sr=rate).audio_codes[0][:, 0].tolist()
            frame = rate * FRAME_MS // 1000
            onset_frame = int(record["onset_peak_ms"] // FRAME_MS)
            for i, token in enumerate(codes):
                rms = 20 * np.log10(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) + 1e-9)
                if i < onset_frame:
                    silence[(token, min(onset_frame - i, 4))] += 1
                elif rms > SPEECH_RMS_DBFS:
                    speech[token] += 1
        deep = {k: v for k, v in silence.items() if k[1] == 4}
        print(json.dumps({
            "grid": grid_dir.name if grid_dir.name != "grid" else grid_dir.parent.name,
            "covers_all_leading_silence": round(sum(v for (t, _), v in silence.items() if t in shipped) / sum(silence.values()), 4),
            "covers_silence_4plus_frames_before_onset": round(sum(v for (t, _), v in deep.items() if t in shipped) / sum(deep.values()), 4),
            "share_of_speech_frames": round(sum(v for t, v in speech.items() if t in shipped) / sum(speech.values()), 5),
            "speech_frames": sum(speech.values()),
        }))


if __name__ == "__main__":
    main()
