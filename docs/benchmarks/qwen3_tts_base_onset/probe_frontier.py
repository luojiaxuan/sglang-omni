"""Coverage frontier of noise-probe silence sets against generated leading silence and speech.

For each noise color and RMS ceiling, the probe set is the union of codebook-0 ids the codec
assigns to that noise at every level up to the ceiling; it is scored against the leading-silence
and speech frames of an onset grid, re-encoded with the same codec.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import soundfile

from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

FRAME_MS = 80
SPEECH_RMS_DBFS = -35.0
LEVELS_DBFS = (-90, -80, -70, -65, -60, -55, -50, -45, -40)
PROBE_SECONDS = 2.0


def colored_noise(color: str, num_samples: int, generator: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(generator.standard_normal(num_samples))
    frequencies = np.fft.rfftfreq(num_samples)
    frequencies[0] = frequencies[1]
    exponent = {"white": 0.0, "pink": 0.5, "brown": 1.0}[color]
    noise = np.fft.irfft(spectrum / frequencies**exponent, n=num_samples)
    return (noise / np.sqrt(np.mean(noise**2))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()

    tokenizer = load_qwen3_tts_tokenizer(
        args.model, device="cuda:0", dtype="bfloat16", attn_implementation=None
    )
    sample_rate = tokenizer.get_input_sample_rate()

    silence = collections.Counter()
    speech = collections.Counter()
    for line in (args.grid_dir / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        audio, rate = soundfile.read(args.grid_dir / "wav" / record["wav"], dtype="float32")
        codes = tokenizer.encode([audio], sr=rate).audio_codes[0][:, 0].tolist()
        frame = rate * FRAME_MS // 1000
        onset_frame = int(record["onset_peak_ms"] // FRAME_MS)
        for i, token in enumerate(codes):
            rms = 20 * np.log10(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) + 1e-9)
            if i < onset_frame:
                silence[token] += 1
            elif rms > SPEECH_RMS_DBFS:
                speech[token] += 1

    def share(counter: collections.Counter, ids: set[int]) -> float:
        return sum(c for t, c in counter.items() if t in ids) / max(sum(counter.values()), 1)

    generator = np.random.default_rng(0)
    num_samples = int(PROBE_SECONDS * sample_rate)
    for color in ("white", "pink", "brown", "all"):
        colors = ("white", "pink", "brown") if color == "all" else (color,)
        ids: set[int] = set()
        for level in LEVELS_DBFS:
            waveforms = [np.float32(10 ** (level / 20)) * colored_noise(c, num_samples, generator) for c in colors]
            for code in tokenizer.encode(waveforms, sr=sample_rate).audio_codes:
                ids.update(code[:, 0].tolist())
            print(json.dumps({
                "color": color, "ceiling_dbfs": level, "ids": len(ids),
                "covers_silence": round(share(silence, ids), 4),
                "share_of_speech": round(share(speech, ids), 4),
            }))


if __name__ == "__main__":
    main()
