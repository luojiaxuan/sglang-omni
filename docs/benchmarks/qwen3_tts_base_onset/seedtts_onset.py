"""Print one JSON line with the audible onset of every WAV under a SeedTTS output dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile
from onset_client import peak_onset_ms, rms_onset_ms


def main() -> None:
    root = Path(sys.argv[1])
    for wav_path in sorted(root.rglob("*.wav")):
        audio, sr = soundfile.read(wav_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        print(
            json.dumps(
                {
                    "wav": str(wav_path.relative_to(root)),
                    "duration_s": round(len(audio) / sr, 4),
                    "onset_peak_ms": peak_onset_ms(audio, sr),
                    "onset_rms40_ms": rms_onset_ms(audio, sr),
                }
            )
        )


if __name__ == "__main__":
    main()
