"""Markdown table of every pulled sweep job: onset grid, SeedTTS WER, similarity, SeedTTS onset, runaways."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUNAWAY_SECONDS = 163.8


def onset_cells(values: list[float]) -> list[str]:
    v = np.array(values)
    return [f"{np.median(v):.0f}", f"{np.percentile(v, 90):.0f}", f"{np.mean(v > 160):.0%}"]


def main() -> None:
    print("| job | grid median | grid p90 | grid > 160 ms | SeedTTS WER | SIM | SeedTTS median | SeedTTS p90 | SeedTTS > 160 ms | runaways |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for job_dir in map(Path, sys.argv[1:]):
        grid = [json.loads(line)["onset_peak_ms"] for line in (job_dir / "grid" / "records.jsonl").read_text().splitlines()]
        seedtts = job_dir / "seedtts"
        wer = json.loads((seedtts / "wer_results.json").read_text())["summary"]["wer_corpus"]
        similarity = json.loads((seedtts / "similarity_results.json").read_text())["summary"]["speaker_similarity_mean"]
        onsets = [json.loads(line)["onset_peak_ms"] for line in (job_dir / "seedtts_onset.jsonl").read_text().splitlines()]
        generated = json.loads((seedtts / "generated.json").read_text())
        runaways = sum(1 for item in generated if (item["audio_duration_s"] or 0) >= RUNAWAY_SECONDS)
        cells = [job_dir.name, *onset_cells(grid), f"{100 * wer:.3f}%", f"{similarity:.2f}",
                 *onset_cells([o for o in onsets if o is not None]), str(runaways)]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
