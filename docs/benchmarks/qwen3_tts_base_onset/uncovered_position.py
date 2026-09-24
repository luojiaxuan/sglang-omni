"""Where in the leading silence the noise-probe set misses: coverage by frames before onset."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile
from probe_frontier import FRAME_MS, colored_noise

from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def main() -> None:
    model, grid_dir, ceiling_dbfs = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
    tokenizer = load_qwen3_tts_tokenizer(model, device="cuda:0", dtype="bfloat16", attn_implementation=None)
    sample_rate = tokenizer.get_input_sample_rate()
    generator = np.random.default_rng(0)
    probe_ids: set[int] = set()
    for level in (float("-inf"), *range(-90, int(ceiling_dbfs) + 1, 5)):
        waveforms = [np.float32(10 ** (level / 20)) * colored_noise(c, 2 * sample_rate, generator) for c in ("white", "pink", "brown")]
        for code in tokenizer.encode(waveforms, sr=sample_rate).audio_codes:
            probe_ids.update(code[:, 0].tolist())
    by_distance = collections.defaultdict(lambda: [0, 0])
    missed = collections.Counter()
    for line in (grid_dir / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        audio, rate = soundfile.read(grid_dir / "wav" / record["wav"], dtype="float32")
        codes = tokenizer.encode([audio], sr=rate).audio_codes[0][:, 0].tolist()
        onset_frame = int(record["onset_peak_ms"] // FRAME_MS)
        for i in range(min(onset_frame, len(codes))):
            distance = min(onset_frame - i, 4)
            by_distance[distance][0] += 1
            by_distance[distance][1] += codes[i] in probe_ids
            if codes[i] not in probe_ids:
                missed[(codes[i], distance)] += 1
    print(json.dumps({"probe_ids": len(probe_ids), "ceiling_dbfs": ceiling_dbfs}))
    for distance in sorted(by_distance):
        total, hit = by_distance[distance]
        print(json.dumps({"frames_before_onset": distance if distance < 4 else ">=4", "frames": total, "covered": round(hit / total, 4)}))
    print(json.dumps({"top_missed_(id,distance)": missed.most_common(15)}))


if __name__ == "__main__":
    main()
