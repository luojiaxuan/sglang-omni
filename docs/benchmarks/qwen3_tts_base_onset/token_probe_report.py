"""Join the logged first codebook-0 ids with onset records, one request at a time, and tabulate.

The probe grid is sent with concurrency 1, so the i-th logged line belongs to the i-th record.
Reports the frame-0 ids of silent starts and speech starts, and the ids the talker emits inside the
leading silence, against the silence sets derived from the codec encoder.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

FRAME_MS = 80


def main() -> None:
    probe_dir = Path(sys.argv[1])
    silence_sets = {name: set(json.loads(ids)) for name, ids in (arg.split("=", 1) for arg in sys.argv[2:])}
    records = [json.loads(line) for line in (probe_dir / "grid" / "records.jsonl").read_text().splitlines()]
    codes = [json.loads(line.split(" ", 1)[1]) for line in (probe_dir / "codes0.txt").read_text().splitlines()]
    codes = codes[-len(records):]
    assert len(codes) == len(records), (len(codes), len(records))

    silent_start = collections.Counter()
    speech_start = collections.Counter()
    in_silence = collections.Counter()
    for record, first_codes in zip(records, codes):
        onset_frame = int(record["onset_peak_ms"] // FRAME_MS)
        if onset_frame >= 3:
            silent_start[first_codes[0]] += 1
        elif onset_frame == 0:
            speech_start[first_codes[0]] += 1
        in_silence.update(first_codes[:onset_frame])

    def share(counter: collections.Counter, ids: set[int]) -> float:
        return round(sum(c for t, c in counter.items() if t in ids) / max(sum(counter.values()), 1), 4)

    print(json.dumps({
        "probe": probe_dir.name,
        "requests": len(records),
        "silent_starts": sum(silent_start.values()),
        "speech_starts": sum(speech_start.values()),
        "silent_start_frame0_top": silent_start.most_common(12),
        "speech_start_frame0_top": speech_start.most_common(12),
        "leading_silence_ids_top": in_silence.most_common(20),
        "leading_silence_distinct_ids": len(in_silence),
        "coverage": {
            name: {
                "silent_start_frame0": share(silent_start, ids),
                "speech_start_frame0": share(speech_start, ids),
                "leading_silence_frames": share(in_silence, ids),
            }
            for name, ids in silence_sets.items()
        },
    }))


if __name__ == "__main__":
    main()
