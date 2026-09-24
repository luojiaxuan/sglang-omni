"""Run the SeedTTS similarity phase one clip at a time, so runaway clips capped at max_new_tokens fit in memory."""

from __future__ import annotations

import benchmarks.tasks.tts as tts_tasks
from benchmarks.eval.benchmark_tts_seedtts import main

tts_tasks.SPEAKER_SIMILARITY_BATCH_SIZE = 1

if __name__ == "__main__":
    main()
