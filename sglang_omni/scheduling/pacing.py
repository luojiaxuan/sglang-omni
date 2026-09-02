# SPDX-License-Identifier: Apache-2.0
"""Playback-lead pacing policy for streaming AR requests.

A streaming TTS request that has already emitted audio far ahead of the
client's playback position gains nothing from decoding further right now,
while its decode slot is scarce for streams near their playback deadline and
for requests still waiting for first audio. The pacer tracks how far ahead of
playback each emitting request is and parks the comfortable ones — the
scheduler keeps their KV resident and simply leaves them out of the decode
batch until their lead drains.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from itertools import count
from typing import Any


@dataclass(frozen=True)
class PacingConfig:
    """Playback-lead pacing thresholds.

    ``lead_s`` parks a request once its emitted audio runs this far ahead of
    realtime playback; ``resume_lead_s`` is the lead at which it becomes
    eligible to decode again. Parked requests wake by ascending deadline, at
    most ``max_resume_per_step`` per scheduling step, so a burst that parked
    together does not rejoin as one oversized batch.
    """

    lead_s: float
    resume_lead_s: float
    frame_duration_s: float
    max_resume_per_step: int = 4

    def __post_init__(self) -> None:
        if self.lead_s <= 0:
            raise ValueError("pacing lead_s must be > 0")
        if not 0 < self.resume_lead_s < self.lead_s:
            raise ValueError("pacing resume_lead_s must be in (0, lead_s)")
        if self.frame_duration_s <= 0:
            raise ValueError("pacing frame_duration_s must be > 0")
        if self.max_resume_per_step <= 0:
            raise ValueError("pacing max_resume_per_step must be > 0")


class PlaybackLeadPacer:
    """Holds parked requests and decides pace/resume per scheduling step."""

    def __init__(self, config: PacingConfig) -> None:
        self.config = config
        self._heap: list[tuple[float, int, Any]] = []
        self._held_rids: set[str] = set()
        self._dropped_rids: set[str] = set()
        self._tiebreak = count()
        self.paced_total = 0
        self.resumed_total = 0

    def __len__(self) -> int:
        return len(self._held_rids)

    def lead_s(self, data: Any, now: float | None = None) -> float | None:
        """Playback lead of a request, or None when it is not paceable.

        A request is paceable only after its first emitted chunk (protecting
        time to first audio) and only when it streams codec output — a
        non-streaming request has no playback clock to pace against.
        """
        if data is None:
            return None
        if not data.first_emit_s:
            return None
        # note (luojiaxuan): only Qwen3-TTS request data carries this field;
        # other AR models' requests are simply never paceable.
        if not getattr(data, "stream_codec_output", False):
            return None
        if now is None:
            now = time.perf_counter()
        generated_s = data.generation_steps * self.config.frame_duration_s
        return generated_s - (now - data.first_emit_s)

    def should_pace(self, data: Any, now: float | None = None) -> float | None:
        """Return the lead when the request should be parked, else None."""
        lead = self.lead_s(data, now)
        if lead is None or lead <= self.config.lead_s:
            return None
        return lead

    def hold(self, req: Any, lead: float, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        rid = req.rid
        if rid in self._held_rids:
            return
        wake_at = now + max(0.0, lead - self.config.resume_lead_s)
        heapq.heappush(self._heap, (wake_at, next(self._tiebreak), req))
        self._held_rids.add(rid)
        self._dropped_rids.discard(rid)
        self.paced_total += 1

    def drop(self, rid: str) -> None:
        """Mark a parked request as gone (abort); it is skipped on wake."""
        if rid in self._held_rids:
            self._dropped_rids.add(rid)

    def held_rids(self) -> frozenset[str]:
        return frozenset(self._held_rids)

    def pop_resumable(self, now: float | None = None) -> tuple[list[Any], list[Any]]:
        """Pop requests due to wake, capped per step.

        Returns ``(resumable, dropped)``: requests to put back into the decode
        batch, and parked requests that were dropped (aborted) while held —
        the caller owns releasing their resources.
        """
        if now is None:
            now = time.perf_counter()
        resumable: list[Any] = []
        dropped: list[Any] = []
        while self._heap and len(resumable) < self.config.max_resume_per_step:
            wake_at, _, req = self._heap[0]
            rid = req.rid
            if rid in self._dropped_rids:
                heapq.heappop(self._heap)
                self._held_rids.discard(rid)
                self._dropped_rids.discard(rid)
                dropped.append(req)
                continue
            if wake_at > now:
                break
            heapq.heappop(self._heap)
            self._held_rids.discard(rid)
            resumable.append(req)
        self.resumed_total += len(resumable)
        return resumable, dropped

    def drain(self) -> list[Any]:
        """Remove and return every parked request, dropped ones included."""
        reqs = [req for _, _, req in self._heap]
        self._heap.clear()
        self._held_rids.clear()
        self._dropped_rids.clear()
        return reqs
