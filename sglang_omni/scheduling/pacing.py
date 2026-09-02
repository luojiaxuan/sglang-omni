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

    ``chunk_frames`` and ``steady_stride_frames`` describe the vocoder's
    emission plan (first chunk, ramp entries, then the steady stride). The
    client only holds audio up to the last completed chunk boundary, so the
    lead is computed from *routed* frames, not generated frames — generated
    frames still parked inside the vocoder's next chunk are no cushion at
    all. Keep these in sync with the vocoder factory's chunk configuration.
    """

    lead_s: float
    resume_lead_s: float
    frame_duration_s: float
    max_resume_per_step: int = 4
    chunk_frames: tuple[int, ...] = ()
    steady_stride_frames: int = 0
    startup_reserve_slots: int = 4

    def __post_init__(self) -> None:
        if self.lead_s <= 0:
            raise ValueError("pacing lead_s must be > 0")
        if not 0 < self.resume_lead_s < self.lead_s:
            raise ValueError("pacing resume_lead_s must be in (0, lead_s)")
        if self.frame_duration_s <= 0:
            raise ValueError("pacing frame_duration_s must be > 0")
        if self.max_resume_per_step <= 0:
            raise ValueError("pacing max_resume_per_step must be > 0")
        if any(f <= 0 for f in self.chunk_frames):
            raise ValueError("pacing chunk_frames entries must be > 0")
        if self.chunk_frames and self.steady_stride_frames <= 0:
            raise ValueError(
                "pacing steady_stride_frames must be > 0 when chunk_frames is set"
            )
        if self.startup_reserve_slots < 0:
            raise ValueError("pacing startup_reserve_slots must be >= 0")


class PlaybackLeadPacer:
    """Holds parked requests and decides pace/resume per scheduling step."""

    def __init__(self, config: PacingConfig) -> None:
        self.config = config
        self._heap: list[tuple[float, int, Any]] = []
        self._held_rids: set[str] = set()
        self._dropped_rids: set[str] = set()
        self._orphaned_drops: list[Any] = []
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
        routed = self._routed_audio_frames(
            data.generation_steps,
            suppressed=bool(getattr(data, "suppress_bootstrap_silence", False)),
        )
        if routed <= 0:
            return None
        routed_s = routed * self.config.frame_duration_s
        return routed_s - (now - data.first_emit_s)

    def _routed_audio_frames(self, generated: int, *, suppressed: bool) -> int:
        """Audio frames the vocoder has released for ``generated`` frames.

        Chunks flush at cumulative boundaries; a suppressed stream decodes
        one extra frame into its first chunk and withholds one frame of
        audio from it.
        """
        chunk_frames = self.config.chunk_frames
        if not chunk_frames:
            return generated
        bump = 1 if suppressed else 0
        boundary = chunk_frames[0] + bump
        if generated < boundary:
            return 0
        routed = boundary - bump
        for stride in chunk_frames[1:]:
            if generated < boundary + stride:
                return routed
            boundary += stride
            routed += stride
        stride = self.config.steady_stride_frames
        routed += ((generated - boundary) // stride) * stride
        return routed

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

    def pop_resumable(
        self,
        now: float | None = None,
        *,
        max_resume: int | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Pop requests due to wake, capped per step.

        ``max_resume`` lowers this step's cap below the configured one —
        the scheduler passes the decode slots it can spare after reserving
        room for requests still waiting for first audio. Returns
        ``(resumable, dropped)``: requests to put back into the decode
        batch, and parked requests that were dropped (aborted) while held —
        the caller owns releasing their resources.
        """
        if now is None:
            now = time.perf_counter()
        cap = self.config.max_resume_per_step
        if max_resume is not None:
            cap = min(cap, max(0, max_resume))
        resumable: list[Any] = []
        dropped: list[Any] = []
        while self._heap and len(resumable) < cap:
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

    def next_wake_at(self) -> float | None:
        """Earliest pending wake time, skipping dropped entries."""
        while self._heap:
            wake_at, _, req = self._heap[0]
            if req.rid in self._dropped_rids:
                heapq.heappop(self._heap)
                self._held_rids.discard(req.rid)
                self._dropped_rids.discard(req.rid)
                self._orphaned_drops.append(req)
                continue
            return wake_at
        return None

    def take_orphaned_drops(self) -> list[Any]:
        """Dropped entries surfaced by ``next_wake_at`` peeks."""
        drops = self._orphaned_drops
        self._orphaned_drops = []
        return drops

    def drain(self) -> list[Any]:
        """Remove and return every parked request, dropped ones included."""
        reqs = [req for _, _, req in self._heap]
        self._heap.clear()
        self._held_rids.clear()
        self._dropped_rids.clear()
        return reqs
