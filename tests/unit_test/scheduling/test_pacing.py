# SPDX-License-Identifier: Apache-2.0
"""Unit tests for playback-lead pacing policy."""

from dataclasses import dataclass, field

import pytest

from sglang_omni.scheduling.pacing import PacingConfig, PlaybackLeadPacer


@dataclass
class _Data:
    generation_steps: int = 0
    first_emit_s: float = 0.0
    stream_codec_output: bool = True


@dataclass
class _Req:
    rid: str
    _finished: bool = False
    _omni_data: _Data = field(default_factory=_Data)

    def finished(self) -> bool:
        return self._finished


def _pacer(**overrides) -> PlaybackLeadPacer:
    config = dict(
        lead_s=1.0,
        resume_lead_s=0.4,
        frame_duration_s=0.08,
        max_resume_per_step=2,
    )
    config.update(overrides)
    return PlaybackLeadPacer(PacingConfig(**config))


def test_pacing_config_validation() -> None:
    with pytest.raises(ValueError):
        PacingConfig(lead_s=0.0, resume_lead_s=0.4, frame_duration_s=0.08)
    with pytest.raises(ValueError):
        PacingConfig(lead_s=1.0, resume_lead_s=1.5, frame_duration_s=0.08)
    with pytest.raises(ValueError):
        PacingConfig(lead_s=1.0, resume_lead_s=0.4, frame_duration_s=0.0)


def test_lead_requires_first_emit_and_streaming() -> None:
    pacer = _pacer()
    now = 100.0

    assert pacer.lead_s(None, now) is None
    assert pacer.lead_s(_Data(generation_steps=50), now) is None

    silent = _Data(generation_steps=50, first_emit_s=99.0, stream_codec_output=False)
    assert pacer.lead_s(silent, now) is None

    emitting = _Data(generation_steps=50, first_emit_s=99.0)
    assert pacer.lead_s(emitting, now) == pytest.approx(50 * 0.08 - 1.0)


def test_should_pace_only_above_threshold() -> None:
    pacer = _pacer()
    now = 100.0

    behind = _Data(generation_steps=10, first_emit_s=99.5)
    assert pacer.should_pace(behind, now) is None

    ahead = _Data(generation_steps=30, first_emit_s=99.9)
    lead = pacer.should_pace(ahead, now)
    assert lead == pytest.approx(30 * 0.08 - 0.1)


def test_hold_and_resume_by_deadline_with_step_cap() -> None:
    pacer = _pacer(max_resume_per_step=2)
    now = 100.0
    reqs = [_Req(rid=f"r{i}") for i in range(3)]
    for i, req in enumerate(reqs):
        pacer.hold(req, lead=1.2 + 0.1 * i, now=now)

    assert len(pacer) == 3

    early, dropped = pacer.pop_resumable(now=now + 0.1)
    assert early == [] and dropped == []

    first, dropped = pacer.pop_resumable(now=now + 2.0)
    assert dropped == []
    assert [req.rid for req in first] == ["r0", "r1"]

    rest, _ = pacer.pop_resumable(now=now + 2.0)
    assert [req.rid for req in rest] == ["r2"]
    assert len(pacer) == 0
    assert pacer.paced_total == 3
    assert pacer.resumed_total == 3


def test_wake_time_tracks_lead_over_resume_threshold() -> None:
    pacer = _pacer()
    now = 100.0
    req = _Req(rid="r")
    pacer.hold(req, lead=1.5, now=now)

    not_yet, _ = pacer.pop_resumable(now=now + 1.0)
    assert not_yet == []

    woken, _ = pacer.pop_resumable(now=now + 1.11)
    assert [r.rid for r in woken] == ["r"]


def test_dropped_requests_skip_resume_and_are_returned() -> None:
    pacer = _pacer()
    now = 100.0
    keep = _Req(rid="keep")
    gone = _Req(rid="gone")
    pacer.hold(gone, lead=1.1, now=now)
    pacer.hold(keep, lead=1.2, now=now)
    pacer.drop("gone")
    pacer.drop("never-held")

    woken, dropped = pacer.pop_resumable(now=now + 5.0)

    assert [r.rid for r in woken] == ["keep"]
    assert [r.rid for r in dropped] == ["gone"]
    assert len(pacer) == 0


def test_double_hold_is_idempotent() -> None:
    pacer = _pacer()
    req = _Req(rid="r")
    pacer.hold(req, lead=1.5, now=100.0)
    pacer.hold(req, lead=2.0, now=100.0)

    assert len(pacer) == 1
    woken, _ = pacer.pop_resumable(now=200.0)
    assert len(woken) == 1


def test_drain_returns_everything() -> None:
    pacer = _pacer()
    reqs = [_Req(rid=f"r{i}") for i in range(3)]
    for req in reqs:
        pacer.hold(req, lead=1.5, now=100.0)
    pacer.drop("r1")

    drained = pacer.drain()

    assert {r.rid for r in drained} == {"r0", "r1", "r2"}
    assert len(pacer) == 0
    assert pacer.pop_resumable(now=999.0) == ([], [])


def test_routed_audio_frames_follow_chunk_boundaries() -> None:
    pacer = _pacer(chunk_frames=(2, 4, 8), steady_stride_frames=8)

    expect = {1: 0, 2: 2, 5: 2, 6: 6, 13: 6, 14: 14, 21: 14, 22: 22, 30: 30}
    for generated, routed in expect.items():
        assert (
            pacer._routed_audio_frames(generated, suppressed=False) == routed
        ), generated

    suppressed_expect = {2: 0, 3: 2, 6: 2, 7: 6, 14: 6, 15: 14}
    for generated, routed in suppressed_expect.items():
        assert (
            pacer._routed_audio_frames(generated, suppressed=True) == routed
        ), generated


def test_lead_uses_routed_frames_not_generated() -> None:
    pacer = _pacer(chunk_frames=(2, 4, 8), steady_stride_frames=8)
    now = 100.0

    # 13 generated frames, but only 6 released to the client: the five
    # frames parked inside the vocoder's next chunk are no cushion.
    data = _Data(generation_steps=13, first_emit_s=99.9)
    assert pacer.lead_s(data, now) == pytest.approx(6 * 0.08 - 0.1)

    # Below the first chunk boundary nothing has been released.
    early = _Data(generation_steps=1, first_emit_s=99.9)
    assert pacer.lead_s(early, now) is None


def test_qwen3_tts_builder_keeps_pacing_opt_in() -> None:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    default = Qwen3TtsEngineBuilder().extra_scheduler_kwargs()
    assert "pacing_lead_s" not in default

    enabled = Qwen3TtsEngineBuilder(pacing_lead_s=1.0).extra_scheduler_kwargs()
    assert enabled["pacing_lead_s"] == 1.0
    assert enabled["pacing_resume_lead_s"] == 0.4
    assert enabled["pacing_frame_duration_s"] == 0.08
    assert enabled["pacing_max_resume_per_step"] == 4
    assert enabled["pacing_chunk_frames"] == (2, 4, 8)
    assert enabled["pacing_steady_stride_frames"] == 8
