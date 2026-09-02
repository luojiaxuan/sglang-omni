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


def test_qwen3_tts_builder_enables_pacing_by_default() -> None:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    kwargs = Qwen3TtsEngineBuilder().extra_scheduler_kwargs()
    assert kwargs["pacing_lead_s"] == 1.0
    assert kwargs["pacing_resume_lead_s"] == 0.4
    assert kwargs["pacing_frame_duration_s"] == 0.08
    assert kwargs["pacing_max_resume_per_step"] == 4

    disabled = Qwen3TtsEngineBuilder(pacing_lead_s=0.0).extra_scheduler_kwargs()
    assert "pacing_lead_s" not in disabled
