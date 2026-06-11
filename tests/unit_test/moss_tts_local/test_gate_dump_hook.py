# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GATE-ONLY audio_codes dump hook (scripts/gate_dump_hook.py).

Confirms the zero-production-trace contract: without the env var, install() is a
pure no-op (the serving function is untouched); with it, install() wraps the
audio_codes producer to dump per request_id, idempotently.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def test_install_is_noop_without_env(monkeypatch):
    pytest.importorskip("sglang")
    monkeypatch.delenv("MOSS_GATE_DUMP_AUDIO_CODES", raising=False)
    import gate_dump_hook
    from sglang_omni.models.moss_tts_local import request_builders as rb

    before = rb.apply_sglang_moss_tts_local_result
    assert gate_dump_hook.install() is False
    assert rb.apply_sglang_moss_tts_local_result is before  # untouched


def test_install_wraps_and_dumps(monkeypatch, tmp_path):
    pytest.importorskip("sglang")
    import numpy as np
    import torch

    monkeypatch.setenv("MOSS_GATE_DUMP_AUDIO_CODES", str(tmp_path))
    import gate_dump_hook
    from sglang_omni.models.moss_tts_local import request_builders as rb

    codes = torch.arange(6).reshape(2, 3)

    def stub(payload, data):  # stand-in for the real result builder
        data.state.audio_codes = codes
        return "result"

    monkeypatch.setattr(rb, "apply_sglang_moss_tts_local_result", stub)
    assert gate_dump_hook.install() is True
    wrapped = rb.apply_sglang_moss_tts_local_result
    assert wrapped is not stub
    assert gate_dump_hook.install() is True  # idempotent

    # Keyed by the public seed (server ignores client request_id).
    payload = types.SimpleNamespace(request_id="speech-uuid")
    data = types.SimpleNamespace(
        seed=777, state=types.SimpleNamespace(audio_codes=None)
    )
    assert wrapped(payload, data) == "result"  # delegates to the wrapped fn
    saved = np.load(tmp_path / "777.npy")  # seed key, not the server request_id
    assert np.array_equal(saved, codes.numpy())

    # No seed -> falls back to the request_id key.
    data_no_seed = types.SimpleNamespace(
        seed=None, state=types.SimpleNamespace(audio_codes=codes)
    )
    wrapped(types.SimpleNamespace(request_id="rid2"), data_no_seed)
    assert (tmp_path / "rid2.npy").exists()
