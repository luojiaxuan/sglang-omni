# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_model import conftest as model_conftest
from tests.test_model import test_qwen3_omni_videoamme_talker_tp2_ci as stage11


def test_stage11_fp8_tp2_fixture_uses_safe_startup_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield SimpleNamespace(port=8000)

    monkeypatch.setattr(model_conftest, "_start_qwen3_omni_speech_server", fake_start)

    fixture_fn = model_conftest.qwen3_omni_fp8_talker_server_tp2.__wrapped__
    generator = fixture_fn(None)
    assert next(generator).port == 8000
    with pytest.raises(StopIteration):
        next(generator)

    extra_args = captured["extra_args"]
    assert extra_args[extra_args.index("--thinker-mem-fraction-static") + 1] == "0.38"
    assert extra_args[extra_args.index("--talker-mem-fraction-static") + 1] == "0.18"
    assert extra_args[extra_args.index("--thinker-cuda-graph") + 1] == "off"
    assert extra_args[extra_args.index("--partial-start-min-chunks") + 1] == "3"


def test_stage11_videoamme_config_reduces_payload(monkeypatch, tmp_path_factory) -> None:
    captured: dict[str, object] = {}

    async def fake_run_videoamme_eval(config, compute_wer: bool):
        captured["config"] = config
        captured["compute_wer"] = compute_wer
        return {"summary": {}, "speed": {}, "per_sample": []}

    monkeypatch.setattr(stage11, "run_videoamme_eval", fake_run_videoamme_eval)

    artifacts = stage11.talker_eval_artifacts.__wrapped__(
        SimpleNamespace(port=8000),
        tmp_path_factory,
    )

    config = captured["config"]
    assert captured["compute_wer"] is False
    assert artifacts.per_sample == []
    assert config.video_max_pixels == 150_528
    assert config.warmup == 1
    assert config.extra_request_params == {"talker_prefill_user_context": False}
