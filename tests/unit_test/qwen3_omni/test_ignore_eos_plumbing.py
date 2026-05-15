# SPDX-License-Identifier: Apache-2.0
"""Regression test: ``ignore_eos`` threads through the request stack.

The MMMU decode-throughput-parity lane needs the upstream SGLang sampler to
keep emitting tokens until ``max_new_tokens`` regardless of EOS. The flag
flows: ``ChatCompletionRequest.ignore_eos`` →
``_build_chat_generate_request`` → ``GenerateRequest.sampling.ignore_eos``
→ ``SamplingParams.to_dict()`` (consumed by
``build_sglang_thinker_request``) → upstream
``sglang.srt.sampling.sampling_params.SamplingParams(ignore_eos=...)``.

These tests cover the three handoffs that live in pure Python without GPU
dependencies. The final hop into the upstream SGLang ``SamplingParams``
constructor is exercised indirectly: ``build_sglang_thinker_request``
reads the value from the dict, so any plumbing break above this layer
shows up here.
"""
from __future__ import annotations

import pytest

# sglang_omni.serve depends on fastapi which is not always installed on dev
# machines; skip the module-level imports cleanly when fastapi is missing
# so the rest of the unit-test suite still collects.
pytest.importorskip("fastapi")

from sglang_omni.client.types import SamplingParams as ClientSamplingParams  # noqa: E402
from sglang_omni.serve.openai_api import _build_chat_generate_request  # noqa: E402
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage  # noqa: E402


def _make_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="ping")],
        **kwargs,
    )


def test_ignore_eos_default_false() -> None:
    gen_req = _build_chat_generate_request(_make_request())
    assert gen_req.sampling.ignore_eos is False


def test_ignore_eos_threads_through_to_generate_request() -> None:
    gen_req = _build_chat_generate_request(_make_request(ignore_eos=True))
    assert gen_req.sampling.ignore_eos is True


def test_ignore_eos_in_sampling_params_to_dict() -> None:
    sp = ClientSamplingParams(ignore_eos=True)
    d = sp.to_dict()
    assert d["ignore_eos"] is True


def test_ignore_eos_default_omitted_from_payload() -> None:
    # Default-False ignore_eos still serializes to False so build_sglang_thinker_request's
    # params.get("ignore_eos", False) cast yields a stable False.
    sp = ClientSamplingParams()
    d = sp.to_dict()
    assert d["ignore_eos"] is False
