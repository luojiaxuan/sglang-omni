# SPDX-License-Identifier: Apache-2.0
"""Client.completion() surfaces RL-rollout artifacts (logprobs, weight_version)."""

from __future__ import annotations

import asyncio
from typing import Any

from sglang_omni.client import Client
from sglang_omni.client.types import GenerateRequest


class _SubmitStubCoordinator:
    """Non-streaming coordinator stub: completion() only needs submit()."""

    def __init__(self, result: Any) -> None:
        self._result = result

    async def submit(self, request_id: str, omni_request: Any) -> Any:
        del request_id, omni_request
        return self._result


class _StreamStubCoordinator:
    """Streaming coordinator stub: yields the given StreamMessages in order."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def stream(self, request_id: str, omni_request: Any):
        del request_id, omni_request
        for message in self._messages:
            yield message


def test_completion_surfaces_logprobs_and_weight_version() -> None:
    # `output_token_logprobs` carries (logprob, token_id) pairs, mirroring
    # sglang's native meta_info; msgpack transports them as JSON arrays.
    result = {
        "text": "hello",
        "finish_reason": "stop",
        "output_token_logprobs": [[-0.1, 11], [-0.2, 22], [-0.3, 33]],
        "weight_version": "v7",
        "completion_tokens": 3,
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.output_token_logprobs == [[-0.1, 11], [-0.2, 22], [-0.3, 33]]
    assert out.weight_version == "v7"


def test_completion_without_logprobs_leaves_fields_none() -> None:
    result = {"text": "hello", "finish_reason": "stop"}
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.output_token_logprobs is None
    assert out.weight_version is None


def test_completion_surfaces_rollout_from_multiterminal_decode() -> None:
    # qwen3-omni merges a text `decode` terminal with an audio terminal; the
    # rollout artifacts ride on the decode result.
    result = {
        "decode": {
            "text": "hi",
            "finish_reason": "stop",
            "output_token_logprobs": [[-0.5, 9]],
            "weight_version": "v9",
        },
        "code2wav": {"audio_data": [0.0, 0.1, -0.1], "sample_rate": 24000},
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.text == "hi"
    assert out.audio is not None
    assert out.output_token_logprobs == [[-0.5, 9]]
    assert out.weight_version == "v9"


def test_completion_concatenates_streamed_logprobs() -> None:
    from sglang_omni.proto import StreamMessage

    messages = [
        StreamMessage(
            request_id="r1",
            from_stage="decode",
            chunk={
                "text": "he",
                "output_token_logprobs": [[-0.1, 11], [-0.2, 22]],
                "weight_version": "v7",
            },
            stage_name="decode",
            modality="text",
        ),
        StreamMessage(
            request_id="r1",
            from_stage="decode",
            chunk={
                "text": "llo",
                "output_token_logprobs": [[-0.3, 33]],
                "finish_reason": "stop",
                "weight_version": "v7",
            },
            stage_name="decode",
            modality="text",
        ),
    ]
    client = Client(_StreamStubCoordinator(messages))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=True), request_id="r1")
    )

    assert out.text == "hello"
    assert out.output_token_logprobs == [[-0.1, 11], [-0.2, 22], [-0.3, 33]]
    assert out.weight_version == "v7"
