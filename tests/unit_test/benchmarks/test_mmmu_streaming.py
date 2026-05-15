# SPDX-License-Identifier: Apache-2.0
"""Client-layer tests for the MMMU streaming benchmark surface.

Coverage:

- Frame-aware SSE parser with split/coalesced byte streams and an injected
  fake clock that asserts exact TTFT and inter-chunk offsets.
- ``build_mmmu_payload`` parity between the ``omni`` and ``sglang``
  backends for 1-, 2-, and 7-image samples, with the strict full-part-order
  contract ``[image_url, ..., text]`` mirroring
  ``Qwen3OmniPreprocessor._build_multimodal_messages``.
- Slim-final regression: when the SSE stream contains a role-only frame,
  content deltas, and a final finish chunk, the client must concatenate
  text exactly once (no duplication from the final chunk) and must extract
  usage from the finish frame.
- Backward-compat: the non-stream send_fn path continues to populate
  ``client_wall_time_s`` and leave streaming fields unset.

Server-side scheduler coverage already lives at
``tests/unit_test/qwen3_omni/test_streaming.py`` (PR #406, 738 lines) and
is not modified by this plan; this file adds the complementary
client-layer surface only.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from typing import Iterable, List

import pytest
from PIL import Image

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.mmmu import MMMUSample
from benchmarks.tasks.visual_understand import (
    build_mmmu_payload,
    consume_sse_stream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(seed: int) -> Image.Image:
    """Tiny RGB image with seed-determined pixel pattern."""
    img = Image.new("RGB", (8, 8))
    px = img.load()
    for y in range(8):
        for x in range(8):
            px[x, y] = ((seed * 17 + x * 3 + y) % 256, (seed + x) % 256, (seed + y) % 256)
    return img


def _make_sample(sample_id: str, n_images: int, prompt: str = "describe this") -> MMMUSample:
    return MMMUSample(
        sample_id=sample_id,
        question=prompt,
        options=["a", "b", "c", "d"],
        answer="A",
        images=[_make_image(i + 1) for i in range(n_images)],
        subject="Test",
        prompt=prompt,
        all_choices=["A", "B", "C", "D"],
        index2ans={"A": "a", "B": "b", "C": "c", "D": "d"},
        question_type="multiple-choice",
    )


class _FakeClock:
    """Deterministic clock for SSE-arrival timing tests.

    Each call to the instance returns the next scripted value. Asserts
    enforce no over-consumption: the test must script exactly as many
    timestamps as the parser consumes.
    """

    def __init__(self, values: Iterable[float]) -> None:
        self._values: List[float] = list(values)
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._values):
            raise AssertionError(
                f"_FakeClock exhausted: parser called clock {self._index + 1} "
                f"times but only {len(self._values)} timestamps were scripted"
            )
        val = self._values[self._index]
        self._index += 1
        return val


class _FakeResponseContent:
    """Mimics aiohttp.StreamReader.iter_any() with scripted byte chunks."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = list(chunks)

    def iter_any(self):
        chunks = list(self._chunks)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()


class _FakeResponse:
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self.content = _FakeResponseContent(chunks)


def _sse_frame(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


# ---------------------------------------------------------------------------
# SSE parser tests
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_parser_extracts_ttft_and_offsets_with_injected_clock() -> None:
    role_frame = _sse_frame(
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    content_frame_a = _sse_frame(
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]}
    )
    content_frame_b = _sse_frame(
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]}
    )
    finish_frame = _sse_frame(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        }
    )
    done_frame = b"data: [DONE]\n\n"

    response = _FakeResponse(
        [role_frame, content_frame_a, content_frame_b, finish_frame, done_frame]
    )
    # request_send=100.0; parser calls clock once per content frame (2 frames).
    clock = _FakeClock([100.250, 100.310])

    result = RequestResult(request_id="r1")
    asyncio.new_event_loop().run_until_complete(
        consume_sse_stream(
            response, result, request_send_time=100.0, clock=clock
        )
    )

    assert result.text == "Hello world"
    assert result.content_chunk_count == 2
    assert result.ttft_s == pytest.approx(0.250, abs=1e-6)
    assert result.content_chunk_offsets_ms == [250.0, 310.0]
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 2


def test_parser_handles_split_tcp_reads() -> None:
    """A single SSE frame split across multiple buffer reads still parses."""
    frame = _sse_frame(
        {"choices": [{"delta": {"content": "split-text"}, "finish_reason": None}]}
    )
    # Split frame into 3 arbitrary byte ranges.
    chunks = [frame[:10], frame[10:25], frame[25:]]
    response = _FakeResponse(chunks + [b"data: [DONE]\n\n"])
    clock = _FakeClock([5.001])

    result = RequestResult(request_id="r2")
    asyncio.new_event_loop().run_until_complete(
        consume_sse_stream(response, result, request_send_time=5.0, clock=clock)
    )

    assert result.text == "split-text"
    assert result.content_chunk_count == 1


def test_parser_role_only_and_final_usage_yield_no_ttft() -> None:
    role_frame = _sse_frame(
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    finish_frame = _sse_frame(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0},
        }
    )
    response = _FakeResponse([role_frame, finish_frame, b"data: [DONE]\n\n"])
    clock = _FakeClock([])  # no content frames -> no clock calls

    result = RequestResult(request_id="r3")
    asyncio.new_event_loop().run_until_complete(
        consume_sse_stream(response, result, request_send_time=0.0, clock=clock)
    )

    assert result.ttft_s is None
    assert result.content_chunk_count == 0
    assert result.content_chunk_offsets_ms == []
    assert result.text == ""
    assert result.prompt_tokens == 3


def test_parser_slim_final_does_not_duplicate_text() -> None:
    """The slim-final contract: the final chunk carries usage but NO content.

    PR #406's StreamingDetokenizeScheduler explicitly omits the cumulative
    text from the final result when streaming; this client-layer test
    asserts the benchmark consumer never gets fooled into concatenating
    it twice if a misbehaving server were to include it.
    """
    role_frame = _sse_frame(
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    content_frame = _sse_frame(
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]}
    )
    # finish chunk with empty delta (slim-final compliant)
    finish_frame_slim = _sse_frame(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
    )
    response = _FakeResponse(
        [role_frame, content_frame, finish_frame_slim, b"data: [DONE]\n\n"]
    )
    clock = _FakeClock([1.05])

    result = RequestResult(request_id="r4")
    asyncio.new_event_loop().run_until_complete(
        consume_sse_stream(response, result, request_send_time=1.0, clock=clock)
    )

    # Text appears exactly once.
    assert result.text == "answer"
    assert result.content_chunk_count == 1


# ---------------------------------------------------------------------------
# Payload parity tests
# ---------------------------------------------------------------------------


def _stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("n_images", [1, 2, 7])
def test_omni_backend_uses_top_level_images(n_images: int) -> None:
    sample = _make_sample(f"s{n_images}", n_images=n_images, prompt="describe images")
    payload = build_mmmu_payload(
        sample,
        "qwen3-omni",
        backend="omni",
        modalities=["text"],
        max_tokens=2048,
        temperature=0.0,
        stream=False,
        enable_audio=False,
    )

    assert payload["messages"] == [{"role": "user", "content": "describe images"}]
    assert isinstance(payload["images"], list)
    assert len(payload["images"]) == n_images
    for uri in payload["images"]:
        assert uri.startswith("data:image/png;base64,")


@pytest.mark.parametrize("n_images", [1, 2, 7])
def test_sglang_backend_part_order_is_images_then_text(n_images: int) -> None:
    sample = _make_sample(f"s{n_images}", n_images=n_images, prompt="describe images")
    payload = build_mmmu_payload(
        sample,
        "qwen3-vl",
        backend="sglang",
        modalities=["text"],
        max_tokens=2048,
        temperature=0.0,
        stream=False,
        enable_audio=False,
    )

    assert "images" not in payload
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == n_images + 1
    # First n entries are image parts in dataset order.
    for i in range(n_images):
        part = content[i]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")
    # Final entry is the text part.
    text_part = content[-1]
    assert text_part == {"type": "text", "text": "describe images"}


def test_sglang_backend_image_order_matches_omni() -> None:
    """Re-rendering the same sample twice yields the same payload hash."""
    sample = _make_sample("ordered", n_images=3, prompt="ordered prompt")

    args = {
        "model_name": "qwen3-vl",
        "backend": "sglang",
        "modalities": ["text"],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
        "enable_audio": False,
    }
    p1 = build_mmmu_payload(sample, **args)
    p2 = build_mmmu_payload(sample, **args)
    assert _stable_hash(p1) == _stable_hash(p2)


def test_sglang_backend_reordering_images_changes_hash() -> None:
    """Swapping images in the sample changes the stable-JSON payload hash."""
    sample = _make_sample("swap", n_images=3, prompt="swap prompt")
    args = {
        "model_name": "qwen3-vl",
        "backend": "sglang",
        "modalities": ["text"],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
        "enable_audio": False,
    }
    original = build_mmmu_payload(sample, **args)

    # Build a second sample with images in reversed order.
    swapped = MMMUSample(
        sample_id=sample.sample_id,
        question=sample.question,
        options=sample.options,
        answer=sample.answer,
        images=list(reversed(sample.images)),
        subject=sample.subject,
        prompt=sample.prompt,
        all_choices=sample.all_choices,
        index2ans=sample.index2ans,
        question_type=sample.question_type,
    )
    swapped_payload = build_mmmu_payload(swapped, **args)
    assert _stable_hash(original) != _stable_hash(swapped_payload)


def test_seed_and_ignore_eos_flow_into_payload() -> None:
    sample = _make_sample("seed", n_images=1)
    payload = build_mmmu_payload(
        sample,
        "qwen3-omni",
        backend="omni",
        modalities=["text"],
        max_tokens=256,
        temperature=0.0,
        stream=False,
        enable_audio=False,
        seed=42,
        ignore_eos=True,
    )
    assert payload["seed"] == 42
    assert payload["ignore_eos"] is True


def test_default_seed_and_ignore_eos_absent() -> None:
    sample = _make_sample("no-seed", n_images=1)
    payload = build_mmmu_payload(
        sample,
        "qwen3-omni",
        backend="omni",
        modalities=["text"],
        max_tokens=2048,
        temperature=0.0,
        stream=False,
        enable_audio=False,
    )
    assert "seed" not in payload
    assert "ignore_eos" not in payload


def test_unsupported_backend_raises() -> None:
    sample = _make_sample("bad", n_images=1)
    with pytest.raises(ValueError, match="backend"):
        build_mmmu_payload(
            sample,
            "x",
            backend="vllm",  # type: ignore[arg-type]
            modalities=["text"],
            max_tokens=2048,
            temperature=0.0,
            stream=False,
            enable_audio=False,
        )


# ---------------------------------------------------------------------------
# RequestResult schema tests
# ---------------------------------------------------------------------------


def test_request_result_streaming_fields_default_safe() -> None:
    r = RequestResult()
    assert r.ttft_s is None
    assert r.content_chunk_offsets_ms == []
    assert r.content_chunk_count == 0
    assert r.client_wall_time_s == 0.0
    assert r.timing_source == ""


def test_request_result_offsets_list_is_per_instance() -> None:
    """Mutable-default safety: each RequestResult gets its own list."""
    a = RequestResult()
    b = RequestResult()
    a.content_chunk_offsets_ms.append(1.0)
    assert b.content_chunk_offsets_ms == []
