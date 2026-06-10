# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-TTS Local decode-state pool (PR-A c3).

CPU-only: the pool derives its sizing/placement from a fake model exposing a
``_decode_input_embedding.weight`` tensor, so no CUDA is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts_local.state_pool import (
    MossTTSLocalDecodeJournal,
    MossTTSLocalDecodeStatePool,
)

_HIDDEN = 8


def _model(max_running_requests: int = 4) -> SimpleNamespace:
    """Fake model exposing only what the pool reads."""
    weight = torch.zeros(max_running_requests, _HIDDEN, dtype=torch.bfloat16)
    embedding = SimpleNamespace(weight=weight)
    return SimpleNamespace(_decode_input_embedding=embedding)


def _params(seed: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        text_temperature=0.5,
        text_top_p=0.9,
        text_top_k=40,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        sampling_seed=seed,
    )


def test_pool_dims_derive_from_embedding_weight():
    """P = weight.shape[0] + 1; no literal row count, padding row reserved."""
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=4))
    assert pool.num_rows == 5
    assert pool.padding_row == 4
    assert pool.hidden_size == _HIDDEN
    assert pool.feedback_embeds.shape == (5, _HIDDEN)
    assert pool.feedback_embeds.dtype == torch.bfloat16
    for name in ("text_temp", "text_top_p", "audio_temp", "audio_top_p"):
        assert getattr(pool, name).shape == (5,)
        assert getattr(pool, name).dtype == torch.float32
    for name in ("text_top_k", "audio_top_k", "seeds"):
        assert getattr(pool, name).shape == (5,)
        assert getattr(pool, name).dtype == torch.int64


def test_acquire_is_idempotent_by_rid():
    pool = MossTTSLocalDecodeStatePool(_model())
    first = pool.acquire_row("a")
    again = pool.acquire_row("a")
    assert first == again
    # A second rid takes a different row.
    other = pool.acquire_row("b")
    assert other != first


def test_padding_row_never_acquired():
    """Real rows are 0..P-2; the padding row stays out of every assignment."""
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=4))
    acquired = {pool.acquire_row(f"r{i}") for i in range(4)}
    assert acquired == {0, 1, 2, 3}
    assert pool.padding_row not in acquired


def test_pool_exhaustion_raises():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    pool.acquire_row("a")
    pool.acquire_row("b")
    try:
        pool.acquire_row("c")
    except RuntimeError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on pool exhaustion")


def test_release_is_noop_for_unheld_rid():
    pool = MossTTSLocalDecodeStatePool(_model())
    # No row held: release must not raise or perturb the free list.
    free_before = list(pool._free_rows)
    pool.release_row("ghost")
    assert pool._free_rows == free_before


def test_release_frees_and_recycles_row():
    pool = MossTTSLocalDecodeStatePool(_model(max_running_requests=2))
    row_a = pool.acquire_row("a")
    pool.acquire_row("b")
    pool.release_row("a")
    assert pool.row_for("a") is None
    # The freed row is reusable.
    row_c = pool.acquire_row("c")
    assert row_c == row_a


def test_release_resets_row_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=123))
    pool.feedback_embeds[row].fill_(1.0)
    pool.release_row("a")
    assert torch.all(pool.feedback_embeds[row] == 0)
    assert pool.text_temp[row] == 0.0
    assert pool.audio_top_k[row] == 0
    assert pool.seeds[row] == 0


def test_reset_row_zeroes_all_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=99))
    pool.feedback_embeds[row].fill_(2.0)
    pool.reset_row(row)
    assert torch.all(pool.feedback_embeds[row] == 0)
    for name in (
        "text_temp",
        "text_top_p",
        "audio_temp",
        "audio_top_p",
        "text_top_k",
        "audio_top_k",
        "seeds",
    ):
        assert getattr(pool, name)[row] == 0


def test_write_params_writes_request_static_fields():
    pool = MossTTSLocalDecodeStatePool(_model())
    row = pool.acquire_row("a")
    pool.write_params(row, _params(seed=555))
    assert pool.text_temp[row].item() == torch.tensor(0.5, dtype=torch.float32).item()
    assert pool.text_top_p[row].item() == torch.tensor(0.9, dtype=torch.float32).item()
    assert pool.audio_temp[row].item() == torch.tensor(1.7, dtype=torch.float32).item()
    assert pool.audio_top_p[row].item() == torch.tensor(0.8, dtype=torch.float32).item()
    assert int(pool.text_top_k[row]) == 40
    assert int(pool.audio_top_k[row]) == 25
    assert int(pool.seeds[row]) == 555


def test_write_params_does_not_touch_other_rows():
    pool = MossTTSLocalDecodeStatePool(_model())
    row_a = pool.acquire_row("a")
    row_b = pool.acquire_row("b")
    pool.write_params(row_a, _params(seed=1))
    assert pool.seeds[row_b] == 0
    assert pool.text_temp[row_b] == 0.0


def test_row_for_returns_none_when_unheld():
    pool = MossTTSLocalDecodeStatePool(_model())
    assert pool.row_for("nobody") is None
    row = pool.acquire_row("a")
    assert pool.row_for("a") == row


def test_journal_holds_fields():
    rows = torch.arange(2 * 13, dtype=torch.long).reshape(2, 13)
    journal = MossTTSLocalDecodeJournal(rids=["a", "b"], pool_rows=[0, 1], rows=rows)
    assert journal.rids == ["a", "b"]
    assert journal.pool_rows == [0, 1]
    assert torch.equal(journal.rows, rows)
