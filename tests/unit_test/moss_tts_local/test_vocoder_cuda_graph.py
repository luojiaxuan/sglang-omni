# SPDX-License-Identifier: Apache-2.0
"""命门 (lifeline) bit-identity gate for the MOSS streaming-vocoder codec-decode CUDA graph.

The CUDA-graph replay MUST be bit-for-bit identical to the eager codec decode, or streaming audio
changes. Because the codec is STATEFUL (per-slot causal offset/KV), the test decodes MULTI-CHUNK
sequences (so the state must advance correctly across steps) at varied active-slot counts (so the
exec_mask state gating must isolate slots) and T boundaries. Any PCM mismatch fails the gate -- the
cut is dead; do NOT relax the comparison. GPU + the real MOSS-Audio-Tokenizer-v2 codec required.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

CODEC_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/snapshots/*/"
)
N_VQ = 12  # MOSS-TTS-Local v1.5 uses the first 12 RVQ codebooks
STREAM_SLOTS = 8
OFFLINE_SLOTS = 8
# T values the gate exercises (chunk sizes); warmup captures these + remainders.
CHUNK_TS = [1, 5, 25, 100]
_HAS_CUDA = torch.cuda.is_available()


def _codebook_size(codec) -> int:
    q = getattr(codec, "quantizer", None)
    qs = getattr(q, "quantizers", None)
    if qs:
        for attr in ("codebook_size", "n_codes", "num_embeddings", "codebook_dim"):
            v = getattr(qs[0], attr, None)
            if isinstance(v, int) and v > 0:
                return v
    v = getattr(getattr(codec, "config", None), "codebook_size", None)
    return v if isinstance(v, int) and v > 0 else 1024


@pytest.fixture(scope="module")
def session_bundle():
    # Load the codec DIRECTLY (the sglang processor pulls librosa/soxr audio deps absent in this
    # serving container; the codec modeling file is self-contained). streaming_vocoder imports clean.
    os.environ["MOSS_VOCODER_CUDA_GRAPH"] = "1"
    import glob

    from transformers import AutoModel

    from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession

    snaps = glob.glob(CODEC_GLOB)
    if not snaps:
        pytest.skip("MOSS-Audio-Tokenizer-v2 codec snapshot not found")
    codec = (
        AutoModel.from_pretrained(snaps[0], trust_remote_code=True).to("cuda").eval()
    )
    n_vq = N_VQ
    vocab = _codebook_size(codec)
    session = _CodecStreamSession(
        codec, stream_slots=STREAM_SLOTS, offline_slots=OFFLINE_SLOTS, n_vq=n_vq
    )
    # Capture every length the test emits (each chunk_t and its remainder) once, sealed.
    wanted = set()
    for chunk_t in CHUNK_TS:
        total = chunk_t * 3 + max(1, chunk_t // 2)
        pos = 0
        while pos < total:
            wanted.add(min(chunk_t, total - pos))
            pos += min(chunk_t, total - pos)
    captured = session.warmup_cuda_graph(sorted(wanted))
    return session, n_vq, vocab, set(captured)


def _decode_chunks(session, slot_seqs, chunk_t):
    """Decode dict{slot: [n_vq, T_total]} in lockstep chunks of chunk_t. Resets slots first."""
    slots = list(slot_seqs)
    session._reset_slots(slots)
    total = next(iter(slot_seqs.values())).shape[1]
    parts = {s: [] for s in slots}
    pos = 0
    while pos < total:
        t = min(chunk_t, total - pos)
        out = session.step({s: slot_seqs[s][:, pos : pos + t] for s in slots})
        for s in slots:
            parts[s].append(out[s])
        pos += t
    return {s: torch.cat(parts[s], dim=-1) for s in slots}


def test_some_graphs_captured(session_bundle):
    _, _, _, captured = session_bundle
    # If zero captured, the whole line is eager-only (no prize) -- surface it loudly.
    assert (
        captured
    ), "no codec-decode CUDA graphs captured (all shapes fell back to eager)"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize("chunk_t", CHUNK_TS)
@pytest.mark.parametrize("n_active", [1, 8])
def test_streaming_pcm_bit_identical(session_bundle, chunk_t, n_active):
    session, n_vq, vocab, captured = session_bundle
    if chunk_t not in captured:
        pytest.skip(
            f"T={chunk_t} fell back to eager (not captured); nothing to compare"
        )
    torch.manual_seed(1000 * chunk_t + n_active)
    total = chunk_t * 3 + max(1, chunk_t // 2)  # multiple full chunks + a remainder
    slot_seqs = {
        s: torch.randint(0, vocab, (n_vq, total), device="cuda", dtype=torch.long)
        for s in range(n_active)
    }
    runner = session._cg_runner
    session._cg_runner = None  # force eager
    eager = _decode_chunks(session, slot_seqs, chunk_t)
    session._cg_runner = runner  # graph path
    graphed = _decode_chunks(session, slot_seqs, chunk_t)
    for s in range(n_active):
        assert torch.equal(eager[s], graphed[s]), (
            f"streaming PCM not bit-identical (chunk_t={chunk_t}, n_active={n_active}, slot={s}): "
            f"max|delta|={(eager[s] - graphed[s]).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize("chunk_t", [5, 25])
def test_graph_tracks_eager_across_chunkings(session_bundle, chunk_t):
    """Decode the SAME codes at two different chunk boundaries; for each chunking the graph PCM must
    equal the eager PCM bit-for-bit. NOTE: the codec is itself chunk-boundary DEPENDENT (eager T=25
    and T=5 PCM differ by ~0.8 -- different per-step buffer/state evolution), so we do NOT assert
    cross-chunking equality. The gate is that the graph faithfully reproduces eager AT EACH chunking,
    including that chunk-dependence; an extra T (here both 5 and 25) guards against a graph that only
    matches eager at the single T the other test exercises."""
    session, n_vq, vocab, captured = session_bundle
    if chunk_t not in captured:
        pytest.skip(
            f"T={chunk_t} fell back to eager (not captured); nothing to compare"
        )
    torch.manual_seed(7)
    total = 75
    seq = {0: torch.randint(0, vocab, (n_vq, total), device="cuda", dtype=torch.long)}
    runner = session._cg_runner
    session._cg_runner = None  # force eager
    eager = _decode_chunks(session, seq, chunk_t)[0]
    session._cg_runner = runner  # graph path
    graphed = _decode_chunks(session, seq, chunk_t)[0]
    assert torch.equal(eager, graphed), (
        f"graph decode not bit-identical to eager at chunk_t={chunk_t}: "
        f"max|delta|={(eager - graphed).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_replay_failure_disables_runner_and_serves_eager_bit_identical(session_bundle):
    """A replay-path exception must not crash or cascade: the runner is disabled (every future step
    degrades to eager) and that eager output is bit-identical to a pure-eager reference. The failing
    step itself raises (its per-slot state is indeterminate after a partial replay) -- bit-safe, we
    never emit a non-identical output. This is the default-on robustness gate for replay failures.
    """
    session, n_vq, vocab, captured = session_bundle
    chunk_t = next((t for t in (5, 25) if t in captured), None)
    if chunk_t is None:
        pytest.skip("need T=5 or T=25 captured")
    torch.manual_seed(4242)
    seq = {
        0: torch.randint(0, vocab, (n_vq, chunk_t * 3), device="cuda", dtype=torch.long)
    }
    runner = session._cg_runner
    session._cg_runner = None  # pure-eager reference
    eager_ref = _decode_chunks(session, seq, chunk_t)[0]

    session._cg_runner = runner  # graph path, but make the next replay blow up
    session._reset_slots([0])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated replay failure")

    runner.decode_step = boom
    with pytest.raises(RuntimeError):
        session.step({0: seq[0][:, :chunk_t]})
    assert session._cg_runner is None, "runner must be disabled after a replay failure"

    # session is now eager-only -> a fresh decode must be bit-identical to the pure-eager reference
    after = _decode_chunks(session, seq, chunk_t)[0]
    assert torch.equal(after, eager_ref), (
        "post-failure eager output not bit-identical to eager reference: "
        f"max|delta|={(after - eager_ref).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
