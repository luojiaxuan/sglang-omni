# SPDX-License-Identifier: Apache-2.0
"""CUDA graph runner for the MOSS-TTS-Local streaming vocoder (codec) decode.

The streaming codec decode (``_CodecStreamSession.step`` -> ``codec._decode_frame``) is a
RVQ dequant -> ConvTranspose decoder stack that is tiny per call and dominated by kernel-launch
overhead: under streaming the scheduler emits ~one decode per audio frame, so the per-launch floor
is paid thousands of times per request. Capturing the decode into a CUDA graph and replaying it
collapses those launches into one replay.

Adapted from the Higgs vocoder CUDA-graph runner (sgl #581/#729), with two MOSS specifics:

1. **B is fixed at the full slot width (``stream_slots + offline_slots``, default 16); only T varies.**
   The MOSS streaming step always builds a ``[n_vq, B_full, T]`` tensor and selects active slots via
   ``exec_mask`` (vs Higgs capturing B=1 only). So we capture **one graph per T**, B fixed -- no
   bucketing over B, no padding (the ConvTranspose receptive field makes padding non-bit-exact).

2. **The codec decode is STATEFUL** (per-slot causal offset/KV under ``codec.streaming()``), gated by
   ``exec_mask`` via ``torch.where`` (e.g. ``offset = where(exec_mask, offset+T, offset)``). So:
   - We capture with ``exec_mask`` all-active; ``_set_streaming_exec_mask`` (a host-side in-place copy
     into the modules' fixed mask buffers) is called BEFORE replay with the real mask, so the captured
     ``torch.where`` reads the live mask and advances ONLY active slots' state -- inactive (paused/free)
     slots are preserved.
   - Warmup + capture advance state, so the caller MUST reset all slots after ``warmup()`` (see
     ``_CodecStreamSession``); serving then advances from clean state.

Capture happens ONCE at warmup (GPU quiescent) then the runner is sealed -- never captures during
serving (live capture corrupts the co-located AR stage's graph mempool). Capture is best-effort: any
shape that fails to capture (e.g. an unexpected ``.item()`` host sync) is dropped and falls back to
eager, so correctness never depends on capture succeeding. Bit-identity vs eager is gated by
``tests/unit_test/moss_tts_local/test_vocoder_cuda_graph.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


class MossVocoderCudaGraphRunner:
    """Warmup-captured, sealed replay of exact-T CUDA graphs for the MOSS codec decode (B fixed)."""

    def __init__(
        self,
        codec,
        *,
        batch_size: int,
        n_vq: int,
        max_frames: int = 128,
        max_graphs: int = 160,
        warmup_iters: int = 3,
    ) -> None:
        self._codec = codec
        self._batch_size = int(batch_size)
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._max_frames = int(max_frames)
        self._max_graphs = int(max_graphs)
        self._warmup_iters = int(warmup_iters)
        # Min free VRAM to attempt a capture (each graph is multi-GB); below it we skip -> eager,
        # so a VRAM-tight box degrades gracefully instead of OOM-ing. Env-configurable.
        self._min_free_bytes = int(
            float(os.environ.get("MOSS_VOCODER_CUDA_GRAPH_MIN_FREE_GB", "3"))
            * (1024**3)
        )
        self._graphs: dict[int, tuple] = {}
        self._pool = None
        self._sealed = False

    def _eligible(self, t: int) -> bool:
        return 1 <= t <= self._max_frames

    def _enough_free_vram(self) -> bool:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes

    @torch.no_grad()
    def _reset_state(self) -> None:
        """Reset every streaming module's per-slot state to offset 0 (in-place)."""
        reset_mask = torch.ones(self._batch_size, dtype=torch.bool, device=self._device)

        def _r(module) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(reset_mask.to(state.device))

        self._codec.apply(_r)

    @torch.no_grad()
    def _capture_t(self, t: int) -> None:
        b, n = self._batch_size, self._n_vq
        device = self._device
        static_codes = torch.zeros(n, b, t, dtype=torch.long, device=device)
        # Capture all-active; the live exec_mask at replay decides which slots advance.
        static_lengths = torch.full((b,), t, dtype=torch.long, device=device)
        exec_mask = torch.ones(b, dtype=torch.bool, device=device)
        self._codec._set_streaming_exec_mask(exec_mask)
        # Note: (Jiaxin Deng) side-stream warmup forces lazy allocs (conv algo / workspaces) out of the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._codec._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Note: (Jiaxin Deng) reset to offset 0 AFTER warmup, BEFORE capture -- capturing at the
        # warmup-advanced offset bakes a wrong start state (~0.4 PCM error). reset re-activates all slots.
        self._reset_state()
        self._codec._set_streaming_exec_mask(exec_mask)
        # Shared mempool across the T graphs to bound memory (large B=16 intermediates); capture order in warmup.
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            result = self._codec._decode_frame(static_codes, static_lengths)
            static_audio = result.audio
            static_audio_lengths = result.audio_lengths
        self._graphs[t] = (
            graph,
            static_codes,
            static_lengths,
            static_audio,
            static_audio_lengths,
        )
        logger.info(
            "Captured MOSS vocoder CUDA graph T=%d (B=%d) -> audio %s (%d cached)",
            t,
            b,
            tuple(static_audio.shape),
            len(self._graphs),
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int]) -> None:
        """Capture one graph per T, once, then seal. Call at startup while the GPU is quiescent.

        The caller MUST reset all codec slots after this returns (warmup advances per-slot state).
        """
        if self._sealed:
            logger.warning(
                "MossVocoderCudaGraphRunner.warmup called after seal; ignoring"
            )
            return
        # Note: (Jiaxin Deng) capture LARGEST T first -- the graphs share one mempool; capturing a larger
        # graph after a smaller one grows the pool and invalidates earlier graphs' addresses (replay segfaults).
        for t in sorted(dict.fromkeys(int(x) for x in frames), reverse=True):
            if t in self._graphs:
                continue
            if not self._eligible(t):
                logger.warning(
                    "skip MOSS vocoder CG T=%d: outside [1, %d]", t, self._max_frames
                )
                continue
            if len(self._graphs) >= self._max_graphs:
                logger.warning(
                    "MOSS vocoder CG cap %d reached; skipping rest", self._max_graphs
                )
                break
            # Note: (Jiaxin Deng) VRAM headroom guard -- skip capture (-> eager) rather than risk OOM on
            # a tight box. Checked per-T because each capture allocates; free only drops through the loop.
            if not self._enough_free_vram():
                free, _ = torch.cuda.mem_get_info(self._device)
                logger.warning(
                    "MOSS vocoder CG: free VRAM %.1fGB < %.1fGB headroom; skipping T=%d+ (eager)",
                    free / 1024**3,
                    self._min_free_bytes / 1024**3,
                    t,
                )
                break
            try:
                self._capture_t(t)
            except Exception as exc:  # best-effort; uncaptured T falls back to eager
                self._graphs.pop(t, None)
                logger.warning(
                    "MOSS vocoder CG capture failed for T=%d: %s; will use eager",
                    t,
                    exc,
                )
        self._sealed = True
        logger.info(
            "MOSS vocoder CUDA graphs sealed: %d T captured %s",
            len(self._graphs),
            sorted(self._graphs.keys()),
        )

    def captured_frames(self) -> list[int]:
        return sorted(self._graphs.keys())

    @torch.no_grad()
    def decode_step(
        self,
        codes_step: torch.Tensor,
        codes_lengths: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay the captured graph for ``[n_vq, B_full, T]`` codes, else return None (eager).

        Sets the live exec_mask (host-side, into the codec's fixed mask buffers) so the captured
        torch.where gating advances only active slots, copies codes into the static input, replays,
        and returns cloned (audio, audio_lengths) static outputs. ``codes_lengths`` is intentionally
        NOT copied -- capture used all-T lengths and active slots are full-T, so the static lengths
        already match; inactive slots are gated out by exec_mask. NOT re-entrant (single static buffer
        per T); the streaming scheduler drains on one serial loop.
        """
        if not codes_step.is_cuda:
            return None
        n, b, t = codes_step.shape
        if b != self._batch_size or n != self._n_vq:
            return None
        entry = self._graphs.get(int(t))
        if entry is None:
            return None
        graph, static_codes, static_lengths, static_audio, static_audio_lengths = entry
        # Replicate eager inputs exactly (codes, lengths, exec_mask) so replay is bit-for-bit identical.
        self._codec._set_streaming_exec_mask(exec_mask)
        static_codes.copy_(codes_step)
        static_lengths.copy_(codes_lengths)
        graph.replay()
        return static_audio.clone(), static_audio_lengths.clone()
