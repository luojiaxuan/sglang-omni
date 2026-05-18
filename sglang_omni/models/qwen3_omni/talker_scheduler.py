# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were originally requested so the caller can
    re-enable graph capture after the model worker is constructed.
    """

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if feedback_enabled:
        server_args.disable_overlap_schedule = True
        if want_cuda_graph:
            server_args.disable_cuda_graph = True
        server_args.moe_runner_backend = "flashinfer_cutlass"
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = 0
    return want_cuda_graph


# Minimum prefetched thinker chunk count required to enter the partial-start
# path. The 9-row assistant tail that ``build_assistant_part`` constructs
# collapses below 3 chunks. The decode-ready operating point (at least one
# consumable future text row after ``include_assistant_eos=False`` strips the
# trailing EOS row) sits at 5 chunks. Both numbers are pinned by
# ``test_partial_prompt_prefill_layout_invariants``; do not change this floor
# without updating that test.
MIN_PARTIAL_START_CHUNKS = 3


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    def __init__(
        self,
        *args: Any,
        partial_start_min_chunks: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._partial_start_min_chunks = partial_start_min_chunks

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if self._partial_start_min_chunks is None:
            return False
        effective_min = max(self._partial_start_min_chunks, MIN_PARTIAL_START_CHUNKS)
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return len(prefetched) >= effective_min

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        del req_data, payload
        # The talker request builder consumes the full thinker stream up front and
        # seeds pending_text_queue itself, so the scheduler must not replay it.
        return None

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            logger.debug(
                "Deferring decode batch until talker feedback/text input is ready"
            )
            return False
        return True

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
