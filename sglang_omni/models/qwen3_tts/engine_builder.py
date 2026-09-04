# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.qwen3_tts import CAPABILITIES, request_builders
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.config import is_qwen3_tts_base_model
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


# note (luojiaxuan): measured from 1548 prefills at 10 and 20 RPS on H100
# CustomVoice. Real extend token counts are p50=8, p90=19, p99=38, max=252,
# and requests coalesce up to 9-deep without pushing the shape past 256, so the
# buckets are dense where the mass is and keep 384/512 only as headroom. The
# generic ladder starts at 4 and steps by 4, which sends every 1-3 token prefill
# over the padding factor and back to eager: 39.9% of prefills, against 0% here.
QWEN3_TTS_PREFILL_CUDA_GRAPH_BS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    20,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
)


class Qwen3TtsEngineBuilder(TtsEngineBuilder):
    model_name = "Qwen3-TTS"
    context_length = 8192
    model_arch_override = "Qwen3TTSTalker"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        attn_implementation: str | None = None,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
    ) -> None:
        self.attn_implementation = attn_implementation
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.wrapper: Any | None = None
        self._stream_output_builder: Any | None = None
        # note (luojiaxuan): the factory assigns this before generation_defaults
        # runs, but Qwen3TTSPipelineConfig.generation_admission_defaults builds a
        # bare builder just to read the admission keys, so it needs a value.
        self.checkpoint_dir: str = ""

    def resolve_checkpoint(self, model_path: str) -> str:
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen_tts = importlib.import_module("qwen_tts")
        if not hasattr(qwen_tts, "Qwen3TTSModel"):
            raise ImportError("qwen_tts does not expose Qwen3TTSModel")

        return super().resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen3_stages._register_qwen3_tts_hf_config()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": 16,
            "max_queued_requests": 16,
            "cuda_graph_max_bs": 32,
            "torch_compile_max_bs": 32,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }
        if self.checkpoint_dir and not is_qwen3_tts_base_model(self.checkpoint_dir):
            # note (luojiaxuan): the ladder is sized from the text-only prompt
            # length distribution, and under load prefills coalesce into one
            # extend batch, so it must reach well past a single prompt. Base
            # prefills also carry reference audio, giving them a different shape
            # distribution, so they keep the eager path until measured.
            defaults["cuda_graph_backend_prefill"] = CudaGraphBackend.BREAKABLE
            defaults["cuda_graph_bs_prefill"] = list(QWEN3_TTS_PREFILL_CUDA_GRAPH_BS)
        return defaults

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del gpu_id, server_args
        from qwen_tts import Qwen3TTSModel
        from transformers import AutoProcessor

        model = model_worker.model_runner.model
        speech_tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
            checkpoint_dir,
            device=device,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )
        model.load_speech_tokenizer(speech_tokenizer)
        processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            fix_mistral_regex=True,
        )
        self.wrapper = Qwen3TTSModel(
            model=model,
            processor=processor,
            generate_defaults=qwen3_stages._load_qwen3_tts_generate_defaults(
                checkpoint_dir
            ),
        )
        request_builders.set_qwen3_tts_preprocessing_context(
            model=model,
            wrapper=self.wrapper,
        )

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if _is_truthy(overrides.get("enable_torch_compile", False)):
            raise ValueError("Qwen3-TTS torch.compile is not supported")

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.qwen3_tts.model_runner"
        )

        return model_runner_mod.Qwen3TTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        request_builder, result_adapter, self._stream_output_builder = (
            request_builders.make_qwen3_tts_scheduler_adapters(
                model=model,
                wrapper=self.wrapper,
            )
        )
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": self._stream_output_builder,
            "request_build_max_workers": 4,
            "request_build_max_pending": 16,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_qwen3_tts_request
