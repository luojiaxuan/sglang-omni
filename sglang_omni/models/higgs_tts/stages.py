# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

Pipeline shape::

    preprocessing → audio_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text tokenize + (if raw audio path)
  load waveform; fast path also delay-encodes client-supplied
  ``reference_codes`` and builds the prompt. Returns a
  :class:`ThreadedSimpleScheduler` for CPU-heavy work.
- ``create_audio_encoder_executor``: GPU codec encode for the raw-audio
  path → delayed ref codes + prompt assembly. No-op on the fast path.
- ``create_sglang_tts_engine_executor``: runs :class:`HiggsTTSModel` under
  sglang's worker; the model runner computes the fused multi-codebook
  embedding inline in prefill from ``reference_codes_delayed`` and overlays
  it at ``-100`` placeholder positions. Returns a :class:`OmniScheduler`.
- ``create_vocoder_executor``: reverses the delay pattern, decodes via
  :class:`HiggsAudioCodec` into a mono 24 kHz waveform. Returns a
  :class:`SimpleScheduler`.
"""

from __future__ import annotations

import logging
import os
import queue as _queue_mod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torchaudio.functional as F_audio
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import make_higgs_scheduler_adapters
from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    reverse_delay_pattern,
    to_codes_TN,
    truncate_rope_to_bf16,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

logger = logging.getLogger(__name__)


# Codec runs at 75 Hz; chunked prefill of the multi-codebook prompt is unsafe
# (sampler state machine has no rollback) so reject inputs past chunked_prefill_size.
_MAX_REF_AUDIO_SEC = 100


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    max_concurrency: int = 8,
):
    """CPU stage: text tokenize + optional ref-audio file IO.

    Builds the full prompt + delays the codes when the client supplied
    pre-encoded ``reference_codes``. When raw audio is supplied, defers
    codec encoding (and prompt assembly) to the audio_encoder stage —
    only the loaded waveform is shipped forward.
    """
    checkpoint_dir = resolve_checkpoint(model_path)

    # Higgs ckpt tokenizer_config.json uses transformers v5 metadata and crashes
    # transformers<5's from_pretrained; load tokenizer.json directly to avoid it.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        raw_refs = inputs.get("references")
        if raw_refs and isinstance(raw_refs, list):
            first = raw_refs[0]
            if isinstance(first, dict):
                inputs = dict(inputs)
                if first.get("text") and not inputs.get("reference_text"):
                    inputs["reference_text"] = first["text"]
                if inputs.get("reference_audio") is None:
                    if "bytes" in first or "base64" in first or "data" in first:
                        inputs["reference_audio"] = first
                    else:
                        inputs["reference_audio"] = first.get(
                            "audio_path"
                        ) or first.get("path")

        text = inputs.get("input") or inputs.get("text") or ""
        reference_text = inputs.get("reference_text") or None
        ref_codes_TN = to_codes_TN(inputs.get("reference_codes"), num_codebooks)
        if ref_codes_TN is not None and ref_codes_TN.shape[0] > _MAX_REF_AUDIO_SEC * 75:
            raise ValueError(
                f"reference_codes is too long ({ref_codes_TN.shape[0]} frames); "
                f"cap at {_MAX_REF_AUDIO_SEC}s of audio "
                f"(~{_MAX_REF_AUDIO_SEC * 75} frames at 75 Hz)."
            )

        waveform_tensor = None
        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            waveform_np, sample_rate = load_audio_to_24k(inputs["reference_audio"])
            wav = torch.from_numpy(waveform_np)
            if sample_rate != 24000:
                wav = F_audio.resample(wav, sample_rate, 24000)
            if wav.shape[-1] > _MAX_REF_AUDIO_SEC * 24000:
                raise ValueError(
                    f"reference_audio is too long "
                    f"({wav.shape[-1] / 24000:.1f}s); cap at {_MAX_REF_AUDIO_SEC}s."
                )
            waveform_tensor = wav.view(1, 1, -1).contiguous().float()

        if ref_codes_TN is not None:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(
                text,
                num_ref_tokens=delayed.shape[0],
                reference_text=reference_text,
            )
            ref_codes_delayed: list[list[int]] | None = delayed.tolist()
            target_text_for_encoder = None
            reference_text_for_encoder = None
        elif waveform_tensor is None:
            prompt_ids = adapter.build_prompt(
                text, num_ref_tokens=0, reference_text=reference_text
            )
            ref_codes_delayed = None
            target_text_for_encoder = None
            reference_text_for_encoder = None
        else:
            prompt_ids = []
            ref_codes_delayed = None
            target_text_for_encoder = text
            reference_text_for_encoder = reference_text

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            reference_waveform=waveform_tensor,
            target_text=target_text_for_encoder,
            reference_text=reference_text_for_encoder,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    num_codebooks: int = 8,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
):
    """GPU stage: codec-encode raw ref audio → delayed codes + prompt assembly.

    No-op when preprocessing already produced ``reference_codes_delayed`` (the
    client-supplied pre-encoded fast path). Codec weights are extracted from
    the TTS checkpoint itself (bundled at ``tied.embedding.modality_embeddings``).
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec = get_or_load_codec(checkpoint_dir, device, dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        waveform = state.reference_waveform
        if waveform is None:
            return payload

        ref_codes_TN = codec.encode_reference(waveform, sample_rate=24000).to(
            torch.long
        )
        if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != num_codebooks:
            raise ValueError(
                f"codec output must be [T, {num_codebooks}], got "
                f"{tuple(ref_codes_TN.shape)}"
            )
        delayed = apply_delay_pattern(ref_codes_TN)
        state.reference_codes_delayed = delayed.tolist()
        state.prompt_token_ids = adapter.build_prompt(
            state.target_text or "",
            num_ref_tokens=delayed.shape[0],
            reference_text=state.reference_text,
        )
        state.reference_waveform = None
        state.target_text = None
        state.reference_text = None
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int | None = 2048,
    server_args_overrides: dict[str, Any] | None = None,
):
    """sglang-backed AR engine for Higgs TTS."""
    checkpoint_dir = resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": 32,
        "mem_fraction_static": 0.85,
        "max_running_requests": 16,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        # Radix cache is namespaced per ref-audio via Req.extra_key (set in
        # build_sglang_higgs_request); shared -100 placeholder prefixes from
        # different ref audios can't cross-contaminate the KV tree.
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(server_args, gpu_id)

    truncate_rope_to_bf16(model_worker.model_runner.model)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = HiggsTTSModelRunner(model_worker, output_proc)
    model = model_worker.model_runner.model
    request_builder, result_adapter = make_higgs_scheduler_adapters(
        model,
        max_new_tokens_cap=max_new_tokens,
    )

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=model.reset_request,
    )
    model_runner.set_stream_outbox(scheduler.outbox)
    return scheduler


def _delayed_to_waveform(
    codec: HiggsAudioCodec,
    delayed_LN: torch.Tensor,
    codebook_size: int,
) -> torch.Tensor:
    """Reverse delay pattern, zero special tokens, decode to a float32 waveform.

    Returns the waveform on whatever device the codec runs on. Caller is
    responsible for the single GPU→host transfer when serializing.
    """
    codes_TN = reverse_delay_pattern(delayed_LN)
    codec_vocab = int(codebook_size) - 2
    codes_TN = torch.where(
        codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
    )
    return codec.decode(codes_TN).detach().to(torch.float32)


@dataclass
class _HiggsStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    emitted_raw_frames: int = 0
    next_decode_rows: int = 0
    has_emitted: bool = False


class HiggsVocoderScheduler:
    """Vocoder scheduler covering both streaming and non-streaming requests.

    Non-streaming requests are forwarded to ``vocode_fn`` (the original
    Higgs vocoder closure). Streaming requests accumulate per-step delayed
    code rows arriving as ``stream_chunk`` messages and emit audio deltas
    as the codec window crosses stride thresholds; the final ``result``
    is a slim StagePayload carrying just sample rate + usage.
    """

    def __init__(
        self,
        codec: HiggsAudioCodec,
        vocode_fn: Callable[[StagePayload], StagePayload],
        *,
        sample_rate: int,
        stream_stride: int = 75,
        stream_followup_stride: int = 75,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 4,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream_holdback_tokens must be >= 0")

        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._codec = codec
        self._vocode_fn = vocode_fn
        self._sample_rate = sample_rate
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._running = False

        self._payloads: dict[str, StagePayload] = {}
        self._stream_states: dict[str, _HiggsStreamState] = {}
        self._pending_done: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            try:
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_done(msg.request_id)
                else:
                    raise ValueError(f"Unsupported vocoder message type: {msg.type}")
            except Exception as exc:
                logger.exception("Higgs vocoder failed for %s", msg.request_id)
                self.outbox.put(
                    OutgoingMessage(request_id=msg.request_id, type="error", data=exc)
                )
                self.abort(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._payloads.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._pending_done.discard(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        streaming = bool(payload.request.params.get("stream"))
        if not streaming:
            result = self._vocode_fn(payload)
            self.outbox.put(
                OutgoingMessage(request_id=request_id, type="result", data=result)
            )
            return

        self._payloads[request_id] = payload
        self._stream_states.setdefault(request_id, _HiggsStreamState())
        if request_id in self._pending_done:
            self._pending_done.discard(request_id)
            self._finalize_streaming_request(request_id)

    def _on_chunk(self, request_id: str, chunk: Any) -> None:
        # Chunks are only emitted by the engine for streaming requests, so
        # gating happens upstream — accept any chunk that arrives here.
        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        row = getattr(chunk, "data", chunk)
        if not isinstance(row, torch.Tensor):
            row = torch.tensor(row, dtype=torch.long)
        else:
            row = row.detach().to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Higgs stream chunk must be 1-D [N], got {tuple(row.shape)}"
            )
        state.delayed_rows.append(row)

        output = self._decode_delta(state, is_final=False)
        if output is not None:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

    def _on_done(self, request_id: str) -> None:
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return
        self._finalize_streaming_request(request_id)

    def _finalize_streaming_request(self, request_id: str) -> None:
        payload = self._payloads[request_id]
        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        output = self._decode_delta(state, is_final=True)
        if output is not None:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = _build_usage(HiggsTtsState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        self.abort(request_id)

    def _decode_delta(
        self, state: _HiggsStreamState, *, is_final: bool
    ) -> dict[str, Any] | None:
        delayed_count = len(state.delayed_rows)
        if delayed_count == 0:
            return None
        num_codebooks = int(state.delayed_rows[0].shape[0])
        if delayed_count < num_codebooks:
            return None
        raw_total = delayed_count - num_codebooks + 1

        next_decode_rows = state.next_decode_rows or max(
            num_codebooks, self._stream_stride
        )
        if not is_final and delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        emit_until_raw = raw_total
        if not is_final and self._stream_holdback_tokens:
            emit_until_raw = max(0, raw_total - self._stream_holdback_tokens)
        if emit_until_raw <= state.emitted_raw_frames:
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        window_start_raw = max(
            0, state.emitted_raw_frames - self._stream_overlap_tokens
        )
        rows_end = emit_until_raw + num_codebooks - 1
        rows = state.delayed_rows[window_start_raw:rows_end]
        delayed_LN = torch.stack(rows, dim=0).to(torch.long)
        audio = _delayed_to_waveform(self._codec, delayed_LN, codebook_size=1026)

        decoded_raw_frames = emit_until_raw - window_start_raw
        samples_per_frame = max(int(audio.shape[-1]) // max(decoded_raw_frames, 1), 1)
        trim_frames = state.emitted_raw_frames - window_start_raw
        trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[-1]))
        delta = audio[trim_samples:].contiguous()
        if delta.numel() == 0:
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        state.emitted_raw_frames = emit_until_raw
        state.next_decode_rows = delayed_count + self._stream_followup_stride
        state.has_emitted = True
        return {
            "audio_data": delta.cpu().numpy().tolist(),
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }


def _build_usage(state: HiggsTtsState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(state.engine_time_s, 6)
    return usage


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    stream_stride: int = 75,
    stream_followup_stride: int = 75,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 4,
):
    """Decode Higgs delayed codes to a mono 24 kHz waveform.

    Codec weights are extracted from the TTS checkpoint itself.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)
    sample_rate = HiggsAudioCodec.SAMPLE_RATE

    def _vocode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed

        if not delayed_rows:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        if delayed_LN.shape[0] < state.num_codebooks:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        waveform = _delayed_to_waveform(codec, delayed_LN, state.codebook_size)
        payload.data["audio_data"] = waveform.cpu().numpy().tolist()
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    return HiggsVocoderScheduler(
        codec,
        _vocode,
        sample_rate=sample_rate,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
    )


__all__ = [
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
