# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.model_runner.encoder_model_runner import EncoderModelRunner
from sglang_omni.models.qwen3_omni.components.image_encoder import (
    _ensure_sglang_vision_runtime,
    _load_sglang_visual_weights,
)
from sglang_omni.models.qwen3_omni.encoder_model_runner import (
    Qwen3OmniAudioEncoderModelRunner,
    Qwen3OmniImageEncoderModelRunner,
)
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.scheduling.stage_cache import StageOutputCache
from tests.unit_test.fixtures.qwen_fakes import (
    FakeAudioEncoderModel,
    FakeImageEncoderModel,
    make_qwen_payload,
    make_qwen_state,
)


def _set_sglang_mm_attention_backend(backend: str | None) -> None:
    from sglang.srt.server_args import get_global_server_args

    get_global_server_args().mm_attention_backend = backend


def _init_finite_parameters(module: torch.nn.Module) -> None:
    for param in module.parameters():
        torch.nn.init.uniform_(param, -0.01, 0.01)


class _GraphDispatchRunner(EncoderModelRunner):
    def __init__(self, *, use_graph: bool) -> None:
        super().__init__(model=object(), stage_name="test_encoder")
        self.use_graph = use_graph

    def load_state(self, payload: Any) -> Any:
        return payload

    def store_state(self, payload: Any, state: Any) -> Any:
        del state
        return payload

    def build_encoder_request(self, payload: Any, state: Any) -> Any:
        del state
        return payload

    def apply_result(self, state: Any, result: Any) -> None:
        del state, result

    def can_run_cuda_graph(self, prepared: Any) -> bool:
        del prepared
        return self.use_graph

    def forward_eager(self, prepared: Any) -> str:
        del prepared
        return "eager"

    def forward_cuda_graph(self, prepared: Any) -> str:
        del prepared
        return "graph"


class VisionFlash3Attention:
    pass


class _GraphPolicyVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.config = SimpleNamespace(hidden_size=8, num_heads=2)
        self.blocks = [
            SimpleNamespace(
                attn=SimpleNamespace(qkv_backend=VisionFlash3Attention()),
            )
        ]


class _GraphPolicyImageModel:
    def __init__(self, *, spatial_merge_size: int = 1) -> None:
        self.visual = _GraphPolicyVisual().eval()
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = 8
        self.deepstack_layers = 0
        self.visual_dtype_bytes = 4

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}


class _GraphPolicyImageRunner(Qwen3OmniImageEncoderModelRunner):
    def _visual_cuda_graph_supported(self) -> bool:
        return True


def _assert_single_graph_fallback(
    runner: Qwen3OmniImageEncoderModelRunner,
    reason: str,
) -> None:
    assert runner.cuda_graph_fallback_reasons == {reason: 1}
    assert runner.cuda_graph_stats.fallbacks == 1
    assert runner.cuda_graph_stats.fallbacks == sum(
        runner.cuda_graph_fallback_reasons.values()
    )


def test_encoder_model_runner_dispatches_cuda_graph_from_forward() -> None:
    assert _GraphDispatchRunner(use_graph=True).forward({}) == "graph"
    assert _GraphDispatchRunner(use_graph=False).forward({}) == "eager"


def test_encoder_model_runner_reuses_static_buffers_until_capture() -> None:
    runner = _GraphDispatchRunner(use_graph=False)
    first = runner.static_input_buffer(
        "small",
        "input",
        shape=(2, 3),
        dtype=torch.float32,
        device="cpu",
    )
    second = runner.static_input_buffer(
        "small",
        "input",
        shape=(2, 3),
        dtype=torch.float32,
        device="cpu",
    )
    assert second.data_ptr() == first.data_ptr()

    resized = runner.static_input_buffer(
        "small",
        "input",
        shape=(3, 3),
        dtype=torch.float32,
        device="cpu",
    )
    assert resized.shape == (3, 3)

    runner.cuda_graphs["small"] = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Cannot resize CUDA graph buffer"):
        runner.static_input_buffer(
            "small",
            "input",
            shape=(4, 3),
            dtype=torch.float32,
            device="cpu",
        )


def test_qwen_image_encoder_runner_batches_splits_and_uses_cache() -> None:
    model = FakeImageEncoderModel()
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        cache=StageOutputCache(max_size=8, cache_device="cpu"),
    )
    state = make_qwen_state(
        encoder_inputs={
            "image_encoder": {
                "cache_key": "shared-image",
                "pixel_values": torch.ones((2, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            }
        }
    )
    first = make_qwen_payload(state, request_id="image-1")
    duplicate = make_qwen_payload(state, request_id="image-2")

    outputs = runner.execute_batch([first, duplicate])

    assert len(outputs) == 2
    assert len(model.calls) == 1
    for payload in outputs:
        out_state = PipelineState.from_dict(payload.data)
        image_out = out_state.encoder_outs["image_encoder"]
        assert image_out["image_embeds"].shape == (2, 2)
        assert image_out["image_token_counts"].tolist() == [2]
        assert image_out["deepstack_visual_embeds_image"][0].shape == (2, 2)

    cached = runner.execute(make_qwen_payload(state, request_id="image-3"))

    assert len(model.calls) == 1
    cached_state = PipelineState.from_dict(cached.data)
    assert cached_state.encoder_outs["image_encoder"]["image_embeds"].shape == (2, 2)


def test_qwen_image_encoder_graph_body_matches_visual_forward() -> None:
    config_module = pytest.importorskip("sglang.srt.configs.qwen3_omni")
    modeling_module = pytest.importorskip("sglang.srt.models.qwen3_omni_moe")
    _ensure_sglang_vision_runtime("dummy", device="cpu")
    _set_sglang_mm_attention_backend("sdpa")

    config = config_module.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    visual = modeling_module.Qwen3OmniMoeVisionEncoder(config).eval()
    _init_finite_parameters(visual)
    model = SimpleNamespace(
        visual=visual,
        spatial_merge_size=1,
        out_hidden_size=8,
        deepstack_layers=1,
        visual_dtype_bytes=4,
    )
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        enable_cuda_graph=False,
    )
    pixel_values = torch.randn(4, 12)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    with torch.no_grad():
        eager_output = visual(pixel_values, grid_thw)
        eager_embeds = eager_output[:, :8]
        eager_deepstack = [eager_output[:, 8:16]]
        prepared = runner._prepare_visual_forward_inputs(pixel_values, grid_thw)
        graph_body = runner._forward_visual_graph_body(
            hidden_states=prepared["hidden_states"],
            cu_seqlens=prepared["cu_seqlens"],
            position_embeddings=prepared["position_embeddings"],
        )

    torch.testing.assert_close(graph_body["embeds"], eager_embeds)
    assert len(graph_body["deepstack"]) == len(eager_deepstack)
    for graph_tensor, eager_tensor in zip(
        graph_body["deepstack"],
        eager_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)


def test_qwen_image_encoder_sglang_weight_loader_matches_hf_visual() -> None:
    hf_config_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe"
    )
    hf_modeling_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
    )
    sgl_config_module = pytest.importorskip("sglang.srt.configs.qwen3_omni")
    sgl_modeling_module = pytest.importorskip("sglang.srt.models.qwen3_omni_moe")
    _ensure_sglang_vision_runtime("dummy", device="cpu")
    _set_sglang_mm_attention_backend("sdpa")

    config_kwargs = dict(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    hf_visual = hf_modeling_module.Qwen3OmniMoeVisionEncoder(
        hf_config_module.Qwen3OmniMoeVisionEncoderConfig(**config_kwargs)
    ).eval()
    _init_finite_parameters(hf_visual)
    sgl_visual = sgl_modeling_module.Qwen3OmniMoeVisionEncoder(
        sgl_config_module.Qwen3OmniMoeVisionEncoderConfig(**config_kwargs)
    ).eval()

    loaded = _load_sglang_visual_weights(sgl_visual, dict(hf_visual.state_dict()))

    assert len(loaded) == len(dict(sgl_visual.named_parameters(remove_duplicate=False)))

    pixel_values = torch.randn(4, 12)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        hf_embeds, hf_deepstack = hf_visual(pixel_values, grid_thw)
        sgl_output = sgl_visual(pixel_values, grid_thw)

    torch.testing.assert_close(sgl_output[:, :8], hf_embeds, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        sgl_output[:, 8:16],
        hf_deepstack[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_qwen_image_encoder_uses_finite_video_graph_budgets() -> None:
    config_module = pytest.importorskip("sglang.srt.configs.qwen3_omni")
    modeling_module = pytest.importorskip("sglang.srt.models.qwen3_omni_moe")
    _ensure_sglang_vision_runtime("dummy", device="cpu")
    _set_sglang_mm_attention_backend("sdpa")

    config = config_module.Qwen3OmniMoeVisionEncoderConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[],
    )
    visual = modeling_module.Qwen3OmniMoeVisionEncoder(config).eval()
    _init_finite_parameters(visual)
    model = SimpleNamespace(
        visual=visual,
        spatial_merge_size=1,
        out_hidden_size=8,
        deepstack_layers=0,
        visual_dtype_bytes=4,
    )
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        cuda_graph_token_budgets=(64,),
        cuda_graph_sequence_budgets=(8,),
        cuda_graph_max_sequence_token_budgets=(16,),
    )
    first_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    second_grid = torch.tensor([[1, 4, 4], [2, 2, 4]], dtype=torch.long)
    first_pixels = torch.randn(32, 12)
    second_pixels = torch.randn(32, 12)

    first_budget = runner._select_visual_graph_budget(first_pixels, first_grid)
    second_budget = runner._select_visual_graph_budget(second_pixels, second_grid)

    assert first_budget == second_budget
    assert first_budget is not None
    assert first_budget.token_budget == 64
    assert first_budget.sequence_budget == 8
    assert first_budget.max_sequence_token_budget == 16

    first_cu = runner._build_budgeted_cu_seqlens(
        grid_thw=first_grid,
        graph_key=first_budget,
        device=torch.device("cpu"),
    )
    second_cu = runner._build_budgeted_cu_seqlens(
        grid_thw=second_grid,
        graph_key=second_budget,
        device=torch.device("cpu"),
    )

    assert first_cu.shape == second_cu.shape == (9,)
    assert int(first_cu[-1].item()) == 64
    assert int(second_cu[-1].item()) == 64
    assert int((first_cu[1:] - first_cu[:-1]).max().item()) == 16
    assert int((second_cu[1:] - second_cu[:-1]).max().item()) == 16


def test_qwen_image_encoder_factory_exposes_cuda_graph_controls(monkeypatch) -> None:
    from sglang_omni.models.qwen3_omni import stages as qwen_stages

    runner_kwargs: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            runner_kwargs.update(kwargs)

        def execute(self, payload: Any) -> Any:
            return payload

        def execute_batch(self, payloads: list[Any]) -> list[Any]:
            return payloads

        def estimate_payload_cost(self, payload: Any) -> int:
            del payload
            return 0

    monkeypatch.setattr(
        qwen_stages,
        "Qwen3OmniImageEncoder",
        SimpleNamespace,
    )
    monkeypatch.setattr(qwen_stages, "Qwen3OmniImageEncoderModelRunner", FakeRunner)

    qwen_stages.create_image_encoder_executor(
        "dummy",
        enable_cuda_graph=False,
        cuda_graph_token_budgets=(4,),
        cuda_graph_sequence_budgets=(1,),
        cuda_graph_max_sequence_token_budgets=(4,),
        cuda_graph_max_graphs=1,
        cuda_graph_max_buffer_bytes=123,
    )

    assert runner_kwargs["enable_cuda_graph"] is False
    assert runner_kwargs["cuda_graph_token_budgets"] == (4,)
    assert runner_kwargs["cuda_graph_sequence_budgets"] == (1,)
    assert runner_kwargs["cuda_graph_max_sequence_token_budgets"] == (4,)
    assert runner_kwargs["cuda_graph_max_graphs"] == 1
    assert runner_kwargs["cuda_graph_max_buffer_bytes"] == 123


def test_qwen_image_encoder_graph_fallback_stats_cover_planning_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cases = [
        (
            "no_fitting_budget",
            _GraphPolicyImageModel(),
            torch.randn(6, 12),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        ),
        (
            "grid_not_cpu",
            _GraphPolicyImageModel(),
            torch.randn(4, 12),
            torch.empty((1, 3), dtype=torch.long, device="meta"),
        ),
        (
            "pixel_grid_mismatch",
            _GraphPolicyImageModel(),
            torch.randn(3, 12),
            torch.tensor([[1, 2, 2]], dtype=torch.long),
        ),
        (
            "unaligned_token_count",
            _GraphPolicyImageModel(spatial_merge_size=2),
            torch.randn(2, 12),
            torch.tensor([[1, 1, 2]], dtype=torch.long),
        ),
    ]

    for reason, model, pixel_values, grid_thw in cases:
        runner = _GraphPolicyImageRunner(
            model=model,
            cuda_graph_token_budgets=(4,),
            cuda_graph_sequence_budgets=(1,),
            cuda_graph_max_sequence_token_budgets=(4,),
        )

        runner.forward(
            {
                "model_inputs": {
                    "pixel_values": pixel_values,
                    "image_grid_thw": grid_thw,
                },
                "metas": [],
            }
        )

        _assert_single_graph_fallback(runner, reason)


def test_qwen_image_encoder_graph_limit_fallback_stats_match_reason_counts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    max_graph_runner = _GraphPolicyImageRunner(
        model=_GraphPolicyImageModel(),
        cuda_graph_token_budgets=(4, 8),
        cuda_graph_sequence_budgets=(1,),
        cuda_graph_max_sequence_token_budgets=(4, 8),
        cuda_graph_max_graphs=1,
    )
    existing_budget = max_graph_runner._select_visual_graph_budget(
        torch.randn(4, 12),
        torch.tensor([[1, 2, 2]], dtype=torch.long),
    )
    assert existing_budget is not None
    max_graph_runner.cuda_graphs[existing_budget] = object()  # type: ignore[assignment]

    max_graph_runner.forward(
        {
            "model_inputs": {
                "pixel_values": torch.randn(8, 12),
                "image_grid_thw": torch.tensor([[1, 4, 2]], dtype=torch.long),
            },
            "metas": [],
        }
    )

    _assert_single_graph_fallback(max_graph_runner, "max_graphs")

    max_buffer_runner = _GraphPolicyImageRunner(
        model=_GraphPolicyImageModel(),
        cuda_graph_token_budgets=(4,),
        cuda_graph_sequence_budgets=(1,),
        cuda_graph_max_sequence_token_budgets=(4,),
        cuda_graph_max_buffer_bytes=1,
    )

    max_buffer_runner.forward(
        {
            "model_inputs": {
                "pixel_values": torch.randn(4, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            },
            "metas": [],
        }
    )

    _assert_single_graph_fallback(max_buffer_runner, "max_buffer_bytes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_qwen_image_encoder_runner_replays_cuda_graph() -> None:
    config_module = pytest.importorskip("sglang.srt.configs.qwen3_omni")
    modeling_module = pytest.importorskip("sglang.srt.models.qwen3_omni_moe")
    _ensure_sglang_vision_runtime("dummy", device="cuda")
    _set_sglang_mm_attention_backend(None)

    config = config_module.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=64,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    visual = (
        modeling_module.Qwen3OmniMoeVisionEncoder(config)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    _init_finite_parameters(visual)
    model = SimpleNamespace(
        visual=visual,
        spatial_merge_size=1,
        out_hidden_size=64,
        deepstack_layers=1,
        visual_dtype_bytes=4,
    )
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        cuda_graph_token_budgets=(16,),
        cuda_graph_sequence_budgets=(4,),
        cuda_graph_max_sequence_token_budgets=(4,),
    )
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    def run_graph(pixel_values: torch.Tensor) -> dict[str, Any]:
        return runner.forward(
            {
                "model_inputs": {
                    "pixel_values": pixel_values,
                    "image_grid_thw": grid_thw,
                },
                "metas": [],
            }
        )

    with torch.no_grad():
        first_pixels = torch.randn(4, 12)
        first_eager_output = visual(first_pixels.cuda(), grid_thw)
        first_eager = first_eager_output[:, :64]
        first_deepstack = [first_eager_output[:, 64:128]]
        first_graph = run_graph(first_pixels)

        second_pixels = torch.randn(4, 12)
        second_eager_output = visual(second_pixels.cuda(), grid_thw)
        second_eager = second_eager_output[:, :64]
        second_deepstack = [second_eager_output[:, 64:128]]
        second_graph = run_graph(second_pixels)

    assert len(runner.cuda_graphs) == 1
    torch.testing.assert_close(first_graph["image_embeds"], first_eager)
    torch.testing.assert_close(second_graph["image_embeds"], second_eager)
    for graph_tensor, eager_tensor in zip(
        first_graph["deepstack_visual_embeds_image"],
        first_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)
    for graph_tensor, eager_tensor in zip(
        second_graph["deepstack_visual_embeds_image"],
        second_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)


def test_qwen_audio_encoder_runner_pads_and_splits_batch_outputs() -> None:
    model = FakeAudioEncoderModel()
    runner = Qwen3OmniAudioEncoderModelRunner(model=model)
    first = make_qwen_payload(
        make_qwen_state(
            encoder_inputs={
                "audio_encoder": {
                    "input_features": torch.ones((1, 2, 2)),
                    "audio_feature_lengths": torch.tensor([2]),
                }
            }
        ),
        request_id="audio-1",
    )
    second = make_qwen_payload(
        make_qwen_state(
            encoder_inputs={
                "audio_encoder": {
                    "input_features": torch.ones((1, 2, 3)),
                    "audio_feature_lengths": torch.tensor([3]),
                }
            }
        ),
        request_id="audio-2",
    )

    outputs = runner.execute_batch([first, second])

    assert len(outputs) == 2
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["input_features"].shape == (2, 2, 3)
    assert call["audio_feature_lengths"].tolist() == [2, 3]

    first_out = PipelineState.from_dict(outputs[0].data).encoder_outs["audio_encoder"]
    second_out = PipelineState.from_dict(outputs[1].data).encoder_outs["audio_encoder"]
    assert first_out["audio_embeds"].shape == (2, 2)
    assert second_out["audio_embeds"].shape == (3, 2)
    assert first_out["audio_output_lengths"].tolist() == [2]
    assert second_out["audio_output_lengths"].tolist() == [3]
