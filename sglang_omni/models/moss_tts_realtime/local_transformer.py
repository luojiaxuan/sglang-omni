# SPDX-License-Identifier: Apache-2.0
"""Kernel-backed local RVQ transformer for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(
            values.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * values.to(dtype)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LocalAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", hidden_size // self.num_heads))
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        eps = float(config.rms_norm_eps)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(hidden_states.shape[0])
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, self.num_heads, self.head_dim)
        )
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, self.num_key_value_heads, self.head_dim
            )
        )
        value = self.v_proj(hidden_states).view(
            batch, self.num_key_value_heads, self.head_dim
        )
        return query, key, value


class LocalMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class LocalDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = LocalAttention(config)
        self.mlp = LocalMLP(config)
        eps = float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(int(config.hidden_size), eps)
        self.post_attention_layernorm = RMSNorm(int(config.hidden_size), eps)


class LocalBackbone(nn.Module):
    """Incremental Qwen-style decoder over the 16 local codebook positions."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                self.hidden_size // int(config.num_attention_heads),
            )
        )
        self.max_positions = int(getattr(config, "rvq", 16))
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(
                    int(config.audio_vocab_size),
                    self.hidden_size,
                    int(config.audio_pad_token),
                )
                for _ in range(self.max_positions - 1)
            ]
        )
        self.layers = nn.ModuleList(
            [LocalDecoderLayer(config) for _ in range(int(config.num_hidden_layers))]
        )
        self.norm = RMSNorm(self.hidden_size, float(config.rms_norm_eps))

        inv_freq = 1.0 / (
            float(config.rope_theta)
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)
        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kv_capacity = 0
        self._kv_frozen = False

    def _ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._kv_cache
            and self._kv_capacity >= batch_size
            and self._kv_cache[0][0].device == device
            and self._kv_cache[0][0].dtype == dtype
        ):
            return
        if self._kv_frozen:
            raise RuntimeError(
                "MOSS-TTS-Realtime local KV cache cannot grow after graph capture"
            )
        capacity = max(batch_size, self._kv_capacity, 1)
        shape = (
            capacity,
            self.num_key_value_heads,
            self.max_positions,
            self.head_dim,
        )
        self._kv_cache = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]
        self._kv_capacity = capacity

    def freeze_kv_cache(self) -> None:
        self._kv_frozen = True

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if not 0 <= position < self.max_positions:
            raise ValueError(f"local position {position} is out of range")
        batch = int(hidden_states.shape[0])
        self._ensure_kv_cache(batch, hidden_states.device, hidden_states.dtype)
        cos = self.rope_cos[position].to(hidden_states.dtype)
        sin = self.rope_sin[position].to(hidden_states.dtype)

        values = hidden_states
        for layer_index, layer in enumerate(self.layers):
            residual = values
            query, key, value = layer.self_attn.project(
                layer.input_layernorm(values)
            )
            query = query * cos + _rotate_half(query) * sin
            key = key * cos + _rotate_half(key) * sin
            key_cache, value_cache = self._kv_cache[layer_index]
            key_cache[:batch, :, position].copy_(key)
            value_cache[:batch, :, position].copy_(value)
            attended = F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key_cache[:batch, :, : position + 1],
                value_cache[:batch, :, : position + 1],
                enable_gqa=True,
            )
            values = residual + layer.self_attn.o_proj(
                attended.squeeze(2).reshape(batch, self.hidden_size)
            )
            values = values + layer.mlp(layer.post_attention_layernorm(values))
        return self.norm(values)


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Checkpoint-compatible local decoder and its per-codebook LM heads."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = LocalBackbone(config)
        self.local_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    int(config.hidden_size),
                    int(config.audio_vocab_size),
                    bias=False,
                )
                for _ in range(int(config.rvq))
            ]
        )

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        return self.model.step(hidden_states, position)

    def freeze_kv_cache(self) -> None:
        self.model.freeze_kv_cache()

    def ensure_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.model._ensure_kv_cache(batch_size, device, dtype)


__all__ = ["MossTTSRealtimeLocalTransformer"]
