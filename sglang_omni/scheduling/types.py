# SPDX-License-Identifier: Apache-2.0
"""Scheduling types used by OmniScheduler and SGLang components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class SchedulerStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass
class SchedulerRequest:
    request_id: str
    status: SchedulerStatus = SchedulerStatus.WAITING
    data: Any = None
    error: Exception | None = None
    arrival_time: float = 0.0
    finish_time: float | None = None


@dataclass
class SchedulerOutput:
    requests: list[SchedulerRequest]
    batch_data: Any
    step_id: int = 0

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]


@dataclass
class RequestOutput:
    request_id: str
    data: Any = None
    finished: bool = False
    extra: dict[str, Any] | None = None


@dataclass
class ModelRunnerOutput:
    outputs: dict[str, RequestOutput]
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    can_run_cuda_graph: bool = False


@dataclass
class ARRequestData:
    """Backend-neutral autoregressive request state."""

    input_ids: "torch.Tensor | None" = None
    attention_mask: "torch.Tensor | None" = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    # RL-rollout artifacts.
    weight_version: str | None = None
    return_logprob: bool = False
    output_token_logprobs: list[Any] = field(default_factory=list)
    capture_model_output_keys: tuple[str, ...] = ()
    max_new_tokens: int | None = None
    enforce_request_limits: bool = False
    temperature: float = 0.0


def gather_sampled_logprobs(next_token_logits: Any, sampled_ids: Any) -> list | None:
    """Per-row log-prob of each sampled token, for RL rollout.

    ``next_token_logits`` is ``[batch, vocab]`` and ``sampled_ids`` is
    ``[batch]``; returns one ``log_softmax`` value per row (the rollout log-prob
    of the chosen token under the post-penalty distribution), or ``None`` when
    the inputs are unusable.
    """
    import torch

    if next_token_logits is None or sampled_ids is None:
        return None
    logits = next_token_logits.float()
    if logits.ndim != 2:
        return None
    logprobs = torch.log_softmax(logits, dim=-1)
    ids = sampled_ids.to(device=logits.device, dtype=torch.long).view(-1, 1)
    return logprobs.gather(1, ids).squeeze(1).tolist()
