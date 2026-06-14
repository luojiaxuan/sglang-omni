# SPDX-License-Identifier: Apache-2.0
"""gather_sampled_logprobs: per-row logprob of the sampled token (RL rollout)."""

from __future__ import annotations

import math

import torch

from sglang_omni.scheduling.types import gather_sampled_logprobs


def test_uniform_logits_give_log_uniform() -> None:
    logits = torch.zeros(1, 4)  # uniform over 4 -> log(1/4)
    ids = torch.tensor([2])
    out = gather_sampled_logprobs(logits, ids)
    assert out is not None
    assert math.isclose(out[0], math.log(0.25), abs_tol=1e-4)


def test_confident_logits_give_near_zero_logprob() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    ids = torch.tensor([0, 1])
    out = gather_sampled_logprobs(logits, ids)
    assert out[0] > -0.01 and out[1] > -0.01  # picked the confident token


def test_wrong_token_is_very_negative() -> None:
    logits = torch.tensor([[10.0, 0.0]])
    ids = torch.tensor([1])  # the unlikely token
    out = gather_sampled_logprobs(logits, ids)
    assert out[0] < -9.0


def test_none_inputs_return_none() -> None:
    assert gather_sampled_logprobs(None, torch.tensor([0])) is None
    assert gather_sampled_logprobs(torch.zeros(1, 4), None) is None
