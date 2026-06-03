# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.weight_checker import StrictWeightChecker


def test_strict_weight_checker_snapshot_compare_and_checksum() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    runner = SimpleNamespace(model=model)
    checker = StrictWeightChecker(runner)

    snapshot = checker.run("snapshot")
    checksum = checker.run("checksum")
    compare_before = checker.run("compare")

    with torch.no_grad():
        model.weight[0, 0] += 1.0

    compare_after = checker.run("compare")

    assert snapshot["tensor_count"] == 1
    assert set(snapshot["checksums"]) == {"weight"}
    assert checksum["per_gpu_checksum"] == snapshot["per_gpu_checksum"]
    assert compare_before["matched"] is True
    assert compare_after["matched"] is False
    assert compare_after["changed"] == ["weight"]
