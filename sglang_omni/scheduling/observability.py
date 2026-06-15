# SPDX-License-Identifier: Apache-2.0
"""Best-effort scheduler and CUDA telemetry helpers."""

from __future__ import annotations

import os
from typing import Any


def env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def safe_qsize(queue_obj: Any) -> int | None:
    qsize = getattr(queue_obj, "qsize", None)
    if not callable(qsize):
        return None
    try:
        return int(qsize())
    except Exception:
        return None


def batch_size(batch: Any) -> int:
    if batch is None:
        return 0
    return len(getattr(batch, "reqs", ()) or ())


def allocator_available_tokens(allocator: Any) -> int | None:
    available_size = getattr(allocator, "available_size", None)
    if not callable(available_size):
        return None
    try:
        return int(available_size())
    except Exception:
        return None


def cuda_memory_snapshot(gpu_id: int | None) -> dict[str, int | None]:
    snapshot: dict[str, int | None] = {
        "gpu_id": gpu_id,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_max_allocated_bytes": None,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
    }
    if gpu_id is None:
        return snapshot
    try:
        import torch

        if not torch.cuda.is_available():
            return snapshot
        device = int(gpu_id)
        snapshot["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        snapshot["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        snapshot["cuda_max_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot["cuda_free_bytes"] = int(free_bytes)
        snapshot["cuda_total_bytes"] = int(total_bytes)
    except Exception:
        return snapshot
    return snapshot
