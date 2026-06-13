# SPDX-License-Identifier: Apache-2.0
"""Small env/metric helpers shared by the scheduler.

Kept dependency-free so the scheduler hot path can import them cheaply. All env
readers fall back to a default (current behavior) when unset or malformed.
"""

from __future__ import annotations

import os
from typing import Any


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def batch_size(batch: Any) -> int:
    if batch is None:
        return 0
    return len(getattr(batch, "reqs", ()) or ())
