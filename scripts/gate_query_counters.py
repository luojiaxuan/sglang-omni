#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GATE-ONLY async-lookahead query-hit counter dump (PR-B benchmark only).

Standalone, never imported by serving. ONLY used for a SEPARATE instrumented ON
run to read the lookahead hit/miss rate — NOT during the clean measured ABAB
runs, because the monkeypatch adds per-resolve overhead that would bias the ON
arm. When MOSS_GATE_QUERY_DUMP=<file> is set, install() wraps
ModelRunner.execute_resolve to write the cumulative ``_async_query_hit
_async_query_miss`` to <file> every 100 resolves (amortized, tmpfs). The
benchmark reads it after the run; rate = hit / (hit + miss).
"""
from __future__ import annotations

import os


def install() -> bool:
    path = os.environ.get("MOSS_GATE_QUERY_DUMP")
    if not path:
        return False
    from sglang_omni.model_runner.base import ModelRunner

    orig = ModelRunner.execute_resolve
    if getattr(orig, "_gate_q_wrapped", False):
        return True

    state = {"n": 0}

    def wrapped(self, pending):
        r = orig(self, pending)
        state["n"] += 1
        if state["n"] % 100 == 0:
            try:
                with open(path, "w") as fh:
                    fh.write(
                        f"{getattr(self, '_async_query_hit', 0)} "
                        f"{getattr(self, '_async_query_miss', 0)}"
                    )
            except Exception:
                pass
        return r

    wrapped._gate_q_wrapped = True
    ModelRunner.execute_resolve = wrapped
    print(f"[gate_query_counters] installed -> {path}")
    return True


if __name__ == "__main__":
    print("installed" if install() else "no-op (MOSS_GATE_QUERY_DUMP unset)")
