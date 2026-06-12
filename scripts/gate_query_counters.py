#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GATE-ONLY async-lookahead query-hit counter dump (PR-B benchmark only).

Standalone, never imported by serving. ONLY used for a SEPARATE instrumented ON
run to read the lookahead hit/miss rate — NOT during the clean measured ABAB
runs, because the monkeypatch adds per-resolve overhead that would bias the ON
arm. When MOSS_GATE_QUERY_DUMP=<file> is set, install() wraps
ModelRunner.execute_resolve to write the cumulative ``_async_query_hit
_async_query_miss`` to <file> on every resolve (tmpfs). The S1-S5 gate reads it
after each ON arm to assert lookahead actually engaged (hit+miss > 0) — or, for
the sync-routed scenario, did NOT (hit+miss == 0, file stays absent). It writes
every resolve, not every 100th, because a gate scenario is short (tens of
resolves) and would otherwise never flush; the extra file write does not touch
any tensor or RNG, so it leaves the ON-arm audio bit-identical. NOT for the
clean measured ABAB runs (the per-resolve write would bias latency there).
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

    def wrapped(self, pending):
        r = orig(self, pending)
        # Every resolve: a gate scenario is short, so amortizing would never
        # flush. Skip the write when nothing was resolved (pending is None, the
        # first iteration / post-drain no-op) so the counters reflect real
        # lookahead resolves only.
        if pending is not None:
            try:
                with open(path, "w") as fh:
                    fh.write(
                        f"{getattr(self, '_async_query_hit', 0)} "
                        f"{getattr(self, '_async_query_miss', 0)}"
                    )
            except Exception:
                # Best-effort gate-only telemetry: never let a counter-dump I/O
                # error perturb the resolve it wraps (the gate reads whatever the
                # last successful write left).
                pass
        return r

    wrapped._gate_q_wrapped = True
    ModelRunner.execute_resolve = wrapped
    print(f"[gate_query_counters] installed -> {path}")
    return True


if __name__ == "__main__":
    print("installed" if install() else "no-op (MOSS_GATE_QUERY_DUMP unset)")
