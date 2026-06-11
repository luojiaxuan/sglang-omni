#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GATE-ONLY audio_codes dump hook (PR-B S1-S5 gate infrastructure).

This module is NEVER imported by the serving/production code path — only the
gate launcher (``scripts/gate_serve.py``) imports it and calls ``install()``
before the server starts. So the production path carries zero trace of it: no
module load, no per-request branch, nothing. It exists solely to capture the
vocoder-entry ``state.audio_codes`` per request for the ON/OFF bit-identity
comparison.

When ``MOSS_GATE_DUMP_AUDIO_CODES=<dir>`` is set, ``install()`` monkeypatches
``apply_sglang_moss_tts_local_result`` (the function that materializes
``state.audio_codes``) to additionally write ``<dir>/<request_id>.npy``. The
scheduler result-adapter looks the function up as a module global at call time,
so rebinding it takes effect for every subsequent request. When the env var is
unset, ``install()`` is a no-op.
"""
from __future__ import annotations

import os


def install() -> bool:
    """Install the dump wrapper iff MOSS_GATE_DUMP_AUDIO_CODES is set.

    Returns True if installed, False if the env var is absent (no-op). Idempotent.
    """
    dump_dir = os.environ.get("MOSS_GATE_DUMP_AUDIO_CODES")
    if not dump_dir:
        return False
    import numpy as np

    from sglang_omni.models.moss_tts_local import request_builders as rb

    orig = rb.apply_sglang_moss_tts_local_result
    if getattr(orig, "_gate_dump_wrapped", False):
        return True
    os.makedirs(dump_dir, exist_ok=True)

    def wrapped(payload, data):
        result = orig(payload, data)
        try:
            codes = data.state.audio_codes
            arr = codes.detach().cpu().numpy() if hasattr(codes, "detach") else np.asarray(codes)
            # Key by the request's public seed, not the request_id: the server
            # assigns its own (speech-<uuid>) id and ignores client x-request-id,
            # so a seed key is the only thing stable across the ON/OFF arms (same
            # seeds) and concurrency-safe (no send-order dependency). Fall back to
            # request_id when no seed is set.
            seed = getattr(data, "seed", None)
            key = str(seed) if seed is not None else str(payload.request_id)
            np.save(os.path.join(dump_dir, f"{key}.npy"), arr)
        except Exception as exc:  # never break serving on a dump failure
            print(f"[gate_dump_hook] dump failed for {payload.request_id}: {exc}")
        return result

    wrapped._gate_dump_wrapped = True
    rb.apply_sglang_moss_tts_local_result = wrapped
    print(f"[gate_dump_hook] installed; audio_codes -> {dump_dir}")
    return True


if __name__ == "__main__":
    print("installed" if install() else "no-op (MOSS_GATE_DUMP_AUDIO_CODES unset)")
