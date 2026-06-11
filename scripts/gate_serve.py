#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GATE-ONLY server launcher: install the audio_codes dump hook, then run the
normal ``sgl-omni`` CLI. Used only by the PR-B S1-S5 gate driver so the hook
never touches the production serving path.

    PYTHONPATH=<repo> MOSS_GATE_DUMP_AUDIO_CODES=<dir> \\
      python scripts/gate_serve.py serve --config examples/configs/moss_tts_local.yaml \\
      --port 8000 --async-decode on

All args after the script name are forwarded verbatim to the CLI app.
"""
from __future__ import annotations

import os
import sys

# Put scripts/ on the path so the standalone gate hook imports cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gate_dump_hook  # noqa: E402  (gate-only module)

gate_dump_hook.install()

from sglang_omni.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
