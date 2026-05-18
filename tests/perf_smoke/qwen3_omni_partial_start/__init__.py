# SPDX-License-Identifier: Apache-2.0
"""Opt-in performance harness for Qwen3-Omni talker partial-start (issue #473).

These scripts are intentionally NOT collected by pytest in CI. They drive a
running sglang-omni server with the ``partial_start_min_chunks`` knob to
measure audio TTFT and timeline overlap between thinker and talker stages.
"""
