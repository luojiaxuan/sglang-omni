# SPDX-License-Identifier: Apache-2.0
"""Measure audio TTFT for Qwen3-Omni with and without partial-start talker.

This script is NOT a CI test. It is invoked manually inside the
``sglang-omni-hayden`` container on ion8-h200-omni and ion9-h200-omni to
collect the evidence required for the AC-10 acceptance criterion of the
partial-stream talker startup work (issue #473).

The script does not provision a server; it expects the server to already be
running with either the baseline or the enabled configuration (see README).

Usage:
    python -m tests.perf_smoke.qwen3_omni_partial_start.measure_ttft \\
        --base-url http://127.0.0.1:8000 \\
        --label disabled \\
        --output results/disabled.json \\
        --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

# Two fixed-seed synthetic prompts: one that should land near the chunk
# threshold (short) and one that should comfortably exceed it (medium). Each
# prompt asks for a clear single-language audio response so that downstream
# WER scoring can stay simple.
PROMPTS: dict[str, str] = {
    "short": "Please reply: Hello, how are you today?",
    "medium": (
        "Please respond with the following sentence verbatim: "
        "The quick brown fox jumps over the lazy dog while the sun sets "
        "over the quiet hills, and the river continues to flow gently "
        "through the valley."
    ),
}


@dataclass
class RunResult:
    label: str
    prompt_id: str
    repeat: int
    ttft_seconds: float
    total_seconds: float
    body_bytes: int
    status_code: int


@dataclass
class Summary:
    label: str
    base_url: str
    per_run: list[RunResult] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)


async def _measure_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    *,
    request_id_hint: str,
    seed: int,
) -> tuple[float, float, int, int]:
    """Issue one /v1/audio/speech streaming request and time the first byte."""
    url = f"{base_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "model": "qwen3-omni",
        "input": prompt,
        "voice": "alloy",
        "response_format": "wav",
        "stream": True,
        "seed": seed,
        "metadata": {"client_label": request_id_hint},
    }

    start = time.perf_counter()
    ttft: float | None = None
    body_bytes = 0
    status_code = 0

    async with client.stream("POST", url, json=payload, timeout=300.0) as response:
        status_code = response.status_code
        if status_code >= 400:
            text = await response.aread()
            raise RuntimeError(f"server returned {status_code}: {text[:512]!r}")
        async for chunk in response.aiter_bytes():
            if ttft is None and chunk:
                ttft = time.perf_counter() - start
            body_bytes += len(chunk)

    total = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("server returned 200 but no audio bytes")
    return ttft, total, body_bytes, status_code


async def _run(args: argparse.Namespace) -> Summary:
    summary = Summary(label=args.label, base_url=args.base_url)
    async with httpx.AsyncClient(http2=False) as client:
        for prompt_id, prompt_text in PROMPTS.items():
            ttfts: list[float] = []
            for repeat in range(args.repeats):
                seed = 1000 + repeat
                hint = f"{args.label}-{prompt_id}-{repeat}"
                ttft, total, body_bytes, status_code = await _measure_one(
                    client,
                    args.base_url,
                    prompt_text,
                    request_id_hint=hint,
                    seed=seed,
                )
                summary.per_run.append(
                    RunResult(
                        label=args.label,
                        prompt_id=prompt_id,
                        repeat=repeat,
                        ttft_seconds=ttft,
                        total_seconds=total,
                        body_bytes=body_bytes,
                        status_code=status_code,
                    )
                )
                ttfts.append(ttft)
                print(
                    f"[{args.label}] prompt={prompt_id} repeat={repeat} "
                    f"ttft={ttft:.3f}s total={total:.3f}s bytes={body_bytes}"
                )

            summary.aggregate[prompt_id] = {
                "ttft_mean": statistics.fmean(ttfts),
                "ttft_min": min(ttfts),
                "ttft_max": max(ttfts),
                "ttft_stdev": statistics.pstdev(ttfts) if len(ttfts) > 1 else 0.0,
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--label",
        required=True,
        help="Label for this run, e.g. 'disabled' or 'enabled-min5'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write JSON results.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    summary = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": summary.label,
                "base_url": summary.base_url,
                "per_run": [asdict(r) for r in summary.per_run],
                "aggregate": summary.aggregate,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
