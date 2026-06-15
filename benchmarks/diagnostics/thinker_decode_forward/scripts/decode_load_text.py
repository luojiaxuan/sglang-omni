#!/usr/bin/env python3
"""Dependency-free steady-decode load driver for thinker FP8/BF16 A/B.

Fires a fixed number of concurrent text chat-completions and continuously
refills finished slots so the thinker decode batch stays near `--concurrency`
for `--duration` seconds. Use with SGLANG_OMNI_PHASE_PROFILE=1 +
SGLANG_OMNI_PHASE_SYNC=1 + SGLANG_OMNI_DECODE_STATS=1 on the server to read the
``[fwd-by-bs]`` decode-forward GPU-time curve from the server log.

Unlike the SST harness (run_concurrency.py) this needs no simuleval/audio data;
it drives the same thinker prefill+decode path with plain text requests.

Example:
    python decode_load_text.py --base-url http://127.0.0.1:8133 \
        --concurrency 32 --max-tokens 256 --duration 45
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

PROMPT = (
    "Write a long, detailed explanation of how a transformer mixture-of-experts "
    "layer routes tokens to experts, and why decode is memory-bound. Keep going."
)


def _one_request(base_url: str, model: str, max_tokens: int, stats: dict, lock: threading.Lock) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "modalities": ["text"],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        n = int(r.get("usage", {}).get("completion_tokens", 0))
        with lock:
            stats["completed"] += 1
            stats["tokens"] += n
    except Exception as e:  # noqa: BLE001 - keep the load loop alive
        with lock:
            stats["errors"] += 1
            stats["last_error"] = str(e)[:200]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8133")
    ap.add_argument("--model-name", default="qwen3-omni")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=45.0)
    args = ap.parse_args()

    stats = {"completed": 0, "tokens": 0, "errors": 0, "last_error": ""}
    lock = threading.Lock()
    deadline = time.time() + args.duration
    threads: set[threading.Thread] = set()

    def worker() -> None:
        while time.time() < deadline:
            _one_request(args.base_url, args.model_name, args.max_tokens, stats, lock)

    t0 = time.time()
    for _ in range(args.concurrency):
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        threads.add(th)
    for th in threads:
        th.join(timeout=args.duration + 300)
    wall = time.time() - t0

    tps = stats["tokens"] / wall if wall else 0.0
    print(
        f"[decode-load] concurrency={args.concurrency} wall={wall:.1f}s "
        f"completed={stats['completed']} decode_tokens={stats['tokens']} "
        f"errors={stats['errors']} throughput={tps:.1f} tok/s"
    )
    if stats["errors"]:
        print(f"[decode-load] last_error: {stats['last_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
