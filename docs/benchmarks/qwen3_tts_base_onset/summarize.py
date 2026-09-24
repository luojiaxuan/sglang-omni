"""Summarize onset records written by onset_client.py.

Prints, per (model, mode), the pooled onset distribution, the share of
requests whose onset exceeds one and two codec frames (80 ms each at 12.5 Hz),
and how the variance splits between prompts and seeds; then a per-prompt
table of median and range across references and seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME_MS = 80.0


def describe(values: np.ndarray) -> str:
    return (
        f"n={values.size} median={np.median(values):.0f} mean={values.mean():.0f} "
        f"p90={np.percentile(values, 90):.0f} max={values.max():.0f} "
        f">80ms={np.mean(values > FRAME_MS):.0%} >160ms={np.mean(values > 2 * FRAME_MS):.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--metric", default="onset_peak_ms", choices=["onset_peak_ms", "onset_rms40_ms"])
    args = parser.parse_args()

    rows = [json.loads(line) for path in args.records for line in path.read_text().splitlines()]
    failed = [row for row in rows if row["status"] != 200]
    silent = [row for row in rows if row["status"] == 200 and row[args.metric] is None]
    print(f"{len(rows)} records, {len(failed)} failed, {len(silent)} never crossed the threshold")

    groups = defaultdict(list)
    for row in rows:
        if row["status"] == 200 and row[args.metric] is not None:
            groups[(row["model"], row["mode"])].append(row)

    for (model, mode), group in sorted(groups.items()):
        values = np.array([row[args.metric] for row in group])
        print(f"\n## {model} {mode}: {describe(values)}")
        by_prompt = defaultdict(list)
        by_cell = defaultdict(list)
        for row in group:
            by_prompt[row["prompt_id"]].append(row[args.metric])
            by_cell[(row["ref"], row["prompt_id"])].append(row[args.metric])
        within_seed_var = np.mean([np.var(v) for v in by_cell.values()])
        cell_means = np.array([np.mean(v) for v in by_cell.values()])
        print(f"variance split: across seeds (within ref x prompt) {within_seed_var:.0f} ms^2, "
              f"across ref x prompt cell means {np.var(cell_means):.0f} ms^2")
        print("| prompt | median | min | max | >80ms |")
        print("|---|---|---|---|---|")
        for prompt_id in sorted(by_prompt):
            v = np.array(by_prompt[prompt_id])
            text = next(row["text"] for row in group if row["prompt_id"] == prompt_id)
            print(f"| {text} | {np.median(v):.0f} | {v.min():.0f} | {v.max():.0f} | {np.mean(v > FRAME_MS):.0%} |")


if __name__ == "__main__":
    main()
