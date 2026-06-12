#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate PR-B perf ABAB: per-arm mean +/- 95% CI and the ON-vs-OFF delta
(the async-decode speedup). For latency/rtf, ON faster = lower; for throughput,
ON faster = higher."""
from __future__ import annotations
import csv
import math
import sys

_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}


def t975(df):
    if df <= 0:
        return float("nan")
    for k in sorted(_T):
        if df <= k:
            return _T[k]
    return 1.96


def ci(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    m = sum(vals) / n
    if n == 1:
        return (m, float("nan"), 1)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return (m, t975(n - 1) * sd / math.sqrt(n), n)


def main(path):
    rows = list(csv.DictReader(open(path)))
    arms = {}
    for row in rows:
        arms.setdefault(row["arm"], []).append(row)
    metrics = [
        "rtf_mean",
        "out_throughput",
        "latency_mean",
        "latency_p95",
        "latency_p99",
        "qps",
    ]
    lower_is_faster = {"rtf_mean", "latency_mean", "latency_p95", "latency_p99"}
    print(f"\n{'='*70}\nPR-B perf ABAB ({path})")
    present = set(rows[0].keys()) if rows else set()
    agg = {}
    for m in metrics:
        if m not in present:
            continue
        print(f"\n-- {m} --")
        agg[m] = {}
        for arm in sorted(arms):
            vals = [float(r[m]) for r in arms[arm]]
            mean, half, n = ci(vals)
            agg[m][arm] = (mean, half)
            hs = "nan" if math.isnan(half) else f"{half:.4f}"
            print(f"  {arm} (n={n}): {mean:.4f} +/- {hs}")
        if "off" in agg[m] and "on" in agg[m]:
            om, oh = agg[m]["off"]
            nm, nh = agg[m]["on"]
            if m in lower_is_faster:
                delta = (om - nm) / om * 100  # +% = ON faster
                direction = "ON faster" if delta > 0 else "ON slower"
            else:
                delta = (nm - om) / om * 100  # +% = ON higher (faster)
                direction = "ON higher" if delta > 0 else "ON lower"
            sep = ""
            if not (math.isnan(oh) or math.isnan(nh)):
                disjoint = (om + oh < nm - nh) or (nm + nh < om - oh)
                sep = " [CIs disjoint -> significant]" if disjoint else " [CIs overlap -> within noise]"
            print(f"  => delta {delta:+.2f}% ({direction}){sep}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
