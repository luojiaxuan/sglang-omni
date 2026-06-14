#!/usr/bin/env python3
"""Custom all-reduce vs NCCL microbench (eager + CUDA-graph replay).

Matches Qwen3-Omni thinker decode all-reduce shape: hidden=2048, bs=8..32.

Usage (2 GPUs, TP=2):
  MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct \
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 \
    benchmarks/diagnostics/thinker_decode_forward/scripts/car_bench.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
from sglang.srt.distributed import (
    get_tensor_model_parallel_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce as AR,
)
from sglang.srt.distributed.parallel_state import graph_capture as sgl_graph_capture
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

lr = int(os.environ["LOCAL_RANK"])
ws = int(os.environ["WORLD_SIZE"])
model = os.environ.get("MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

torch.cuda.set_device(lr)
init_distributed_environment(
    world_size=ws,
    rank=lr,
    local_rank=lr,
    distributed_init_method="env://",
    backend="nccl",
)
try:
    sa = ServerArgs(model_path=model, tp_size=ws)
    set_global_server_args_for_scheduler(sa)
    if lr == 0:
        print("[sa] ServerArgs set ok")
except Exception as e:
    if lr == 0:
        print("[sa] ServerArgs FAILED:", type(e).__name__, e)
initialize_model_parallel(tensor_model_parallel_size=ws)
grp = get_tensor_model_parallel_group()
ca = getattr(grp, "ca_comm", None)
if lr == 0:
    print(
        f"[init] world={ws} ca_comm={ca} disabled={getattr(ca, 'disabled', None)} "
        f"full_nvlink={getattr(ca, 'full_nvlink', None)}"
    )

SIZES = [(2048, 32), (2048, 24), (2048, 16), (2048, 8), (2048, 1)]


def bench(label: str) -> None:
    for hidden, bs in SIZES:
        x = torch.randn(bs, hidden, dtype=torch.bfloat16, device="cuda")
        used = False
        if ca is not None and not ca.disabled:
            try:
                used = bool(ca.should_custom_ar(x))
            except Exception:
                used = False
        for _ in range(50):
            AR(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        n = 500
        for _ in range(n):
            AR(x)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / n * 1e6
        if lr == 0:
            kb = bs * hidden * 2 / 1024
            print(f"[{label}] hidden={hidden} bs={bs:3d} {kb:5.0f}KB  {us:7.1f} us/AR  custom_used={used}")


def graph_bench(label: str, hidden: int, bs: int) -> None:
    x = torch.randn(bs, hidden, dtype=torch.bfloat16, device="cuda")
    g = torch.cuda.CUDAGraph()
    try:
        with sgl_graph_capture() as gcc:
            st = gcc.stream
            s2 = torch.cuda.Stream()
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(3):
                    AR(x)
            torch.cuda.current_stream().wait_stream(s2)
            with torch.cuda.graph(g, stream=st):
                AR(x)
    except Exception as e:
        if lr == 0:
            print(f"[GRAPH-{label}] capture FAILED bs={bs}: {type(e).__name__} {e}")
        return
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    n = 500
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    if lr == 0:
        print(f"[GRAPH-{label}] hidden={hidden} bs={bs:3d} replay {us:7.1f} us/AR")


def main() -> None:
    bench("DEFAULT")
    if ca is not None:
        ca.disabled = True
        bench("NCCL")
        ca.disabled = False
    dist.barrier()

    for h, b in [(2048, 32), (2048, 24)]:
        graph_bench("CUSTOM", h, b)
    if ca is not None:
        ca.disabled = True
        for h, b in [(2048, 32), (2048, 24)]:
            graph_bench("NCCL", h, b)
        ca.disabled = False
    dist.barrier()
    if lr == 0:
        print("[done]")


if __name__ == "__main__":
    main()
