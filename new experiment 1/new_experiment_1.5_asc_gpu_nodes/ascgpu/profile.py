from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
from torch import distributed as dist
from torch.nn import functional as functional
from torch.distributed.elastic.multiprocessing.errors import record

from .data import write_json
from .model import ForecastModel, ModelConfig, ParallelContext, clip_global_norm


@record
def profile(arguments):
    context = ParallelContext.initialize(8)
    metadata_path = arguments.root / "panel" / "meta.json"
    tickers = json.loads(metadata_path.read_text())["n_tickers"] if metadata_path.exists() else 40000
    config = ModelConfig(width=arguments.width, heads=arguments.width // 64, feedforward=4 * arguments.width,
                         temporal_depth=arguments.temporal, cross_depth=arguments.cross, tickers=tickers)
    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    model = ForecastModel(config, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1, fused=True)
    inputs = torch.randn(arguments.microbatch, 64, 256, device=context.device)
    tickers_input = torch.arange(64, device=context.device).expand(arguments.microbatch, -1)
    positions = torch.zeros(arguments.microbatch, device=context.device, dtype=torch.long)
    targets = torch.zeros(arguments.microbatch, device=context.device)
    durations = []
    for iteration in range(arguments.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(inputs, tickers_input, positions)
            loss = functional.huber_loss(predictions, targets)
        loss.backward()
        norm = clip_global_norm(model, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - started)
        if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(norm)):
            raise RuntimeError("Non-finite capacity-probe loss or gradients")
    capacity = torch.cuda.get_device_properties(context.device).total_memory
    measurements = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
                                 sum(durations[1:] or durations) / len(durations[1:] or durations)],
                                dtype=torch.float64, device=context.device)
    dist.all_reduce(measurements, op=dist.ReduceOp.MAX)
    parameters = model.parameter_count()
    result = {
        "kind": "synthetic_capacity_probe_not_training", "job_id": os.environ.get("SLURM_JOB_ID"),
        "tp": context.tp_size, "dp": context.dp_size, "parameters": parameters,
        "config": asdict(config), "microbatch": arguments.microbatch,
        "peak_allocated_bytes": int(measurements[0]), "peak_reserved_bytes": int(measurements[1]),
        "device_capacity_bytes": capacity, "seconds_per_microbatch": float(measurements[2]),
        "safe_memory_fit": bool(measurements[1] <= capacity * 0.85),
        "latest_checkpoint_bytes_estimate": parameters * 12,
        "best_checkpoint_bytes_estimate": parameters * 4,
        "optimizer": "FP32 parameters, gradients, and AdamW moments; BF16 matrix computation",
    }
    if context.rank == 0:
        write_json(arguments.root / "reports" / f"capacity-{arguments.width}-{arguments.temporal}-{arguments.cross}.json", result)
        print(json.dumps(result), flush=True)
    context.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--temporal", type=int, default=16)
    parser.add_argument("--cross", type=int, default=12)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    profile(parser.parse_args())