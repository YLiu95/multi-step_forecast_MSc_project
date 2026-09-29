from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from safetensors.torch import save_file
import torch
from torch import distributed as dist
from torch.nn import functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .contract import deadline_timestamp
from .data import AnchorDataset, MAG7, REVISION, write_json
from .model import ForecastModel, ModelConfig, ParallelContext, clip_global_norm

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def barrier():
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(root, model, optimizer, state, is_best):
    context = model.context
    directory = root / "checkpoints" / f"step_{state['step']:08d}"
    started = time.monotonic()
    if context.dp_rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"tp-{context.tp_rank:02d}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None}, temporary)
        temporary.replace(target)
        if is_best:
            weights = {
                name: value.detach().to(device="cpu", dtype=torch.bfloat16 if getattr(value, "tp_kind", None) is not None else torch.float32).contiguous()
                for name, value in model.named_parameters()
            }
            target_weights = directory / f"weights-{context.tp_rank:02d}.safetensors"
            temporary_weights = target_weights.with_suffix(".tmp")
            save_file(weights, str(temporary_weights), metadata={"kind": "BF16 inference matrices with FP32 replicated normalization", "step": str(state["step"])})
            temporary_weights.replace(target_weights)
            del weights
        if context.rank == 0:
            write_json(directory / "state.json", state)
            write_json(directory / "config.json", asdict(model.config))
            write_json(directory / "partitions.json", model.partition_metadata())
    barrier()
    if context.rank == 0:
        expected_keys = set(model.state_dict())
        sizes = {}
        for shard in range(context.tp_size):
            path = directory / f"tp-{shard:02d}.pt"
            loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if set(loaded["model"]) != expected_keys or loaded["state"]["step"] != state["step"]:
                raise RuntimeError("Saved checkpoint metadata failed the restore-format check")
            if not loaded["optimizer"]["state"]:
                raise RuntimeError("A resumable checkpoint must contain optimizer moments")
            sizes[path.name] = path.stat().st_size
            del loaded
        write_json(directory / "file_sizes.json", sizes)
        (directory / "COMPLETE").write_text("complete\n")
        pointer = {"directory": directory.relative_to(root).as_posix(), "step": state["step"], "samples": state["samples"]}
        write_json(root / "latest.json", pointer)
        if is_best:
            write_json(root / "best.json", pointer)
        print(json.dumps({"event": "checkpoint_complete", "step": state["step"], "is_best": is_best,
                          "latest_bytes": sum(sizes.values()), "seconds": time.monotonic() - started}), flush=True)
    barrier()
    return time.monotonic() - started


def restore_checkpoint(directory, model, optimizer):
    path = directory / f"tp-{model.context.tp_rank:02d}.pt"
    restored = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = restored["state"]
    if state["tp"] != model.context.tp_size or state["dp"] != model.context.dp_size:
        raise ValueError("Exact pilot continuation requires the recorded TP/DP layout")
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["torch_rng"])
    if model.context.device.type == "cuda":
        torch.cuda.set_rng_state(restored["cuda_rng"], model.context.device)
    return state


def broadcast_batch(dataset, indices, seed, context, mag7_ids):
    batch_size = len(indices)
    shapes = {"inputs": ((batch_size, 64, 256), torch.float32),
              "ticker_ids": ((batch_size, 64), torch.long),
              "target_position": ((batch_size,), torch.long), "label": ((batch_size,), torch.float32),
              "market": ((batch_size,), torch.long), "weight": ((batch_size,), torch.float32),
              "mag7": ((batch_size,), torch.float32)}
    split, sample_indices = indices[0][0], [value[1] for value in indices]
    if context.tp_rank == 0:
        values = dataset.batch(split, sample_indices, seed)
        targets = values["ticker_ids"][np.arange(batch_size), values["target_position"]]
        values["mag7"] = np.isin(targets, mag7_ids).astype(np.float32)
    result = {}
    for name, (shape, dtype) in shapes.items():
        tensor = torch.as_tensor(values[name], dtype=dtype, device=context.device) if context.tp_rank == 0 else torch.empty(shape, dtype=dtype, device=context.device)
        dist.broadcast(tensor, src=context.dp_rank * context.tp_size, group=context.tp_group)
        result[name] = tensor
    return result


def evaluate(model, dataset, monitor, microbatch, context, mag7_ids, market_count):
    model.eval()
    totals = torch.zeros((market_count + 2, 9), dtype=torch.float64, device=context.device)
    rounds = math.ceil(len(monitor) / (microbatch * context.dp_size))
    for iteration in range(rounds):
        start = (iteration * context.dp_size + context.dp_rank) * microbatch
        indices = [("val", monitor[position] if position < len(monitor) else -1) for position in range(start, start + microbatch)]
        batch = broadcast_batch(dataset, indices, 7331, context, mag7_ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(batch["inputs"], batch["ticker_ids"], batch["target_position"])
        target = batch["label"]
        error = prediction - target
        metrics = torch.stack((functional.huber_loss(prediction, target, reduction="none"), error.abs(), error.square(),
                               prediction, target, prediction.square(), target.square(),
                               functional.huber_loss(torch.zeros_like(target), target, reduction="none"), torch.ones_like(target)), dim=1).double()
        weighted = metrics * batch["weight"][:, None]
        totals[0] += weighted.sum(0)
        totals[1] += (weighted * batch["mag7"][:, None]).sum(0)
        for market in range(market_count):
            totals[market + 2] += (weighted * (batch["market"] == market)[:, None]).sum(0)
    dist.all_reduce(totals, group=context.dp_group)
    summaries = []
    for row in totals.cpu().tolist():
        count = row[-1]
        summaries.append({"count": int(count), "huber": row[0] / max(count, 1), "mae": row[1] / max(count, 1),
                          "rmse": math.sqrt(row[2] / max(count, 1)), "prediction_mean": row[3] / max(count, 1),
                          "target_mean": row[4] / max(count, 1),
                          "prediction_std": math.sqrt(max(0, row[5] / max(count, 1) - (row[3] / max(count, 1))**2)),
                          "target_std": math.sqrt(max(0, row[6] / max(count, 1) - (row[4] / max(count, 1))**2)),
                          "zero_huber": row[7] / max(count, 1)})
    model.train()
    return summaries


def train(arguments):
    context = ParallelContext.initialize(8)
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, request_stop)
    panel_root = arguments.panel or arguments.root / "panel"
    metadata = json.loads((panel_root / "meta.json").read_text())
    raw_config = json.loads(arguments.config.read_text())
    config = ModelConfig(**raw_config.get("config", raw_config))
    config.tickers = metadata["n_tickers"]
    dataset = AnchorDataset(panel_root) if context.tp_rank == 0 else None
    monitor = json.loads((panel_root / "monitor.json").read_text())
    if arguments.monitor_limit:
        monitor = monitor[:arguments.monitor_limit]
    vocabulary = json.loads((panel_root / "vocabulary.json").read_text())
    mag7_ids = [index for index, entry in enumerate(vocabulary) if entry["market"] == "US" and entry["ticker"] in MAG7]
    markets = [info["name"] for info in metadata["markets"]]
    total_anchors = sum(info["anchors"]["train"] for info in metadata["markets"])
    stop_at = deadline_timestamp(arguments.train_until)
    torch.manual_seed(config.seed + context.dp_rank)
    model = ForecastModel(config, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.1, fused=True)
    wrapped = DistributedDataParallel(model, device_ids=[context.local_rank], process_group=context.dp_group,
                                      broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=32) if context.dp_size > 1 else model
    state = {"step": 0, "samples": 0, "epoch": 0, "position": 0, "best_loss": None, "best_step": None,
             "tp": 8, "dp": context.dp_size, "global_batch": arguments.global_batch, "seed": config.seed,
             "dataset_revision": REVISION, "monitor_seed": 7331, "monitor_count": len(monitor),
             "monitor_sha256": hashlib.sha256(json.dumps(monitor).encode()).hexdigest(),
             "selection_metric": "fixed_pilot_validation_subset_huber_not_full_validation", "weights": "raw_not_EMA",
             "test_evaluated": False, "next_monitor_samples": 50000, "parameters": model.parameter_count(),
             "train_stop_unix": stop_at, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if arguments.resume:
        state = restore_checkpoint(arguments.resume, model, optimizer)
        state["train_stop_unix"] = stop_at
    arguments.root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(arguments.root / "runs" / arguments.run_id)) if context.rank == 0 else None
    if context.rank == 0:
        write_json(arguments.root / "config.json", {"model": asdict(config), "training": state,
                                                    "torch": torch.__version__, "cuda": torch.version.cuda,
                                                    "dropout": 0.0, "compute_dtype": "bfloat16", "optimizer_dtype": "float32"})
        print(json.dumps({"event": "training_start", **state}), flush=True)
    order = np.load(panel_root / f"train_order_{config.seed}_{state['epoch']}.npy", mmap_mode="r") if context.tp_rank == 0 else None
    last_monitor = time.monotonic()
    estimated_step_seconds = arguments.estimated_step_seconds
    last_saved_step = state["step"] if arguments.resume else -1
    last_summary = None
    while state["step"] < arguments.max_steps:
        stopping = torch.tensor(int(STOP_REQUESTED or time.time() + max(estimated_step_seconds * 1.25, 30) >= stop_at), device=context.device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if stopping.item():
            break
        if state["position"] >= total_anchors:
            state["epoch"] += 1
            state["position"] = 0
            if context.rank == 0:
                dataset.ensure_permutation(state["epoch"], config.seed)
            barrier()
            if context.tp_rank == 0:
                order = np.load(panel_root / f"train_order_{config.seed}_{state['epoch']}.npy", mmap_mode="r")
        real_batch = min(arguments.global_batch, total_anchors - state["position"])
        accumulation = math.ceil(real_batch / (arguments.microbatch * context.dp_size))
        optimizer.zero_grad(set_to_none=True)
        learning_rate = arguments.learning_rate * min(1.0, (state["step"] + 1) / 10)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        started = time.monotonic()
        loss_sum = torch.zeros((), device=context.device)
        for microstep in range(accumulation):
            start = state["position"] + (microstep * context.dp_size + context.dp_rank) * arguments.microbatch
            positions = list(range(start, start + arguments.microbatch))
            indices = [("train", int(order[position]) if context.tp_rank == 0 and position < state["position"] + real_batch else -1) for position in positions]
            batch = broadcast_batch(dataset, indices, config.seed + state["epoch"], context, mag7_ids)
            synchronization = wrapped.no_sync() if context.dp_size > 1 and microstep + 1 < accumulation else nullcontext()
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = wrapped(batch["inputs"], batch["ticker_ids"], batch["target_position"])
                    losses = functional.huber_loss(prediction, batch["label"], reduction="none") * batch["weight"]
                    loss = losses.sum() * context.dp_size / real_batch
                loss.backward()
                loss_sum += losses.detach().sum()
        gradient_norm = clip_global_norm(model, 1.0)
        dist.all_reduce(loss_sum, group=context.dp_group)
        train_loss = float(loss_sum.item()) / real_batch
        if not math.isfinite(train_loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("Non-finite loss or gradient; refusing to publish corrupted training state")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        estimated_step_seconds = max(elapsed, estimated_step_seconds * 0.8)
        state["step"] += 1
        state["position"] += real_batch
        state["samples"] += real_batch
        if context.rank == 0:
            record = {"step": state["step"], "samples": state["samples"], "train_huber": train_loss,
                      "gradient_norm": gradient_norm, "learning_rate": learning_rate,
                      "seconds": elapsed, "unique_examples_per_second": real_batch / elapsed,
                      "epoch_fraction": state["epoch"] + state["position"] / total_anchors}
            with (arguments.root / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            for key, value in record.items():
                writer.add_scalar("train/" + key, value, state["samples"])
            writer.add_scalar("performance/peak_gpu_reserved_gib", torch.cuda.max_memory_reserved() / 2**30, state["samples"])
            writer.flush()
            print(json.dumps({"event": "optimizer_step", **record}), flush=True)
        monitor_due = state["step"] == 1 or state["samples"] >= state["next_monitor_samples"] or time.monotonic() - last_monitor >= arguments.monitor_seconds
        decision = torch.tensor(int(monitor_due) if context.rank == 0 else 0, device=context.device)
        dist.broadcast(decision, src=0)
        if decision.item():
            last_summary = evaluate(model, dataset, monitor, arguments.microbatch, context, mag7_ids, len(markets))
            best = state["best_loss"] is None or last_summary[0]["huber"] < state["best_loss"]
            if best:
                state["best_loss"], state["best_step"] = last_summary[0]["huber"], state["step"]
            while state["samples"] >= state["next_monitor_samples"]:
                state["next_monitor_samples"] += 50000
            if context.rank == 0:
                for label, summary in zip(["overall", "mag7", *markets], last_summary):
                    for key, value in summary.items():
                        writer.add_scalar(f"pilot_validation/{label}/{key}", value, state["samples"])
                writer.flush()
                print(json.dumps({"event": "pilot_validation", "step": state["step"], "metrics": last_summary[0], "is_best": best}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint(arguments.root, model, optimizer, state, best)
            last_saved_step = state["step"]
            last_monitor = time.monotonic()
    if state["step"] == 0:
        raise RuntimeError("No optimizer update completed before the deadline")
    if state["step"] != last_saved_step:
        last_summary = evaluate(model, dataset, monitor, arguments.microbatch, context, mag7_ids, len(markets))
        best = state["best_loss"] is None or last_summary[0]["huber"] < state["best_loss"]
        if best:
            state["best_loss"], state["best_step"] = last_summary[0]["huber"], state["step"]
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(arguments.root, model, optimizer, state, best)
    if context.rank == 0:
        if last_summary is not None:
            for key, value in last_summary[0].items():
                writer.add_scalar("pilot_validation/final/" + key, value, state["samples"])
        writer.close()
        write_json(arguments.root / "reports" / "training_summary.json", {**state, "last_monitor": last_summary,
                    "elapsed_training_stop_unix": time.time(), "full_epoch_completed": state["epoch"] > 0,
                    "full_validation_completed": False, "pilot_only": True})
        write_json(arguments.root / "TRAINING_DONE.json", {"step": state["step"], "samples": state["samples"], "completed_at_unix": time.time()})
        print(json.dumps({"event": "training_done", "step": state["step"], "samples": state["samples"]}), flush=True)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-until", default="2026-09-29T15:15:00+02:00")
    parser.add_argument("--run-id", default="athena-20260929")
    parser.add_argument("--global-batch", type=int, default=320)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--monitor-seconds", type=float, default=300)
    parser.add_argument("--monitor-limit", type=int, default=0)
    parser.add_argument("--estimated-step-seconds", type=float, default=30)
    parser.add_argument("--resume", type=Path)
    train(parser.parse_args())