from __future__ import annotations

import math
import os
import time
from copy import deepcopy

import torch
import torch.distributed as distributed

from .losses import batch_metric_sums, finalize_metrics


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl")
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if distributed.is_available() and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def learning_rate(cfg, global_step: int) -> float:
    warmup = cfg.warmup_epochs * cfg.steps_per_epoch
    total = cfg.epochs * cfg.steps_per_epoch
    if global_step < warmup:
        return cfg.lr * (global_step + 1) / max(warmup, 1)
    progress = min(max((global_step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cosine)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = deepcopy(unwrap(model)).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = unwrap(model).state_dict()
        for name, target in self.shadow.state_dict().items():
            if target.dtype.is_floating_point:
                target.lerp_(source[name], 1 - self.decay)
            else:
                target.copy_(source[name])


def train_epoch(cfg, model, panel, optimizer, scaler, criterion, ema, state,
                generator, callbacks, device):
    model.train()
    accumulated = torch.zeros(3, device=device)
    started = time.time()
    seen = 0
    for batch in panel.random_batches("train", cfg.batch_size,
                                      cfg.steps_per_epoch, generator):
        rate = learning_rate(cfg, state["global_step"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            prediction = model(batch["x"], batch["ticker_ids"],
                               batch["target_position"])
            loss, components = criterion(prediction, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        accumulated += torch.stack((loss.detach(), components["magnitude"],
                                    components["direction"]))
        seen += len(batch["x"])
        state["global_step"] += 1
        if state["is_main"] and state["global_step"] % cfg.log_every_steps == 0:
            state.update({
                "step_loss": loss.item(),
                "step_magnitude_loss": components["magnitude"].item(),
                "step_direction_loss": components["direction"].item(),
                "lr": rate,
                "gradient_norm": gradient_norm.item(),
                "loss_scale": scaler.get_scale(),
                "samples_per_second": seen * state["world_size"] / max(time.time() - started, 1e-9),
            })
            callbacks.on_step_end(state)
    averages = accumulated / cfg.steps_per_epoch
    return {
        "loss": averages[0].item(),
        "magnitude_loss": averages[1].item(),
        "direction_loss": averages[2].item(),
    }


@torch.no_grad()
def evaluate(cfg, model, panel, criterion, device, rank: int, world_size: int,
             split: str = "val", n_batches: int | None = None):
    model.eval()
    loss_sums = torch.zeros(4, device=device, dtype=torch.float64)
    metric_sums: dict[str, torch.Tensor] = {}
    batch_count = cfg.val_batches if n_batches is None else n_batches
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    for batch in panel.fixed_batches(split, cfg.batch_size, batch_count,
                                     seed=cfg.seed + split_offset + rank):
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            prediction = model(batch["x"], batch["ticker_ids"],
                               batch["target_position"])
            loss, components = criterion(prediction, batch)
        count = len(batch["x"])
        loss_sums += torch.tensor([loss.item() * count,
                                  components["magnitude"].item() * count,
                                  components["direction"].item() * count,
                                  count], device=device, dtype=torch.float64)
        current = batch_metric_sums(prediction, batch)
        for name, value in current.items():
            metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
    if world_size > 1:
        distributed.all_reduce(loss_sums)
        for value in metric_sums.values():
            distributed.all_reduce(value)
    metrics = finalize_metrics(metric_sums)
    denominator = loss_sums[3].clamp_min(1)
    metrics.update({
        "loss": (loss_sums[0] / denominator).item(),
        "magnitude_loss": (loss_sums[1] / denominator).item(),
        "direction_loss": (loss_sums[2] / denominator).item(),
    })
    return metrics


def checkpoint_payload(cfg, model, optimizer, scaler, ema, state,
                       n_tickers: int, include_optimizer: bool) -> dict:
    payload = {
        "config": cfg.to_dict(),
        "model": unwrap(model).state_dict(),
        "ema": ema.shadow.state_dict() if ema else None,
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "best_value": state["best_value"],
        "n_tickers": n_tickers,
    }
    if include_optimizer:
        payload.update({"optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict()})
    return payload


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None,
                    map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(payload["model"])
    if ema is not None and payload.get("ema") is not None:
        ema.shadow.load_state_dict(payload["ema"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload