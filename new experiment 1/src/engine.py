"""Training engine: DDP, fp16 AMP, cosine schedule, EMA, resume.

Distributed strategy
--------------------
`DistributedDataParallel` with one process per GPU, launched by `torchrun`.
The tempting alternative, `nn.DataParallel`, runs both GPUs from a single
Python process: the GIL serialises the forward passes, gradients are gathered
on GPU 0 (so its memory fills first), and you typically get ~1.3x rather than
~1.95x from two cards. DDP runs a real process per GPU and overlaps the
gradient all-reduce with the backward pass.

Mixed precision
---------------
A T4 is `sm_75`. It has fp16 tensor cores but **no bfloat16**, so we use fp16
plus a `GradScaler`. fp16 has a narrow exponent range; the scaler multiplies
the loss by a large factor before `backward()` so small gradients do not flush
to zero, then divides it out before the optimiser step. Without it a deep
fp16 model silently trains to nothing.

What an "epoch" means here
--------------------------
There are ~15M anchors, far more than we want to touch between validations, so
an epoch is a fixed budget of `steps_per_epoch` randomly sampled batches per
rank. That decouples "how often do I checkpoint" from "how big is the dataset",
which is what you want when the dataset can change size.
"""
from __future__ import annotations

import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import Config
from .losses import ForecastLoss, metrics, point_forecast


# --------------------------------------------------------------------------- #
#  Distributed helpers
# --------------------------------------------------------------------------- #
def setup_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank); initialise NCCL if launched by torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world, local


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
#  Schedule
# --------------------------------------------------------------------------- #
def lr_at(cfg: Config, step: int, steps_per_epoch: int) -> float:
    """Linear warmup then cosine decay to `min_lr_frac * lr`.

    Warmup matters for Transformers: the attention softmax is very sensitive
    early on, and a full-size first step can push it into a degenerate state
    it never recovers from.
    """
    warmup = cfg.warmup_epochs * steps_per_epoch
    total = cfg.epochs * steps_per_epoch
    if step < warmup:
        return cfg.lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cos)


# --------------------------------------------------------------------------- #
#  EMA
# --------------------------------------------------------------------------- #
class ModelEMA:
    """Exponential moving average of the weights.

    SGD on a noisy objective wanders around the minimum rather than sitting in
    it. Averaging recent weights is a nearly free variance reduction and on
    financial targets it is usually worth several basis points of val loss.

    The update is fused with `torch._foreach_*`. The obvious loop over
    `state_dict()` issues ~300 tiny CUDA kernels every step, which on a 0.5 s
    step is a measurable tax; the foreach version issues 2.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = deepcopy(_unwrap(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._src: list[torch.Tensor] | None = None
        self._dst: list[torch.Tensor] = []
        self._copy: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _bind(self, model: torch.nn.Module) -> None:
        msd = _unwrap(model).state_dict()
        self._src, self._dst, self._copy = [], [], []
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                self._dst.append(v)
                self._src.append(msd[k])
            else:
                self._copy.append((v, msd[k]))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if self._src is None:
            self._bind(model)
        torch._foreach_mul_(self._dst, self.decay)
        torch._foreach_add_(self._dst, self._src, alpha=1 - self.decay)
        for dst, src in self._copy:
            dst.copy_(src)

    def load_state_dict(self, sd) -> None:
        self.shadow.load_state_dict(sd)
        self._src = None                     # rebind: the tensors were replaced


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)      # undo torch.compile wrapper


# --------------------------------------------------------------------------- #
#  Loops
# --------------------------------------------------------------------------- #
def train_one_epoch(cfg, model, panel, opt, scaler, criterion, ema, state,
                    generator, callbacks, device, steps_per_epoch):
    model.train()
    # Accumulate the loss on the GPU. Calling .item() every step would force a
    # host sync, which stalls the CPU until the GPU drains and destroys the
    # overlap between the backward pass and the DDP all-reduce. We only sync on
    # logging steps and once at the end of the epoch.
    loss_accum = torch.zeros((), device=device)
    seen = 0
    t0 = time.time()
    batches = panel.random_batches("train", cfg.batch_size,
                                   steps_per_epoch * cfg.accum_steps, generator)

    opt.zero_grad(set_to_none=True)
    for micro, (x, y, _sigma) in enumerate(batches):
        is_update = (micro + 1) % cfg.accum_steps == 0
        lr = lr_at(cfg, state["global_step"], steps_per_epoch)
        for g in opt.param_groups:
            g["lr"] = lr

        # In the accumulation micro-steps we must NOT all-reduce gradients.
        ctx = (model.no_sync() if (not is_update and hasattr(model, "no_sync"))
               else _null())
        with ctx:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                pred = model(x)
                loss = criterion(pred, y) / cfg.accum_steps
            scaler.scale(loss).backward()

        loss_accum += loss.detach() * cfg.accum_steps
        seen += x.shape[0]

        if not is_update:
            continue

        # unscale before clipping, otherwise you would clip the *scaled* norm
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

        state["global_step"] += 1
        if state["is_main"] and state["global_step"] % cfg.log_every_steps == 0:
            state["loss"] = loss.item() * cfg.accum_steps      # the only sync
            state["lr"] = lr
            state["grad_norm"] = gnorm.item()
            state["loss_scale"] = float(scaler.get_scale()) if cfg.amp else 0.0
            state["samples_per_sec"] = (seen * state["world_size"]
                                        / max(time.time() - t0, 1e-9))
            callbacks.on_step_end(state)

    n_updates = max(steps_per_epoch, 1)
    dt = max(time.time() - t0, 1e-9)
    state["lr"] = lr
    return loss_accum.item() / (n_updates * cfg.accum_steps), seen * state["world_size"] / dt


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@torch.no_grad()
def evaluate(cfg, model, panel, criterion, device, split="val",
             max_batches=None, rank=0, world_size=1, collect_examples=0):
    model.eval()
    losses, preds, targets = [], [], []
    ex_p, ex_y = None, None
    sq_err = torch.zeros(cfg.n_steps_out, device=device)
    n_seen = 0

    for x, y, _sigma in panel.sequential_batches(split, cfg.batch_size,
                                                 max_batches, rank, world_size):
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            pred = model(x)
            loss = criterion(pred, y)
        losses.append(loss.float().item())
        p = point_forecast(cfg, pred).float()
        sq_err += ((p - y.float()) ** 2).sum(0)
        n_seen += x.shape[0]
        # Subsample what we keep: 200 batches x 512 x 20 floats would be fine,
        # but keeping it bounded makes the metric cost independent of val size.
        if len(preds) < 40:
            preds.append(p)
            targets.append(y.float())
        if collect_examples and ex_p is None:
            ex_p = p[:collect_examples].cpu().numpy()
            ex_y = y[:collect_examples].float().cpu().numpy()

    p_all = torch.cat(preds)
    y_all = torch.cat(targets)
    out = metrics(cfg, p_all, y_all)
    out["loss"] = float(np.mean(losses))

    # Average the metrics across ranks so every process agrees on "is this the
    # best epoch" -- otherwise rank 0 could checkpoint and rank 1 early-stop.
    for k in list(out):
        out[k] = all_reduce_mean(out[k], device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sq_err, op=dist.ReduceOp.SUM)
        n_t = torch.tensor([n_seen], device=device, dtype=torch.float64)
        dist.all_reduce(n_t)
        n_seen = int(n_t.item())
    horizon_rmse = torch.sqrt(sq_err / max(n_seen, 1)).cpu().numpy()
    return out, horizon_rmse, (ex_y, ex_p)


# --------------------------------------------------------------------------- #
#  Checkpoints
# --------------------------------------------------------------------------- #
def make_checkpoint_fn(cfg, model, opt, scaler, ema, state, meta):
    def fn(include_optimizer: bool = True) -> dict:
        payload = {
            "config": cfg.to_dict(),
            "model": _unwrap(model).state_dict(),
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "n_features": meta["n_features"],
            "feature_names": meta["feature_names"],
            "torch_version": torch.__version__,
        }
        if ema is not None:
            payload["ema"] = ema.shadow.state_dict()
        if include_optimizer:
            payload["optimizer"] = opt.state_dict()
            payload["scaler"] = scaler.state_dict()
            payload["rng"] = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
            }
        return payload
    return fn


def load_checkpoint(path, model, opt=None, scaler=None, ema=None,
                    map_location="cpu", strict=True) -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    _unwrap(model).load_state_dict(ck["model"], strict=strict)
    if ema is not None and "ema" in ck:
        ema.load_state_dict(ck["ema"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    return ck
