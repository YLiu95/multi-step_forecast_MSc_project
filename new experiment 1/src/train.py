"""Training entrypoint. Launch with torchrun so both T4s are used:

    torchrun --nproc_per_node=2 -m src.train --run-name my_run

Any `Config` field can be overridden from the command line, e.g.
`--set batch_size=768 --set depth=16`.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .callbacks import (CallbackList, CheckpointCallback, ConsoleLogger,
                        EarlyStopping, GitBackupCallback, HistoryLogger,
                        TensorBoardLogger, forecast_figure)
from .config import Config, REPO_ROOT, pretty
from .dataset import GPUPanel
from .engine import (ModelEMA, all_reduce_mean, cleanup_distributed, evaluate,
                     load_checkpoint, make_checkpoint_fn, setup_distributed,
                     train_one_epoch, _unwrap)
from .losses import ForecastLoss
from .model import build_model
from .prepare_data import load_panel


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None,
                    help="path to a saved config.json")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="'auto', or a path to a .pt checkpoint")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE", help="override any Config field")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--no-hf", action="store_true")
    return ap.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    overrides = {}
    for item in args.set:
        k, _, v = item.partition("=")
        try:
            overrides[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k.strip()] = v
    if args.run_name:
        overrides["run_name"] = args.run_name
    return cfg.override(**overrides) if overrides else cfg


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    rank, world, local = setup_distributed()
    main_proc = rank == 0
    device = torch.device(f"cuda:{local}")
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True

    if main_proc:
        cfg.make_dirs()
        cfg.save(cfg.paths["run"] / "config.json")
        print(pretty(cfg), flush=True)
        print(f"\nworld_size={world}  device={torch.cuda.get_device_name(local)}",
              flush=True)

    # ---------------------------------------------------------------- data --
    arrays, anchors, meta = load_panel(cfg, mmap=True)
    panel = GPUPanel(cfg, arrays, anchors, device)
    del arrays
    n_features = panel.n_features
    if main_proc:
        print(f"panel in VRAM   : {panel.vram_gb():.2f} GB  "
              f"({panel.feat.shape[0]:,} tickers x {panel.feat.shape[1]:,} days "
              f"x {n_features} features)")
        print(f"anchors         : train {panel.size('train'):,} | "
              f"val {panel.size('val'):,} | test {panel.size('test'):,}", flush=True)

    # --------------------------------------------------------------- model --
    model = build_model(cfg, n_features).to(device)
    n_params = model.n_params()
    summary = (f"PatchForecaster | {n_params / 1e6:.1f}M params | "
               f"d_model={cfg.d_model} depth={cfg.depth} heads={cfg.n_heads} "
               f"d_ff={cfg.d_ff} | {cfg.n_patches} patches of {cfg.patch_len} days "
               f"| in ({cfg.n_steps_in},{n_features}) -> out ({cfg.n_steps_out},"
               f"{cfg.n_outputs_per_step})")
    if main_proc:
        print(summary, flush=True)

    if cfg.compile_model:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"[compile] disabled: {type(exc).__name__}: {exc}")

    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local,
                    gradient_as_bucket_view=True)

    opt = torch.optim.AdamW(_unwrap(model).param_groups(cfg.weight_decay),
                            lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    criterion = ForecastLoss(cfg).to_device(device)

    state = {"epoch": 0, "global_step": 0, "world_size": world,
             "is_main": main_proc, "n_epochs": cfg.epochs,
             "model_summary": summary, "best_value": float("inf"),
             "loss": 0.0, "lr": cfg.lr, "grad_norm": 0.0,
             "samples_per_sec": 0.0, "loss_scale": 0.0}

    # -------------------------------------------------------------- resume --
    start_epoch = 0
    ckpt_path = _resolve_resume(cfg, args.resume)
    if ckpt_path:
        ck = load_checkpoint(ckpt_path, model, opt, scaler, ema,
                             map_location=device)
        start_epoch = int(ck.get("epoch", 0))
        state["global_step"] = int(ck.get("global_step", 0))
        state["best_value"] = float(ck.get("best_value", float("inf")))
        state["epoch"] = start_epoch
        if main_proc:
            print(f"resumed from {ckpt_path} at epoch {start_epoch} "
                  f"(step {state['global_step']}, best {state['best_value']:.6f})",
                  flush=True)

    # ----------------------------------------------------------- callbacks --
    cbs: list = []
    if main_proc:
        cbs = [TensorBoardLogger(cfg), HistoryLogger(cfg)]
        ckpt_cb = CheckpointCallback(cfg)
        if args.no_hf:
            ckpt_cb.hf.enabled = False
        cbs.append(ckpt_cb)
        if not args.no_git:
            cbs.append(GitBackupCallback(cfg, REPO_ROOT.parent))
        cbs += [EarlyStopping(cfg), ConsoleLogger()]
    callbacks = CallbackList(cbs)
    state["checkpoint_fn"] = make_checkpoint_fn(
        cfg, model, opt, scaler, ema, state,
        {"n_features": n_features, "feature_names": meta["feature_names"]})
    callbacks.on_train_begin(state)

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed * 1000 + rank)

    # ------------------------------------------------------------- the loop --
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        state["epoch"] = epoch
        gen.manual_seed(cfg.seed * 1000 + rank + epoch * 7919)
        t0 = time.time()

        train_loss, sps = train_one_epoch(
            cfg, model, panel, opt, scaler, criterion, ema, state, gen,
            callbacks, device, cfg.steps_per_epoch)

        eval_model = ema.shadow if ema is not None else model
        val_metrics, horizon_rmse, (ex_y, ex_p) = evaluate(
            cfg, eval_model, panel, criterion, device, "val",
            cfg.val_batches, rank, world, collect_examples=6)

        state["epoch_train_loss"] = all_reduce_mean(train_loss, device)
        state["val_metrics"] = val_metrics
        state["horizon_rmse"] = horizon_rmse
        state["epoch_seconds"] = time.time() - t0
        state["samples_per_sec"] = sps
        state["fig"] = (forecast_figure(cfg, ex_y, ex_p)
                        if (main_proc and ex_y is not None) else None)

        if main_proc:
            callbacks.on_epoch_end(state)
            if state["fig"] is not None:
                import matplotlib.pyplot as plt
                plt.close(state["fig"])

        if _should_stop(state, device, world):
            break

    state["best_metrics"] = state.get("val_metrics", {})
    if main_proc:
        callbacks.on_train_end(state)
        print("\ntraining finished. best value:", state["best_value"], flush=True)
    cleanup_distributed()


def _resolve_resume(cfg: Config, spec: str | None) -> Path | None:
    if not spec:
        return None
    if spec == "auto":
        latest = cfg.paths["ckpt"] / "latest.pt"
        return latest if latest.exists() else None
    p = Path(spec)
    return p if p.exists() else None


def _should_stop(state: dict, device, world: int) -> bool:
    """Broadcast rank 0's early-stop decision so every rank exits together.

    If one rank breaks out of the loop and the others do not, the next
    all-reduce hangs forever -- a classic and very confusing DDP deadlock.
    """
    flag = torch.tensor([1.0 if state.get("stop") else 0.0], device=device)
    if world > 1:
        dist.broadcast(flag, src=0)
    return bool(flag.item())


if __name__ == "__main__":
    main()
