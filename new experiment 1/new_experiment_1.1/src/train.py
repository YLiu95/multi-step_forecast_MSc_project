from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .callbacks import (CallbackList, CheckpointCallback, ConsoleLogger,
                        EarlyStopping, GitBackupCallback, HistoryLogger,
                        SelfEvolutionLogger, TensorBoardLogger)
from .config import Config, pretty
from .dataset import GPUBasketPanel
from .engine import (ModelEMA, checkpoint_payload, cleanup_distributed, evaluate,
                     load_checkpoint, setup_distributed, train_epoch, unwrap)
from .losses import DualTaskLoss
from .model import build_model
from .prepare_data import load_panel


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="Use 'auto' or a checkpoint path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    return parser.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value
    if args.run_name:
        overrides["run_name"] = args.run_name
    return cfg.override(**overrides) if overrides else cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    rank, world_size, local_rank = setup_distributed()
    main_process = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True

    if main_process:
        cfg.make_dirs()
        cfg.save(cfg.paths["run"] / "config.json")
        print(pretty(cfg), flush=True)
    arrays, anchors, metadata = load_panel(cfg)
    panel = GPUBasketPanel(cfg, arrays, anchors, device)
    del arrays

    model = build_model(cfg, panel.n_tickers).to(device)
    model_summary = (f"Hierarchical Patch Transformer: {model.n_params() / 1e6:.1f}M parameters; "
                     f"{cfg.n_tickers_per_sample} tickers x {cfg.n_steps_in} days; "
                     f"{cfg.n_patches} patches; one target; two prediction heads.")
    if main_process:
        print(model_summary)
        print(f"resident panel: {panel.vram_gb():.2f} GB per GPU; universe: "
              f"{panel.n_tickers:,}; targets: {len(panel.target_indices)}", flush=True)
    if cfg.compile_model:
        model = torch.compile(model)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay else None
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(unwrap(model).param_groups(cfg.weight_decay),
                                  lr=cfg.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    criterion = DualTaskLoss(cfg)

    state = {"epoch": 0, "global_step": 0, "best_value": float("inf"),
             "is_main": main_process, "world_size": world_size,
             "n_epochs": cfg.epochs, "model_summary": model_summary,
             "samples_per_second": 0.0, "messages": []}
    start_epoch = 0
    resume = cfg.paths["ckpt"] / "latest.pt" if args.resume == "auto" else (
        Path(args.resume) if args.resume else None)
    if resume and resume.exists():
        payload = load_checkpoint(resume, model, optimizer, scaler, ema, device)
        start_epoch = payload["epoch"]
        state.update(epoch=start_epoch, global_step=payload["global_step"],
                     best_value=payload.get("best_value", float("inf")))

    callbacks = []
    if main_process:
        callbacks = [TensorBoardLogger(cfg), HistoryLogger(cfg), SelfEvolutionLogger(cfg)]
        checkpoint = CheckpointCallback(cfg)
        if args.no_hf:
            checkpoint.hf.api = None
        callbacks.append(checkpoint)
        if not args.no_git:
            callbacks.append(GitBackupCallback(cfg))
        callbacks.extend((EarlyStopping(cfg), ConsoleLogger()))
    callback_list = CallbackList(callbacks)
    state["checkpoint_fn"] = lambda include_optimizer: checkpoint_payload(
        cfg, model, optimizer, scaler, ema, state, panel.n_tickers, include_optimizer)
    callback_list.on_train_begin(state)

    generator = torch.Generator(device=device)
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        state["epoch"] = epoch
        generator.manual_seed(cfg.seed * 1000 + rank + epoch * 7919)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        state["training"] = train_epoch(cfg, model, panel, optimizer, scaler,
                                        criterion, ema, state, generator,
                                        callback_list, device)
        validation_model = ema.shadow if ema else model
        state["validation"] = evaluate(cfg, validation_model, panel, criterion,
                                       device, rank, world_size)
        state["epoch_seconds"] = time.time() - started
        if main_process:
            callback_list.on_epoch_end(state)
        stop = torch.tensor([bool(state.get("stop"))], device=device)
        if world_size > 1:
            distributed.broadcast(stop, src=0)
        if stop.item():
            break
    if main_process:
        callback_list.on_train_end(state)
    cleanup_distributed()


if __name__ == "__main__":
    main()