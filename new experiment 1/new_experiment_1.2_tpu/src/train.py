from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import jax
import numpy as np

from .callbacks import (
    CheckpointManager,
    ExperimentDiary,
    HistoryLogger,
    TrainingMonitor,
    hbm_used_gb,
)
from .config import Config, EXPERIMENT_ROOT
from .dataset import GlobalBasketSampler, prefetch_batches
from .engine import (
    build_eval_step,
    build_train_step,
    create_mesh,
    create_train_state,
    estimated_model_flops,
    evaluate,
    read_loss_weights,
    replicate_state,
    shard_batch,
)
from .hub import GitBackup
from .model import CrossTickerPatchTransformer, count_parameters


def mirror_run_logs(cfg: Config) -> None:
    destination = EXPERIMENT_ROOT / "logs" / cfg.run_name
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.jsonl"):
        for source in cfg.paths["run"].glob(pattern):
            shutil.copy2(source, destination / source.name)
    tensorboard = cfg.paths["tensorboard"]
    if tensorboard.exists():
        shutil.copytree(tensorboard, destination / "tensorboard", dirs_exist_ok=True)


def mean_metrics(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


def train(cfg: Config, resume: bool = False, backups: bool = True,
          stop_after_epoch: int | None = None,
          enable_early_stopping: bool = True) -> None:
    jax.config.update("jax_threefry_partitionable", True)
    cfg.validate()
    cfg.make_dirs()
    cfg.save(cfg.paths["run"] / "config.json")
    sampler = GlobalBasketSampler(cfg)
    model = CrossTickerPatchTransformer(cfg, n_tickers=sampler.n_tickers)
    state, schedule = create_train_state(cfg, model, jax.random.key(cfg.seed))
    parameter_count = count_parameters({"params": state.params})
    model_summary = (
        f"Hierarchical Patch Transformer with {parameter_count:,} parameters. "
        f"Each sample contains {cfg.n_tickers_per_sample} tickers x "
        f"{cfg.n_steps_in} daily returns, represented as {cfg.n_patches} patches."
    )
    print(model_summary, flush=True)

    checkpoint_manager = CheckpointManager(cfg, enable_remote=backups)
    start_epoch = 1
    best_value = float("inf")
    best_epoch = 0
    patience_wait = 0
    if resume:
        restored = checkpoint_manager.restore_auto(state)
        if restored:
            state, metadata = restored
            start_epoch = int(metadata["epoch"]) + 1
            best_value = float(metadata["best_value"])
            best_epoch = int(metadata.get("best_epoch", metadata["epoch"]))
            patience_wait = int(metadata.get("patience_wait", 0))
            print(f"Resumed after epoch {start_epoch - 1}", flush=True)
        else:
            print("No checkpoint found; starting from epoch 1", flush=True)

    mesh = create_mesh()
    state = replicate_state(state, mesh)
    train_step = build_train_step(cfg, schedule)
    eval_step = build_eval_step(cfg, model)
    monitor = TrainingMonitor(cfg, model_summary)
    history = HistoryLogger(cfg)
    diary = ExperimentDiary(cfg)
    git_backup = GitBackup(cfg.github_repo)
    peak_flops = 197e12 * len(jax.devices())
    flops_per_step = estimated_model_flops(cfg, cfg.batch_size)
    global_step = int(jax.device_get(state.step))

    try:
        final_epoch = min(cfg.epochs, stop_after_epoch or cfg.epochs)
        for epoch in range(start_epoch, final_epoch + 1):
            epoch_started = time.time()
            weights = read_loss_weights(cfg)
            replicated = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            )
            device_weights = jax.device_put(weights, replicated)
            accumulated = {"loss": 0.0, "magnitude_loss": 0.0,
                           "direction_loss": 0.0}
            interval_started = time.time()
            interval_samples = 0
            batches = sampler.batches(
                "train",
                cfg.batch_size,
                cfg.steps_per_epoch,
                seed=cfg.seed + epoch,
            )
            for epoch_step, host_batch in enumerate(prefetch_batches(batches), start=1):
                device_batch = shard_batch(host_batch, mesh)
                dropout_key = jax.device_put(
                    jax.random.key(cfg.seed + global_step + 1), replicated
                )
                state, device_metrics = train_step(
                    state, device_batch, dropout_key, device_weights
                )
                host_metrics = {
                    key: float(value)
                    for key, value in jax.device_get(device_metrics).items()
                }
                global_step += 1
                interval_samples += cfg.batch_size
                for key in accumulated:
                    accumulated[key] += host_metrics[key]
                if global_step % cfg.log_every_steps == 0:
                    elapsed = max(time.time() - interval_started, 1e-9)
                    seconds_per_step = elapsed / max(
                        interval_samples / cfg.batch_size, 1
                    )
                    performance = {
                        "samples_per_second": interval_samples / elapsed,
                        "steps_per_second": 1.0 / seconds_per_step,
                        "hbm_used_gb": hbm_used_gb(),
                        "mfu_percent": 100.0 * flops_per_step
                        / seconds_per_step / peak_flops,
                    }
                    monitor.log_step(global_step, host_metrics, performance)
                    interval_started = time.time()
                    interval_samples = 0

            training_metrics = mean_metrics(accumulated, cfg.steps_per_epoch)
            validation = evaluate(
                cfg,
                state,
                model,
                sampler,
                mesh,
                weights,
                split="val",
                eval_step=eval_step,
            )
            epoch_seconds = time.time() - epoch_started
            value = validation["overall"]["loss"]
            improved = value < best_value - 1e-5
            if improved:
                best_value = value
                best_epoch = epoch
                patience_wait = 0
            else:
                patience_wait += 1

            monitor.log_epoch(
                epoch, global_step, training_metrics, validation, weights.tolist()
            )
            history.append(
                epoch, global_step, epoch_seconds, training_metrics, validation,
                weights.tolist(),
            )
            diary.record_epoch(
                epoch, training_metrics, validation, weights.tolist(), epoch_seconds,
                best_epoch,
            )
            metadata = {
                "epoch": epoch,
                "global_step": global_step,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "patience_wait": patience_wait,
                "n_tickers": sampler.n_tickers,
                "parameter_count": parameter_count,
                "loss_weights": weights.tolist(),
            }
            messages = []
            if improved:
                messages.append(checkpoint_manager.save_best(state.ema_params, metadata))
            if epoch % cfg.backup_every_epochs == 0:
                messages.extend(checkpoint_manager.save_periodic(state, metadata))
                mirror_run_logs(cfg)
                if backups:
                    messages.append(git_backup.push(
                        cfg.github_subdir,
                        f"backup: experiment 1.2 epoch {epoch} val_loss={value:.6f}",
                    ))
            overall = validation["overall"]
            marker = " BEST" if improved else ""
            print(
                f"epoch {epoch:>3}/{cfg.epochs} | train {training_metrics['loss']:.5f} | "
                f"val {value:.5f} | direction {overall['direction_accuracy']:.3f} | "
                f"magnitude MAE {overall['magnitude_mae_bp']:.1f} bp | "
                f"{epoch_seconds / 60:.1f} min{marker}",
                flush=True,
            )
            for message in messages:
                print(f"  {message}", flush=True)
            if enable_early_stopping and patience_wait >= cfg.early_stop_patience:
                print(
                    f"Early stopping: no validation improvement for "
                    f"{cfg.early_stop_patience} epochs; best was epoch {best_epoch}.",
                    flush=True,
                )
                break
    finally:
        monitor.close()
        mirror_run_logs(cfg)
        if backups:
            print(git_backup.push(
                cfg.github_subdir, "backup: experiment 1.2 training state"
            ), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 1.2 on TPU v5e-8")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-backups", action="store_true")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Run through the configured final epoch while still saving the best model",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop cleanly after this epoch without changing the 60-epoch LR schedule",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    if args.smoke:
        cfg = cfg.override(
            run_name=f"{cfg.run_name}_smoke",
            epochs=2,
            steps_per_epoch=5,
            val_batches=2,
            warmup_epochs=1,
            backup_every_epochs=1,
            early_stop_patience=2,
        )
    train(
        cfg,
        resume=args.resume,
        backups=not args.skip_backups,
        stop_after_epoch=args.stop_after_epoch,
        enable_early_stopping=not args.no_early_stopping,
    )


if __name__ == "__main__":
    main()