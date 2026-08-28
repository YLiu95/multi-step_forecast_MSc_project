from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import jax
from tensorboardX import SummaryWriter

from .config import Config, EXPERIMENT_ROOT
from .engine import (
    ExperimentTrainState,
    restore_params,
    restore_state,
    save_params,
    save_state,
)
from .hub import GitBackup, HFBackup


class TrainingMonitor:
    def __init__(self, cfg: Config, model_summary: str):
        self.cfg = cfg
        self.writer = SummaryWriter(str(cfg.paths["tensorboard"]))
        self.writer.add_text("00_START_HERE/model", model_summary, 0)
        self.writer.add_text(
            "00_START_HERE/loss",
            "Total loss = magnitude weight x Huber loss + direction weight x binary "
            "cross-entropy. Lower validation loss is better; read both task metrics too.",
            0,
        )
        self.writer.add_text(
            "00_START_HERE/mag7_warning",
            "Mag7 contains only seven tickers. Its curve is useful for monitoring but is "
            "noisier than the full validation curve and is not separate proof of performance.",
            0,
        )
        self.writer.add_text(
            "00_START_HERE/data_warning",
            "The source dataset contains survivors only. Results can overestimate returns "
            "and underestimate failures; this model is not a trading recommendation.",
            0,
        )
        self.writer.add_text(
            "config/resolved", "```json\n" + json.dumps(cfg.to_dict(), indent=2) + "\n```", 0
        )

    def log_step(self, step: int, metrics: dict[str, float], performance: dict[str, float]) -> None:
        names = {
            "loss": "training/total_loss",
            "magnitude_loss": "training/magnitude_huber",
            "direction_loss": "training/direction_bce",
            "learning_rate": "training/learning_rate",
            "gradient_norm": "training/gradient_norm",
            "magnitude_head_gradient_norm": "gradnorm/magnitude_head",
            "direction_head_gradient_norm": "gradnorm/direction_head",
        }
        for key, tag in names.items():
            self.writer.add_scalar(tag, metrics[key], step)
        for key, value in performance.items():
            self.writer.add_scalar(f"performance/{key}", value, step)

    def log_epoch(self, epoch: int, step: int, training: dict[str, float],
                  validation: dict[str, dict[str, float]], weights: list[float]) -> None:
        for name, value in training.items():
            self.writer.add_scalar(f"epoch_training/{name}", value, step)
        for group, metrics in validation.items():
            if group == "overall":
                prefix = "validation"
            elif group.startswith("country/"):
                prefix = f"validation_country/{group.split('/', 1)[1]}"
            else:
                prefix = f"validation_cohort/{group}"
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
        self.writer.add_scalar("loss_weights/magnitude", weights[0], step)
        self.writer.add_scalar("loss_weights/direction", weights[1], step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class ExperimentDiary:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = EXPERIMENT_ROOT / "selfevo_process.md"

    def record_epoch(self, epoch: int, training: dict[str, float],
                     validation: dict[str, dict[str, float]], weights: list[float],
                     epoch_seconds: float, best_epoch: int) -> None:
        overall = validation["overall"]
        if epoch == 1:
            action = "Keep the initial settings until at least three comparable validation points exist."
        elif epoch == best_epoch:
            action = "Keep the current settings because the declared validation objective improved."
        elif training["loss"] < 0.8 * overall["loss"]:
            action = "Watch for overfitting; do not add capacity, and let early stopping protect the run."
        else:
            action = "Continue unchanged; no single epoch justifies a hyperparameter intervention."
        text = (
            f"\n## Epoch {epoch}\n\n"
            f"- Observation: train loss `{training['loss']:.5f}`; validation loss "
            f"`{overall['loss']:.5f}`; magnitude MAE `{overall['magnitude_mae_bp']:.1f}` bp; "
            f"direction accuracy `{overall['direction_accuracy']:.3f}`; epoch "
            f"`{epoch_seconds / 60:.1f}` minutes.\n"
            f"- Loss weights: magnitude `{weights[0]:.3f}`, direction `{weights[1]:.3f}`.\n"
            f"- Action: {action}\n"
            "- Lesson: compare validation trends and naive baselines, not training loss alone.\n"
        )
        with self.path.open("a") as handle:
            handle.write(text)


class HistoryLogger:
    def __init__(self, cfg: Config):
        self.path = cfg.paths["history"]

    def append(self, epoch: int, step: int, epoch_seconds: float,
               training: dict[str, float], validation: dict[str, dict[str, float]],
               weights: list[float]) -> None:
        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": step,
            "epoch_seconds": epoch_seconds,
            "wall_time": time.time(),
            "loss_weights": weights,
            "training": training,
            "validation": validation,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


class CheckpointManager:
    def __init__(self, cfg: Config, enable_remote: bool = True):
        self.cfg = cfg
        self.root = cfg.paths["checkpoints"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.hf = HFBackup(cfg.hf_repo_id) if enable_remote else None

    def save_periodic(self, state: ExperimentTrainState, metadata: dict) -> list[str]:
        epoch = int(metadata["epoch"])
        directory = self.root / f"epoch_{epoch:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        save_state(state, directory / "state.msgpack")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.root / "latest.json").write_text(json.dumps({"directory": directory.name}) + "\n")
        messages = [f"Saved local checkpoint {directory.name}"]
        if self.hf is not None:
            messages.append(self.hf.upload_folder(
                directory, f"checkpoints/{directory.name}", f"Checkpoint epoch {epoch}"
            ))
            messages.extend(self.hf.rotate_checkpoints(self.cfg.keep_last_n_checkpoints))
        directories = sorted(self.root.glob("epoch_*"))
        for old in directories[:-self.cfg.keep_last_n_checkpoints]:
            shutil.rmtree(old)
        return messages

    def save_best(self, params: Any, metadata: dict) -> str:
        directory = self.root / "best model"
        directory.mkdir(parents=True, exist_ok=True)
        save_params(params, directory / "params.msgpack")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if self.hf is None:
            return "Saved local best model"
        return self.hf.upload_folder(directory, "best model", "Update best model")

    def restore_auto(self, template: ExperimentTrainState) \
            -> tuple[ExperimentTrainState, dict] | None:
        pointer = self.root / "latest.json"
        directory = None
        if pointer.exists():
            directory = self.root / json.loads(pointer.read_text())["directory"]
        if directory is None or not (directory / "state.msgpack").exists():
            directory = self.hf.download_latest(self.root) if self.hf is not None else None
        if directory is None:
            return None
        state = restore_state(template, directory / "state.msgpack")
        metadata = json.loads((directory / "metadata.json").read_text())
        return state, metadata

    def restore_best(self, template_params: Any) -> tuple[Any, dict] | None:
        directory = self.root / "best model"
        if not (directory / "params.msgpack").exists():
            downloaded = self.hf.download_best(directory) if self.hf is not None else None
            if downloaded is None:
                return None
        params = restore_params(template_params, directory / "params.msgpack")
        metadata = json.loads((directory / "metadata.json").read_text())
        return params, metadata


def hbm_used_gb() -> float:
    values = [
        (device.memory_stats() or {}).get("bytes_in_use", 0)
        for device in jax.devices()
    ]
    return max(values, default=0) / 1e9