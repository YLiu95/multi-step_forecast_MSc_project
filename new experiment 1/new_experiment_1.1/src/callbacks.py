from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from .config import Config, EXPERIMENT_ROOT, REPO_ROOT
from .hub import GitBackup, HFBackup


class Callback:
    def on_train_begin(self, state: dict) -> None: ...
    def on_step_end(self, state: dict) -> None: ...
    def on_epoch_end(self, state: dict) -> None: ...
    def on_train_end(self, state: dict) -> None: ...


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def _call(self, name: str, state: dict) -> None:
        for callback in self.callbacks:
            getattr(callback, name)(state)

    def on_train_begin(self, state): self._call("on_train_begin", state)
    def on_step_end(self, state): self._call("on_step_end", state)
    def on_epoch_end(self, state): self._call("on_epoch_end", state)
    def on_train_end(self, state): self._call("on_train_end", state)


class TensorBoardLogger(Callback):
    def __init__(self, cfg: Config):
        from torch.utils.tensorboard import SummaryWriter
        self.cfg = cfg
        self.writer = SummaryWriter(str(cfg.paths["tb"]))

    def on_train_begin(self, state):
        self.writer.add_text("guide/01_model", state["model_summary"], 0)
        self.writer.add_text("guide/02_loss",
                             "total = 0.7 x magnitude Huber + 0.3 x direction BCE", 0)
        self.writer.add_text("guide/03_read_first",
                             "Watch validation/loss, validation/direction_accuracy, "
                             "and validation/magnitude_mae_bp together.", 0)
        self.writer.add_text("config/resolved", "```json\n" +
                             json.dumps(self.cfg.to_dict(), indent=2) + "\n```", 0)

    def on_step_end(self, state):
        step = state["global_step"]
        writer = self.writer
        writer.add_scalar("training/total_loss", state["step_loss"], step)
        writer.add_scalar("training/magnitude_huber", state["step_magnitude_loss"], step)
        writer.add_scalar("training/direction_bce", state["step_direction_loss"], step)
        writer.add_scalar("training/learning_rate", state["lr"], step)
        writer.add_scalar("training/gradient_norm", state["gradient_norm"], step)
        writer.add_scalar("performance/samples_per_second", state["samples_per_second"], step)
        writer.add_scalar("performance/gpu_memory_allocated_GB",
                          torch.cuda.memory_allocated() / 1e9, step)
        writer.add_scalar("performance/gpu_memory_reserved_GB",
                          torch.cuda.memory_reserved() / 1e9, step)
        writer.add_scalar("performance/fp16_loss_scale", state["loss_scale"], step)

    def on_epoch_end(self, state):
        step = state["global_step"]
        for name, value in state["validation"].items():
            self.writer.add_scalar(f"validation/{name}", value, step)
        for name, value in state["training"].items():
            self.writer.add_scalar(f"epoch_training/{name}", value, step)
        self.writer.flush()

    def on_train_end(self, state):
        self.writer.close()


class HistoryLogger(Callback):
    def __init__(self, cfg: Config):
        self.path = cfg.paths["history"]

    def on_epoch_end(self, state):
        row = {"epoch": state["epoch"], "global_step": state["global_step"],
               "epoch_seconds": state["epoch_seconds"], "wall_time": time.time()}
        row.update({f"train_{key}": value for key, value in state["training"].items()})
        row.update({f"val_{key}": value for key, value in state["validation"].items()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


class SelfEvolutionLogger(Callback):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = EXPERIMENT_ROOT / "selfevo_process.md"

    def on_epoch_end(self, state):
        validation = state["validation"]
        observation = (
            f"train loss {state['training']['loss']:.5f}; validation loss "
            f"{validation['loss']:.5f}; direction accuracy "
            f"{validation['direction_accuracy']:.3f}; magnitude MAE "
            f"{validation['magnitude_mae_bp']:.1f} bp; peak allocated VRAM "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB on rank 0."
        )
        if state["epoch"] == 1:
            action = "Keep the initial settings until a trend is visible; one epoch is not evidence for retuning."
        elif validation["direction_accuracy"] < 0.49:
            action = "Inspect class balance and BCE learning before increasing model size."
        elif state["training"]["loss"] < 0.8 * validation["loss"]:
            action = "Generalization gap is widening; prefer stronger regularization or earlier stopping."
        else:
            action = "No automatic hyperparameter change; continue collecting comparable validation points."
        with self.path.open("a") as handle:
            handle.write(f"\n## Epoch {state['epoch']}\n\n"
                         f"- Observation: {observation}\n"
                         f"- Action: {action}\n"
                         f"- Lesson: Decisions require validation trends, not training loss alone.\n")


class CheckpointCallback(Callback):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.directory = cfg.paths["ckpt"]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hf = HFBackup(cfg.hf_repo_id)

    def on_epoch_end(self, state):
        epoch = state["epoch"]
        value = state["validation"]["loss"]
        improved = value < state["best_value"]
        if improved:
            state["best_value"] = value
        payload = state["checkpoint_fn"](True)
        path = self.directory / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save(payload, path)
        shutil.copy2(path, self.directory / "latest.pt")

        if improved:
            best = self.directory / "best.pt"
            torch.save(state["checkpoint_fn"](False), best)
            remote = f"{self.cfg.hf_experiment_dir}/best model/best.pt"
            state.setdefault("messages", []).append(self.hf.upload(
                best, remote, f"best model epoch {epoch}"))
        state["is_best"] = improved

        if epoch % self.cfg.hf_push_every_epochs == 0:
            prefix = f"{self.cfg.hf_experiment_dir}/checkpoints"
            remote = f"{prefix}/{path.name}"
            state.setdefault("messages", []).append(self.hf.upload(
                path, remote, f"checkpoint epoch {epoch}"))
            state.setdefault("messages", []).extend(
                self.hf.rotate(prefix, self.cfg.keep_last_n_ckpts))
        files = sorted(self.directory.glob("ckpt_epoch_*.pt"))
        for old in files[:-self.cfg.keep_last_n_ckpts]:
            old.unlink()


class EarlyStopping(Callback):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.best = float("inf")
        self.wait = 0

    def on_train_begin(self, state):
        self.best = state["best_value"]

    def on_epoch_end(self, state):
        value = state["validation"]["loss"]
        if value < self.best - 1e-5:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
        state["patience_left"] = self.cfg.early_stop_patience - self.wait
        state["stop"] = self.wait >= self.cfg.early_stop_patience


class GitBackupCallback(Callback):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backup = GitBackup(REPO_ROOT, cfg.github_repo)
        self.log_directory = EXPERIMENT_ROOT / "logs" / cfg.run_name

    def _mirror(self):
        self.log_directory.mkdir(parents=True, exist_ok=True)
        for source in (self.cfg.paths["history"], self.cfg.paths["run"] / "config.json"):
            if source.exists():
                shutil.copy2(source, self.log_directory / source.name)

    def on_epoch_end(self, state):
        if state["epoch"] % self.cfg.git_push_every_epochs:
            return
        self._mirror()
        message = (f"backup: experiment 1.1 epoch {state['epoch']} "
                   f"val_loss={state['validation']['loss']:.6f}")
        state.setdefault("messages", []).append(
            self.backup.push(message, self.cfg.github_subdir))

    def on_train_end(self, state):
        self._mirror()
        state.setdefault("messages", []).append(self.backup.push(
            "backup: experiment 1.1 training finished", self.cfg.github_subdir))


class ConsoleLogger(Callback):
    def on_epoch_end(self, state):
        validation = state["validation"]
        marker = " BEST" if state.get("is_best") else ""
        print(f"epoch {state['epoch']:>3}/{state['n_epochs']} | "
              f"train {state['training']['loss']:.5f} | val {validation['loss']:.5f} | "
              f"direction {validation['direction_accuracy']:.3f} | "
              f"magnitude MAE {validation['magnitude_mae_bp']:.1f} bp | "
              f"{state['samples_per_second']:,.0f} samples/s | "
              f"{state['epoch_seconds']:.0f}s{marker}", flush=True)
        for message in state.pop("messages", []):
            print(f"  {message}", flush=True)