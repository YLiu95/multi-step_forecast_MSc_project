"""Training callbacks: TensorBoard, checkpointing, early stopping, backups.

Everything that is not "compute a gradient" lives here, behind a small hook
protocol, so `engine.py` stays readable. Only rank 0 ever does I/O.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .hub import GitBackup, HFBackup


class Callback:
    def on_train_begin(self, state: dict) -> None: ...
    def on_step_end(self, state: dict) -> None: ...
    def on_epoch_end(self, state: dict) -> None: ...
    def on_train_end(self, state: dict) -> None: ...


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def _fan(self, hook: str, state: dict) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(state)

    def on_train_begin(self, state): self._fan("on_train_begin", state)
    def on_step_end(self, state): self._fan("on_step_end", state)
    def on_epoch_end(self, state): self._fan("on_epoch_end", state)
    def on_train_end(self, state): self._fan("on_train_end", state)


# --------------------------------------------------------------------------- #
#  TensorBoard
# --------------------------------------------------------------------------- #
class TensorBoardLogger(Callback):
    """Writes scalars, histograms and a forecast figure to `runs/<name>/tb`.

    The `hparams` block at the end is what lets TensorBoard's HPARAMS tab
    compare this run against every previous one side by side.
    """

    def __init__(self, cfg: Config):
        from torch.utils.tensorboard import SummaryWriter
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=str(cfg.paths["tb"]))

    def on_train_begin(self, state):
        self.writer.add_text("config",
                             "```json\n" + json.dumps(self.cfg.to_dict(),
                                                      indent=2, default=list)
                             + "\n```", 0)
        self.writer.add_text("model/summary", state.get("model_summary", ""), 0)

    def on_step_end(self, state):
        step = state["global_step"]
        if step % self.cfg.log_every_steps:
            return
        w = self.writer
        w.add_scalar("train/loss", state["loss"], step)
        w.add_scalar("train/lr", state["lr"], step)
        w.add_scalar("train/grad_norm", state["grad_norm"], step)
        w.add_scalar("perf/samples_per_sec", state["samples_per_sec"], step)
        w.add_scalar("perf/gpu_mem_alloc_GB",
                     torch.cuda.memory_allocated() / 1e9, step)
        w.add_scalar("perf/gpu_mem_reserved_GB",
                     torch.cuda.memory_reserved() / 1e9, step)
        if state.get("loss_scale"):
            w.add_scalar("train/amp_loss_scale", state["loss_scale"], step)

    def on_epoch_end(self, state):
        step = state["global_step"]
        for k, v in state["val_metrics"].items():
            self.writer.add_scalar(f"val/{k}", v, step)
        for k, v in state.get("train_metrics", {}).items():
            self.writer.add_scalar(f"train_eval/{k}", v, step)
        self.writer.add_scalar("epoch", state["epoch"], step)
        self.writer.add_scalar("perf/epoch_seconds", state["epoch_seconds"], step)
        if state.get("fig") is not None:
            self.writer.add_figure("val/forecast_examples", state["fig"], step)
        if state.get("horizon_rmse") is not None:
            for h, v in enumerate(state["horizon_rmse"], start=1):
                self.writer.add_scalar("val_horizon_rmse/step", v, h)
        self.writer.flush()

    def on_train_end(self, state):
        hp = {k: v for k, v in self.cfg.to_dict().items()
              if isinstance(v, (int, float, str, bool))}
        best = {f"hparam/{k}": v for k, v in state.get("best_metrics", {}).items()
                if isinstance(v, (int, float))}
        if best:
            self.writer.add_hparams(hp, best, run_name=".")
        self.writer.close()


# --------------------------------------------------------------------------- #
#  History (for resume + cross-run plots)
# --------------------------------------------------------------------------- #
class HistoryLogger(Callback):
    """Append one JSON line per epoch. Survives a crash; trivially re-readable."""

    def __init__(self, cfg: Config):
        self.path = cfg.paths["history"]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, state):
        row = {"epoch": state["epoch"], "global_step": state["global_step"],
               "train_loss": state["epoch_train_loss"],
               "val_loss": state["val_metrics"]["loss"],
               "lr": state["lr"], "epoch_seconds": state["epoch_seconds"],
               "samples_per_sec": state["samples_per_sec"],
               "wall_time": time.time()}
        row.update({f"val_{k}": v for k, v in state["val_metrics"].items()})
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #
class CheckpointCallback(Callback):
    """Local rotation + Hugging Face upload.

    Two different artefacts, on purpose:

    * ``ckpt_epoch_XXX.pt`` -- model **and** optimiser, scaler, EMA, RNG state.
      This is what `resume training.ipynb` needs. It is ~3x the model size
      because Adam keeps two moments per parameter.
    * ``best.pt`` -- weights only. Small, and all you need for inference.

    Full checkpoints are uploaded every `hf_push_every_epochs` (they are big);
    the best model is uploaded the moment it improves (it is cheap and it is
    the artefact you would be sad to lose).
    """

    def __init__(self, cfg: Config, monitor: str = "loss", mode: str = "min"):
        self.cfg = cfg
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.dir = cfg.paths["ckpt"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hf = HFBackup(cfg.hf_repo_id)
        self.log: list[str] = []

    def _better(self, v: float) -> bool:
        return v < self.best if self.mode == "min" else v > self.best

    def on_train_begin(self, state):
        self.best = state.get("best_value", self.best)

    def on_epoch_end(self, state):
        cfg, ep = self.cfg, state["epoch"]
        value = state["val_metrics"][self.monitor]
        payload = state["checkpoint_fn"](include_optimizer=True)
        payload["best_value"] = self.best

        path = self.dir / f"ckpt_epoch_{ep:03d}.pt"
        torch.save(payload, path)
        latest = self.dir / "latest.pt"
        shutil.copy2(path, latest)
        state.setdefault("messages", []).append(
            f"saved {path.name} ({path.stat().st_size / 1e6:,.0f} MB)")

        if self._better(value):
            self.best = value
            payload["best_value"] = value
            best_path = self.dir / "best.pt"
            torch.save(state["checkpoint_fn"](include_optimizer=False)
                       | {"best_value": value, "epoch": ep}, best_path)
            msg = self.hf.upload(best_path, f"{cfg.hf_best_dir}/best.pt",
                                 f"best {self.monitor}={value:.6f} @ epoch {ep}")
            state["messages"].append(f"NEW BEST {self.monitor}={value:.6f} | {msg}")
            state["is_best"] = True
        else:
            state["is_best"] = False
        state["best_value"] = self.best

        if ep % cfg.hf_push_every_epochs == 0:
            msg = self.hf.upload(path, f"{cfg.hf_ckpt_dir}/{path.name}",
                                 f"checkpoint epoch {ep}")
            state["messages"].append(msg)
            if cfg.paths["history"].exists():
                state["messages"].append(
                    self.hf.upload(cfg.paths["history"],
                                   f"{cfg.hf_ckpt_dir}/{cfg.run_name}_history.jsonl",
                                   f"history epoch {ep}"))

        self._rotate()

    def _rotate(self) -> None:
        files = sorted(self.dir.glob("ckpt_epoch_*.pt"))
        for old in files[:-self.cfg.keep_last_n_ckpts]:
            old.unlink(missing_ok=True)

    def on_train_end(self, state):
        p = self.dir / "latest.pt"
        if p.exists():
            state.setdefault("messages", []).append(
                self.hf.upload(p, f"{self.cfg.hf_ckpt_dir}/latest.pt",
                               "final checkpoint"))


# --------------------------------------------------------------------------- #
#  Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping(Callback):
    """Stop when validation stops improving.

    On noisy financial targets the val curve is jagged, so `patience` is
    generous and `min_delta` rejects improvements that are pure luck.
    """

    def __init__(self, cfg: Config, monitor: str = "loss", mode: str = "min",
                 min_delta: float = 1e-5):
        self.cfg, self.monitor, self.mode, self.min_delta = cfg, monitor, mode, min_delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.wait = 0

    def on_epoch_end(self, state):
        v = state["val_metrics"][self.monitor]
        improved = (v < self.best - self.min_delta if self.mode == "min"
                    else v > self.best + self.min_delta)
        if improved:
            self.best, self.wait = v, 0
        else:
            self.wait += 1
            if self.wait >= self.cfg.early_stop_patience:
                state["stop"] = True
                state.setdefault("messages", []).append(
                    f"early stop: no {self.monitor} improvement in "
                    f"{self.wait} epochs (best {self.best:.6f})")
        state["patience_left"] = self.cfg.early_stop_patience - self.wait


# --------------------------------------------------------------------------- #
#  GitHub backup
# --------------------------------------------------------------------------- #
class GitBackupCallback(Callback):
    """Periodically commit code + the run history so a crash loses nothing."""

    def __init__(self, cfg: Config, repo_dir: str | Path):
        self.cfg = cfg
        self.git = GitBackup(repo_dir, cfg.github_repo)
        self.mirror = Path(repo_dir) / cfg.github_subdir / "logs" / cfg.run_name
        self.mirror.mkdir(parents=True, exist_ok=True)

    def _sync(self) -> None:
        hist = self.cfg.paths["history"]
        if hist.exists():
            shutil.copy2(hist, self.mirror / "history.jsonl")
        cfg_path = self.cfg.paths["run"] / "config.json"
        if cfg_path.exists():
            shutil.copy2(cfg_path, self.mirror / "config.json")

    def on_epoch_end(self, state):
        if state["epoch"] % self.cfg.git_push_every_epochs:
            return
        self._sync()
        msg = self.git.push(
            f"backup: {self.cfg.run_name} epoch {state['epoch']} "
            f"val_loss={state['val_metrics']['loss']:.6f}")
        state.setdefault("messages", []).append(msg)

    def on_train_end(self, state):
        self._sync()
        state.setdefault("messages", []).append(
            self.git.push(f"backup: {self.cfg.run_name} finished"))


# --------------------------------------------------------------------------- #
#  Console
# --------------------------------------------------------------------------- #
class ConsoleLogger(Callback):
    def on_epoch_end(self, state):
        m = state["val_metrics"]
        flag = "  <-- BEST" if state.get("is_best") else ""
        print(f"epoch {state['epoch']:>3}/{state['n_epochs']} | "
              f"train {state['epoch_train_loss']:.5f} | "
              f"val {m['loss']:.5f} | r2 {m['r2_vs_zero']:+.4f} | "
              f"rank_ic {m['rank_ic']:+.4f} | dir {m['dir_acc']:.3f} | "
              f"lr {state['lr']:.2e} | {state['samples_per_sec']:,.0f} samp/s | "
              f"{state['epoch_seconds']:.0f}s{flag}", flush=True)
        for msg in state.pop("messages", []):
            print(f"        {msg}", flush=True)


# --------------------------------------------------------------------------- #
#  Diagnostic figure
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forecast_figure(cfg: Config, y_true: np.ndarray, y_pred: np.ndarray,
                    n: int = 6):
    """Predicted vs actual cumulative return path, in volatility units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(y_true))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6), squeeze=False)
    h = np.arange(1, cfg.n_steps_out + 1)
    for k, ax in enumerate(axes.ravel()[:n]):
        ax.plot(h, np.cumsum(y_true[k]), "o-", ms=3, label="actual", color="k")
        ax.plot(h, np.cumsum(y_pred[k]), "s--", ms=3, label="predicted",
                color="tab:red")
        ax.axhline(0, lw=0.8, color="grey", ls=":")
        ax.set_title(f"val sample {k}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("cumulative return (sigma units)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Multi-step forecast: cumulative path over the horizon")
    fig.tight_layout()
    return fig
