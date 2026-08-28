from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get("ARTIFACT_ROOT", "/root/artifacts/new_experiment_1.2_tpu")
)

MARKETS = ("AU", "CA", "CH", "CN", "DE", "FR", "GB", "HK", "IN", "JP", "KR", "NL", "US")
MAGNIFICENT_SEVEN = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")


@dataclass(frozen=True)
class Config:
    run_name: str = "patchtst_global_tpu_v1_2"
    seed: int = 1337

    dataset_repo: str = "YL95/new_experiment_1-data"
    price_column: str = "adj_close_clean"
    markets: tuple[str, ...] = MARKETS
    n_steps_in: int = 256
    patch_len: int = 8
    patch_stride: int = 8
    horizon: int = 7
    n_tickers_per_sample: int = 64
    return_clip: float = 8.0
    train_end: str = "2018-12-31"
    val_end: str = "2022-12-31"
    embargo_sessions: int = 7

    d_model: int = 1024
    n_heads: int = 16
    d_ff: int = 4096
    temporal_depth: int = 12
    cross_ticker_depth: int = 8
    dropout: float = 0.15
    remat: bool = True

    magnitude_loss_weight: float = 0.7
    direction_loss_weight: float = 0.3
    magnitude_huber_delta_pct: float = 1.0

    epochs: int = 60
    steps_per_epoch: int = 500
    val_batches: int = 80
    batch_size: int = 320
    learning_rate: float = 2e-4
    min_lr_fraction: float = 0.02
    weight_decay: float = 0.1
    warmup_epochs: int = 2
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    early_stop_patience: int = 12
    log_every_steps: int = 20
    backup_every_epochs: int = 5
    keep_last_n_checkpoints: int = 2

    hf_repo_id: str = "YL95/new_experiment_1.2_tpu"
    github_repo: str = "YLiu95/multi-step_forecast_MSc_project"
    github_subdir: str = "new experiment 1/new_experiment_1.2_tpu"
    artifact_root: str = str(ARTIFACT_ROOT)
    loss_weights_path: str = str(EXPERIMENT_ROOT / "loss_weights.json")

    @property
    def n_patches(self) -> int:
        return (self.n_steps_in - self.patch_len) // self.patch_stride + 1

    @property
    def paths(self) -> dict[str, Path]:
        root = Path(self.artifact_root)
        run = root / "runs" / self.run_name
        return {
            "root": root,
            "cache": root / "cache",
            "panel": root / "panel",
            "run": run,
            "tensorboard": run / "tensorboard",
            "history": run / "history.jsonl",
            "checkpoints": root / "checkpoints" / self.run_name,
            "loss_weights": Path(self.loss_weights_path),
        }

    def validate(self) -> None:
        if self.patch_stride != self.patch_len:
            raise ValueError("This experiment requires non-overlapping patches")
        if self.n_steps_in % self.patch_len:
            raise ValueError("n_steps_in must be divisible by patch_len")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.batch_size % 8:
            raise ValueError("batch_size must divide evenly across 8 TPU cores")
        if self.n_tickers_per_sample < 2:
            raise ValueError("Each basket needs one target and at least one context ticker")

    def make_dirs(self) -> None:
        for name, path in self.paths.items():
            if name not in {"history", "loss_weights"}:
                path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["markets"] = list(self.markets)
        return values

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        valid = {field.name for field in fields(cls)}
        if "markets" in raw:
            raw["markets"] = tuple(raw["markets"])
        return cls(**{key: value for key, value in raw.items() if key in valid})

    def override(self, **changes: Any) -> "Config":
        return replace(self, **changes)