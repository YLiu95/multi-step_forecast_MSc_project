from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "/root/artifacts/new_experiment_1.1"))


@dataclass
class Config:
    run_name: str = "patchtst_cross_ticker_full_v1_1"
    seed: int = 1337

    # Data universe. Public exchange directories describe currently listed
    # securities; a true point-in-time, delisting-inclusive universe requires
    # a licensed source and is deliberately not claimed here.
    start_date: str = "1990-01-01"
    end_date: str | None = None
    min_history_days: int = 300
    exclude_etfs: bool = True
    download_chunk_size: int = 200

    # Reproducible target composition: 7 + 10 + 50 + 100 = 167 unique names.
    n_nasdaq100_targets: int = 10
    n_sp500_targets: int = 50
    n_outside_targets: int = 100

    n_steps_in: int = 256
    patch_len: int = 8
    patch_stride: int = 8
    horizon: int = 7
    n_tickers_per_sample: int = 64
    n_target_tickers_per_sample: int = 1
    return_clip: float = 8.0

    train_end: str = "2018-12-31"
    val_end: str = "2022-12-31"

    # Temporal attention runs within each ticker; cross-ticker attention then
    # mixes 64 compact ticker summaries. This avoids attention over 64*32 tokens.
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    temporal_depth: int = 4
    cross_ticker_depth: int = 4
    dropout: float = 0.15

    magnitude_loss_weight: float = 0.7
    direction_loss_weight: float = 0.3
    magnitude_huber_delta_pct: float = 1.0

    epochs: int = 60
    steps_per_epoch: int = 500
    val_batches: int = 80
    batch_size: int = 80
    lr: float = 2e-4
    min_lr_frac: float = 0.02
    weight_decay: float = 0.1
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    compile_model: bool = True
    ema_decay: float = 0.999

    early_stop_patience: int = 12
    log_every_steps: int = 20
    hf_push_every_epochs: int = 5
    git_push_every_epochs: int = 5
    keep_last_n_ckpts: int = 2

    hf_repo_id: str = "YL95/new_experiment_1"
    hf_experiment_dir: str = "new_experiment_1.1"
    github_repo: str = "YLiu95/multi-step_forecast_MSc_project"
    github_subdir: str = "new experiment 1/new_experiment_1.1"
    artifact_root: str = str(ARTIFACT_ROOT)

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
            "tb": run / "tb",
            "history": run / "history.jsonl",
            "ckpt": root / "checkpoints" / self.run_name,
        }

    def make_dirs(self) -> None:
        for name, path in self.paths.items():
            if name != "history":
                path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in valid})

    def override(self, **kwargs: Any) -> "Config":
        values = self.to_dict()
        unknown = set(kwargs) - set(values)
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        values.update(kwargs)
        return Config(**values)


def pretty(cfg: Config) -> str:
    rows = ["Experiment 1.1 configuration", "=" * 64]
    rows.extend(f"  {field.name:30s} {getattr(cfg, field.name)}" for field in fields(cfg))
    rows.append(f"  {'n_patches':30s} {cfg.n_patches}")
    return "\n".join(rows)