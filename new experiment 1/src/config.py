"""Single source of truth for every tunable knob.

Anything you might want to change lives in the `Config` dataclass below.
Both `prepare_data.py` and `train.py` read from it, and the resolved config is
written into every checkpoint so a run can always be reproduced or resumed.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Where things live.  ARTIFACT_ROOT is deliberately OUTSIDE the git repo:
#  multi-GB parquet caches and checkpoints must never enter version control.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "/root/artifacts"))


@dataclass
class Config:
    # ---------------------------------------------------------------- run ids
    run_name: str = "patchtst_us_equities_v1"
    seed: int = 1337

    # ------------------------------------------------------------- universe #
    # The full NASDAQ-traded symbol directory is ~13k symbols. Most are ETFs,
    # warrants, units and preferred shares that carry no useful price dynamics,
    # so we filter to common stock and then keep the most liquid `max_tickers`.
    start_date: str = "1990-01-01"
    end_date: str | None = None            # None -> today
    max_tickers: int = 3000                # kept AFTER the liquidity ranking
    min_history_days: int = 750            # ~3 years; shorter listings are dropped
    min_median_dollar_volume: float = 1e6  # illiquid names are mostly noise
    exclude_etfs: bool = True
    context_tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "HYG")

    # ------------------------------------------------------------- windowing #
    n_steps_in: int = 256                  # look-back, trading days (~1 year)
    n_steps_out: int = 20                  # forecast horizon (~1 month)
    vol_window: int = 60                   # trailing window for the vol scaler
    clip_sigma: float = 8.0                # winsorise standardised values

    # ------------------------------------------------- chronological splits #
    # Split by DATE across the whole panel, never by sample. A purge gap of
    # (n_steps_in + n_steps_out) bars sits between the splits so no validation
    # window can share a single bar with a training window.
    train_end: str = "2018-12-31"
    val_end: str = "2022-12-31"            # test = val_end -> today

    # ----------------------------------------------------------- model size #
    # PatchTST-style encoder. Sized so a single T4 (16GB) is comfortably busy
    # under fp16 AMP while leaving headroom for a large batch.
    patch_len: int = 8
    patch_stride: int = 8
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    depth: int = 8
    dropout: float = 0.2
    head_dropout: float = 0.2

    # ------------------------------------------------- anti-memorisation -- #
    # The market/calendar channels are identical for every ticker on a given
    # day, so a look-back window of them fingerprints the DATE. With only
    # ~7,300 distinct training dates a large model memorises the date -> future
    # mapping outright. These three knobs break that shortcut.
    disable_features: tuple[str, ...] = ("dow", "month_sin", "month_cos")
    shared_group_dropout: float = 0.5   # blank ALL shared channels, per sample
    input_noise: float = 0.05           # gaussian jitter on the inputs

    # ---------------------------------------------------------------- loss  #
    loss: str = "huber"                    # 'huber' | 'mse' | 'quantile'
    huber_delta: float = 1.0
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    horizon_decay: float = 0.0             # >0 down-weights far horizons

    # ------------------------------------------------------------ optimiser #
    # Measured on one T4: 38M params, bs=1024, fp16 + torch.compile ->
    # ~1.8k samples/s/GPU, ~9.7 GB peak. Adding the ~0.6 GB resident panel and
    # the EMA copy lands around 10.5 of 15.4 GB, i.e. VRAM well used with
    # enough headroom that allocator fragmentation cannot OOM the run.
    epochs: int = 60
    steps_per_epoch: int = 500             # per rank; an "epoch" is a fixed budget
    val_batches: int = 60                  # per rank
    batch_size: int = 1024                 # per GPU
    lr: float = 2e-4                       # lowered after the baseline overfit
    min_lr_frac: float = 0.02
    weight_decay: float = 0.15             # raised for the same reason
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    accum_steps: int = 1
    amp: bool = True                       # T4 = sm_75 -> fp16 only, no bf16
    compile_model: bool = True             # measured +23% throughput
    ema_decay: float = 0.999               # 0 disables the EMA shadow weights

    # ------------------------------------------------------------ callbacks #
    early_stop_patience: int = 12          # epochs without val improvement
    ckpt_every_epochs: int = 1             # local checkpoint cadence
    hf_push_every_epochs: int = 5          # full-checkpoint upload cadence
    git_push_every_epochs: int = 5         # code + history backup cadence
    keep_last_n_ckpts: int = 3             # local disk hygiene
    log_every_steps: int = 25

    # --------------------------------------------------------------- remote #
    hf_repo_id: str = "YL95/new_experiment_1"
    hf_ckpt_dir: str = "checkpoints"
    hf_best_dir: str = "best model"
    github_repo: str = "YLiu95/multi-step_forecast_MSc_project"
    github_subdir: str = "new experiment 1"
    # Working notebooks live outside the repo; every periodic git backup copies
    # them in so the committed copy can never go stale.
    notebook_dir: str = "/root"

    # ---------------------------------------------------------------- paths #
    artifact_root: str = str(ARTIFACT_ROOT)

    # ------------------------------------------------------------- derived  #
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    # --------------------------------------------------------------------- #
    @property
    def paths(self) -> dict[str, Path]:
        root = Path(self.artifact_root)
        return {
            "root": root,
            "cache": root / "cache",
            "panel": root / "panel",
            "runs": root / "runs",
            "run": root / "runs" / self.run_name,
            "tb": root / "runs" / self.run_name / "tb",
            "ckpt": root / "checkpoints" / self.run_name,
            "history": root / "runs" / self.run_name / "history.jsonl",
        }

    def make_dirs(self) -> None:
        for key, p in self.paths.items():
            if key != "history":
                p.mkdir(parents=True, exist_ok=True)

    @property
    def n_patches(self) -> int:
        return (self.n_steps_in - self.patch_len) // self.patch_stride + 1

    @property
    def n_outputs_per_step(self) -> int:
        return len(self.quantiles) if self.loss == "quantile" else 1

    # --------------------------------------------------------- (de)serialise #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=list))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        valid = {f.name for f in fields(cls)}
        tuple_fields = {
            f.name for f in fields(cls) if "tuple" in str(f.type).lower()
        }
        clean = {k: (tuple(v) if k in tuple_fields and isinstance(v, list) else v)
                 for k, v in d.items() if k in valid}
        return cls(**clean)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def override(self, **kwargs: Any) -> "Config":
        d = self.to_dict()
        unknown = set(kwargs) - set(d)
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        d.update(kwargs)
        return Config.from_dict(d)


def pretty(cfg: Config) -> str:
    lines = ["Config", "=" * 62]
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "feature_names" and len(v) > 6:
            v = f"({len(v)} features)"
        lines.append(f"  {f.name:26s} {v}")
    lines += ["-" * 62,
              f"  n_patches (derived)        {cfg.n_patches}",
              f"  outputs/step (derived)     {cfg.n_outputs_per_step}"]
    return "\n".join(lines)
