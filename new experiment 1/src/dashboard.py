"""Offline TensorBoard replacement: read the event files, plot with matplotlib.

VS Code tunnel port-forwarding is not always available, and TensorBoard is only
a *viewer* -- the data is already on disk in `runs/<name>/tb/events.out.*`. This
module reads those files directly, so you get the same curves inside a notebook
with **no server, no port and no forwarding**.

    from src.dashboard import read_runs, plot_runs, summary_table
    runs = read_runs("/root/artifacts/runs")
    summary_table(runs)
    plot_runs(runs)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# The panels worth looking at, in order of usefulness, with a reference line.
DEFAULT_PANELS = [
    ("val/loss",               "validation loss",                            None),
    ("train/loss",             "training loss (per step)",                   None),
    ("val/r2_vs_zero",         "val $R^2$ vs zero   (+0.005..0.02 = good)",  0.0),
    ("val/rank_ic",            "val rank IC   (0.02..0.05 = real signal)",   0.0),
    ("val/dir_acc",            "val directional accuracy",                   0.5),
    ("val/std_ratio",          "val std_ratio   (-> 0 = model gave up)",     None),
    ("train/lr",               "learning rate",                              None),
    ("train/grad_norm",        "gradient norm",                              None),
    ("perf/samples_per_sec",   "throughput (samples/s, both GPUs)",          None),
]


def read_runs(runs_dir: str | Path = "/root/artifacts/runs",
              size_guidance: int = 100_000) -> dict[str, dict[str, pd.DataFrame]]:
    """Load every scalar from every run under `runs_dir`.

    Returns {run_name: {tag: DataFrame[step, value]}}.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for tb_dir in sorted(Path(runs_dir).glob("*/tb")):
        name = tb_dir.parent.name
        ea = EventAccumulator(str(tb_dir),
                              size_guidance={"scalars": size_guidance})
        try:
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        out[name] = {
            t: pd.DataFrame([(e.step, e.value) for e in ea.Scalars(t)],
                            columns=["step", "value"])
            for t in tags
        }
    return out


def summary_table(runs: dict[str, dict[str, pd.DataFrame]],
                  monitor: str = "val/loss") -> pd.DataFrame:
    """Best-epoch summary, one row per run, sorted by the monitored metric."""
    rows = []
    for name, tags in runs.items():
        if monitor not in tags or tags[monitor].empty:
            continue
        df = tags[monitor]
        i = int(df["value"].idxmin())
        best_step = int(df.loc[i, "step"])
        row = {"run": name, "points": len(df),
               "best_step": best_step, "best_val_loss": df.loc[i, "value"]}
        # value of every other val metric at the same step
        for tag, short in [("val/r2_vs_zero", "r2"), ("val/rank_ic", "rank_ic"),
                           ("val/dir_acc", "dir_acc"), ("val/std_ratio", "std_ratio")]:
            if tag in tags and not tags[tag].empty:
                d = tags[tag]
                row[short] = float(d.iloc[(d["step"] - best_step).abs().argmin()]["value"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)


def plot_runs(runs: dict[str, dict[str, pd.DataFrame]],
              panels=None, smooth: int = 1, figsize=(16, 10)):
    """Overlay every run on a grid of panels. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    panels = panels or DEFAULT_PANELS
    available = [p for p in panels if any(p[0] in t for t in runs.values())]
    if not available:
        print("No scalars found - has the first epoch finished?")
        return None

    ncol = 3
    nrow = (len(available) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for ax, (tag, title, ref) in zip(axes.ravel(), available):
        for name, tags in sorted(runs.items()):
            if tag not in tags or tags[tag].empty:
                continue
            df = tags[tag]
            y = (df["value"].rolling(smooth, min_periods=1).mean()
                 if smooth > 1 else df["value"])
            ax.plot(df["step"], y, label=name, lw=1.5)
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("global step")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("All runs - read straight from the TensorBoard event files "
                 "(no server required)", y=1.00)
    fig.tight_layout()
    return fig
