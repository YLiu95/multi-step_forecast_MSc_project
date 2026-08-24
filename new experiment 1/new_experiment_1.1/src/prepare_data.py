from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, pretty
from .download import (MAGNIFICENT_SEVEN, assemble_close, build_universe,
                       download_adjusted_close, select_targets)


def eligible_target_symbols(cfg: Config, close: np.ndarray,
                            dates: pd.DatetimeIndex, tickers: list[str]) -> set[str]:
    """Return symbols with at least one legal target anchor in every split."""
    with np.errstate(divide="ignore", invalid="ignore"):
        returns_valid = np.isfinite(np.diff(np.log(close), axis=1, prepend=np.nan))
    cumulative = np.pad(np.cumsum(returns_valid.astype(np.int32), axis=1),
                        ((0, 0), (1, 0)))
    days = np.arange(cfg.n_steps_in - 1, close.shape[1] - cfg.horizon,
                     dtype=np.int32)
    legal = ((cumulative[:, days + 1] -
              cumulative[:, days + 1 - cfg.n_steps_in] == cfg.n_steps_in) &
             (cumulative[:, days + 1 + cfg.horizon] -
              cumulative[:, days + 1] == cfg.horizon))
    date_values = np.asarray(dates)
    train_end = int(np.searchsorted(date_values, np.datetime64(cfg.train_end), "right")) - 1
    val_end = int(np.searchsorted(date_values, np.datetime64(cfg.val_end), "right")) - 1
    split_day_masks = (
        days <= train_end - cfg.horizon,
        (days >= train_end + cfg.horizon) & (days <= val_end - cfg.horizon),
        days >= val_end + cfg.horizon,
    )
    eligible = np.logical_and.reduce([legal[:, mask].any(axis=1)
                                      for mask in split_day_masks])
    return {ticker for ticker, keep in zip(tickers, eligible) if keep}


def build_return_arrays(cfg: Config, close: np.ndarray, dates: pd.DatetimeIndex,
                        tickers: list[str], targets: pd.DataFrame) -> tuple[dict, dict]:
    """Convert adjusted prices into percent log returns and legal target anchors."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
        returns = np.diff(log_close, axis=1, prepend=np.nan) * 100.0
    del log_close
    returns[~np.isfinite(returns)] = np.nan
    valid = np.isfinite(returns)

    train_columns = dates <= pd.Timestamp(cfg.train_end)
    train_values = returns[:, train_columns]
    scale = float(np.nanstd(train_values))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Could not estimate a positive training return scale")
    normalized = np.clip(returns / scale, -cfg.return_clip, cfg.return_clip)
    normalized = np.nan_to_num(normalized, nan=0.0).astype(np.float16)

    cumulative = np.pad(np.cumsum(valid.astype(np.int32), axis=1), ((0, 0), (1, 0)))
    window_valid = np.zeros_like(valid)
    days = np.arange(cfg.n_steps_in - 1, valid.shape[1])
    window_valid[:, days] = (
        cumulative[:, days + 1] - cumulative[:, days + 1 - cfg.n_steps_in]
        == cfg.n_steps_in
    )
    del cumulative

    ticker_to_index = {symbol: index for index, symbol in enumerate(tickers)}
    target_indices = np.array([ticker_to_index[symbol] for symbol in targets["symbol"]],
                              dtype=np.int32)
    anchors = build_target_anchors(cfg, valid, returns, dates, target_indices)
    arrays = {
        "returns": normalized,
        "raw_returns_pct": np.nan_to_num(returns, nan=0.0).astype(np.float16),
        "valid": valid,
        "window_valid": window_valid,
        "target_indices": target_indices,
        "return_scale_pct": np.array(scale, dtype=np.float32),
    }
    return arrays, anchors


def build_target_anchors(cfg: Config, valid: np.ndarray, returns: np.ndarray,
                         dates: pd.DatetimeIndex,
                         target_indices: np.ndarray) -> dict[str, np.ndarray]:
    """A legal target has a full input window and seven future returns."""
    length, horizon = cfg.n_steps_in, cfg.horizon
    selected = valid[target_indices].astype(np.int32)
    cumulative = np.pad(np.cumsum(selected, axis=1), ((0, 0), (1, 0)))
    n_days = valid.shape[1]
    days = np.arange(length - 1, n_days - horizon, dtype=np.int32)
    input_ok = (cumulative[:, days + 1] - cumulative[:, days + 1 - length]) == length
    future_ok = (cumulative[:, days + 1 + horizon] - cumulative[:, days + 1]) == horizon
    target_row, day_col = np.nonzero(input_ok & future_ok)
    ticker_index = target_indices[target_row].astype(np.int32)
    anchor_day = days[day_col].astype(np.int32)

    date_values = np.asarray(dates)
    train_end = int(np.searchsorted(date_values, np.datetime64(cfg.train_end), "right")) - 1
    val_end = int(np.searchsorted(date_values, np.datetime64(cfg.val_end), "right")) - 1
    split_masks = {
        "train": anchor_day <= train_end - horizon,
        "val": (anchor_day >= train_end + horizon) & (anchor_day <= val_end - horizon),
        "test": anchor_day >= val_end + horizon,
    }
    result: dict[str, np.ndarray] = {}
    for split, mask in split_masks.items():
        result[f"{split}_i"] = ticker_index[mask]
        result[f"{split}_t"] = anchor_day[mask]
        print(f"{split:>5} anchors: {mask.sum():,}")
    return result


def prepare(cfg: Config, force: bool = False, symbol_limit: int | None = None) -> Path:
    cfg.make_dirs()
    output = cfg.paths["panel"]
    marker = output / "meta.json"
    if marker.exists() and not force:
        print(f"Panel already exists at {output}; use --force to rebuild")
        return output

    print(pretty(cfg))
    universe = build_universe(cfg)
    symbols = universe["symbol"].tolist()
    if symbol_limit is not None:
        symbols = sorted(set(symbols[:symbol_limit]) | set(MAGNIFICENT_SEVEN))
        print(f"Smoke limit plus Mag7: {len(symbols):,} symbols")
    paths = download_adjusted_close(cfg, symbols)
    close, dates, tickers = assemble_close(cfg, paths, set(MAGNIFICENT_SEVEN))
    eligible = eligible_target_symbols(cfg, close, dates, tickers)
    target_universe = universe[universe["symbol"].isin(eligible)]
    targets = select_targets(cfg, target_universe)
    print(f"target-eligible universe: {len(eligible):,}; selected {len(targets)}")
    arrays, anchors = build_return_arrays(cfg, close, dates, tickers, targets)
    del close

    output.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
    for name, array in anchors.items():
        np.save(output / f"anchor_{name}.npy", array)
    targets.to_csv(output / "target_tickers.csv", index=False)
    metadata = {
        "tickers": tickers,
        "dates": [str(date.date()) for date in dates],
        "shape": list(arrays["returns"].shape),
        "target_count": len(targets),
        "target_selection_seed": cfg.seed,
        "return_scale_pct": float(arrays["return_scale_pct"]),
        "universe_limitation": (
            "Currently listed NASDAQ and NYSE common stocks with Yahoo history; "
            "delisted securities require licensed point-in-time data."
        ),
        "config": cfg.to_dict(),
    }
    marker.write_text(json.dumps(metadata, indent=2))
    print(f"Saved panel to {output} ({sum(p.stat().st_size for p in output.glob('*.npy')) / 1e6:.1f} MB)")
    return output


def load_panel(cfg: Config, mmap: bool = True):
    root = cfg.paths["panel"]
    mode = "r" if mmap else None
    metadata = json.loads((root / "meta.json").read_text())
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode=mode)
              for name in ("returns", "raw_returns_pct", "valid", "window_valid",
                           "target_indices")}
    anchors = {name: np.load(root / f"anchor_{name}.npy", mmap_mode=mode)
               for name in ("train_i", "train_t", "val_i", "val_t", "test_i", "test_t")}
    return arrays, anchors, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--symbol-limit", type=int)
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    prepare(config, force=args.force, symbol_limit=args.symbol_limit)