from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import Config, MAGNIFICENT_SEVEN
from .download import download_market_files


@dataclass(frozen=True)
class SplitBounds:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def price_to_log_returns(prices: np.ndarray) -> np.ndarray:
    returns = np.full(prices.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:, 1:] = 100.0 * (
            np.log(prices[:, 1:]) - np.log(prices[:, :-1])
        )
    returns[~np.isfinite(returns)] = np.nan
    return returns


def split_anchor_masks(dates: np.ndarray, cfg: Config) -> SplitBounds:
    train_boundary = np.searchsorted(
        dates, np.datetime64(cfg.train_end), side="right"
    ) - 1
    val_boundary = np.searchsorted(
        dates, np.datetime64(cfg.val_end), side="right"
    ) - 1
    days = np.arange(len(dates), dtype=np.int32)
    return SplitBounds(
        train=days <= train_boundary - cfg.horizon,
        val=(days >= train_boundary + cfg.embargo_sessions)
        & (days <= val_boundary - cfg.horizon),
        test=days >= val_boundary + cfg.embargo_sessions,
    )


def rolling_window_valid(valid: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros(valid.shape, dtype=np.bool_)
    for start in range(0, valid.shape[0], 512):
        block = valid[start:start + 512].astype(np.int32)
        cumulative = np.pad(np.cumsum(block, axis=1), ((0, 0), (1, 0)))
        days = np.arange(length - 1, valid.shape[1], dtype=np.int32)
        output[start:start + len(block), days] = (
            cumulative[:, days + 1] - cumulative[:, days + 1 - length] == length
        )
    return output


def future_valid(valid: np.ndarray, horizon: int) -> np.ndarray:
    output = np.zeros(valid.shape, dtype=np.bool_)
    for start in range(0, valid.shape[0], 512):
        block = valid[start:start + 512].astype(np.int32)
        cumulative = np.pad(np.cumsum(block, axis=1), ((0, 0), (1, 0)))
        days = np.arange(0, valid.shape[1] - horizon, dtype=np.int32)
        output[start:start + len(block), days] = (
            cumulative[:, days + 1 + horizon] - cumulative[:, days + 1] == horizon
        )
    return output


def mask_to_csr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = mask.sum(axis=1, dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()))
    _, days = np.nonzero(mask)
    if mask.shape[1] >= np.iinfo(np.int16).max:
        raise ValueError("Market calendar is too long for int16 anchor days")
    return offsets, days.astype(np.int16)


def table_to_price_matrix(paths: list[Path], price_column: str) \
        -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pa.concat_tables(
        [pq.read_table(path, columns=["ticker", "date", price_column]) for path in paths]
    ).combine_chunks()
    encoded = pc.dictionary_encode(table["ticker"].chunk(0))
    tickers = encoded.dictionary.to_pylist()
    ticker_index = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    row_dates = table["date"].to_numpy(zero_copy_only=False).astype("datetime64[D]")
    dates = np.unique(row_dates)
    day_index = np.searchsorted(dates, row_dates)
    prices = np.full((len(tickers), len(dates)), np.nan, dtype=np.float32)
    values = table[price_column].to_numpy(zero_copy_only=False).astype(np.float32)
    values[(values <= 0) | ~np.isfinite(values)] = np.nan
    prices[ticker_index, day_index] = values
    return prices, dates, tickers


def prepare_raw_market(cfg: Config, market: str, paths: list[Path], output: Path,
                       global_offset: int) -> dict:
    market_dir = output / market
    market_dir.mkdir(parents=True, exist_ok=True)
    prices, dates, tickers = table_to_price_matrix(paths, cfg.price_column)
    returns = price_to_log_returns(prices)
    del prices
    train_mask = dates <= np.datetime64(cfg.train_end)
    train_values = returns[:, train_mask]
    finite = np.isfinite(train_values)
    values = train_values[finite].astype(np.float64)
    statistics = {
        "sum": float(values.sum()),
        "sum_squares": float(np.square(values).sum()),
        "count": int(values.size),
    }
    del train_values, finite, values
    np.save(market_dir / "raw_returns_pct.npy", returns.astype(np.float16))
    np.save(market_dir / "dates.npy", dates)
    np.save(market_dir / "global_ids.npy",
            np.arange(global_offset, global_offset + len(tickers), dtype=np.int32))
    pd.DataFrame({
        "ticker": tickers,
        "market": market,
        "local_id": np.arange(len(tickers), dtype=np.int32),
        "global_id": np.arange(global_offset, global_offset + len(tickers), dtype=np.int32),
    }).to_csv(market_dir / "tickers.csv", index=False)
    print(
        f"{market}: {len(tickers):,} series, {len(dates):,} sessions, "
        f"{len(returns):,} rows prepared",
        flush=True,
    )
    return {
        "market": market,
        "n_tickers": len(tickers),
        "n_dates": len(dates),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "global_offset": global_offset,
        "statistics": statistics,
    }


def finalize_market(cfg: Config, market_info: dict, output: Path,
                    return_scale: float) -> dict:
    market = market_info["market"]
    market_dir = output / market
    raw = np.load(market_dir / "raw_returns_pct.npy", mmap_mode="r")
    dates = np.load(market_dir / "dates.npy")
    valid = np.isfinite(raw)
    window_valid = rolling_window_valid(valid, cfg.n_steps_in)
    enough_context = window_valid.sum(axis=0) >= cfg.n_tickers_per_sample
    target_valid = window_valid & future_valid(valid, cfg.horizon)
    target_valid &= enough_context[None, :]
    bounds = split_anchor_masks(dates, cfg)

    normalized = np.nan_to_num(
        np.clip(raw.astype(np.float32) / return_scale, -cfg.return_clip, cfg.return_clip),
        nan=0.0,
    ).astype(np.float16)
    np.save(market_dir / "returns.npy", normalized)
    np.save(market_dir / "window_valid.npy", window_valid)
    del normalized

    ticker_table = pd.read_csv(market_dir / "tickers.csv")
    anchor_counts: dict[str, int] = {}
    eligibility: dict[str, np.ndarray] = {}
    for split, split_mask in (
        ("train", bounds.train), ("val", bounds.val), ("test", bounds.test)
    ):
        mask = target_valid & split_mask[None, :]
        offsets, days = mask_to_csr(mask)
        np.save(market_dir / f"anchor_{split}_offsets.npy", offsets)
        np.save(market_dir / f"anchor_{split}_days.npy", days)
        eligibility[split] = np.diff(offsets) > 0
        ticker_table[f"eligible_{split}"] = eligibility[split]
        anchor_counts[split] = int(len(days))
        del mask, offsets, days
    ticker_table["always_eligible"] = (
        eligibility["train"] & eligibility["val"] & eligibility["test"]
    )
    ticker_table["is_mag7"] = (
        (ticker_table["market"] == "US")
        & ticker_table["ticker"].isin(MAGNIFICENT_SEVEN)
    )
    ticker_table.to_csv(market_dir / "tickers.csv", index=False)
    return {**market_info, "anchors": anchor_counts}


def prepare(cfg: Config, force: bool = False) -> Path:
    cfg.validate()
    cfg.make_dirs()
    output = cfg.paths["panel"]
    marker = output / "meta.json"
    if marker.exists() and not force:
        print(f"Panel already exists at {output}; use --force to rebuild")
        return output
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    paths_by_market = download_market_files(cfg)
    market_info = []
    global_offset = 0
    for market in cfg.markets:
        info = prepare_raw_market(
            cfg, market, paths_by_market[market], output, global_offset
        )
        market_info.append(info)
        global_offset += info["n_tickers"]

    total_count = sum(info["statistics"]["count"] for info in market_info)
    total_sum = sum(info["statistics"]["sum"] for info in market_info)
    total_sum_squares = sum(info["statistics"]["sum_squares"] for info in market_info)
    mean = total_sum / total_count
    return_scale = float(np.sqrt(total_sum_squares / total_count - mean * mean))
    if not np.isfinite(return_scale) or return_scale <= 0:
        raise RuntimeError("Could not estimate a positive global training return scale")
    print(f"Global train-only return scale: {return_scale:.6f}%", flush=True)

    finalized = [finalize_market(cfg, info, output, return_scale) for info in market_info]
    ticker_tables = [pd.read_csv(output / market / "tickers.csv") for market in cfg.markets]
    tickers = pd.concat(ticker_tables, ignore_index=True)
    tickers.to_csv(output / "tickers.csv", index=False)
    tickers[tickers["is_mag7"]].to_csv(output / "mag7.csv", index=False)
    metadata = {
        "dataset_repo": cfg.dataset_repo,
        "price_column": cfg.price_column,
        "n_tickers": int(len(tickers)),
        "return_scale_pct": return_scale,
        "markets": finalized,
        "config": cfg.to_dict(),
    }
    marker.write_text(json.dumps(metadata, indent=2) + "\n")
    cfg.save(output / "config.json")
    print(f"Prepared {len(tickers):,} global series at {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Experiment 1.2 market panels")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    prepare(cfg, force=args.force)


if __name__ == "__main__":
    main()