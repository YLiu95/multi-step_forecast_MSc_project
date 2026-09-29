from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from .contract import DataContract, future_valid, history_valid, price_to_returns, signed_label, split_masks

DATASET = "YL95/new_experiment_1-data"
REVISION = "bcbbefdbe2313673895eb1a0d354747a9f1624fa"
MARKETS = ("AU", "CA", "CH", "CN", "DE", "FR", "GB", "HK", "IN", "JP", "KR", "NL", "US")
MAG7 = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_market(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    tickers = set()
    calendars = []
    observations = 0
    for path in paths:
        table = pq.read_table(path, columns=["ticker", "date"])
        tickers.update(pc.unique(table["ticker"]).to_pylist())
        calendars.append(pc.unique(table["date"]).to_numpy(zero_copy_only=False).astype("datetime64[D]"))
        observations += len(table)
    symbols = sorted(tickers)
    dates = np.unique(np.concatenate(calendars))
    prices = np.full((len(symbols), len(dates)), np.nan, dtype=np.float64)
    seen = np.zeros(prices.size, dtype=bool)
    invalid_prices = 0
    for path in paths:
        table = pq.read_table(path, columns=["ticker", "date", "adj_close_clean"])
        ticker_index = pd.Categorical(table["ticker"].to_pandas(), categories=symbols).codes.astype(np.int64)
        date_index = np.searchsorted(dates, table["date"].to_numpy(zero_copy_only=False).astype("datetime64[D]"))
        linear = ticker_index * len(dates) + date_index
        if len(np.unique(linear)) != len(linear) or seen[linear].any():
            raise ValueError(f"Duplicate ticker/session observations in {path.name}; audit required")
        seen[linear] = True
        values = table["adj_close_clean"].to_numpy(zero_copy_only=False).astype(np.float64)
        invalid = ~np.isfinite(values) | (values <= 0)
        invalid_prices += int(invalid.sum())
        values[invalid] = np.nan
        prices[ticker_index, date_index] = values
    return prices, dates, symbols, {"observations": observations, "invalid_prices": invalid_prices, "duplicates": 0}


def prepare(root: Path, contract: DataContract = DataContract()) -> Path:
    panel_root = root / "panel"
    marker = panel_root / "meta.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing["revision"] != REVISION or existing["contract"] != asdict(contract):
            raise ValueError("Prepared data provenance differs from the requested experiment")
        return panel_root
    load_dotenv(Path.home() / ".env", override=False)
    snapshot = Path(snapshot_download(
        repo_id=DATASET, repo_type="dataset", revision=REVISION,
        allow_patterns=[f"data/{market}/*.parquet" for market in MARKETS],
        token=os.environ["HF_TOKEN"], cache_dir=root / "cache" / "hf", max_workers=8,
    ))
    panel_root.mkdir(parents=True, exist_ok=True)
    market_info = []
    vocabulary = []
    for market in MARKETS:
        directory = panel_root / market
        directory.mkdir(exist_ok=True)
        raw_marker = directory / "raw_info.json"
        if raw_marker.exists():
            info = json.loads(raw_marker.read_text())
            symbols = json.loads((directory / "symbols.json").read_text())
        else:
            paths = sorted((snapshot / "data" / market).glob("*.parquet"))
            if not paths:
                raise ValueError(f"No parquet shards for {market}")
            prices, dates, symbols, audit = read_market(paths)
            returns = price_to_returns(prices)
            del prices
            training_values = returns[:, dates <= np.datetime64(contract.train_end)]
            finite_values = training_values[np.isfinite(training_values)].astype(np.float64)
            info = {
                "name": market, "tickers": len(symbols), "sessions": len(dates),
                "offset": len(vocabulary), "first_date": str(dates[0]), "last_date": str(dates[-1]),
                "sum": float(finite_values.sum()), "sum_squares": float(np.square(finite_values).sum()),
                "count": int(finite_values.size), **audit,
            }
            np.save(directory / "raw.npy", returns, allow_pickle=False)
            np.save(directory / "dates.npy", dates, allow_pickle=False)
            write_json(directory / "symbols.json", symbols)
            write_json(raw_marker, info)
            del returns, training_values, finite_values
        if info["offset"] != len(vocabulary):
            raise ValueError("Cached vocabulary offsets are inconsistent")
        vocabulary.extend({"market": market, "ticker": symbol} for symbol in symbols)
        market_info.append(info)
        print(json.dumps({"stage": "raw_panel", **info}), flush=True)
    count = sum(info["count"] for info in market_info)
    mean = sum(info["sum"] for info in market_info) / count
    scale = float(np.sqrt(sum(info["sum_squares"] for info in market_info) / count - mean * mean))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only normalization scale")
    for info in market_info:
        directory = panel_root / info["name"]
        raw = np.load(directory / "raw.npy", mmap_mode="r", allow_pickle=False)
        dates = np.load(directory / "dates.npy", allow_pickle=False)
        valid = np.isfinite(raw)
        histories = history_valid(valid, contract.window)
        targets = histories & future_valid(valid, contract.horizon)
        targets &= (histories.sum(axis=0) >= contract.basket)[None, :]
        normalized = np.nan_to_num(np.clip(raw / scale, -contract.clip, contract.clip), nan=0.0).astype(np.float16)
        np.save(directory / "inputs.npy", normalized, allow_pickle=False)
        np.save(directory / "history_valid.npy", histories, allow_pickle=False)
        info["anchors"] = {}
        info["eligible_tickers"] = {}
        for split, split_mask in split_masks(dates, contract).items():
            mask = targets & split_mask[None, :]
            counts = mask.sum(axis=1, dtype=np.int64)
            offsets = np.concatenate(([0], counts.cumsum()))
            if len(dates) >= 32768:
                raise ValueError("Calendar exceeds the supported int16 anchor-day range")
            days = np.nonzero(mask)[1].astype(np.int16)
            np.save(directory / f"{split}_offsets.npy", offsets, allow_pickle=False)
            np.save(directory / f"{split}_days.npy", days, allow_pickle=False)
            info["anchors"][split] = int(len(days))
            info["eligible_tickers"][split] = int((counts > 0).sum())
        write_json(directory / "info.json", info)
        print(json.dumps({"stage": "eligible_anchors", "market": info["name"], "anchors": info["anchors"]}), flush=True)
        del raw, valid, histories, targets, normalized, mask, days
    metadata = {
        "dataset": DATASET, "revision": REVISION, "contract": asdict(contract),
        "return_scale_pct": scale, "n_tickers": len(vocabulary), "markets": market_info,
        "raw_return_dtype": "float32", "input_dtype": "float16",
    }
    write_json(panel_root / "vocabulary.json", vocabulary)
    write_json(marker, metadata)
    dataset = AnchorDataset(panel_root)
    dataset.ensure_permutation(0, 1337)
    monitor = dataset.monitor_indices(64, 7331)
    write_json(panel_root / "monitor.json", monitor)
    write_json(root / "reports" / "data_audit.json", {
        **metadata, "monitor_count": len(monitor),
        "monitor_sha256": hashlib.sha256(json.dumps(monitor).encode()).hexdigest(),
        "test_evaluated": False,
    })
    return panel_root


class AnchorDataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "meta.json").read_text())
        self.contract = DataContract(**self.metadata["contract"])
        self.panels = []
        for info in self.metadata["markets"]:
            directory = self.root / info["name"]
            panel = {"info": info}
            for name in ("raw", "inputs", "history_valid", "train_offsets", "train_days", "val_offsets", "val_days"):
                panel[name] = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            self.panels.append(panel)
        self.cumulative = {
            split: np.cumsum([0] + [len(panel[f"{split}_days"]) for panel in self.panels], dtype=np.int64)
            for split in ("train", "val")
        }

    def count(self, split: str) -> int:
        return int(self.cumulative[split][-1])

    def ensure_permutation(self, epoch: int, seed: int) -> Path:
        path = self.root / f"train_order_{seed}_{epoch}.npy"
        if not path.exists():
            total = self.count("train")
            if total >= 2**32:
                raise ValueError("Anchor permutation exceeds the explicit uint32 budget")
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(total).astype(np.uint32)
            temporary = path.with_name(path.stem + ".tmp.npy")
            np.save(temporary, order, allow_pickle=False)
            temporary.replace(path)
        return path

    def sample(self, split: str, index: int, seed: int) -> dict[str, np.ndarray]:
        if split not in ("train", "val") or not 0 <= index < self.count(split):
            raise ValueError("Invalid training/validation anchor")
        market_index = int(np.searchsorted(self.cumulative[split], index, side="right") - 1)
        panel = self.panels[market_index]
        local_index = index - int(self.cumulative[split][market_index])
        target = int(np.searchsorted(panel[f"{split}_offsets"], local_index, side="right") - 1)
        day = int(panel[f"{split}_days"][local_index])
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        choices = np.flatnonzero(panel["history_valid"][:, day])
        choices = choices[choices != target]
        context = rng.choice(choices, self.contract.basket - 1, replace=False)
        basket = np.concatenate(([target], context))
        rng.shuffle(basket)
        target_position = int(np.flatnonzero(basket == target)[0])
        days = day + np.arange(-self.contract.window + 1, 1)
        inputs = np.asarray(panel["inputs"][basket[:, None], days[None, :]], dtype=np.float32)
        return {
            "inputs": inputs,
            "ticker_ids": (basket + panel["info"]["offset"]).astype(np.int64),
            "target_position": np.array(target_position, dtype=np.int64),
            "label": np.array(signed_label(panel["raw"], target, day, self.contract.horizon), dtype=np.float32),
            "market": np.array(market_index, dtype=np.int64),
            "weight": np.array(1.0, dtype=np.float32),
        }

    def batch(self, split: str, indices: list[int], seed: int) -> dict[str, np.ndarray]:
        samples = [self.sample(split, max(index, 0), seed) for index in indices]
        for sample, index in zip(samples, indices):
            if index < 0:
                sample["weight"] = np.array(0.0, dtype=np.float32)
        return {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}

    def monitor_indices(self, per_market: int, seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        indices = []
        for market_index, panel in enumerate(self.panels):
            count = len(panel["val_days"])
            if count:
                indices.extend((rng.choice(count, min(per_market, count), replace=False) + self.cumulative["val"][market_index]).tolist())
            if panel["info"]["name"] == "US":
                symbols = json.loads((self.root / "US" / "symbols.json").read_text())
                for local_id, symbol in enumerate(symbols):
                    offsets = panel["val_offsets"]
                    if symbol in MAG7 and offsets[local_id + 1] > offsets[local_id]:
                        indices.append(int(self.cumulative["val"][market_index] + offsets[local_id]))
        return sorted(set(indices))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare(arguments.root), flush=True)


if __name__ == "__main__":
    main()