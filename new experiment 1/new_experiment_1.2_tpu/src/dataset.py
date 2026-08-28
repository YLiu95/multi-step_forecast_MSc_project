from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class MarketPanel:
    name: str
    returns: np.ndarray
    raw_returns_pct: np.ndarray
    window_valid: np.ndarray
    global_ids: np.ndarray
    is_mag7: np.ndarray
    always_eligible: np.ndarray
    anchor_offsets: dict[str, np.ndarray]
    anchor_days: dict[str, np.ndarray]

    @classmethod
    def load(cls, panel_root: Path, market: str) -> "MarketPanel":
        directory = panel_root / market
        ticker_table = pd.read_csv(directory / "tickers.csv")
        return cls(
            name=market,
            returns=np.load(directory / "returns.npy", mmap_mode="r"),
            raw_returns_pct=np.load(
                directory / "raw_returns_pct.npy", mmap_mode="r"
            ),
            window_valid=np.load(directory / "window_valid.npy", mmap_mode="r"),
            global_ids=np.load(directory / "global_ids.npy", mmap_mode="r"),
            is_mag7=ticker_table["is_mag7"].to_numpy(dtype=np.bool_),
            always_eligible=ticker_table["always_eligible"].to_numpy(dtype=np.bool_),
            anchor_offsets={
                split: np.load(
                    directory / f"anchor_{split}_offsets.npy", mmap_mode="r"
                )
                for split in ("train", "val", "test")
            },
            anchor_days={
                split: np.load(
                    directory / f"anchor_{split}_days.npy", mmap_mode="r"
                )
                for split in ("train", "val", "test")
            },
        )


class GlobalBasketSampler:
    def __init__(self, cfg: Config, panel_root: str | Path | None = None):
        self.cfg = cfg
        self.panel_root = Path(panel_root or cfg.paths["panel"])
        metadata = json.loads((self.panel_root / "meta.json").read_text())
        self.market_names = tuple(item["market"] for item in metadata["markets"])
        self.panels = [MarketPanel.load(self.panel_root, name) for name in self.market_names]
        self.target_pairs: dict[str, np.ndarray] = {}
        self.targets_by_market: dict[str, list[np.ndarray]] = {}
        self.mag7_pairs: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            by_market = []
            pairs = []
            mag7_pairs = []
            for market_index, panel in enumerate(self.panels):
                eligible = np.flatnonzero(np.diff(panel.anchor_offsets[split]) > 0)
                by_market.append(eligible.astype(np.int32))
                pairs.extend((market_index, int(local_id)) for local_id in eligible)
                mag7_pairs.extend(
                    (market_index, int(local_id))
                    for local_id in eligible
                    if panel.is_mag7[local_id]
                )
            self.targets_by_market[split] = by_market
            self.target_pairs[split] = np.asarray(pairs, dtype=np.int32)
            self.mag7_pairs[split] = np.asarray(mag7_pairs, dtype=np.int32).reshape(-1, 2)
            if not pairs:
                raise RuntimeError(f"No eligible target tickers in {split}")
        self.input_offsets = np.arange(-cfg.n_steps_in + 1, 1, dtype=np.int32)
        self.future_offsets = np.arange(1, cfg.horizon + 1, dtype=np.int32)

    @property
    def n_tickers(self) -> int:
        return sum(len(panel.global_ids) for panel in self.panels)

    def _stratified_pairs(self, split: str, batch_size: int,
                          rng: np.random.Generator) -> np.ndarray:
        available = [index for index, values in enumerate(self.targets_by_market[split])
                     if len(values)]
        mag7_count = min(8, batch_size // 20, len(self.mag7_pairs[split]))
        floor = min(8, (batch_size - mag7_count) // len(available))
        market_choices = np.repeat(available, floor)
        remaining = batch_size - mag7_count - len(market_choices)
        counts = np.array(
            [len(self.targets_by_market[split][index]) for index in available],
            dtype=np.float64,
        )
        if remaining:
            extra = rng.choice(available, size=remaining, p=counts / counts.sum())
            market_choices = np.concatenate((market_choices, extra))
        rng.shuffle(market_choices)
        pairs = np.empty((len(market_choices), 2), dtype=np.int32)
        pairs[:, 0] = market_choices
        for row, market_index in enumerate(market_choices):
            pairs[row, 1] = rng.choice(self.targets_by_market[split][market_index])
        if mag7_count:
            chosen_mag7 = self.mag7_pairs[split][
                rng.integers(0, len(self.mag7_pairs[split]), size=mag7_count)
            ]
            pairs = np.concatenate((pairs, chosen_mag7))
            rng.shuffle(pairs)
        return pairs

    def _target_pairs(self, split: str, batch_size: int, rng: np.random.Generator,
                      stratified: bool) -> np.ndarray:
        if stratified:
            return self._stratified_pairs(split, batch_size, rng)
        candidates = self.target_pairs[split]
        return candidates[rng.integers(0, len(candidates), size=batch_size)]

    def sample_batch(self, split: str, batch_size: int, rng: np.random.Generator,
                     stratified: bool = False) -> dict[str, np.ndarray]:
        selected = self._target_pairs(split, batch_size, rng, stratified)
        basket_size = self.cfg.n_tickers_per_sample
        inputs = np.empty(
            (batch_size, basket_size, self.cfg.n_steps_in), dtype=np.float16
        )
        ticker_ids = np.empty((batch_size, basket_size), dtype=np.int32)
        target_position = np.empty(batch_size, dtype=np.int32)
        magnitude = np.empty(batch_size, dtype=np.float32)
        direction = np.empty(batch_size, dtype=np.float32)
        signed_return = np.empty(batch_size, dtype=np.float32)
        market_id = selected[:, 0].astype(np.int32)
        is_mag7 = np.empty(batch_size, dtype=np.bool_)
        always_eligible = np.empty(batch_size, dtype=np.bool_)

        for row, (market_index, target_local) in enumerate(selected):
            panel = self.panels[int(market_index)]
            offsets = panel.anchor_offsets[split]
            start, stop = int(offsets[target_local]), int(offsets[target_local + 1])
            anchor_index = rng.integers(start, stop)
            anchor_day = int(panel.anchor_days[split][anchor_index])

            candidates = np.flatnonzero(panel.window_valid[:, anchor_day])
            candidates = candidates[candidates != target_local]
            context = rng.choice(candidates, size=basket_size - 1, replace=False)
            position = int(rng.integers(0, basket_size))
            basket = np.insert(context, position, target_local).astype(np.int32)

            days = anchor_day + self.input_offsets
            inputs[row] = panel.returns[basket[:, None], days[None, :]]
            ticker_ids[row] = panel.global_ids[basket]
            target_position[row] = position
            future_days = anchor_day + self.future_offsets
            total = float(
                panel.raw_returns_pct[target_local, future_days].astype(np.float32).sum()
            )
            signed_return[row] = total
            magnitude[row] = abs(total)
            direction[row] = float(total > 0)
            is_mag7[row] = panel.is_mag7[target_local]
            always_eligible[row] = panel.always_eligible[target_local]

        return {
            "inputs": inputs,
            "ticker_ids": ticker_ids,
            "target_position": target_position,
            "magnitude_pct": magnitude,
            "direction": direction,
            "signed_return_pct": signed_return,
            "market_id": market_id,
            "is_mag7": is_mag7,
            "always_eligible": always_eligible,
        }

    def batches(self, split: str, batch_size: int, count: int, seed: int,
                stratified: bool = False) -> Iterator[dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        for _ in range(count):
            yield self.sample_batch(split, batch_size, rng, stratified)


def prefetch_batches(batches: Iterator[dict[str, np.ndarray]],
                     depth: int = 2) -> Iterator[dict[str, np.ndarray]]:
    work_queue: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def produce() -> None:
        try:
            for batch in batches:
                work_queue.put(batch)
        except BaseException as error:
            work_queue.put(error)
        finally:
            work_queue.put(sentinel)

    thread = threading.Thread(target=produce, name="batch-prefetch", daemon=True)
    thread.start()
    while True:
        item = work_queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item