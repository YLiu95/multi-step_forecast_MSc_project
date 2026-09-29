from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class DataContract:
    window: int = 256
    horizon: int = 7
    basket: int = 64
    train_end: str = "2018-12-31"
    val_end: str = "2022-12-31"
    embargo: int = 7
    clip: float = 8.0


def price_to_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    valid = np.isfinite(prices) & (prices > 0)
    result = np.full(prices.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        differences = 100.0 * np.diff(np.log(prices), axis=1)
    result[:, 1:] = np.where(valid[:, 1:] & valid[:, :-1], differences, np.nan)
    return result


def history_valid(valid: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(valid.shape, dtype=bool)
    if valid.shape[1] < window:
        return result
    for start in range(0, len(valid), 256):
        block = valid[start:start + 256]
        cumulative = np.pad(np.cumsum(block, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        result[start:start + len(block), window - 1:] = (
            cumulative[:, window:] - cumulative[:, :-window] == window
        )
    return result


def future_valid(valid: np.ndarray, horizon: int) -> np.ndarray:
    result = np.zeros(valid.shape, dtype=bool)
    if valid.shape[1] <= horizon:
        return result
    for start in range(0, len(valid), 256):
        block = valid[start:start + 256]
        cumulative = np.pad(np.cumsum(block, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        result[start:start + len(block), :-horizon] = (
            cumulative[:, horizon + 1:] - cumulative[:, 1:-horizon] == horizon
        )
    return result


def split_masks(dates: np.ndarray, contract: DataContract) -> dict[str, np.ndarray]:
    train_boundary = np.searchsorted(dates, np.datetime64(contract.train_end), side="right") - 1
    val_boundary = np.searchsorted(dates, np.datetime64(contract.val_end), side="right") - 1
    positions = np.arange(len(dates))
    return {
        "train": positions <= train_boundary - contract.horizon,
        "val": (positions >= train_boundary + contract.embargo)
        & (positions <= val_boundary - contract.horizon),
        "test": positions >= val_boundary + contract.embargo,
    }


def signed_label(returns: np.ndarray, target: int, anchor: int, horizon: int = 7) -> float:
    values = returns[target, anchor + 1:anchor + 1 + horizon]
    if len(values) != horizon or not np.isfinite(values).all():
        raise ValueError("A signed label requires every future market session")
    return float(values.astype(np.float64).sum())


def deadline_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Operational deadlines must include an explicit timezone")
    return parsed.timestamp()