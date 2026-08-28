import numpy as np

from src.config import Config
from src.prepare_data import (
    future_valid,
    mask_to_csr,
    price_to_log_returns,
    rolling_window_valid,
    split_anchor_masks,
)


def test_log_returns_and_missing_gap_are_handled():
    prices = np.array([[100.0, 102.0, 101.0], [20.0, np.nan, 21.0]], dtype=np.float32)
    returns = price_to_log_returns(prices)

    assert np.isclose(returns[0, 1], 100 * np.log(1.02), atol=1e-5)
    assert np.isnan(returns[1, 1:]).all()


def test_window_and_future_validity_do_not_cross_gaps():
    valid = np.array([[False, True, True, True, True, False, True, True]], dtype=bool)
    windows = rolling_window_valid(valid, length=3)
    future = future_valid(valid, horizon=2)

    assert windows.tolist() == [[False, False, False, True, True, False, False, False]]
    assert future[0, 1]
    assert not future[0, 3]


def test_split_masks_enforce_horizon_and_embargo():
    dates = np.arange(
        np.datetime64("2018-12-01"), np.datetime64("2023-02-01"), dtype="datetime64[D]"
    )
    cfg = Config().override(horizon=7, embargo_sessions=7)
    masks = split_anchor_masks(dates, cfg)
    train_days = dates[masks.train]
    val_days = dates[masks.val]
    test_days = dates[masks.test]

    assert train_days[-1] <= np.datetime64("2018-12-24")
    assert val_days[0] >= np.datetime64("2019-01-07")
    assert val_days[-1] <= np.datetime64("2022-12-24")
    assert test_days[0] >= np.datetime64("2023-01-07")


def test_anchor_mask_round_trips_to_csr():
    mask = np.array([[False, True, False], [True, False, True]])
    offsets, days = mask_to_csr(mask)

    assert offsets.tolist() == [0, 1, 3]
    assert days.tolist() == [1, 0, 2]