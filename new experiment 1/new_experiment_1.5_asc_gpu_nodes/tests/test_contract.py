import numpy as np
import pytest

from ascgpu.contract import DataContract, deadline_timestamp, future_valid, history_valid, price_to_returns, signed_label, split_masks


def test_signed_seven_session_label():
    prices = np.exp(np.arange(20)[None, :] * -0.01) * 100
    returns = price_to_returns(prices)
    assert signed_label(returns, 0, 5) == pytest.approx(-7.0)
    assert signed_label(returns, 0, 5) == pytest.approx(100 * np.log(prices[0, 12] / prices[0, 5]))


def test_missing_session_invalidates_adjacent_returns_and_windows():
    prices = np.arange(1, 21, dtype=float)[None, :]
    prices[0, 7] = np.nan
    returns = price_to_returns(prices)
    assert np.isnan(returns[0, 7:9]).all()
    valid = np.isfinite(returns)
    assert not history_valid(valid, 4)[0, 10]
    assert history_valid(valid, 4)[0, 12]
    assert not future_valid(valid, 7)[0, 4]
    with pytest.raises(ValueError):
        signed_label(returns, 0, 4)


def test_session_index_split_and_label_boundaries():
    dates = np.arange("2018-12-01", "2023-02-01", dtype="datetime64[D]")
    contract = DataContract()
    masks = split_masks(dates, contract)
    train_boundary = np.flatnonzero(dates == np.datetime64(contract.train_end))[0]
    val_boundary = np.flatnonzero(dates == np.datetime64(contract.val_end))[0]
    assert np.flatnonzero(masks["train"])[-1] == train_boundary - 7
    assert np.flatnonzero(masks["val"])[0] == train_boundary + 7
    assert np.flatnonzero(masks["val"])[-1] == val_boundary - 7
    assert np.flatnonzero(masks["test"])[0] == val_boundary + 7
    assert not np.any(masks["train"] & masks["val"])


def test_deadlines_are_timezone_explicit():
    assert deadline_timestamp("2026-09-29T15:15:00+02:00") < deadline_timestamp("2026-09-29T16:00:00+02:00")
    with pytest.raises(ValueError):
        deadline_timestamp("2026-09-29T15:15:00")


def test_prepared_anchor_sampler_is_distinct_deterministic_and_masked(tmp_path):
    from dataclasses import asdict
    from ascgpu.data import AnchorDataset, write_json

    contract = DataContract(window=4, horizon=2, basket=3)
    directory = tmp_path / "AA"
    directory.mkdir()
    raw = np.full((4, 12), -0.5, dtype=np.float32)
    np.save(directory / "raw.npy", raw)
    np.save(directory / "inputs.npy", raw)
    np.save(directory / "history_valid.npy", np.ones(raw.shape, dtype=bool))
    for split in ("train", "val"):
        np.save(directory / f"{split}_offsets.npy", np.array([0, 2, 4, 6, 8]))
        np.save(directory / f"{split}_days.npy", np.tile([5, 6], 4).astype(np.int16))
    write_json(tmp_path / "meta.json", {"contract": asdict(contract), "markets": [{"name": "AA", "offset": 0}]})
    dataset = AnchorDataset(tmp_path)
    first = dataset.sample("train", 3, 42)
    second = dataset.sample("train", 3, 42)
    assert len(np.unique(first["ticker_ids"])) == 3
    assert first["ticker_ids"][first["target_position"]] == 1
    assert first["label"] == -1.0
    np.testing.assert_array_equal(first["ticker_ids"], second["ticker_ids"])
    order = np.load(dataset.ensure_permutation(0, 42))
    assert sorted(order.tolist()) == list(range(8))
    batch = dataset.batch("train", [3, -1], 42)
    np.testing.assert_array_equal(batch["weight"], [1.0, 0.0])