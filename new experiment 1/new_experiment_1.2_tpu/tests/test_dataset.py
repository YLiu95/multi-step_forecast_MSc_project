import json

import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import GlobalBasketSampler, prefetch_batches


def make_panel(tmp_path):
    cfg = Config().override(
        markets=("AA",),
        n_steps_in=4,
        patch_len=2,
        patch_stride=2,
        horizon=2,
        n_tickers_per_sample=3,
        batch_size=8,
        artifact_root=str(tmp_path),
    )
    root = tmp_path / "panel"
    market = root / "AA"
    market.mkdir(parents=True)
    returns = np.arange(6 * 12, dtype=np.float32).reshape(6, 12) / 100
    raw = returns.copy()
    window_valid = np.ones((6, 12), dtype=bool)
    np.save(market / "returns.npy", returns.astype(np.float16))
    np.save(market / "raw_returns_pct.npy", raw.astype(np.float16))
    np.save(market / "window_valid.npy", window_valid)
    np.save(market / "global_ids.npy", np.arange(20, 26, dtype=np.int32))
    ticker_table = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(6)],
        "market": "AA",
        "local_id": np.arange(6),
        "global_id": np.arange(20, 26),
        "eligible_train": True,
        "eligible_val": True,
        "eligible_test": True,
        "always_eligible": [True] * 6,
        "is_mag7": [False, False, True, False, False, False],
    })
    ticker_table.to_csv(market / "tickers.csv", index=False)
    for split in ("train", "val", "test"):
        days = np.tile(np.array([5, 6, 7, 8, 9], dtype=np.int16), 6)
        offsets = np.arange(0, 31, 5, dtype=np.int64)
        np.save(market / f"anchor_{split}_days.npy", days)
        np.save(market / f"anchor_{split}_offsets.npy", offsets)
    (root / "meta.json").write_text(json.dumps({"markets": [{"market": "AA"}]}))
    return cfg, root


def test_sampler_keeps_target_once_and_uses_same_market(tmp_path):
    cfg, root = make_panel(tmp_path)
    sampler = GlobalBasketSampler(cfg, root)
    batch = sampler.sample_batch("train", 8, np.random.default_rng(7))

    assert batch["inputs"].shape == (8, 3, 4)
    assert batch["ticker_ids"].min() >= 20
    assert batch["ticker_ids"].max() <= 25
    selected = batch["ticker_ids"][np.arange(8), batch["target_position"]]
    for row, target in enumerate(selected):
        assert np.count_nonzero(batch["ticker_ids"][row] == target) == 1
    assert np.isfinite(batch["inputs"]).all()
    assert np.isfinite(batch["magnitude_pct"]).all()


def test_fixed_seed_and_prefetch_are_deterministic(tmp_path):
    cfg, root = make_panel(tmp_path)
    sampler = GlobalBasketSampler(cfg, root)
    first = list(prefetch_batches(sampler.batches("val", 4, 2, seed=11), depth=1))
    second = list(sampler.batches("val", 4, 2, seed=11))

    for left, right in zip(first, second):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


def test_stratified_validation_reserves_mag7_rows(tmp_path):
    cfg, root = make_panel(tmp_path)
    sampler = GlobalBasketSampler(cfg, root)
    batch = sampler.sample_batch(
        "val", 20, np.random.default_rng(19), stratified=True
    )

    assert batch["is_mag7"].sum() >= 1