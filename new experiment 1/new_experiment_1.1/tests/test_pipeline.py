import unittest

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.dataset import GPUBasketPanel
from src.losses import DualTaskLoss
from src.model import build_model
from src.prepare_data import build_target_anchors


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(
            n_steps_in=16,
            patch_len=4,
            patch_stride=4,
            horizon=3,
            n_tickers_per_sample=6,
            d_model=48,
            n_heads=6,
            d_ff=96,
            temporal_depth=1,
            cross_ticker_depth=1,
            batch_size=4,
            compile_model=False,
        )

    def test_anchor_excludes_missing_bars(self):
        dates = pd.bdate_range("2018-01-01", periods=1200)
        valid = np.ones((2, len(dates)), dtype=bool)
        valid[1, 400] = False
        anchors = build_target_anchors(
            self.cfg, valid, np.ones_like(valid, dtype=np.float32), dates,
            np.array([1], dtype=np.int32))
        for split in ("train", "val", "test"):
            for ticker, day in zip(anchors[f"{split}_i"], anchors[f"{split}_t"]):
                self.assertTrue(valid[ticker, day - 15:day + 4].all())

    def test_sample_forward_loss_backward(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(size=(20, 50)).astype(np.float16)
        arrays = {
            "returns": returns,
            "raw_returns_pct": returns,
            "window_valid": np.ones((20, 50), dtype=bool),
            "target_indices": np.array([2, 7], dtype=np.int32),
        }
        anchors = {}
        for split in ("train", "val", "test"):
            anchors[f"{split}_i"] = np.array([2, 2, 7, 7], dtype=np.int32)
            anchors[f"{split}_t"] = np.array([20, 25, 20, 25], dtype=np.int32)
        panel = GPUBasketPanel(self.cfg, arrays, anchors, torch.device("cpu"))
        generator = torch.Generator().manual_seed(8)
        batch = next(panel.random_batches("train", 4, 1, generator))
        selected = batch["ticker_ids"].gather(
            1, batch["target_position"][:, None]).squeeze(1)
        for row, target in zip(batch["ticker_ids"], selected):
            self.assertEqual(int((row == target).sum()), 1)

        model = build_model(self.cfg, panel.n_tickers)
        prediction = model(batch["x"], batch["ticker_ids"],
                           batch["target_position"])
        loss, _ = DualTaskLoss(self.cfg)(prediction, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(prediction["magnitude_pct"].shape), (4,))


if __name__ == "__main__":
    unittest.main()