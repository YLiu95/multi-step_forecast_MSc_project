from __future__ import annotations

import numpy as np
import torch

from .config import Config


def _upload(array, dtype: torch.dtype, device: torch.device,
            chunk_rows: int = 256) -> torch.Tensor:
    output = torch.empty(array.shape, dtype=dtype, device=device)
    for start in range(0, array.shape[0], chunk_rows):
        block = np.array(array[start:start + chunk_rows], copy=True, order="C")
        output[start:start + chunk_rows] = torch.from_numpy(block).to(device, dtype=dtype)
    return output


class GPUBasketPanel:
    """GPU-resident return panel that constructs baskets without CPU workers."""

    def __init__(self, cfg: Config, arrays: dict, anchors: dict,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.returns = _upload(arrays["returns"], torch.float16, device)
        self.raw_returns_pct = _upload(arrays["raw_returns_pct"], torch.float16, device)
        self.window_valid = _upload(arrays["window_valid"], torch.bool, device)
        self.target_indices = _upload(arrays["target_indices"], torch.int64, device)
        self.anchors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.groups: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for split in ("train", "val", "test"):
            ticker = _upload(anchors[f"{split}_i"].astype(np.int64), torch.int64, device)
            day = _upload(anchors[f"{split}_t"].astype(np.int64), torch.int64, device)
            self.anchors[split] = ticker, day
            unique, counts = torch.unique_consecutive(ticker, return_counts=True)
            starts = torch.cat((torch.zeros(1, device=device, dtype=torch.int64),
                                counts.cumsum(0)[:-1]))
            self.groups[split] = unique, torch.stack((starts, counts), dim=1)
        self.input_offsets = torch.arange(-cfg.n_steps_in + 1, 1, device=device)
        self.future_offsets = torch.arange(1, cfg.horizon + 1, device=device)

    @property
    def n_tickers(self) -> int:
        return self.returns.shape[0]

    def vram_gb(self) -> float:
        tensors = (self.returns, self.raw_returns_pct, self.window_valid)
        return sum(t.numel() * t.element_size() for t in tensors) / 1e9

    def _balanced_anchor_indices(self, split: str, batch_size: int,
                                 generator: torch.Generator) -> torch.Tensor:
        _, bounds = self.groups[split]
        group = torch.randint(0, len(bounds), (batch_size,), device=self.device,
                              generator=generator)
        starts, counts = bounds[group, 0], bounds[group, 1]
        within = (torch.rand(batch_size, device=self.device, generator=generator)
                  * counts).long()
        return starts + within

    def gather(self, target_ticker: torch.Tensor, anchor_day: torch.Tensor,
               generator: torch.Generator) -> dict[str, torch.Tensor]:
        batch_size = target_ticker.numel()
        basket_size = self.cfg.n_tickers_per_sample
        if basket_size > self.n_tickers:
            raise ValueError(f"basket size {basket_size} exceeds universe {self.n_tickers}")

        # Random scores + top-k samples unique context tickers uniformly from
        # those with a complete 256-day window at each sample's anchor date.
        valid = self.window_valid[:, anchor_day].T
        valid.scatter_(1, target_ticker[:, None], False)
        available = valid.sum(1)
        if torch.any(available < basket_size - 1):
            minimum = int(available.min().item())
            raise RuntimeError(f"Only {minimum} valid context tickers for at least one anchor")
        scores = torch.rand((batch_size, self.n_tickers), device=self.device,
                            generator=generator)
        scores.masked_fill_(~valid, -1.0)
        context = scores.topk(basket_size - 1, dim=1).indices

        target_position = torch.randint(0, basket_size, (batch_size,),
                                        device=self.device, generator=generator)
        basket = torch.empty((batch_size, basket_size), dtype=torch.long,
                             device=self.device)
        positions = torch.arange(basket_size, device=self.device).expand(batch_size, -1)
        non_target = positions != target_position[:, None]
        context_position = positions - (positions > target_position[:, None]).long()
        arranged_context = context.gather(1, context_position.clamp_max(basket_size - 2))
        basket[non_target] = arranged_context[non_target]
        basket.scatter_(1, target_position[:, None], target_ticker[:, None])

        input_days = anchor_day[:, None, None] + self.input_offsets[None, None, :]
        inputs = self.returns[basket[:, :, None], input_days]
        future_days = anchor_day[:, None] + self.future_offsets[None, :]
        cumulative_pct = self.raw_returns_pct[target_ticker[:, None], future_days].float().sum(1)
        return {
            "x": inputs.float(),
            "ticker_ids": basket,
            "target_position": target_position,
            "magnitude_pct": cumulative_pct.abs(),
            "direction": (cumulative_pct > 0).float(),
            "signed_return_pct": cumulative_pct,
        }

    def random_batches(self, split: str, batch_size: int, n_batches: int,
                       generator: torch.Generator):
        ticker, day = self.anchors[split]
        for _ in range(n_batches):
            selected = self._balanced_anchor_indices(split, batch_size, generator)
            yield self.gather(ticker[selected], day[selected], generator)

    def fixed_batches(self, split: str, batch_size: int, n_batches: int,
                      seed: int):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        yield from self.random_batches(split, batch_size, n_batches, generator)