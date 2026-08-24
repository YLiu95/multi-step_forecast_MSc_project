"""GPU-resident panel sampler -- the reason this trains fast on a 4-core box.

Why not a normal `DataLoader`?
------------------------------
The obvious design is a `Dataset` whose `__getitem__` slices a 256-day window
out of a numpy array, wrapped in a `DataLoader` with several workers. On this
machine that design fails badly:

* Materialising every window up front is impossible: ~15M windows x 256 steps
  x 18 features x 4 bytes is roughly **250 GB**. There is 31 GB of RAM.
* Slicing lazily on CPU works, but at the throughput two T4s want (~15k
  samples/s) it needs to copy ~1 GB/s through 4 CPU cores and then over PCIe.
  The GPUs would sit idle waiting, and the worker processes are exactly the
  kind of thing that blows up RAM and kills the environment.

The fix: the *entire* feature panel is only

    3,000 tickers x 9,200 days x 18 features x 2 bytes (fp16) = 1.0 GB

which fits in a T4's 16 GB with room to spare. So we upload it once, and build
each batch by **advanced-indexing on the GPU**. There are zero worker
processes, zero CPU RAM pressure, and zero host-to-device copies in the loop.

A batch is built from two integer vectors:

    X[b] = feat[ticker[b], t[b]-L+1 : t[b]+1]          -> (B, L, F)
    Y[b] = ret [ticker[b], t[b]+1   : t[b]+H+1] / sig[ticker[b], t[b]]

The division by the trailing volatility *at the anchor* is what makes a $3
stock and a $500 stock contribute comparable gradients.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import Config


class GPUPanel:
    """Holds the whole dataset in VRAM and emits batches by index arithmetic."""

    def __init__(self, cfg: Config, arrays: dict, anchors: dict,
                 device: torch.device, split_seed: int = 0):
        self.cfg = cfg
        self.device = device
        L, H = cfg.n_steps_in, cfg.n_steps_out

        # Chunked upload: np.load(mmap_mode='r') means the array is on disk, so
        # copying it in slices keeps transient CPU RAM to a few tens of MB.
        self.feat = _to_gpu(arrays["feat"], torch.float16, device)
        self.ret = _to_gpu(arrays["ret"], torch.float32, device)
        self.sig = _to_gpu(arrays["sig"], torch.float32, device)

        self.idx = {}
        for split in ("train", "val", "test"):
            i = torch.from_numpy(anchors[f"{split}_i"].astype(np.int32)).to(device)
            t = torch.from_numpy(anchors[f"{split}_t"].astype(np.int32)).to(device)
            self.idx[split] = (i, t)

        # Precomputed offset vectors; broadcasting these against the anchor
        # gives the full window without any Python-level loop.
        self.off_in = torch.arange(-L + 1, 1, device=device, dtype=torch.int32)
        self.off_out = torch.arange(1, H + 1, device=device, dtype=torch.int32)
        self.split_seed = split_seed

    # ------------------------------------------------------------------ info
    @property
    def n_features(self) -> int:
        return self.feat.shape[2]

    def size(self, split: str) -> int:
        return int(self.idx[split][0].numel())

    def vram_gb(self) -> float:
        return sum(t.numel() * t.element_size()
                   for t in (self.feat, self.ret, self.sig)) / 1e9

    # ---------------------------------------------------------------- gather
    def gather(self, ti: torch.Tensor, tt: torch.Tensor):
        """(anchor tickers, anchor days) -> (X, Y, sigma) on the GPU."""
        rows = ti.long().unsqueeze(1)                      # (B, 1)
        cols_in = (tt.unsqueeze(1) + self.off_in).long()   # (B, L)
        cols_out = (tt.unsqueeze(1) + self.off_out).long()  # (B, H)

        x = self.feat[rows, cols_in].float()               # (B, L, F)
        sigma = self.sig[ti.long(), tt.long()].unsqueeze(1)  # (B, 1)
        y = self.ret[rows, cols_out] / sigma                # (B, H)
        y = y.clamp(-self.cfg.clip_sigma, self.cfg.clip_sigma)
        return x, y, sigma.squeeze(1)

    # --------------------------------------------------------------- batches
    def random_batches(self, split: str, batch_size: int, n_batches: int,
                       generator: torch.Generator):
        """Random sampling with replacement -- the standard choice when the
        anchor pool is far larger than one epoch's compute budget."""
        i, t = self.idx[split]
        n = i.numel()
        for _ in range(n_batches):
            sel = torch.randint(0, n, (batch_size,), device=self.device,
                                generator=generator)
            yield self.gather(i[sel], t[sel])

    def sequential_batches(self, split: str, batch_size: int,
                           max_batches: int | None = None,
                           rank: int = 0, world_size: int = 1):
        """Deterministic sharded pass -- used for validation so the number is
        comparable between epochs and across ranks."""
        i, t = self.idx[split]
        n = i.numel()
        # A fixed stride keeps each rank's shard spread across the whole period
        # instead of giving rank 0 only the earliest dates.
        sel_all = torch.arange(rank, n, world_size, device=self.device)
        if max_batches is not None:
            want = max_batches * batch_size
            if sel_all.numel() > want:
                step = sel_all.numel() // want
                sel_all = sel_all[::step][:want]
        for s in range(0, sel_all.numel(), batch_size):
            sel = sel_all[s:s + batch_size]
            if sel.numel() == 0:
                break
            yield self.gather(i[sel], t[sel])


def _to_gpu(arr, dtype: torch.dtype, device: torch.device,
            chunk_rows: int = 256) -> torch.Tensor:
    """Copy a (possibly memory-mapped) numpy array to the GPU in row chunks."""
    out = torch.empty(arr.shape, dtype=dtype, device=device)
    for s in range(0, arr.shape[0], chunk_rows):
        block = np.ascontiguousarray(arr[s:s + chunk_rows])
        out[s:s + chunk_rows] = torch.from_numpy(block).to(device, dtype=dtype)
        del block
    return out
