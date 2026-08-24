"""PatchTST-style Transformer encoder for multi-step forecasting.

Why a patch Transformer and not the LSTM from the original notebook
-------------------------------------------------------------------
An LSTM over a 256-day window is 256 *sequentially dependent* small matrix
multiplies. The GPU cannot start step t+1 until step t is finished, so a T4
spends most of its time on kernel-launch latency and its tensor cores are
essentially idle -- you will typically see 10-20% utilisation no matter how
large the batch is.

Patching removes the sequential dependency:

    (B, 256, 18)  --reshape-->  (B, 32 patches, 8 days x 18 feats)
                  --linear--->  (B, 32, d_model)
                  --attention-> all 32 tokens processed in PARALLEL

Every layer is now a large batched GEMM, which is exactly what fp16 tensor
cores are built for. The same parameter budget therefore delivers far more
useful FLOPs per second, and both GPUs stay saturated under DDP.

Patching also shortens the sequence by `patch_stride`, so attention costs
O((L/8)^2) instead of O(L^2) -- a 64x saving that lets us afford a deep model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Config


class PatchEmbed(nn.Module):
    """Slice the look-back into overlapping/adjacent patches and project them.

    Unlike vanilla PatchTST we mix the feature channels *inside* the patch
    projection. PatchTST keeps channels independent because its channels are
    parallel copies of the same kind of series. Ours are heterogeneous
    (a return, a volatility, a market-breadth reading), so letting the first
    layer combine them is strictly more expressive and costs one small matrix.
    """

    def __init__(self, cfg: Config, n_features: int):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.stride = cfg.patch_stride
        self.n_patches = cfg.n_patches
        self.proj = nn.Linear(cfg.patch_len * n_features, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, L, F)
        # unfold gives a view, so no data is copied here.
        p = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        p = p.permute(0, 1, 3, 2).flatten(2)              # (B, n_patch, P*F)
        return self.proj(p) + self.pos


class EncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer block.

    Pre-LN (norm *before* the sublayer) rather than the original post-LN: it
    keeps the residual stream un-normalised, which makes a 12-layer stack train
    stably without a long warmup and without gradient spikes under fp16.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class InputRegulariser(nn.Module):
    """Break the date-fingerprint shortcut before the model can exploit it.

    Seven of the eighteen channels (market context + calendar) are identical for
    every ticker on a given day, so a 256-day window of them identifies the date
    exactly. There are only ~7,300 training dates and ~1,000 tickers share each
    one, so memorising `date -> future market move` drives the training loss
    down fast and generalises to nothing. That is precisely what the first
    baseline run did: train 0.376 -> 0.326 while val rose 0.444 -> 0.454 and
    rank_ic fell from +0.032 to negative.

    Three defences, all training-time only:

    * `disable_features` permanently zeroes pure date identifiers (day-of-week,
      month) which are almost all fingerprint and almost no signal.
    * `shared_group_dropout` blanks the *entire* shared group for a random half
      of samples. The model must therefore be able to forecast from the
      ticker's own history alone, and can only use market context as a weak
      prior on top.
    * `input_noise` jitters the inputs so an exact window cannot be matched to
      an exact date.
    """

    def __init__(self, cfg: Config, feature_names: tuple[str, ...]):
        super().__init__()
        from .features import SHARED_FEATURES

        self.p_group = cfg.shared_group_dropout
        self.noise = cfg.input_noise

        keep = torch.ones(len(feature_names))
        for i, n in enumerate(feature_names):
            if n in set(cfg.disable_features):
                keep[i] = 0.0
        shared = torch.zeros(len(feature_names))
        for i, n in enumerate(feature_names):
            if n in set(SHARED_FEATURES) and keep[i] > 0:
                shared[i] = 1.0
        self.register_buffer("keep", keep.view(1, 1, -1))
        self.register_buffer("shared", shared.view(1, 1, -1))
        self.n_disabled = int((keep == 0).sum())
        self.n_shared = int(shared.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.keep
        if not self.training:
            return x
        if self.p_group > 0 and self.n_shared:
            # one Bernoulli draw per SAMPLE, applied to the whole shared group -
            # dropping channels independently would leave enough of the
            # fingerprint intact to still identify the date.
            b = x.shape[0]
            drop = (torch.rand(b, 1, 1, device=x.device) < self.p_group).float()
            x = x * (1.0 - drop * self.shared)
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        return x


class PatchForecaster(nn.Module):
    """(B, L, F) -> (B, H) point forecast, or (B, H, Q) quantile forecast."""

    def __init__(self, cfg: Config, n_features: int,
                 feature_names: tuple[str, ...] | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_out = cfg.n_outputs_per_step
        self.horizon = cfg.n_steps_out

        if feature_names is None:
            from .features import FEATURE_NAMES
            feature_names = FEATURE_NAMES
        self.reg = InputRegulariser(cfg, tuple(feature_names))
        self.embed = PatchEmbed(cfg, n_features)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head_drop = nn.Dropout(cfg.head_dropout)
        # Flatten-head: every patch token contributes to every horizon step.
        # A single Linear here beats an autoregressive decoder for multi-step
        # forecasting because it cannot accumulate its own errors.
        self.head = nn.Linear(cfg.n_patches * cfg.d_model,
                              cfg.n_steps_out * self.n_out)

        self.apply(self._init)
        # Scale residual-branch output projections by 1/sqrt(2*depth) so the
        # variance of the residual stream does not grow with depth.
        scale = 1.0 / math.sqrt(2 * cfg.depth)
        for blk in self.blocks:
            blk.ff[-1].weight.data.mul_(scale)
            blk.attn.out_proj.weight.data.mul_(scale)
        # Start with near-zero predictions: for a near-zero-mean target that is
        # already the optimal constant, so the model begins at a sane loss.
        nn.init.zeros_(self.head.bias)
        self.head.weight.data.mul_(0.01)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(self.reg(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).flatten(1)
        out = self.head(self.head_drop(h))
        if self.n_out == 1:
            return out                                   # (B, H)
        return out.view(-1, self.horizon, self.n_out)    # (B, H, Q)

    # ------------------------------------------------------------------ utils
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, weight_decay: float):
        """Never weight-decay biases, LayerNorm gains or the positional table.

        Decaying a 1-D parameter shrinks a *scale*, which just fights the
        optimiser. Only weight matrices should be regularised.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith("pos") else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def build_model(cfg: Config, n_features: int,
                feature_names: tuple[str, ...] | None = None) -> PatchForecaster:
    return PatchForecaster(cfg, n_features, feature_names)
