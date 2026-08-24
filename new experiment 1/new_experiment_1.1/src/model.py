from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Config


class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attention = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     need_weights=False)
        values = values + self.dropout(attended)
        return values + self.dropout(self.feed_forward(self.norm2(values)))


class CrossTickerPatchTransformer(nn.Module):
    def __init__(self, cfg: Config, n_tickers: int):
        super().__init__()
        self.cfg = cfg
        self.patch_projection = nn.Linear(cfg.patch_len, cfg.d_model)
        self.patch_position = nn.Parameter(torch.zeros(1, 1, cfg.n_patches, cfg.d_model))
        self.ticker_embedding = nn.Embedding(n_tickers, cfg.d_model)
        self.role_embedding = nn.Embedding(2, cfg.d_model)
        self.temporal_blocks = nn.ModuleList(
            EncoderBlock(cfg) for _ in range(cfg.temporal_depth))
        self.cross_ticker_blocks = nn.ModuleList(
            EncoderBlock(cfg) for _ in range(cfg.cross_ticker_depth))
        self.temporal_norm = nn.LayerNorm(cfg.d_model)
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.head_norm = nn.LayerNorm(cfg.d_model * 2)
        self.magnitude_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1), nn.Softplus())
        self.direction_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1))
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        scale = 1.0 / math.sqrt(2 * (cfg.temporal_depth + cfg.cross_ticker_depth))
        for block in [*self.temporal_blocks, *self.cross_ticker_blocks]:
            block.feed_forward[-1].weight.data.mul_(scale)
            block.attention.out_proj.weight.data.mul_(scale)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor,
                target_position: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, basket_size, _ = x.shape
        patches = x.unfold(2, self.cfg.patch_len, self.cfg.patch_stride)
        tokens = self.patch_projection(patches)
        ticker_identity = self.ticker_embedding(ticker_ids)
        roles = torch.zeros_like(ticker_ids)
        roles.scatter_(1, target_position[:, None], 1)
        tokens = (tokens + self.patch_position
                  + ticker_identity[:, :, None, :]
                  + self.role_embedding(roles)[:, :, None, :])
        tokens = tokens.reshape(batch_size * basket_size, self.cfg.n_patches,
                                self.cfg.d_model)
        for block in self.temporal_blocks:
            tokens = block(tokens)
        ticker_tokens = self.temporal_norm(tokens).mean(1).reshape(
            batch_size, basket_size, self.cfg.d_model)
        for block in self.cross_ticker_blocks:
            ticker_tokens = block(ticker_tokens)
        ticker_tokens = self.cross_norm(ticker_tokens)

        batch_index = torch.arange(batch_size, device=x.device)
        target_state = ticker_tokens[batch_index, target_position]
        target_id = ticker_ids[batch_index, target_position]
        # Directly concatenate identity at the heads. Even if attention learns
        # to ignore identity, both predictions remain explicitly conditioned on it.
        conditioned = self.head_norm(torch.cat(
            (target_state, self.ticker_embedding(target_id)), dim=-1))
        return {
            "magnitude_pct": self.magnitude_head(conditioned).squeeze(-1),
            "direction_logits": self.direction_head(conditioned).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def param_groups(self, weight_decay: float) -> list[dict]:
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if parameter.ndim <= 1 or "embedding" in name or "position" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def build_model(cfg: Config, n_tickers: int) -> CrossTickerPatchTransformer:
    return CrossTickerPatchTransformer(cfg, n_tickers)