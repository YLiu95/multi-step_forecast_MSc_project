"""Losses and evaluation metrics for volatility-scaled multi-step returns.

On the choice of loss
---------------------
Daily equity returns are heavy-tailed: a handful of days in every decade move
10%+. Plain MSE squares those, so a single earnings gap can contribute more
gradient than a thousand ordinary days and the model ends up fitting outliers.

* **Huber** is quadratic inside +-delta and linear outside it, so a 10-sigma day
  contributes a bounded gradient. It is the safe default.
* **Quantile (pinball)** loss predicts the 10th/50th/90th percentile instead of
  a single number, which is a far more honest description of a forecast whose
  signal-to-noise ratio is genuinely low. It is what practitioners want and it
  makes for much stronger dissertation material.

On the metrics
--------------
R^2 against a zero baseline is the honest yardstick for returns. Note the
*sign*: a small **positive** R^2 (0.005-0.02) on out-of-sample equity returns is
a genuinely good result. If you ever see 0.9, you have a look-ahead bug.
`rank_ic` (Spearman correlation between prediction and outcome, computed within
each batch) is the standard cross-sectional metric in quantitative finance.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config


def horizon_weights(cfg: Config, device: torch.device) -> torch.Tensor:
    """Optionally down-weight distant horizons, which are intrinsically noisier."""
    h = torch.arange(cfg.n_steps_out, device=device, dtype=torch.float32)
    if cfg.horizon_decay <= 0:
        w = torch.ones_like(h)
    else:
        w = torch.exp(-cfg.horizon_decay * h)
    return w / w.mean()


class ForecastLoss(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg.loss
        self.register_buffer("q", torch.tensor(cfg.quantiles, dtype=torch.float32))
        self.register_buffer("hw", torch.ones(cfg.n_steps_out))

    def to_device(self, device: torch.device) -> "ForecastLoss":
        self.hw = horizon_weights(self.cfg, device)
        self.q = self.q.to(device)
        return self.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "quantile":
            # pred (B, H, Q), target (B, H) -> broadcast over the quantile axis
            err = target.unsqueeze(-1) - pred
            per = torch.maximum(self.q * err, (self.q - 1.0) * err)  # pinball
            per = per.mean(-1)                                       # (B, H)
        elif self.kind == "mse":
            per = (pred - target) ** 2
        else:
            per = F.huber_loss(pred, target, reduction="none",
                               delta=self.cfg.huber_delta)
        return (per * self.hw).mean()


def point_forecast(cfg: Config, pred: torch.Tensor) -> torch.Tensor:
    """Collapse a quantile output to the median so metrics stay comparable."""
    if cfg.loss != "quantile":
        return pred
    mid = min(range(len(cfg.quantiles)),
              key=lambda i: abs(cfg.quantiles[i] - 0.5))
    return pred[..., mid]


@torch.no_grad()
def metrics(cfg: Config, pred: torch.Tensor, target: torch.Tensor
            ) -> dict[str, float]:
    """All metrics are in volatility units, so they are comparable across tickers."""
    p = point_forecast(cfg, pred).float()
    y = target.float()
    out: dict[str, float] = {}

    out["mae"] = (p - y).abs().mean().item()
    out["rmse"] = torch.sqrt(((p - y) ** 2).mean()).item()

    # R^2 vs the zero forecast. Zero -- not the mean -- is the right baseline:
    # "the price does not move" is the honest naive forecast for a return.
    ss_res = ((p - y) ** 2).sum()
    ss_tot = (y ** 2).sum().clamp_min(1e-12)
    out["r2_vs_zero"] = (1.0 - ss_res / ss_tot).item()

    # Direction of the CUMULATIVE move over the whole horizon -- the quantity a
    # trader actually cares about, and far less noisy than per-day direction.
    cp, cy = p.sum(-1), y.sum(-1)
    out["dir_acc"] = ((cp.sign() == cy.sign()) & (cy != 0)).float().mean().item()

    out["ic"] = _corr(cp, cy)
    out["rank_ic"] = _corr(_rank(cp), _rank(cy))
    out["pred_std"] = p.std().item()
    out["target_std"] = y.std().item()
    # A model that has collapsed to predicting ~0 shows up here as a tiny ratio.
    out["std_ratio"] = out["pred_std"] / max(out["target_std"], 1e-9)
    return out


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return (a @ b / denom).item()


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
    return ranks


@torch.no_grad()
def to_price_path(last_close: torch.Tensor, sigma: torch.Tensor,
                  y_scaled: torch.Tensor) -> torch.Tensor:
    """Undo the volatility scaling and the log-return transform.

        P_{t+h} = C_t * exp( sigma_t * cumsum(y_scaled)_h )

    This is how a scale-free training target becomes a dollar price chart.
    """
    cum = (y_scaled * sigma.unsqueeze(-1)).cumsum(-1)
    return last_close.unsqueeze(-1) * torch.exp(cum)
