from __future__ import annotations

import torch
import torch.nn.functional as functional

from .config import Config


class DualTaskLoss(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) \
            -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        magnitude = functional.huber_loss(
            prediction["magnitude_pct"], batch["magnitude_pct"],
            delta=self.cfg.magnitude_huber_delta_pct)
        direction = functional.binary_cross_entropy_with_logits(
            prediction["direction_logits"], batch["direction"])
        total = (self.cfg.magnitude_loss_weight * magnitude
                 + self.cfg.direction_loss_weight * direction)
        return total, {"magnitude": magnitude.detach(), "direction": direction.detach()}


@torch.no_grad()
def batch_metric_sums(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) \
        -> dict[str, torch.Tensor]:
    magnitude = prediction["magnitude_pct"].float()
    target_magnitude = batch["magnitude_pct"].float()
    probability = prediction["direction_logits"].float().sigmoid()
    target_direction = batch["direction"].float()
    count = torch.tensor(float(len(magnitude)), device=magnitude.device)
    return {
        "count": count,
        "absolute_error": (magnitude - target_magnitude).abs().sum(),
        "squared_error": ((magnitude - target_magnitude) ** 2).sum(),
        "direction_correct": ((probability >= 0.5) == target_direction.bool()).sum(),
        "brier": ((probability - target_direction) ** 2).sum(),
        "positive": target_direction.sum(),
        "predicted_positive": probability.sum(),
        "magnitude_sum": magnitude.sum(),
        "magnitude_sq_sum": (magnitude ** 2).sum(),
    }


def finalize_metrics(sums: dict[str, torch.Tensor]) -> dict[str, float]:
    count = sums["count"].clamp_min(1)
    mean = sums["magnitude_sum"] / count
    variance = (sums["magnitude_sq_sum"] / count - mean ** 2).clamp_min(0)
    return {
        "magnitude_mae_pct": (sums["absolute_error"] / count).item(),
        "magnitude_mae_bp": (100 * sums["absolute_error"] / count).item(),
        "magnitude_rmse_pct": torch.sqrt(sums["squared_error"] / count).item(),
        "direction_accuracy": (sums["direction_correct"] / count).item(),
        "direction_brier": (sums["brier"] / count).item(),
        "actual_up_rate": (sums["positive"] / count).item(),
        "predicted_up_probability": (sums["predicted_positive"] / count).item(),
        "predicted_magnitude_std_pct": torch.sqrt(variance).item(),
    }