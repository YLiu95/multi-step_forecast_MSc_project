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
        "target_magnitude_sum": target_magnitude.sum(),
        "target_magnitude_sq_sum": (target_magnitude ** 2).sum(),
    }


def finalize_metrics(sums: dict[str, torch.Tensor]) -> dict[str, float]:
    count = sums["count"].clamp_min(1)
    mean = sums["magnitude_sum"] / count
    variance = (sums["magnitude_sq_sum"] / count - mean ** 2).clamp_min(0)
    up_rate = sums["positive"] / count
    magnitude_mae = sums["absolute_error"] / count
    zero_magnitude_mae = sums["target_magnitude_sum"] / count
    direction_accuracy = sums["direction_correct"] / count
    majority_accuracy = torch.maximum(up_rate, 1 - up_rate)
    return {
        "magnitude_mae_pct": magnitude_mae.item(),
        "magnitude_mae_bp": (100 * magnitude_mae).item(),
        "magnitude_rmse_pct": torch.sqrt(sums["squared_error"] / count).item(),
        "zero_magnitude_mae_bp": (100 * zero_magnitude_mae).item(),
        "zero_magnitude_rmse_pct": torch.sqrt(
            sums["target_magnitude_sq_sum"] / count).item(),
        "magnitude_mae_improvement_vs_zero_pct": (
            100 * (1 - magnitude_mae / zero_magnitude_mae.clamp_min(1e-12))).item(),
        "direction_accuracy": direction_accuracy.item(),
        "majority_direction_accuracy": majority_accuracy.item(),
        "direction_accuracy_lift_pp": (
            100 * (direction_accuracy - majority_accuracy)).item(),
        "direction_brier": (sums["brier"] / count).item(),
        "prevalence_brier": (up_rate * (1 - up_rate)).item(),
        "actual_up_rate": up_rate.item(),
        "predicted_up_probability": (sums["predicted_positive"] / count).item(),
        "predicted_magnitude_std_pct": torch.sqrt(variance).item(),
    }