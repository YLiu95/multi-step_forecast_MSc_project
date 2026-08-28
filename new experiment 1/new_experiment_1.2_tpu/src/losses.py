from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax


def huber_per_example(prediction: jax.Array, target: jax.Array,
                      delta: float) -> jax.Array:
    difference = jnp.abs(prediction - target)
    quadratic = jnp.minimum(difference, delta)
    linear = difference - quadratic
    return 0.5 * quadratic**2 + delta * linear


def dual_task_loss(prediction: Mapping[str, jax.Array], batch: Mapping[str, jax.Array],
                   weights: jax.Array, huber_delta: float) \
        -> tuple[jax.Array, dict[str, jax.Array]]:
    magnitude_per_example = huber_per_example(
        prediction["magnitude_pct"], batch["magnitude_pct"], huber_delta
    )
    direction_per_example = optax.sigmoid_binary_cross_entropy(
        prediction["direction_logits"], batch["direction"]
    )
    total_per_example = (
        weights[0] * magnitude_per_example + weights[1] * direction_per_example
    )
    return total_per_example.mean(), {
        "total_per_example": total_per_example,
        "magnitude_per_example": magnitude_per_example,
        "direction_per_example": direction_per_example,
        "magnitude": magnitude_per_example.mean(),
        "direction": direction_per_example.mean(),
    }


def empty_metric_sums() -> dict[str, float]:
    return {
        "count": 0.0,
        "loss": 0.0,
        "magnitude_loss": 0.0,
        "direction_loss": 0.0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "direction_correct": 0.0,
        "brier": 0.0,
        "positive": 0.0,
        "predicted_positive": 0.0,
        "magnitude_sum": 0.0,
        "magnitude_sq_sum": 0.0,
        "target_magnitude_sum": 0.0,
        "target_magnitude_sq_sum": 0.0,
    }


def update_metric_sums(sums: dict[str, float], prediction: Mapping[str, np.ndarray],
                       batch: Mapping[str, np.ndarray], losses: Mapping[str, np.ndarray],
                       mask: np.ndarray | None = None) -> None:
    magnitude = np.asarray(prediction["magnitude_pct"], dtype=np.float64)
    target_magnitude = np.asarray(batch["magnitude_pct"], dtype=np.float64)
    probability = 1.0 / (
        1.0 + np.exp(-np.asarray(prediction["direction_logits"], dtype=np.float64))
    )
    target_direction = np.asarray(batch["direction"], dtype=np.float64)
    selected = np.ones(len(magnitude), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not selected.any():
        return
    magnitude = magnitude[selected]
    target_magnitude = target_magnitude[selected]
    probability = probability[selected]
    target_direction = target_direction[selected]
    count = float(selected.sum())
    sums["count"] += count
    sums["loss"] += float(np.asarray(losses["total_per_example"])[selected].sum())
    sums["magnitude_loss"] += float(
        np.asarray(losses["magnitude_per_example"])[selected].sum()
    )
    sums["direction_loss"] += float(
        np.asarray(losses["direction_per_example"])[selected].sum()
    )
    sums["absolute_error"] += float(np.abs(magnitude - target_magnitude).sum())
    sums["squared_error"] += float(np.square(magnitude - target_magnitude).sum())
    sums["direction_correct"] += float(
        ((probability >= 0.5) == target_direction.astype(bool)).sum()
    )
    sums["brier"] += float(np.square(probability - target_direction).sum())
    sums["positive"] += float(target_direction.sum())
    sums["predicted_positive"] += float(probability.sum())
    sums["magnitude_sum"] += float(magnitude.sum())
    sums["magnitude_sq_sum"] += float(np.square(magnitude).sum())
    sums["target_magnitude_sum"] += float(target_magnitude.sum())
    sums["target_magnitude_sq_sum"] += float(np.square(target_magnitude).sum())


def finalize_metrics(sums: Mapping[str, float]) -> dict[str, float]:
    count = max(sums["count"], 1.0)
    predicted_mean = sums["magnitude_sum"] / count
    predicted_variance = max(
        sums["magnitude_sq_sum"] / count - predicted_mean**2, 0.0
    )
    up_rate = sums["positive"] / count
    magnitude_mae = sums["absolute_error"] / count
    zero_magnitude_mae = sums["target_magnitude_sum"] / count
    direction_accuracy = sums["direction_correct"] / count
    majority_accuracy = max(up_rate, 1.0 - up_rate)
    return {
        "count": sums["count"],
        "loss": sums["loss"] / count,
        "magnitude_loss": sums["magnitude_loss"] / count,
        "direction_loss": sums["direction_loss"] / count,
        "magnitude_mae_pct": magnitude_mae,
        "magnitude_mae_bp": 100.0 * magnitude_mae,
        "magnitude_rmse_pct": np.sqrt(sums["squared_error"] / count),
        "zero_magnitude_mae_bp": 100.0 * zero_magnitude_mae,
        "zero_magnitude_rmse_pct": np.sqrt(sums["target_magnitude_sq_sum"] / count),
        "magnitude_mae_improvement_vs_zero_pct": 100.0 * (
            1.0 - magnitude_mae / max(zero_magnitude_mae, 1e-12)
        ),
        "direction_accuracy": direction_accuracy,
        "majority_direction_accuracy": majority_accuracy,
        "direction_accuracy_lift_pp": 100.0 * (
            direction_accuracy - majority_accuracy
        ),
        "direction_brier": sums["brier"] / count,
        "prevalence_brier": up_rate * (1.0 - up_rate),
        "actual_up_rate": up_rate,
        "predicted_up_probability": sums["predicted_positive"] / count,
        "predicted_magnitude_std_pct": np.sqrt(predicted_variance),
    }