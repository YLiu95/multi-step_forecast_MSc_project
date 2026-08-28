import jax.numpy as jnp
import numpy as np

from src.losses import (
    dual_task_loss,
    empty_metric_sums,
    finalize_metrics,
    update_metric_sums,
)


def test_dual_loss_and_metrics_match_hand_calculation():
    prediction = {
        "magnitude_pct": jnp.array([1.0, 3.0]),
        "direction_logits": jnp.array([10.0, -10.0]),
    }
    batch = {
        "magnitude_pct": jnp.array([1.0, 1.0]),
        "direction": jnp.array([1.0, 0.0]),
    }
    total, losses = dual_task_loss(prediction, batch, jnp.array([0.7, 0.3]), 1.0)
    sums = empty_metric_sums()
    update_metric_sums(
        sums,
        {key: np.asarray(value) for key, value in prediction.items()},
        {key: np.asarray(value) for key, value in batch.items()},
        {key: np.asarray(value) for key, value in losses.items()},
    )
    metrics = finalize_metrics(sums)

    assert np.isclose(float(total), 0.525, atol=1e-3)
    assert metrics["magnitude_mae_pct"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["count"] == 2.0


def test_masked_metrics_only_count_selected_rows():
    prediction = {
        "magnitude_pct": np.array([1.0, 9.0]),
        "direction_logits": np.array([9.0, 9.0]),
    }
    batch = {"magnitude_pct": np.array([1.0, 0.0]), "direction": np.array([1.0, 0.0])}
    losses = {
        "total_per_example": np.array([0.0, 9.0]),
        "magnitude_per_example": np.array([0.0, 9.0]),
        "direction_per_example": np.array([0.0, 9.0]),
    }
    sums = empty_metric_sums()
    update_metric_sums(sums, prediction, batch, losses, mask=np.array([True, False]))
    metrics = finalize_metrics(sums)

    assert metrics["count"] == 1.0
    assert metrics["magnitude_mae_pct"] == 0.0
    assert metrics["direction_accuracy"] == 1.0