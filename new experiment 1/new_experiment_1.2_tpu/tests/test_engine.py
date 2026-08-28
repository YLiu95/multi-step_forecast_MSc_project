import json

import jax
import jax.numpy as jnp
import numpy as np

from src.config import Config
from src.engine import (
    build_train_step,
    create_train_state,
    read_loss_weights,
    restore_state,
    save_state,
)
from src.model import CrossTickerPatchTransformer


def tiny_config(tmp_path) -> Config:
    return Config().override(
        n_steps_in=16,
        patch_len=4,
        patch_stride=4,
        n_tickers_per_sample=4,
        d_model=16,
        n_heads=4,
        d_ff=32,
        temporal_depth=1,
        cross_ticker_depth=1,
        dropout=0.0,
        batch_size=8,
        epochs=2,
        steps_per_epoch=2,
        warmup_epochs=1,
        artifact_root=str(tmp_path),
    )


def test_compiled_train_step_and_checkpoint_round_trip(tmp_path):
    cfg = tiny_config(tmp_path)
    model = CrossTickerPatchTransformer(cfg, n_tickers=20)
    state, schedule = create_train_state(cfg, model, jax.random.key(0))
    batch = {
        "inputs": jnp.ones((8, 4, 16), dtype=jnp.float32),
        "ticker_ids": jnp.tile(jnp.arange(4), (8, 1)),
        "target_position": jnp.zeros(8, dtype=jnp.int32),
        "magnitude_pct": jnp.ones(8),
        "direction": jnp.ones(8),
    }
    train_step = build_train_step(cfg, schedule)
    updated, metrics = train_step(
        state, batch, jax.random.key(1), jnp.array([0.7, 0.3])
    )
    jax.block_until_ready(metrics)

    assert int(updated.step) == 1
    assert np.isfinite(float(metrics["loss"]))
    path = tmp_path / "state.msgpack"
    save_state(updated, path)
    restored = restore_state(state, path)
    assert int(restored.step) == 1


def test_live_loss_weights_are_normalized(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.paths["loss_weights"].write_text(json.dumps({
        "magnitude_loss_weight": 2.0,
        "direction_loss_weight": 1.0,
    }))
    weights = read_loss_weights(cfg)
    np.testing.assert_allclose(weights, [2 / 3, 1 / 3])