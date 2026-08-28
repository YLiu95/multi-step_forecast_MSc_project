import jax
import jax.numpy as jnp

from src.config import Config
from src.model import CrossTickerPatchTransformer, count_parameters


def tiny_config() -> Config:
    return Config().override(
        n_steps_in=16,
        patch_len=4,
        patch_stride=4,
        n_tickers_per_sample=4,
        d_model=16,
        n_heads=4,
        d_ff=32,
        temporal_depth=2,
        cross_ticker_depth=2,
        dropout=0.0,
        batch_size=8,
    )


def test_model_outputs_have_expected_shape_and_range():
    cfg = tiny_config()
    model = CrossTickerPatchTransformer(cfg, n_tickers=12)
    inputs = jnp.ones((2, 4, 16), dtype=jnp.float32)
    ticker_ids = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
    target_position = jnp.array([2, 0], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        inputs,
        ticker_ids,
        target_position,
        deterministic=True,
    )
    prediction = model.apply(
        variables, inputs, ticker_ids, target_position, deterministic=True
    )

    assert prediction["magnitude_pct"].shape == (2,)
    assert prediction["direction_logits"].shape == (2,)
    assert jnp.all(prediction["magnitude_pct"] >= 0)
    assert count_parameters(variables) > 0


def test_config_rejects_invalid_production_shapes():
    cfg = tiny_config().override(patch_stride=2)
    try:
        cfg.validate()
    except ValueError as error:
        assert "non-overlapping" in str(error)
    else:
        raise AssertionError("Expected invalid patch stride to fail")