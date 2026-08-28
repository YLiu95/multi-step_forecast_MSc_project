from __future__ import annotations

from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import Config


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    deterministic: bool

    @nn.compact
    def __call__(self, values: jax.Array, unused: None) -> tuple[jax.Array, None]:
        normalized = nn.LayerNorm(dtype=jnp.float32)(values)
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(normalized, normalized, deterministic=self.deterministic)
        values = values + nn.Dropout(self.dropout)(attended, deterministic=self.deterministic)
        normalized = nn.LayerNorm(dtype=jnp.float32)(values)
        hidden = nn.Dense(self.d_ff, dtype=jnp.bfloat16, param_dtype=jnp.float32)(normalized)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(self.dropout)(hidden, deterministic=self.deterministic)
        hidden = nn.Dense(self.d_model, dtype=jnp.bfloat16, param_dtype=jnp.float32)(hidden)
        values = values + nn.Dropout(self.dropout)(hidden, deterministic=self.deterministic)
        return values, None


class CrossTickerPatchTransformer(nn.Module):
    cfg: Config
    n_tickers: int

    def _stack(self, values: jax.Array, depth: int, name: str,
               deterministic: bool) -> jax.Array:
        block = nn.remat(EncoderBlock, prevent_cse=False) if self.cfg.remat else EncoderBlock
        scanned = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=depth,
        )
        values, _ = scanned(
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            d_ff=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            deterministic=deterministic,
            name=name,
        )(values, None)
        return values

    @nn.compact
    def __call__(self, inputs: jax.Array, ticker_ids: jax.Array,
                 target_position: jax.Array, *, deterministic: bool) -> dict[str, jax.Array]:
        batch_size, basket_size, window_length = inputs.shape
        if window_length != self.cfg.n_steps_in:
            raise ValueError(f"Expected window {self.cfg.n_steps_in}, got {window_length}")

        patches = inputs.reshape(
            batch_size, basket_size, self.cfg.n_patches, self.cfg.patch_len
        )
        tokens = nn.Dense(
            self.cfg.d_model,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            name="patch_projection",
        )(patches)
        patch_position = self.param(
            "patch_position",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.cfg.n_patches, self.cfg.d_model),
        )
        ticker_embedding = nn.Embed(
            self.n_tickers,
            self.cfg.d_model,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            name="ticker_embedding",
        )
        ticker_identity = ticker_embedding(ticker_ids)
        roles = jnp.zeros_like(ticker_ids).at[
            jnp.arange(batch_size), target_position
        ].set(1)
        role_identity = nn.Embed(
            2,
            self.cfg.d_model,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            name="role_embedding",
        )(roles)
        tokens = (
            tokens
            + patch_position.astype(jnp.bfloat16)
            + ticker_identity[:, :, None, :]
            + role_identity[:, :, None, :]
        )
        tokens = tokens.reshape(
            batch_size * basket_size, self.cfg.n_patches, self.cfg.d_model
        )
        tokens = self._stack(tokens, self.cfg.temporal_depth, "temporal_blocks", deterministic)
        ticker_states = nn.LayerNorm(dtype=jnp.float32, name="temporal_norm")(tokens)
        ticker_states = ticker_states.mean(axis=1).reshape(
            batch_size, basket_size, self.cfg.d_model
        ).astype(jnp.bfloat16)
        ticker_states = self._stack(
            ticker_states, self.cfg.cross_ticker_depth, "cross_ticker_blocks", deterministic
        )
        ticker_states = nn.LayerNorm(dtype=jnp.float32, name="cross_ticker_norm")(
            ticker_states
        )

        row = jnp.arange(batch_size)
        target_state = ticker_states[row, target_position]
        target_id = ticker_ids[row, target_position]
        conditioned = jnp.concatenate((target_state, ticker_embedding(target_id)), axis=-1)
        conditioned = nn.LayerNorm(dtype=jnp.float32, name="head_norm")(conditioned)

        magnitude_hidden = nn.Dense(
            self.cfg.d_model, dtype=jnp.bfloat16, name="magnitude_hidden"
        )(conditioned)
        magnitude_hidden = nn.gelu(magnitude_hidden)
        magnitude_hidden = nn.Dropout(self.cfg.dropout)(
            magnitude_hidden, deterministic=deterministic
        )
        magnitude = nn.Dense(1, dtype=jnp.float32, name="magnitude_output")(
            magnitude_hidden
        ).squeeze(-1)

        direction_hidden = nn.Dense(
            self.cfg.d_model, dtype=jnp.bfloat16, name="direction_hidden"
        )(conditioned)
        direction_hidden = nn.gelu(direction_hidden)
        direction_hidden = nn.Dropout(self.cfg.dropout)(
            direction_hidden, deterministic=deterministic
        )
        direction = nn.Dense(1, dtype=jnp.float32, name="direction_output")(
            direction_hidden
        ).squeeze(-1)
        return {
            "magnitude_pct": jax.nn.softplus(magnitude),
            "direction_logits": direction,
        }


def count_parameters(variables: Mapping) -> int:
    return sum(value.size for value in jax.tree.leaves(variables["params"]))