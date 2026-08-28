from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import flax
from flax import serialization, traverse_util
from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config
from .losses import dual_task_loss, empty_metric_sums, finalize_metrics, update_metric_sums
from .model import CrossTickerPatchTransformer


class ExperimentTrainState(train_state.TrainState):
    ema_params: Any


def learning_rate_schedule(cfg: Config) -> optax.Schedule:
    warmup_steps = cfg.warmup_epochs * cfg.steps_per_epoch
    total_steps = cfg.epochs * cfg.steps_per_epoch
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.learning_rate * cfg.min_lr_fraction,
    )


def weight_decay_mask(params: Any) -> Any:
    is_frozen = isinstance(params, FrozenDict)
    flat = traverse_util.flatten_dict(flax.core.unfreeze(params))
    mask = {}
    for path, value in flat.items():
        joined = "/".join(path).lower()
        excluded = (
            value.ndim <= 1
            or "embedding" in joined
            or "position" in joined
            or "layernorm" in joined
            or "_norm" in joined
        )
        mask[path] = not excluded
    result = traverse_util.unflatten_dict(mask)
    return flax.core.freeze(result) if is_frozen else result


def create_train_state(cfg: Config, model: CrossTickerPatchTransformer,
                       rng: jax.Array) -> tuple[ExperimentTrainState, optax.Schedule]:
    params_key, dropout_key = jax.random.split(rng)
    sample_inputs = jnp.zeros(
        (1, cfg.n_tickers_per_sample, cfg.n_steps_in), dtype=jnp.bfloat16
    )
    sample_tickers = jnp.zeros((1, cfg.n_tickers_per_sample), dtype=jnp.int32)
    sample_target_position = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(
        {"params": params_key, "dropout": dropout_key},
        sample_inputs,
        sample_tickers,
        sample_target_position,
        deterministic=False,
    )
    schedule = learning_rate_schedule(cfg)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clip),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=cfg.weight_decay,
            mask=weight_decay_mask(variables["params"]),
        ),
    )
    state = ExperimentTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        ema_params=variables["params"],
    )
    return state, schedule


def _named_gradient_norm(grads: FrozenDict, prefixes: tuple[str, ...]) -> jax.Array:
    leaves = [
        value
        for path, value in traverse_util.flatten_dict(grads).items()
        if path[0].startswith(prefixes)
    ]
    return optax.global_norm(leaves) if leaves else jnp.array(0.0)


def build_train_step(cfg: Config, schedule: optax.Schedule) -> Callable:
    @jax.jit
    def train_step(state: ExperimentTrainState, batch: dict[str, jax.Array],
                   dropout_key: jax.Array, weights: jax.Array):
        def objective(params):
            prediction = state.apply_fn(
                {"params": params},
                batch["inputs"],
                batch["ticker_ids"],
                batch["target_position"],
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            loss, components = dual_task_loss(
                prediction, batch, weights, cfg.magnitude_huber_delta_pct
            )
            return loss, components

        (loss, components), grads = jax.value_and_grad(objective, has_aux=True)(
            state.params
        )
        gradient_norm = optax.global_norm(grads)
        magnitude_head_norm = _named_gradient_norm(
            grads, ("magnitude_hidden", "magnitude_output")
        )
        direction_head_norm = _named_gradient_norm(
            grads, ("direction_hidden", "direction_output")
        )
        state = state.apply_gradients(grads=grads)
        ema_params = jax.tree.map(
            lambda ema, current: (
                cfg.ema_decay * ema + (1.0 - cfg.ema_decay) * current
                if jnp.issubdtype(current.dtype, jnp.inexact)
                else current
            ),
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema_params)
        metrics = {
            "loss": loss,
            "magnitude_loss": components["magnitude"],
            "direction_loss": components["direction"],
            "gradient_norm": gradient_norm,
            "magnitude_head_gradient_norm": magnitude_head_norm,
            "direction_head_gradient_norm": direction_head_norm,
            "learning_rate": schedule(state.step - 1),
        }
        return state, metrics

    return train_step


def build_eval_step(cfg: Config, model: CrossTickerPatchTransformer) -> Callable:
    @jax.jit
    def eval_step(params: FrozenDict, batch: dict[str, jax.Array], weights: jax.Array):
        prediction = model.apply(
            {"params": params},
            batch["inputs"],
            batch["ticker_ids"],
            batch["target_position"],
            deterministic=True,
        )
        _, components = dual_task_loss(
            prediction, batch, weights, cfg.magnitude_huber_delta_pct
        )
        return prediction, {
            "total_per_example": components["total_per_example"],
            "magnitude_per_example": components["magnitude_per_example"],
            "direction_per_example": components["direction_per_example"],
        }

    return eval_step


def create_mesh() -> jax.sharding.Mesh:
    devices = np.asarray(jax.devices())
    if len(devices) != 8:
        raise RuntimeError(f"Expected 8 TPU devices, found {len(devices)}")
    return jax.sharding.Mesh(devices, axis_names=("data",))


def replicate_state(state: ExperimentTrainState, mesh: jax.sharding.Mesh) \
        -> ExperimentTrainState:
    return jax.device_put(state, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))


MODEL_BATCH_KEYS = (
    "inputs",
    "ticker_ids",
    "target_position",
    "magnitude_pct",
    "direction",
)


def shard_batch(batch: dict[str, np.ndarray], mesh: jax.sharding.Mesh) \
        -> dict[str, jax.Array]:
    batch_size = len(batch["inputs"])
    if batch_size % len(mesh.devices.flat):
        raise ValueError("Global batch size must divide evenly across TPU devices")
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    return {
        key: jax.device_put(np.asarray(batch[key]), sharding)
        for key in MODEL_BATCH_KEYS
    }


def read_loss_weights(cfg: Config) -> np.ndarray:
    path = cfg.paths["loss_weights"]
    if path.exists():
        values = json.loads(path.read_text())
        weights = np.array(
            [values["magnitude_loss_weight"], values["direction_loss_weight"]],
            dtype=np.float32,
        )
    else:
        weights = np.array(
            [cfg.magnitude_loss_weight, cfg.direction_loss_weight], dtype=np.float32
        )
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError(f"Invalid loss weights in {path}: {weights.tolist()}")
    return weights / weights.sum()


def evaluate(cfg: Config, state: ExperimentTrainState,
             model: CrossTickerPatchTransformer, sampler: Any,
             mesh: jax.sharding.Mesh, weights: np.ndarray, split: str = "val",
             n_batches: int | None = None, eval_step: Callable | None = None) \
        -> dict[str, dict[str, float]]:
    eval_step = eval_step or build_eval_step(cfg, model)
    groups = {"overall": empty_metric_sums(), "mag7": empty_metric_sums(),
              "always_eligible": empty_metric_sums()}
    groups.update({f"country/{market}": empty_metric_sums()
                   for market in sampler.market_names})
    count = cfg.val_batches if n_batches is None else n_batches
    for host_batch in sampler.batches(
        split, cfg.batch_size, count, seed=cfg.seed + 10_000, stratified=True
    ):
        device_batch = shard_batch(host_batch, mesh)
        prediction, losses = eval_step(
            state.ema_params, device_batch, jax.device_put(weights)
        )
        host_prediction, host_losses = jax.device_get((prediction, losses))
        update_metric_sums(groups["overall"], host_prediction, host_batch, host_losses)
        update_metric_sums(
            groups["mag7"], host_prediction, host_batch, host_losses,
            host_batch["is_mag7"],
        )
        update_metric_sums(
            groups["always_eligible"], host_prediction, host_batch, host_losses,
            host_batch["always_eligible"],
        )
        for market_index, market in enumerate(sampler.market_names):
            update_metric_sums(
                groups[f"country/{market}"], host_prediction, host_batch, host_losses,
                host_batch["market_id"] == market_index,
            )
    return {
        name: finalize_metrics(values)
        for name, values in groups.items()
        if values["count"] > 0
    }


def save_state(state: ExperimentTrainState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(jax.device_get(state)))


def restore_state(template: ExperimentTrainState, path: str | Path) \
        -> ExperimentTrainState:
    return serialization.from_bytes(template, Path(path).read_bytes())


def save_params(params: FrozenDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(jax.device_get(params)))


def restore_params(template: Any, path: str | Path) -> Any:
    return serialization.from_bytes(template, Path(path).read_bytes())


def estimated_model_flops(cfg: Config, batch_size: int) -> float:
    tokens = batch_size * cfg.n_tickers_per_sample * cfg.n_patches
    temporal = tokens * cfg.temporal_depth * (
        4 * cfg.d_model**2 + 2 * cfg.d_model * cfg.d_ff
    )
    cross = batch_size * cfg.n_tickers_per_sample * cfg.cross_ticker_depth * (
        4 * cfg.d_model**2 + 2 * cfg.d_model * cfg.d_ff
    )
    attention = (
        batch_size * cfg.n_tickers_per_sample * cfg.temporal_depth
        * 2 * cfg.n_patches**2 * cfg.d_model
        + batch_size * cfg.cross_ticker_depth * 2
        * cfg.n_tickers_per_sample**2 * cfg.d_model
    )
    multiplier = 8.0 if cfg.remat else 6.0
    return multiplier * (temporal + cross + attention)