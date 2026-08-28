from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from .callbacks import hbm_used_gb
from .config import Config
from .dataset import GlobalBasketSampler
from .engine import (
    build_train_step,
    create_mesh,
    create_train_state,
    estimated_model_flops,
    replicate_state,
    shard_batch,
)
from .model import CrossTickerPatchTransformer, count_parameters


def benchmark(cfg: Config, steps: int) -> dict[str, float]:
    sampler = GlobalBasketSampler(cfg)
    model = CrossTickerPatchTransformer(cfg, sampler.n_tickers)
    state, schedule = create_train_state(cfg, model, jax.random.key(cfg.seed))
    parameter_count = count_parameters({"params": state.params})
    mesh = create_mesh()
    state = replicate_state(state, mesh)
    train_step = build_train_step(cfg, schedule)
    weights = jax.device_put(
        np.array([cfg.magnitude_loss_weight, cfg.direction_loss_weight], dtype=np.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    host_batch = sampler.sample_batch(
        "train", cfg.batch_size, np.random.default_rng(cfg.seed)
    )
    batch = shard_batch(host_batch, mesh)
    key_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    key = jax.device_put(jax.random.key(cfg.seed), key_sharding)

    compile_started = time.time()
    state, metrics = train_step(state, batch, key, weights)
    jax.block_until_ready(metrics)
    compile_seconds = time.time() - compile_started
    started = time.time()
    for index in range(steps):
        key = jax.device_put(jax.random.key(cfg.seed + index + 1), key_sharding)
        state, metrics = train_step(state, batch, key, weights)
    jax.block_until_ready(metrics)
    seconds_per_step = (time.time() - started) / steps
    peak_flops = 197e12 * len(jax.devices())
    result = {
        "parameters": float(parameter_count),
        "compile_seconds": compile_seconds,
        "seconds_per_step": seconds_per_step,
        "samples_per_second": cfg.batch_size / seconds_per_step,
        "series_per_second": (
            cfg.batch_size * cfg.n_tickers_per_sample / seconds_per_step
        ),
        "hbm_used_gb_per_core": hbm_used_gb(),
        "estimated_mfu_percent": 100.0 * estimated_model_flops(cfg, cfg.batch_size)
        / seconds_per_step / peak_flops,
    }
    output = cfg.paths["run"] / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Experiment 1.2 on TPU")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    benchmark(cfg, args.steps)


if __name__ == "__main__":
    main()