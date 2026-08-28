from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from .callbacks import CheckpointManager
from .config import Config
from .dataset import GlobalBasketSampler
from .engine import create_mesh, create_train_state, evaluate, replicate_state
from .model import CrossTickerPatchTransformer


def run_evaluation(cfg: Config, split: str, batches: int | None = None) -> dict:
    sampler = GlobalBasketSampler(cfg)
    model = CrossTickerPatchTransformer(cfg, sampler.n_tickers)
    state, _ = create_train_state(cfg, model, jax.random.key(cfg.seed))
    restored = CheckpointManager(cfg).restore_best(state.params)
    if restored is None:
        raise FileNotFoundError("No local or Hugging Face best model is available")
    params, metadata = restored
    state = state.replace(ema_params=params)
    mesh = create_mesh()
    state = replicate_state(state, mesh)
    weights = metadata.get(
        "loss_weights", [cfg.magnitude_loss_weight, cfg.direction_loss_weight]
    )
    metrics = evaluate(
        cfg, state, model, sampler, mesh, weights, split=split, n_batches=batches
    )
    output = cfg.paths["run"] / f"{split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 1.2 best model")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    run_evaluation(cfg, args.split, args.batches)


if __name__ == "__main__":
    main()