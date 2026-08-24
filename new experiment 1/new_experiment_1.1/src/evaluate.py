from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config, EXPERIMENT_ROOT
from .dataset import GPUBasketPanel
from .engine import cleanup_distributed, evaluate, setup_distributed
from .losses import DualTaskLoss
from .model import build_model
from .prepare_data import load_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batches", type=int, default=80)
    args = parser.parse_args()

    cfg = Config()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    arrays, anchors, _ = load_panel(cfg)
    panel = GPUBasketPanel(cfg, arrays, anchors, device)
    model = build_model(cfg, panel.n_tickers).to(device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.paths["ckpt"] / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = checkpoint.get("ema") or checkpoint["model"]
    model.load_state_dict(weights)
    metrics = evaluate(cfg, model, panel, DualTaskLoss(cfg), device,
                       rank, world_size, split=args.split,
                       n_batches=args.batches)
    if rank == 0:
        result = {
            "split": args.split,
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": checkpoint["epoch"],
            "batches_per_rank": args.batches,
            "metrics": metrics,
        }
        output = EXPERIMENT_ROOT / "logs" / cfg.run_name / f"{args.split}_metrics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    cleanup_distributed()


if __name__ == "__main__":
    main()