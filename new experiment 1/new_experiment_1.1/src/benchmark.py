from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .config import Config
from .engine import cleanup_distributed, setup_distributed
from .losses import DualTaskLoss
from .model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tickers-in-universe", type=int, default=6000)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    cfg = Config(batch_size=args.batch_size, compile_model=False)
    model = build_model(cfg, args.tickers_in_universe).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda")
    criterion = DualTaskLoss(cfg)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + rank)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.time()

    for _ in range(args.steps):
        x = torch.randn(args.batch_size, cfg.n_tickers_per_sample, cfg.n_steps_in,
                        device=device, dtype=torch.float16, generator=generator)
        ticker_ids = torch.randint(0, args.tickers_in_universe,
                                   (args.batch_size, cfg.n_tickers_per_sample),
                                   device=device, generator=generator)
        target_position = torch.randint(0, cfg.n_tickers_per_sample,
                                        (args.batch_size,), device=device,
                                        generator=generator)
        batch = {
            "x": x,
            "ticker_ids": ticker_ids,
            "target_position": target_position,
            "magnitude_pct": torch.rand(args.batch_size, device=device,
                                         generator=generator) * 5,
            "direction": torch.randint(0, 2, (args.batch_size,), device=device,
                                        generator=generator).float(),
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(x, ticker_ids, target_position)
            loss, _ = criterion(prediction, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    print(f"rank={rank} gpu={torch.cuda.get_device_name(device)} batch={args.batch_size} "
          f"peak_allocated={peak_allocated:.2f}GB peak_reserved={peak_reserved:.2f}GB "
          f"throughput={args.steps * args.batch_size / elapsed:.1f} samples/s")
    if world_size > 1:
        distributed.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()