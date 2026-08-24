"""End-to-end data preparation: universe -> download -> features -> .npy panel.

Run once:  python -m src.prepare_data
Re-runs are cheap because every download chunk is cached to parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, pretty
from .download import assemble, build_universe, download_chunks
from .features import FEATURE_NAMES, build_anchor_index, build_features


def prepare(cfg: Config, chunk_size: int = 250, force: bool = False,
            symbol_limit: int | None = None) -> Path:
    cfg.make_dirs()
    out_dir = cfg.paths["panel"]
    marker = out_dir / "meta.json"
    if marker.exists() and not force:
        print(f"panel already built at {out_dir}; pass --force to rebuild")
        return out_dir

    t0 = time.time()
    print(pretty(cfg), "\n")

    print("[1/4] building universe")
    uni = build_universe(cfg)
    symbols = uni["symbol"].tolist()
    if symbol_limit:                       # smoke-test escape hatch
        keep = set(symbols[:symbol_limit]) | set(cfg.context_tickers)
        symbols = [s for s in symbols if s in keep]
        print(f"  symbol_limit={symbol_limit} -> {len(symbols)} symbols")

    print(f"\n[2/4] downloading {len(symbols):,} symbols in chunks of {chunk_size}")
    chunk_paths = download_chunks(cfg, symbols, chunk_size=chunk_size)

    print("\n[3/4] assembling + liquidity screen")
    panels, dates, tickers = assemble(cfg, chunk_paths)

    print("\n[4/4] engineering features")
    arrays = build_features(cfg, panels, dates, tickers)
    del panels
    anchors = build_anchor_index(cfg, arrays["good"], dates)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "feat.npy", arrays["feat"])
    np.save(out_dir / "ret.npy", arrays["ret"])
    np.save(out_dir / "sig.npy", arrays["sig"])
    for k, v in anchors.items():
        np.save(out_dir / f"anchor_{k}.npy", v)
    np.save(out_dir / "norm_mu.npy", arrays["mu"])
    np.save(out_dir / "norm_sd.npy", arrays["sd"])

    meta = {
        "tickers": tickers,
        "dates": [str(d.date()) for d in dates],
        "feature_names": list(FEATURE_NAMES),
        "shape": list(arrays["feat"].shape),
        "n_train": int(len(anchors["train_i"])),
        "n_val": int(len(anchors["val_i"])),
        "n_test": int(len(anchors["test_i"])),
        "config": cfg.to_dict(),
    }
    marker.write_text(json.dumps(meta, indent=2, default=list))

    mb = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e6
    print(f"\nsaved {mb:,.0f} MB to {out_dir}  "
          f"(total {time.time() - t0:.0f}s)")
    return out_dir


def load_panel(cfg: Config, mmap: bool = True):
    """Load the prepared panel. `mmap=True` keeps CPU RAM near zero."""
    d = cfg.paths["panel"]
    mode = "r" if mmap else None
    meta = json.loads((d / "meta.json").read_text())
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode=mode)
              for k in ("feat", "ret", "sig")}
    anchors = {k: np.load(d / f"anchor_{k}.npy")
               for k in ("train_i", "train_t", "val_i", "val_t",
                         "test_i", "test_t")}
    return arrays, anchors, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--symbol-limit", type=int, default=None)
    ap.add_argument("--artifact-root", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = Config.load(a.config) if a.config else Config()
    if a.max_tickers:
        cfg = cfg.override(max_tickers=a.max_tickers)
    if a.artifact_root:
        cfg = cfg.override(artifact_root=a.artifact_root)
    prepare(cfg, chunk_size=a.chunk_size, force=a.force,
            symbol_limit=a.symbol_limit)
