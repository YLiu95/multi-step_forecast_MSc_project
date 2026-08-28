from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .config import Config
from .secrets import get_token


def list_market_files(cfg: Config) -> dict[str, list[str]]:
    token = get_token("HF_TOKEN")
    files = HfApi(token=token).list_repo_files(cfg.dataset_repo, repo_type="dataset")
    grouped: dict[str, list[str]] = defaultdict(list)
    allowed = set(cfg.markets)
    for filename in files:
        parts = filename.split("/")
        if len(parts) == 3 and parts[0] == "data" and parts[1] in allowed:
            if filename.endswith(".parquet"):
                grouped[parts[1]].append(filename)
    missing = allowed - set(grouped)
    if missing:
        raise RuntimeError(f"Dataset is missing market files: {sorted(missing)}")
    return {market: sorted(grouped[market]) for market in cfg.markets}


def download_market_files(cfg: Config) -> dict[str, list[Path]]:
    token = get_token("HF_TOKEN")
    local_root = cfg.paths["cache"] / "dataset"
    local_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for market, filenames in list_market_files(cfg).items():
        print(f"Downloading {market}: {len(filenames)} parquet shard(s)", flush=True)
        result[market] = [
            Path(
                hf_hub_download(
                    cfg.dataset_repo,
                    filename,
                    repo_type="dataset",
                    token=token,
                    local_dir=local_root,
                )
            )
            for filename in filenames
        ]
    return result