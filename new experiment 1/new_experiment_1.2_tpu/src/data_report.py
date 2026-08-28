from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import Config, MAGNIFICENT_SEVEN


def build_report(cfg: Config) -> dict:
    panel = cfg.paths["panel"]
    metadata = json.loads((panel / "meta.json").read_text())
    tickers = pd.read_csv(panel / "tickers.csv")
    markets = []
    for item in metadata["markets"]:
        market = item["market"]
        rows = tickers[tickers["market"] == market]
        markets.append({
            "market": market,
            "series": int(len(rows)),
            "observations": int(item.get("n_observations", 0)),
            "sessions": int(item["n_dates"]),
            "first_date": item["first_date"],
            "last_date": item["last_date"],
            "train_targets": int(rows["eligible_train"].sum()),
            "val_targets": int(rows["eligible_val"].sum()),
            "test_targets": int(rows["eligible_test"].sum()),
            "always_eligible": int(rows["always_eligible"].sum()),
            "train_anchors": int(item["anchors"]["train"]),
            "val_anchors": int(item["anchors"]["val"]),
            "test_anchors": int(item["anchors"]["test"]),
        })
    mag7 = sorted(
        tickers.loc[
            (tickers["market"] == "US") & tickers["ticker"].isin(MAGNIFICENT_SEVEN),
            "ticker",
        ].tolist()
    )
    report = {
        "dataset_repo": metadata["dataset_repo"],
        "price_column": metadata["price_column"],
        "global_return_scale_pct": metadata["return_scale_pct"],
        "series": int(len(tickers)),
        "markets": markets,
        "mag7_found": mag7,
        "mag7_missing": sorted(set(MAGNIFICENT_SEVEN) - set(mag7)),
        "totals": {
            key: int(sum(row[key] for row in markets))
            for key in (
                "train_targets", "val_targets", "test_targets", "always_eligible",
                "train_anchors", "val_anchors", "test_anchors",
            )
        },
    }
    output = cfg.paths["run"] / "data_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the prepared Experiment 1.2 panel")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    build_report(cfg)


if __name__ == "__main__":
    main()