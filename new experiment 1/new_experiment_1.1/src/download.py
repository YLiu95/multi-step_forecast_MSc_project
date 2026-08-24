from __future__ import annotations

import io
import json
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NASDAQ_100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MAGNIFICENT_SEVEN = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")


def _read_pipe_table(url: str) -> pd.DataFrame:
    raw = urllib.request.urlopen(url, timeout=90).read().decode("utf-8", "replace")
    frame = pd.read_csv(io.StringIO(raw), sep="|")
    return frame[~frame.iloc[:, 0].astype(str).str.startswith("File Creation")]


def _yf_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def build_universe(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Currently listed NASDAQ and NYSE common stocks.

    Free exchange directories do not contain delisted companies. The saved
    metadata states this limitation instead of presenting a survivorship-biased
    universe as historical point-in-time membership.
    """
    nasdaq = _read_pipe_table(NASDAQ_LISTED).rename(
        columns={"Symbol": "symbol", "Security Name": "name"})
    nasdaq["exchange"] = "NASDAQ"
    other = _read_pipe_table(OTHER_LISTED).rename(
        columns={"NASDAQ Symbol": "symbol", "Security Name": "name"})
    other = other[other["Exchange"].astype(str).str.upper() == "N"]
    other["exchange"] = "NYSE"

    columns = ["symbol", "name", "exchange", "ETF", "Test Issue"]
    universe = pd.concat([nasdaq[columns], other[columns]], ignore_index=True)
    universe = universe[universe["Test Issue"].astype(str).str.upper() != "Y"]
    if cfg.exclude_etfs:
        universe = universe[universe["ETF"].astype(str).str.upper() != "Y"]
    universe["symbol"] = universe["symbol"].astype(str).str.strip().map(_yf_symbol)
    universe = universe[universe["symbol"].str.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?")]

    names = universe["name"].astype(str).str.upper()
    excluded = names.str.contains(
        r"WARRANT|RIGHT|\bUNIT\b|PREFERRED|DEPOSITARY SHARE|NOTE DUE|"
        r"SUBORDINATED|ACQUISITION CORP|BENEFICIAL INTEREST",
        regex=True,
    )
    universe = (universe[~excluded]
                .drop_duplicates("symbol")
                .sort_values("symbol")
                .reset_index(drop=True))
    if verbose:
        counts = universe.groupby("exchange")["symbol"].size().to_dict()
        print(f"current exchange universe: {len(universe):,} symbols {counts}")
    return universe[["symbol", "name", "exchange"]]


def _find_symbol_column(tables: list[pd.DataFrame], candidates: tuple[str, ...]) -> pd.Series:
    for table in tables:
        flattened = [" ".join(map(str, col)) if isinstance(col, tuple) else str(col)
                     for col in table.columns]
        for candidate in candidates:
            for index, name in enumerate(flattened):
                if candidate.lower() == name.lower() or candidate.lower() in name.lower():
                    return table.iloc[:, index].astype(str).str.strip().map(_yf_symbol)
    raise RuntimeError(f"Could not find any of {candidates} in index constituent tables")


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 experiment-research/1.1"
    })
    html = urllib.request.urlopen(request, timeout=90).read().decode("utf-8", "replace")
    return pd.read_html(io.StringIO(html))


def fetch_index_members() -> tuple[set[str], set[str]]:
    request = urllib.request.Request(NASDAQ_100_URL, headers={
        "User-Agent": "Mozilla/5.0 experiment-research/1.1",
        "Accept": "application/json",
    })
    payload = json.load(urllib.request.urlopen(request, timeout=90))
    nasdaq100 = {_yf_symbol(row["symbol"])
                 for row in payload["data"]["data"]["rows"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sp500 = set(_find_symbol_column(_read_html_tables(SP500_URL), ("Symbol",)))
    return nasdaq100, sp500


def select_targets(cfg: Config, universe: pd.DataFrame) -> pd.DataFrame:
    """Create four non-overlapping target strata with a seeded RNG."""
    available = set(universe["symbol"])
    nasdaq100, sp500 = fetch_index_members()
    missing_mag7 = set(MAGNIFICENT_SEVEN) - available
    if missing_mag7:
        raise RuntimeError(f"Mag7 missing from exchange universe: {sorted(missing_mag7)}")

    rng = np.random.default_rng(cfg.seed)
    used = set(MAGNIFICENT_SEVEN)

    def choose(pool: set[str], count: int, group: str) -> list[dict[str, str]]:
        candidates = sorted((pool & available) - used)
        if len(candidates) < count:
            raise RuntimeError(f"{group}: need {count} symbols, found {len(candidates)}")
        picked = rng.choice(candidates, size=count, replace=False).tolist()
        used.update(picked)
        return [{"symbol": symbol, "group": group} for symbol in picked]

    rows = [{"symbol": symbol, "group": "magnificent_seven"}
            for symbol in MAGNIFICENT_SEVEN]
    rows += choose(nasdaq100, cfg.n_nasdaq100_targets, "nasdaq_100")
    rows += choose(sp500, cfg.n_sp500_targets, "sp500_ex_nasdaq100")
    rows += choose(available - nasdaq100 - sp500,
                   cfg.n_outside_targets, "outside_major_indices")
    targets = pd.DataFrame(rows)
    if len(targets) != 167 or targets["symbol"].nunique() != 167:
        raise AssertionError("Target selection must contain exactly 167 unique symbols")
    return targets


def download_adjusted_close(cfg: Config, symbols: list[str],
                            chunk_size: int | None = None) -> list[Path]:
    """Download one field only and cache long-format parquet chunks."""
    import yfinance as yf

    chunk_size = chunk_size or cfg.download_chunk_size
    cache = cfg.paths["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    chunks = [symbols[start:start + chunk_size]
              for start in range(0, len(symbols), chunk_size)]
    paths: list[Path] = []
    started = time.time()

    for index, chunk in enumerate(chunks):
        path = cache / f"adjusted_close_{index:04d}.parquet"
        paths.append(path)
        if path.exists():
            print(f"[{index + 1:>3}/{len(chunks)}] cached {path.name}")
            continue
        try:
            raw = yf.download(chunk, start=cfg.start_date, end=cfg.end_date,
                              auto_adjust=False, actions=False, progress=False,
                              threads=True, group_by="column")
        except Exception as exc:
            print(f"[{index + 1:>3}/{len(chunks)}] failed: {exc}")
            paths.pop()
            continue
        if raw is None or raw.empty:
            paths.pop()
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
            close = raw[field]
        else:
            field = "Adj Close" if "Adj Close" in raw.columns else "Close"
            close = raw[[field]].rename(columns={field: chunk[0]})
        long = close.astype("float32").stack(future_stack=True).rename("adjusted_close")
        long.index.names = ["date", "symbol"]
        long.reset_index().to_parquet(path, index=False, compression="zstd")
        elapsed = time.time() - started
        eta = elapsed / (index + 1) * (len(chunks) - index - 1)
        print(f"[{index + 1:>3}/{len(chunks)}] {len(chunk):>3} symbols, "
              f"{len(long):>9,} rows, ETA {eta / 60:.1f} min")
        del raw, close, long
    return paths


def assemble_close(cfg: Config, paths: list[Path], required: set[str]) \
        -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Two-pass assembly bounds peak CPU RAM to one chunk plus the dense panel."""
    live = [path for path in paths if path.exists()]
    if not live:
        raise RuntimeError("No downloaded parquet chunks were found")

    counts: list[pd.Series] = []
    all_dates: set[np.datetime64] = set()
    for path in live:
        frame = pd.read_parquet(path)
        frame = frame.dropna(subset=["adjusted_close"])
        counts.append(frame.groupby("symbol", observed=True).size())
        all_dates.update(frame["date"].to_numpy())
    history = pd.concat(counts).groupby(level=0).sum()
    missing = required - set(history.index)
    if missing:
        raise RuntimeError(f"Target symbols returned no data: {sorted(missing)}")
    keep = set(history[history >= cfg.min_history_days].index) | required
    tickers = sorted(keep)
    dates = pd.DatetimeIndex(sorted(all_dates))

    ticker_position = pd.Series(np.arange(len(tickers), dtype=np.int32), index=tickers)
    date_position = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    close = np.full((len(tickers), len(dates)), np.nan, dtype=np.float32)
    for path in live:
        frame = pd.read_parquet(path)
        frame = frame[frame["symbol"].isin(keep)].dropna(subset=["adjusted_close"])
        rows = ticker_position.reindex(frame["symbol"]).to_numpy()
        cols = date_position.reindex(pd.DatetimeIndex(frame["date"])).to_numpy()
        close[rows, cols] = frame["adjusted_close"].to_numpy(dtype=np.float32)
    print(f"assembled adjusted close: {close.shape}, {np.isfinite(close).mean():.1%} populated")
    return close, dates, tickers