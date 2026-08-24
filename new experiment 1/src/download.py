"""Build the ticker universe and download daily bars, chunk by chunk.

Two ideas matter here:

1.  **Survivorship bias.** If you take today's S&P 500 constituents you have
    silently selected 500 companies that *survived*. A model trained on that
    universe has never seen a firm collapse and will be over-optimistic. We
    start from the full NASDAQ-traded symbol directory instead, which is much
    closer to the investable universe of the day.

2.  **Memory.** 3,000 tickers x 9,000 days x 5 fields in float64 is ~1.1 GB in
    pandas, and yfinance builds wide MultiIndex frames that transiently cost
    several times that. We therefore download in chunks, immediately downcast
    to float32, and persist each chunk to parquet. Peak RAM stays well under
    2 GB, which matters on a 4-core / 31 GB Kaggle box.
"""
from __future__ import annotations

import io
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

warnings.filterwarnings("ignore")

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
#  1. Universe
# --------------------------------------------------------------------------- #
def _read_symbol_file(url: str) -> pd.DataFrame:
    raw = urllib.request.urlopen(url, timeout=90).read().decode("utf-8", "replace")
    df = pd.read_csv(io.StringIO(raw), sep="|")
    return df[~df.iloc[:, 0].astype(str).str.startswith("File Creation")]


def build_universe(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Return a DataFrame of candidate symbols with `symbol` and `name`."""
    nas = _read_symbol_file(NASDAQ_LISTED)
    oth = _read_symbol_file(OTHER_LISTED)

    nas = nas.rename(columns={"Symbol": "symbol", "Security Name": "name"})
    nas["exchange"] = "NASDAQ"
    oth = oth.rename(columns={"NASDAQ Symbol": "symbol", "Security Name": "name",
                              "Exchange": "exchange"})
    keep = ["symbol", "name", "exchange", "ETF", "Test Issue"]
    both = pd.concat([nas[keep], oth[keep]], ignore_index=True)

    n0 = len(both)
    both = both[both["Test Issue"].astype(str).str.upper() != "Y"]
    if cfg.exclude_etfs:
        both = both[both["ETF"].astype(str).str.upper() != "Y"]

    both["symbol"] = both["symbol"].astype(str).str.strip()
    # '$' and '.' in a NASDAQ symbol mark warrants, units, rights and preferred
    # share classes. Those instruments have no meaningful price dynamics of the
    # kind we are modelling, and yfinance mostly returns empty frames for them.
    both = both[both["symbol"].str.fullmatch(r"[A-Z]{1,5}")]
    name_u = both["name"].astype(str).str.upper()
    junk = name_u.str.contains(
        "WARRANT|UNIT|RIGHT| PFD|PREFERRED|DEPOSITARY SHARE|NOTE DUE|"
        "%|TRUST PREFERRED|SUBORDINATED|ACQUISITION CORP", regex=True)
    both = both[~junk]
    both = both.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

    # Context ETFs are ALWAYS downloaded: they become market-wide features that
    # every ticker sees, not samples in their own right.
    ctx = pd.DataFrame({"symbol": list(cfg.context_tickers),
                        "name": "CONTEXT", "exchange": "CTX",
                        "ETF": "Y", "Test Issue": "N"})
    both = pd.concat([both[~both["symbol"].isin(cfg.context_tickers)], ctx],
                     ignore_index=True)

    if verbose:
        print(f"symbol directory : {n0} rows -> {len(both)} candidate symbols "
              f"({len(cfg.context_tickers)} of them market-context ETFs)")
    return both[["symbol", "name", "exchange"]]


# --------------------------------------------------------------------------- #
#  2. Chunked download
# --------------------------------------------------------------------------- #
def download_chunks(cfg: Config, symbols: list[str], chunk_size: int = 250,
                    verbose: bool = True) -> list[Path]:
    """Download `symbols` in chunks, caching each chunk as float32 parquet.

    Re-running is cheap: an existing chunk file is skipped, so a crashed
    download resumes from where it stopped.
    """
    import yfinance as yf

    cache = cfg.paths["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    out: list[Path] = []
    t_start = time.time()

    for k, chunk in enumerate(chunks):
        path = cache / f"chunk_{k:04d}.parquet"
        out.append(path)
        if path.exists():
            if verbose:
                print(f"  [{k + 1:>3}/{len(chunks)}] cached  {path.name}")
            continue
        t0 = time.time()
        try:
            raw = yf.download(chunk, start=cfg.start_date, end=cfg.end_date,
                              interval="1d", auto_adjust=True, progress=False,
                              threads=True, group_by="column")
        except Exception as exc:                      # network hiccup -> skip
            print(f"  [{k + 1:>3}/{len(chunks)}] FAILED {type(exc).__name__}: {exc}")
            out.pop()
            continue

        if raw is None or raw.empty:
            out.pop()
            continue
        if not isinstance(raw.columns, pd.MultiIndex):    # single-symbol chunk
            raw.columns = pd.MultiIndex.from_product([raw.columns, chunk[:1]])

        # long format: (date, symbol) x 5 fields, float32. Far smaller on disk
        # and it makes the later concat trivial.
        frames = []
        for fld in FIELDS:
            if fld not in raw.columns.get_level_values(0):
                continue
            sub = raw[fld].astype("float32").stack(future_stack=True).rename(fld)
            frames.append(sub)
        if not frames:
            out.pop()
            continue
        long = pd.concat(frames, axis=1).dropna(how="all")
        long.index.names = ["date", "symbol"]
        long.reset_index().to_parquet(path, index=False, compression="zstd")

        if verbose:
            elapsed = time.time() - t_start
            done = k + 1
            eta = elapsed / done * (len(chunks) - done)
            print(f"  [{done:>3}/{len(chunks)}] {len(chunk):>3} syms "
                  f"-> {long.shape[0]:>8,} rows  "
                  f"{time.time() - t0:5.1f}s   ETA {eta / 60:5.1f} min")
        del raw, long, frames
    return out


# --------------------------------------------------------------------------- #
#  3. Assemble + liquidity filter
# --------------------------------------------------------------------------- #
def assemble(cfg: Config, chunk_paths: list[Path], verbose: bool = True
             ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, list[str]]:
    """Load every chunk, apply the liquidity screen, and return dense panels.

    Done in **two passes** on purpose. Concatenating every chunk into one long
    DataFrame would hold ~20M rows whose `symbol` column alone costs over a GB
    of Python strings. Instead pass 1 keeps only per-symbol summary statistics,
    and pass 2 streams each chunk straight into preallocated dense arrays. Peak
    RAM is therefore one chunk (~50 MB) plus the final panel (~0.6 GB).

    Returns
    -------
    panels : dict field -> float32 array of shape (n_tickers, n_days)
    dates  : the shared trading calendar
    tickers: row order of the panels
    """
    live = [p for p in chunk_paths if p.exists()]

    # ---------------- pass 1: statistics only --------------------------------
    stats_parts, all_dates = [], set()
    for p in live:
        df = pd.read_parquet(p, columns=["date", "symbol", "Close", "Volume"])
        df = df.dropna(subset=["Close"])
        all_dates.update(df["date"].unique())
        g = df.assign(dv=df["Close"] * df["Volume"]).groupby("symbol", observed=True)
        stats_parts.append(pd.DataFrame({"n_days": g["Close"].size(),
                                         "med_dv": g["dv"].median()}))
        del df, g
    stats = pd.concat(stats_parts).groupby(level=0).agg(
        {"n_days": "sum", "med_dv": "max"})
    del stats_parts

    ctx = set(cfg.context_tickers)
    ok = stats[(stats["n_days"] >= cfg.min_history_days) &
               (stats["med_dv"] >= cfg.min_median_dollar_volume)]
    keep = set(ok.sort_values("med_dv", ascending=False).head(cfg.max_tickers).index)
    keep |= (ctx & set(stats.index))
    tickers = sorted(keep)
    dates = pd.DatetimeIndex(sorted(all_dates))
    if verbose:
        print(f"liquidity screen : {len(stats):,} -> {len(tickers):,} tickers "
              f"(>= {cfg.min_history_days} days, median $vol >= "
              f"{cfg.min_median_dollar_volume:,.0f})")
        print(f"calendar         : {len(dates):,} trading days "
              f"{dates[0].date()} -> {dates[-1].date()}")

    # ---------------- pass 2: stream into dense arrays -----------------------
    t_pos = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    s_pos = pd.Series(np.arange(len(tickers), dtype=np.int32), index=tickers)
    panels = {f: np.full((len(tickers), len(dates)), np.nan, dtype=np.float32)
              for f in FIELDS}

    for p in live:
        df = pd.read_parquet(p)
        df = df[df["symbol"].isin(keep)].dropna(subset=["Close"])
        if df.empty:
            continue
        ri = s_pos.reindex(df["symbol"]).to_numpy()
        ci = t_pos.reindex(pd.DatetimeIndex(df["date"])).to_numpy()
        for fld in FIELDS:
            if fld in df.columns:
                panels[fld][ri, ci] = df[fld].to_numpy(dtype=np.float32)
        del df, ri, ci

    if verbose:
        cover = np.isfinite(panels["Close"]).mean()
        print(f"panel shape      : {panels['Close'].shape} "
              f"(tickers x days), {cover:.1%} populated")
        print(f"panel memory     : "
              f"{sum(a.nbytes for a in panels.values()) / 1e9:.2f} GB")
    return panels, dates, tickers
