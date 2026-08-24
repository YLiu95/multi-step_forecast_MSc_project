"""Turn raw OHLCV panels into stationary, cross-sectionally comparable features.

The single most important idea in this file
-------------------------------------------
The original notebook forecast *price levels* after MinMax-scaling on the train
split. Prices are non-stationary: they trend, and the test period contains
values the scaler never saw, so the network is *structurally* unable to predict
a new all-time high. Worse, a $3 stock and a $500 stock cannot share a model.

We therefore work in **volatility-scaled log returns**:

    r_t     = log(C_t / C_{t-1})                     stationary, additive
    sigma_t = std(r_{t-59..t})                       trailing, causal
    z_t     = r_t / sigma_t                          scale-free across tickers
    y_h     = r_{t+h} / sigma_t                      the target, h = 1..H

`sigma_t` uses only information available at time `t`, so nothing leaks. Because
the *same* `sigma_t` divides every horizon, the target is a proper multi-step
path that we can convert straight back to dollars:

    P_{t+h} = C_t * exp( sigma_t * cumsum(y_hat)_h )

Every other feature is scaled the same way, which is why one model can learn
from a mega-cap and a micro-cap simultaneously.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Volatility floor / cap. Without the floor a stock that went flat for 60 days
# gets sigma ~ 0 and z blows up to +-1e6, which destroys the loss.
SIG_FLOOR, SIG_CAP = 0.004, 0.25

FEATURE_NAMES = (
    # ---- per-ticker, all volatility-scaled so they are unit-free ----------
    "z_ret", "z_gap", "z_oc", "z_range", "log_sigma",
    "z_mom5", "z_mom20", "z_mom60", "z_mom120", "z_dist252", "vol_surprise",
    # ---- market-wide context, identical for every ticker on a given day ---
    "mkt_ret", "mkt_mom20", "breadth", "dispersion",
    # ---- calendar --------------------------------------------------------
    "dow", "month_sin", "month_cos",
)

# Channels that are byte-identical for every ticker on a given day.
#
# These carry real information (market beta, regime), but they are also a
# LIABILITY: a 256-day window of them uniquely identifies the DATE. There are
# only ~7,300 distinct training dates, and all ~1,000 tickers sharing a date
# also share the same future market move. A large model therefore memorises
# "this window is October 2017" and recalls what happened next -- which looks
# like learning on the training set and generalises to nothing.
#
# `Config.shared_group_dropout` randomly blanks this whole group during
# training so the model cannot build a strategy on the fingerprint.
SHARED_FEATURES = ("mkt_ret", "mkt_mom20", "breadth", "dispersion",
                   "dow", "month_sin", "month_cos")

# Pure date identifiers: essentially no predictive value, maximum fingerprint.
# Disabled by default via `Config.disable_features`.
CALENDAR_FEATURES = ("dow", "month_sin", "month_cos")


def _df(arr: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """(tickers, days) array -> (days, tickers) frame so pandas can roll it."""
    return pd.DataFrame(arr.T, index=dates)


def build_features(cfg: Config, panels: dict[str, np.ndarray],
                   dates: pd.DatetimeIndex, tickers: list[str],
                   verbose: bool = True) -> dict[str, np.ndarray]:
    """Return {'feat','ret','sig','valid'} plus per-feature train statistics."""
    n_t, n_d = panels["Close"].shape
    W = cfg.vol_window

    C = _df(panels["Close"], dates)
    O = _df(panels["Open"], dates)
    Hi = _df(panels["High"], dates)
    Lo = _df(panels["Low"], dates)
    V = _df(panels["Volume"], dates)

    logC = np.log(C.where(C > 0))
    r = logC.diff()                                      # daily log return
    sig = r.rolling(W, min_periods=W // 2).std().clip(SIG_FLOOR, SIG_CAP)

    feats: dict[str, pd.DataFrame] = {}
    feats["z_ret"] = r / sig
    feats["z_gap"] = (np.log(O.where(O > 0)) - logC.shift()) / sig
    feats["z_oc"] = (logC - np.log(O.where(O > 0))) / sig
    feats["z_range"] = (np.log(Hi.where(Hi > 0)) - np.log(Lo.where(Lo > 0))) / sig
    feats["log_sigma"] = np.log(sig)

    # Momentum over k days is a sum of k returns, so its natural scale is
    # sigma * sqrt(k) (random-walk scaling). Dividing by that makes the
    # different look-backs directly comparable to each other.
    for k in (5, 20, 60, 120):
        feats[f"z_mom{k}"] = (logC - logC.shift(k)) / (sig * np.sqrt(k))
    feats["z_dist252"] = ((logC - logC.rolling(252, min_periods=60).max())
                          / (sig * np.sqrt(252)))
    del logC

    logV = np.log1p(V.where(V > 0))
    feats["vol_surprise"] = ((logV - logV.rolling(W, min_periods=W // 2).mean())
                             / logV.rolling(W, min_periods=W // 2).std().clip(1e-3))
    del logV, O, Hi, Lo, V

    # ---- market context, derived from the panel itself ----------------------
    # Using an equal-weighted cross-sectional aggregate rather than SPY means
    # the feature exists back to 1990 and never has a NaN gap.
    mkt = r.median(axis=1)
    mkt_sig = mkt.rolling(W, min_periods=W // 2).std().clip(SIG_FLOOR, SIG_CAP)
    z_mkt = (mkt / mkt_sig).to_frame()
    mkt_mom = ((mkt.rolling(20).sum()) / (mkt_sig * np.sqrt(20))).to_frame()
    breadth = (r > 0).sum(axis=1) / r.notna().sum(axis=1).clip(lower=1)
    breadth = (breadth - 0.5).to_frame() * 2.0
    disp = (r.std(axis=1) / mkt_sig).to_frame()

    ones = pd.DataFrame(1.0, index=dates, columns=C.columns, dtype="float32")
    for name, col in [("mkt_ret", z_mkt), ("mkt_mom20", mkt_mom),
                      ("breadth", breadth), ("dispersion", disp)]:
        feats[name] = ones.mul(col.iloc[:, 0].to_numpy()[:, None])

    # ---- calendar -----------------------------------------------------------
    feats["dow"] = ones.mul((dates.dayofweek.to_numpy() / 2.0 - 1.0)[:, None])
    feats["month_sin"] = ones.mul(np.sin(2 * np.pi * dates.month.to_numpy() / 12)[:, None])
    feats["month_cos"] = ones.mul(np.cos(2 * np.pi * dates.month.to_numpy() / 12)[:, None])
    del ones, C

    assert tuple(feats) == FEATURE_NAMES, (tuple(feats), FEATURE_NAMES)

    # ---- stack into (tickers, days, features) -------------------------------
    F = len(FEATURE_NAMES)
    feat = np.empty((n_t, n_d, F), dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        feat[:, :, j] = feats[name].to_numpy(dtype=np.float32).T
    del feats

    ret = r.to_numpy(dtype=np.float32).T
    sigma = sig.to_numpy(dtype=np.float32).T
    del r, sig

    # ---- global standardisation, fitted on the TRAIN dates only -------------
    train_mask = dates <= pd.Timestamp(cfg.train_end)
    sub = feat[:, train_mask, :].reshape(-1, F)
    finite = np.isfinite(sub)
    mu = np.where(finite, sub, 0).sum(0) / np.maximum(finite.sum(0), 1)
    var = (np.where(finite, (sub - mu) ** 2, 0).sum(0)
           / np.maximum(finite.sum(0), 1))
    sd = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    del sub, finite

    good = np.isfinite(feat).all(axis=2) & np.isfinite(ret) & np.isfinite(sigma)
    np.subtract(feat, mu.astype(np.float32), out=feat)
    np.divide(feat, sd, out=feat)
    np.clip(feat, -cfg.clip_sigma, cfg.clip_sigma, out=feat)
    feat[~good] = 0.0                       # never sampled, but keep it finite
    ret = np.nan_to_num(ret, nan=0.0)
    sigma = np.nan_to_num(sigma, nan=SIG_FLOOR)

    if verbose:
        print(f"features         : {feat.shape} (tickers, days, {F} features)")
        print(f"usable cells     : {good.mean():.1%} of the panel")
        print(pd.DataFrame({"train_mean": mu, "train_std": sd},
                           index=list(FEATURE_NAMES)).round(3).to_string())

    return {
        # float16 halves the GPU footprint of the resident panel; the values are
        # standardised and clipped to +-8 so fp16 has ample precision for them.
        "feat": feat.astype(np.float16),
        "ret": ret.astype(np.float32),
        "sig": np.clip(sigma, SIG_FLOOR, SIG_CAP).astype(np.float32),
        "good": good,
        "mu": mu.astype(np.float32),
        "sd": sd,
    }


def build_anchor_index(cfg: Config, good: np.ndarray, dates: pd.DatetimeIndex
                       ) -> dict[str, np.ndarray]:
    """Enumerate every legal (ticker, t) anchor and split it chronologically.

    An anchor `t` is legal when the whole input window `[t-L+1, t]` **and** the
    whole target window `[t+1, t+H]` consist of usable bars. We test that with
    two cumulative sums instead of a Python loop over 27 million candidates.
    """
    L, H = cfg.n_steps_in, cfg.n_steps_out
    n_t, n_d = good.shape
    g = good.astype(np.int32)
    cum = np.zeros((n_t, n_d + 1), dtype=np.int32)
    np.cumsum(g, axis=1, out=cum[:, 1:])

    t = np.arange(n_d)
    ok = np.zeros((n_t, n_d), dtype=bool)
    lo, hi = L - 1, n_d - H - 1
    idx = t[(t >= lo) & (t <= hi)]
    ok[:, idx] = ((cum[:, idx + 1] - cum[:, idx + 1 - L] == L) &
                  (cum[:, idx + 1 + H] - cum[:, idx + 1] == H))
    del cum, g

    ti, tt = np.nonzero(ok)
    ti, tt = ti.astype(np.int32), tt.astype(np.int32)

    # Chronological split of the ANCHOR date.
    #
    # A train anchor at t_tr <= i_train_end - H has its target window entirely
    # inside the train period, and a val anchor at t_v >= i_train_end has its
    # target window entirely after it -- so the target windows are already
    # disjoint with zero purge. The `purge = H` embargo on top of that handles
    # serial correlation: adjacent target windows overlap each other, so we
    # skip one full horizon before the next split begins.
    #
    # Note we deliberately do NOT purge the full look-back L. A validation input
    # window may contain bars from the training period -- that is just history,
    # exactly what a live model would have. Leakage means seeing a future
    # *label*, not a past *input*.
    d = np.asarray(dates)
    train_end = np.datetime64(pd.Timestamp(cfg.train_end))
    val_end = np.datetime64(pd.Timestamp(cfg.val_end))
    anchor_date = d[tt]
    purge = H

    i_train_end = int(np.searchsorted(d, train_end, "right")) - 1
    i_val_end = int(np.searchsorted(d, val_end, "right")) - 1

    m_train = tt <= (i_train_end - H)
    m_val = (tt >= i_train_end + purge) & (tt <= i_val_end - H)
    m_test = tt >= i_val_end + purge

    out = {}
    for name, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
        out[f"{name}_i"] = ti[m]
        out[f"{name}_t"] = tt[m]
    print(f"anchors          : train {m_train.sum():,} | val {m_val.sum():,} "
          f"| test {m_test.sum():,}   (purge gap {purge} bars)")
    print(f"anchor dates     : train -> {pd.Timestamp(anchor_date[m_train].max()).date()} | "
          f"val {pd.Timestamp(anchor_date[m_val].min()).date()} -> "
          f"{pd.Timestamp(anchor_date[m_val].max()).date()} | "
          f"test {pd.Timestamp(anchor_date[m_test].min()).date()} ->")
    return out
