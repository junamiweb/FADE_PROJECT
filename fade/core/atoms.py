"""Atom computation - the strict feature set.

Only these five atoms are allowed in v0.1:
    return_1h, return_6h, volatility (rolling std 24h),
    volume z-score (24h), trend slope (6-24h).

All operations are vectorised pandas/numpy. No loops over rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fade.config import ATOM_COLUMNS, Config
from fade.core.candle_patterns import compute_candle_patterns
from fade.core.tiktok_chart_patterns import compute_tiktok_chart_patterns

# Bump whenever compute_atom_pool's set of output columns changes, so callers
# that cache the pool by content hash (e.g. fade.pipeline.main) invalidate
# stale entries instead of silently returning a pool with missing columns.
POOL_SCHEMA_VERSION = 2


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Vectorised rolling OLS slope, normalised by window mean price.

    slope_t = cov(x, y) / var(x) over the trailing ``window`` bars, where
    x = [0, 1, ..., window-1]. Normalising by the window mean turns the raw
    slope into a per-bar relative trend so it is comparable across price levels.
    """
    y = series.to_numpy(dtype=float)
    n = y.shape[0]
    out = np.full(n, np.nan)
    if n >= window:
        win = sliding_window_view(y, window)          # (n-window+1, window)
        x = np.arange(window, dtype=float)
        xc = x - x.mean()
        denom = (xc * xc).sum()
        yc = win - win.mean(axis=1, keepdims=True)
        slope = (yc * xc).sum(axis=1) / denom
        norm = win.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(norm != 0, slope / norm, np.nan)
        out[window - 1:] = slope
    return pd.Series(out, index=series.index, name="trend_slope")


def _signed_run_length(returns: pd.Series) -> pd.Series:
    """Signed consecutive-direction run length INCLUSIVE of the current bar.

    At bar t the value is +m if the last m bars (including t) were all up, -m if
    all down, 0 if the current return is flat/NaN. This is a *path* feature: it
    encodes the recent trajectory, not just the instantaneous state. It is fully
    causal — it uses only returns up to and including t (the close at t is known
    at t), and the prediction target is the return at t+1, so there is no
    look-ahead.
    """
    s = np.sign(returns.to_numpy(dtype=float))
    s = np.nan_to_num(s, nan=0.0)
    out = np.zeros(len(s))
    run = 0
    for i in range(len(s)):
        d = s[i]
        if d == 0:
            run = 0
        elif run != 0 and np.sign(run) == d:
            run += int(d)
        else:
            run = int(d)
        out[i] = run
    return pd.Series(out, index=returns.index, name="streak_signed")


def _signed_magnitude_run(returns: pd.Series, vol: pd.Series, k: float) -> pd.Series:
    """Signed run length counting only *large* same-direction bars.

    A bar is "big" when |return| exceeds ``k * trailing_volatility`` at that bar.
    The run counts consecutive big bars of the same sign and RESETS on any small
    / flat / opposite bar. This isolates the hypothesis that a streak of large
    moves mean-reverts harder than a streak of small ones — a magnitude-aware
    path feature the plain sign-streak cannot express.

    Causal: uses only returns and trailing volatility up to and including t; the
    target is the return at t+1, so there is no look-ahead. ``vol`` is a trailing
    rolling std (already causal). Bars before volatility warms up are treated as
    small (run 0) since the threshold is NaN there.
    """
    r = returns.to_numpy(dtype=float)
    v = vol.to_numpy(dtype=float)
    out = np.zeros(len(r))
    run = 0
    for i in range(len(r)):
        thr = k * v[i]
        if not np.isfinite(r[i]) or not np.isfinite(thr) or abs(r[i]) <= thr:
            run = 0  # small / flat / undefined -> streak of big moves breaks
        else:
            d = 1 if r[i] > 0 else -1
            if run != 0 and np.sign(run) == d:
                run += d
            else:
                run = d
        out[i] = run
    return pd.Series(out, index=returns.index, name="streak_big")


def compute_atom_pool(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Compute the full candidate-atom pool (all OHLCV-only, no look-ahead).

    The core 5 plus extra candidates used by the grid search. Every atom is a
    trailing-window statistic; none reads the future.
    """
    config = config or Config()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    return_1h = close.pct_change(1)
    return_6h = close.pct_change(config.return_6h_window)
    volatility = return_1h.rolling(config.volatility_window).std()

    vol_mean = volume.rolling(config.volume_window).mean()
    vol_std = volume.rolling(config.volume_window).std()
    volume_zscore = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

    trend_slope = _rolling_slope(close, config.trend_window)

    # --- extra candidates -------------------------------------------------
    # Intrabar range as a fraction of price (pure per-bar volatility).
    range_pct = (high - low) / close.replace(0.0, np.nan)
    # Where the close sits within the bar range: 0=at low, 1=at high.
    span = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / span
    # Return acceleration: change in 1-bar return (momentum of momentum).
    return_accel = return_1h.diff()
    # Volume trend: normalised rolling slope of volume.
    volume_trend = _rolling_slope(volume, config.trend_window)

    # --- path / temporal atoms (encode trajectory, not just state) --------
    # Signed consecutive-direction run length (inclusive, causal). The recent
    # sequence work showed intraday BTC mean-reverts after streaks, a mechanism
    # the memoryless state atoms cannot express.
    streak_signed = _signed_run_length(return_1h)
    # Magnitude-conditioned run: streak of LARGE moves only (tests whether big-
    # move streaks reverse harder). Threshold = k * trailing volatility.
    streak_big = _signed_magnitude_run(return_1h, volatility, config.streak_big_k)

    # --- named candlestick patterns (0/1 flags, causal) -------------------
    candles = compute_candle_patterns(df)
    tiktok = compute_tiktok_chart_patterns(df)

    pool = pd.DataFrame(
        {
            "return_1h": return_1h,
            "return_6h": return_6h,
            "volatility": volatility,
            "volume_zscore": volume_zscore,
            "trend_slope": trend_slope,
            "range_pct": range_pct,
            "close_pos": close_pos,
            "return_accel": return_accel,
            "volume_trend": volume_trend,
            "streak_signed": streak_signed,
            "streak_big": streak_big,
            "doji": candles["doji"],
            "bullish_engulfing": candles["bullish_engulfing"],
            "bearish_engulfing": candles["bearish_engulfing"],
            "breakout_up": tiktok["breakout_up"],
            "breakout_down": tiktok["breakout_down"],
            "bull_flag": tiktok["bull_flag"],
            "bear_flag": tiktok["bear_flag"],
            "double_top": tiktok["double_top"],
            "double_bottom": tiktok["double_bottom"],
            "trend_hh_hl": tiktok["trend_hh_hl"],
            "trend_lh_ll": tiktok["trend_lh_ll"],
        },
        index=df.index,
    )

    # Carry over news atoms if the caller pre-attached them onto df (e.g. via
    # fade.core.news_features.attach_news_to_pool before loading). They are
    # not derivable from OHLCV alone, so the "news_dl" atom set only resolves
    # when the input frame already has them.
    for col in ("news_tone", "news_tone_chg", "news_vol_z"):
        if col in df.columns:
            pool[col] = df[col].to_numpy()

    return pool


def compute_atoms(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Compute the active atom set (``config.atom_columns``) from an OHLCV frame.

    Returns a frame with only the configured atoms. Warm-up rows containing
    NaNs (from rolling windows) are dropped.
    """
    config = config or Config()
    cols = list(config.atom_columns)
    pool = compute_atom_pool(df, config)
    atoms = pool[cols]
    return atoms.dropna()


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Realised return ``horizon`` bars into the future (the prediction target)."""
    close = df["close"]
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.rename("forward_return")
