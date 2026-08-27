"""TikTok-popular chart pattern flags — causal OHLC geometry only.

Maps names commonly used in short-form trading content (breakout, bull flag,
double top, etc.) to rule-based detectors on historical bars. We do NOT scrape
TikTok or use social feeds — only offline CSV OHLCV (SCOPE_GUARD invariant 3).

Each flag at bar t uses data through t only; target is t+1 return (no look-ahead).

Patterns (pre-registered set for holdout A/B vs path_lean3):
    breakout_up / breakout_down   — close clears rolling N-bar high/low
    bull_flag / bear_flag         — impulse + tight consolidation
    double_top / double_bottom    — two touches of rolling extreme band
    trend_hh_hl / trend_lh_ll     — higher-highs+higher-lows / lower structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 24
IMPULSE_BARS = 5
FLAG_BARS = 5
IMPULSE_PCT = 0.015
FLAG_VOL_RATIO = 0.55


def compute_tiktok_chart_patterns(
    df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """Binary 0/1 flags aligned to df.index."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ret = close.pct_change()

    roll_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    roll_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
    breakout_valid = roll_high.notna() & roll_low.notna()

    breakout_up = (close > roll_high).astype(float).where(breakout_valid)
    breakout_down = (close < roll_low).astype(float).where(breakout_valid)

    impulse_up = close.pct_change(IMPULSE_BARS) >= IMPULSE_PCT
    impulse_dn = close.pct_change(IMPULSE_BARS) <= -IMPULSE_PCT
    vol_long = ret.rolling(lookback, min_periods=lookback).std()
    vol_short = ret.rolling(FLAG_BARS, min_periods=FLAG_BARS).std()
    tight = vol_short <= (vol_long * FLAG_VOL_RATIO)
    flag_valid = vol_long.notna() & vol_short.notna()
    bull_flag = (impulse_up & tight).astype(float).where(flag_valid)
    bear_flag = (impulse_dn & tight).astype(float).where(flag_valid)

    # Shifted by 1, like roll_high/roll_low above: the "established"
    # resistance/support must come from bars BEFORE the current one, not
    # include the current bar's own high/low (which would trivially make
    # every fresh extreme "near" itself).
    rh = high.rolling(lookback, min_periods=lookback).max().shift(1)
    rl = low.rolling(lookback, min_periods=lookback).min().shift(1)
    band = (rh - rl).replace(0, np.nan)
    double_valid = rh.notna() & rl.notna() & band.notna()
    near_high = high >= (rh - 0.08 * band)
    near_low = low <= (rl + 0.08 * band)
    # Reject below the established rolling resistance/support (rh/rl), not
    # just below the current bar's own high/low - a bar's close is almost
    # always below its own high, which made this condition nearly always
    # true and left double_top/bottom driven only by the touch count.
    double_top = (
        near_high
        & (near_high.rolling(lookback, min_periods=lookback).sum() >= 2)
        & (close < rh * 0.999)
    ).astype(float).where(double_valid)
    double_bottom = (
        near_low
        & (near_low.rolling(lookback, min_periods=lookback).sum() >= 2)
        & (close > rl * 1.001)
    ).astype(float).where(double_valid)

    hh = high.diff() > 0
    hl = low.diff() > 0
    lh = high.diff() < 0
    ll = low.diff() < 0
    win = lookback // 2
    hh_mean = hh.rolling(win, min_periods=win).mean()
    hl_mean = hl.rolling(win, min_periods=win).mean()
    lh_mean = lh.rolling(win, min_periods=win).mean()
    ll_mean = ll.rolling(win, min_periods=win).mean()
    trend_valid = hh_mean.notna() & hl_mean.notna() & lh_mean.notna() & ll_mean.notna()
    trend_hh_hl = ((hh_mean >= 0.55) & (hl_mean >= 0.55)).astype(float).where(trend_valid)
    trend_lh_ll = ((lh_mean >= 0.55) & (ll_mean >= 0.55)).astype(float).where(trend_valid)

    # Warm-up bars (where the defining rolling stat isn't settled yet) are
    # left as NaN rather than fabricated as "pattern absent" - callers
    # already dropna() the atom pool before use, same as every other atom.
    return pd.DataFrame(
        {
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "bull_flag": bull_flag,
            "bear_flag": bear_flag,
            "double_top": double_top,
            "double_bottom": double_bottom,
            "trend_hh_hl": trend_hh_hl,
            "trend_lh_ll": trend_lh_ll,
        },
        index=df.index,
    )
