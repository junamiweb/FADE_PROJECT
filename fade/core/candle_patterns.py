"""Named candlestick pattern flags (OHLC geometry, fully causal).

Definitions follow standard rule-based candlestick literature (MDPI-style
geometry on real bodies, no look-ahead):

    doji              : |close - open| < 10% of (high - low)
    bullish_engulfing : prior bar bearish, current bullish, current body
                        engulfs prior body (open <= prev_close, close >= prev_open)
    bearish_engulfing : prior bar bullish, current bearish, current body
                        engulfs prior body (open >= prev_close, close <= prev_open)

Each pattern at bar *t* uses only OHLC through *t* (engulfing also uses *t-1*).
The prediction target is the return at *t+1*, so there is no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised 0/1 flags for named candlestick patterns."""
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    span = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    body_ratio = body / span
    doji = (body_ratio < 0.10).astype(float)
    doji = doji.where(body_ratio.notna(), 0.0)

    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    prev_bear = prev_close < prev_open
    prev_bull = prev_close > prev_open
    curr_bull = close > open_
    curr_bear = close < open_

    bullish_engulfing = (
        prev_bear
        & curr_bull
        & (open_ <= prev_close)
        & (close >= prev_open)
    ).astype(float)
    bullish_engulfing = bullish_engulfing.fillna(0.0)

    bearish_engulfing = (
        prev_bull
        & curr_bear
        & (open_ >= prev_close)
        & (close <= prev_open)
    ).astype(float)
    bearish_engulfing = bearish_engulfing.fillna(0.0)

    return pd.DataFrame(
        {
            "doji": doji,
            "bullish_engulfing": bullish_engulfing,
            "bearish_engulfing": bearish_engulfing,
        },
        index=df.index,
    )
