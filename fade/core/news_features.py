"""News feature engineering — attach daily GDELT news to hourly atom pool.

Reads news_btc.csv (date, news_tone, news_volume), computes causal features,
and merges them onto the hourly index via forward-fill.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fade.utils.logging import get_logger

log = get_logger("news_features")

VOL_Z_WINDOW = 30

# News coverage is currently BTC-only (see download_news.py), keyed by asset
# symbol rather than the OHLCV filename's timeframe suffix.
_NEWS_FILES_BY_SYMBOL = {"btc": "news_btc.csv"}
_TIMEFRAME_SUFFIX = re.compile(r"_(1s|1m|5m|15m|30m|1h|4h|1d)$")


def resolve_news_csv(asset: str, repo_root: str | Path) -> Path | None:
    """Resolve the news CSV for an asset (e.g. ``"eth_1h"``), anchored to
    ``repo_root`` rather than the process's current working directory.

    Returns ``None`` when no news file is available for this asset's symbol
    (news coverage is currently BTC-only, so a non-BTC asset never gets
    another asset's news attached to it).
    """
    symbol = _TIMEFRAME_SUFFIX.sub("", asset.lower())
    name = _NEWS_FILES_BY_SYMBOL.get(symbol)
    if name is None:
        return None
    path = Path(repo_root) / name
    return path if path.exists() else None


def load_news(news_csv: str) -> pd.DataFrame:
    df = pd.read_csv(news_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "news_tone" not in df.columns and "tone" in df.columns:
        df = df.rename(columns={"tone": "news_tone"})
    if "news_volume" not in df.columns and "volume" in df.columns:
        df = df.rename(columns={"volume": "news_volume"})
    for need in ("date", "news_tone", "news_volume"):
        if need not in df.columns:
            raise ValueError(f"news CSV missing column: {need}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    return df.set_index("date")[["news_tone", "news_volume"]].astype(float)


def compute_news_features(news_csv: str) -> pd.DataFrame:
    news = load_news(news_csv)
    tone = news["news_tone"]
    volume = news["news_volume"]
    vmean = volume.rolling(VOL_Z_WINDOW).mean()
    vstd = volume.rolling(VOL_Z_WINDOW).std().replace(0.0, np.nan)
    vol_z = (volume - vmean) / vstd
    feats = pd.DataFrame(
        {
            "news_tone": tone,
            "news_tone_chg": tone.diff(),
            "news_vol_z": vol_z,
        },
        index=news.index,
    )
    # Day D's aggregate news is only fully known at day-end, so it becomes
    # available starting day D+1 — lag the index by one day (no look-ahead).
    feats.index = feats.index + pd.Timedelta(days=1)
    return feats


def attach_news_to_pool(pool: pd.DataFrame, news_csv: str) -> pd.DataFrame:
    feats = compute_news_features(news_csv)
    pool = pool.copy()
    day_idx = (
        pool.index.tz_convert("UTC").normalize()
        if pool.index.tz
        else pd.to_datetime(pool.index).tz_localize("UTC").normalize()
    )
    aligned = feats.reindex(day_idx).ffill()
    pool["news_tone"] = aligned["news_tone"].to_numpy()
    pool["news_tone_chg"] = aligned["news_tone_chg"].to_numpy()
    pool["news_vol_z"] = aligned["news_vol_z"].to_numpy()
    return pool
