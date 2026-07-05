"""OHLCV data loading and validation.

Single asset, 1H timeframe only (v0.1 restriction). We keep the raw frame lean:
only the six canonical columns are retained, sorted by time and de-duplicated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV into a clean, time-indexed DataFrame.

    Expected columns (case-insensitive): timestamp, open, high, low, close,
    volume. Returns a frame indexed by a sorted, unique DatetimeIndex.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    ts_col = next((c for c in ("timestamp", "time", "date", "datetime") if c in df.columns), None)
    if ts_col is None:
        raise ValueError("CSV must contain a timestamp/time/date column.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df[list(REQUIRED_COLUMNS)].astype(float)
    df = df.dropna()
    if df.empty:
        raise ValueError("No valid rows after cleaning.")
    return df


def generate_synthetic_ohlcv(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Generate BTC-like 1H OHLCV data for reproducible demos/tests.

    Uses a regime-switching geometric random walk with volatility clustering so
    that *some* structure exists to be discovered, without hard-coding signal.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")

    # Volatility clustering via a slow-moving latent state.
    vol_state = np.zeros(n)
    v = 0.004
    for i in range(n):
        v = 0.9 * v + 0.1 * abs(rng.normal(0.006, 0.003))
        vol_state[i] = max(v, 0.001)

    # Mild autocorrelation in returns (the "atomic structure" to detect).
    rets = np.zeros(n)
    prev = 0.0
    for i in range(n):
        shock = rng.normal(0.0, vol_state[i])
        rets[i] = 0.05 * prev + shock
        prev = rets[i]

    close = 20000.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    spread = np.abs(rng.normal(0.0, vol_state)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=6.0, sigma=0.5, size=n) * (1 + 5 * vol_state)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
