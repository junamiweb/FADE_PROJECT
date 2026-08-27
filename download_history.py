"""Historical OHLCV downloader for FADE (past data only, no live feeds).

Sources:
  * Binance public data endpoint (data-api.binance.vision) — full hourly / minute
    history back to 2017. No API key, historical klines only.
  * Yahoo Finance (yfinance) — full daily history back to 2014.

Usage:
    python download_history.py hourly     -> btc_1h.csv   (Binance, ~2017->now)
    python download_history.py daily      -> btc_daily.csv (Yahoo, max)
    python download_history.py minutevol  -> btc_1m_vol.csv (1m around high-vol)
    python download_history.py all

    # Arbitrary Binance interval (continuous full history):
    python download_history.py interval 5m    -> btc_5m.csv
    python download_history.py interval 15m   -> btc_15m.csv
    python download_history.py interval 30m   -> btc_30m.csv

    # Any asset (btc/eth) + interval:
    python download_history.py asset eth 15m  -> eth_15m.csv
    python download_history.py ethall         -> eth_5m/15m/30m/1h

    # Resample an existing finer file to 10-minute bars:
    python download_history.py resample10m    -> btc_10m.csv (from btc_5m.csv)

    # Seconds: recent continuous window only (limited history on Binance):
    python download_history.py seconds [days] -> btc_1s.csv (default 7 days)

    # Incremental refresh of all standard BTC+ETH files:
    python download_history.py refresh
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

BINANCE = "https://data-api.binance.vision/api/v3/klines"
# Polite pacing: data-api is generous but bulk alt downloads can 429 without gaps.
BINANCE_PAGE_SLEEP_S = 0.35
BINANCE_429_BASE_SLEEP_S = 5.0
SYMBOL = "BTCUSDT"
ASSETS = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# Full-history start per interval (kept early so downloads cover all regimes).
_INTERVAL_START = {
    "5m": "2017-08-01",
    "15m": "2017-08-01",
    "30m": "2017-08-01",
}


def _parse_retry_after(value: str | None, default: float) -> float:
    """Parse a Retry-After header (delay-seconds only; RFC 9110 also allows an
    HTTP-date, which we don't need to support here) - fall back safely on
    anything else instead of crashing the download."""
    if value is None:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default


def _get_with_retry(params: dict, max_retries: int = 8) -> requests.Response:
    """GET with backoff on 429/418, transient network errors, and 5xx."""
    backoff = BINANCE_429_BASE_SLEEP_S
    last_status: int | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(BINANCE, params=params, timeout=60)
            if resp.status_code == 429:
                last_status = 429
                wait = _parse_retry_after(resp.headers.get("Retry-After"), min(120, backoff))
                print(f"  Binance 429 rate limit, sleep {wait}s...")
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                continue
            if resp.status_code == 418:
                last_status = 418
                wait = min(180, backoff * 2)
                print(f"  Binance 418 (ban/throttle), sleep {wait}s...")
                time.sleep(wait)
                backoff = min(backoff * 2, 180)
                continue
            resp.raise_for_status()
            return resp
        except requests.HTTPError:
            # 5xx is the server's problem and usually transient - retry it
            # like any other transient error. 4xx (other than 429/418, which
            # are handled above) means the request itself is bad - fail fast.
            if 500 <= resp.status_code < 600 and attempt < max_retries - 1:
                wait = min(2 ** attempt, 60)
                print(f"  retry {attempt + 1}/{max_retries} after {wait}s "
                      f"(HTTP {resp.status_code})")
                time.sleep(wait)
                continue
            raise
        except (requests.RequestException, ConnectionError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 60)
            print(f"  retry {attempt + 1}/{max_retries} after {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(
        f"Binance rate limit (HTTP {last_status}) persisted for {max_retries} "
        "consecutive attempts - giving up."
    )


def fetch_binance(interval: str, start: str, end: str | None = None,
                  symbol: str = SYMBOL, limit: int = 1000,
                  page_sleep: float = BINANCE_PAGE_SLEEP_S) -> pd.DataFrame:
    """Paginated Binance klines download into a clean OHLCV frame."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end
                  else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)

    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "limit": limit}
        resp = _get_with_retry(params)
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][6] + 1  # last close-time + 1ms
        if len(batch) < limit:
            break
        if len(rows) % 50000 == 0:
            print(f"  ... {len(rows):,} rows fetched")
        time.sleep(page_sleep)

    df = pd.DataFrame(rows, columns=[
        "openTime", "open", "high", "low", "close", "volume", "closeTime",
        "qav", "trades", "tbav", "tqav", "ignore"])
    df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df = df[_COLS].astype({c: float for c in _COLS[1:]})
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_yahoo_daily() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("BTC-USD", interval="1d", period="max", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "timestamp", "Datetime": "timestamp", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[_COLS].dropna().reset_index(drop=True)


def fetch_minute_high_vol(top_windows: int = 40, window_hours: int = 12,
                          hourly_csv: str = "btc_1h.csv") -> pd.DataFrame:
    """Fetch 1-minute bars around the most volatile hourly moments.

    High-volatility periods are where fine structure (if any) should be richest.
    We locate them from hourly data, then pull 1m klines for a window around
    each. A ``segment`` id marks each contiguous block so downstream code never
    computes returns across the time gaps between windows.
    """
    hourly = pd.read_csv(hourly_csv, parse_dates=["timestamp"])
    ret = hourly["close"].pct_change().abs()
    vol = ret.rolling(24).std()
    hourly = hourly.assign(vol=vol).dropna(subset=["vol"])

    # Non-overlapping top windows: greedily pick highest-vol hours, skip if too
    # close to an already-chosen center.
    order = hourly.sort_values("vol", ascending=False)
    centers: list[pd.Timestamp] = []
    min_gap = pd.Timedelta(hours=window_hours * 2)
    for ts in order["timestamp"]:
        if all(abs(ts - c) > min_gap for c in centers):
            centers.append(ts)
        if len(centers) >= top_windows:
            break
    centers.sort()

    frames = []
    half = pd.Timedelta(hours=window_hours)
    for seg_id, center in enumerate(centers):
        start = (center - half).strftime("%Y-%m-%d %H:%M:%S")
        end = (center + half).strftime("%Y-%m-%d %H:%M:%S")
        block = fetch_binance("1m", start, end)
        block["segment"] = seg_id
        block["center"] = center
        frames.append(block)
        time.sleep(0.2)

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["timestamp"]).sort_values(["segment", "timestamp"]).reset_index(drop=True)


def fetch_seconds(days: int = 7, symbol: str = SYMBOL) -> pd.DataFrame:
    """Recent continuous 1-second window (Binance keeps limited 1s history)."""
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return fetch_binance("1s", start, symbol=symbol)


def resample_to_10m(src: str = "btc_5m.csv") -> pd.DataFrame:
    """Aggregate 5-minute bars into 10-minute bars (2:1) with correct OHLCV."""
    df = pd.read_csv(src, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    agg = df.resample("10min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"})
    agg = agg.dropna(subset=["open", "close"]).reset_index()
    return agg[_COLS]


def _save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    span = f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
    print(f"  saved {path}: {len(df):,} rows   [{span}]")


def _fetch_and_save(path: str, interval: str, start: str, symbol: str = SYMBOL) -> None:
    """Download full history or append from existing file's last bar."""
    p = Path(path)
    if p.exists():
        existing = pd.read_csv(p, parse_dates=["timestamp"])
        last = existing["timestamp"].max()
        start = (last + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  existing {path}: {len(existing):,} rows, appending from {start}...")
        new = fetch_binance(interval, start, symbol=symbol)
        if new.empty:
            print(f"  already up to date")
            return
        df = pd.concat([existing, new], ignore_index=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        df = fetch_binance(interval, start, symbol=symbol)
    _save(df, path)


_REFRESH_JOBS = [
    ("btc", "1h", "2017-08-01"),
    ("btc", "15m", "2017-08-01"),
    ("btc", "30m", "2017-08-01"),
    ("btc", "5m", "2017-08-01"),
    ("eth", "1h", "2017-08-01"),
    ("eth", "15m", "2017-08-01"),
    ("eth", "30m", "2017-08-01"),
    ("eth", "5m", "2017-08-01"),
]


def _refresh_all() -> None:
    for asset, interval, start in _REFRESH_JOBS:
        symbol = ASSETS.get(asset, f"{asset.upper()}USDT")
        path = f"{asset}_{interval}.csv"
        print(f"Refresh {symbol} {interval}...")
        _fetch_and_save(path, interval, start, symbol=symbol)


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("hourly", "all"):
        print("Binance hourly (full history)...")
        _save(fetch_binance("1h", "2017-08-01"), "btc_1h.csv")

    if what in ("daily", "all"):
        print("Yahoo daily (max)...")
        _save(fetch_yahoo_daily(), "btc_daily.csv")

    if what in ("minutevol", "all"):
        print("Binance 1-minute around high-volatility windows...")
        _save(fetch_minute_high_vol(), "btc_1m_vol.csv")

    if what == "interval":
        interval = sys.argv[2]
        start = _INTERVAL_START.get(interval, "2017-08-01")
        path = f"btc_{interval}.csv"
        print(f"Binance {interval} (continuous from {start})...")
        _fetch_and_save(path, interval, start)

    if what == "asset":
        asset = sys.argv[2].lower()
        interval = sys.argv[3]
        symbol = ASSETS.get(asset, f"{asset.upper()}USDT")
        start = _INTERVAL_START.get(interval, "2017-08-01")
        path = f"{asset}_{interval}.csv"
        print(f"Binance {symbol} {interval} (from {start})...")
        _fetch_and_save(path, interval, start, symbol=symbol)

    if what == "ethall":
        for interval in ("5m", "15m", "30m", "1h"):
            symbol = ASSETS["eth"]
            start = _INTERVAL_START.get(interval, "2017-08-01")
            print(f"Binance ETHUSDT {interval} (from {start})...")
            _save(fetch_binance(interval, start, symbol=symbol), f"eth_{interval}.csv")

    if what == "resample10m":
        print("Resampling btc_5m.csv -> 10-minute bars...")
        _save(resample_to_10m(), "btc_10m.csv")

    if what == "seconds":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        print(f"Binance 1-second (last {days} days)...")
        _save(fetch_seconds(days), "btc_1s.csv")

    if what == "refresh":
        _refresh_all()


if __name__ == "__main__":
    main()
