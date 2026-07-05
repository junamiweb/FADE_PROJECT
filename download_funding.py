"""Historical perpetual-futures funding rate downloader (Binance, no key).

Funding rate is an ORTHOGONAL, potentially LEADING signal — unlike news, which
is reactive. Mechanically: when leverage crowds one side, that side pays funding
to the other. Extreme positive funding = over-leveraged longs (squeeze risk down);
extreme negative = crowded shorts (squeeze risk up). So it has a causal basis for
predicting mean-reversion, which is exactly what makes it worth testing.

Source: https://fapi.binance.com/fapi/v1/fundingRate
  * BTCUSDT perpetual, full history from 2019-09-10, one point every 8 hours.
  * Fields kept: fundingTime (UTC), fundingRate, markPrice.

Open interest history is intentionally NOT downloaded: Binance only serves ~30
days of it, which is useless for a multi-year honest backtest.

Usage:
    python download_funding.py                 # BTCUSDT -> funding_btc.csv
    python download_funding.py --symbol ETHUSDT --out funding_eth.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
INCEPTION = "2019-09-01"


def fetch_funding(symbol: str = "BTCUSDT", start: str = INCEPTION,
                  end: str | None = None, limit: int = 1000) -> pd.DataFrame:
    """Paginated funding-rate download into a clean time-indexed frame."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end
                  else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)

    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "limit": limit}
        resp = requests.get(FAPI, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if last <= cursor:                 # no forward progress -> stop
            break
        cursor = last + 1
        if len(batch) < limit:
            break
        time.sleep(0.25)                   # polite; fapi limits are generous

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    # markPrice is blank in some early rows; keep as NaN (price comes from OHLCV).
    df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
    df = df.dropna(subset=["funding_rate"])
    df = df[["timestamp", "funding_rate", "mark_price"]]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    span = f"{df['timestamp'].min()} -> {df['timestamp'].max()}" if not df.empty else "empty"
    print(f"  saved {path}: {len(df):,} rows   [{span}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance funding-rate history")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=INCEPTION)
    parser.add_argument("--out", default="funding_btc.csv")
    args = parser.parse_args()
    print(f"Binance funding rate: {args.symbol} from {args.start}")
    df = fetch_funding(args.symbol, args.start)
    _save(df, args.out)


if __name__ == "__main__":
    main()
