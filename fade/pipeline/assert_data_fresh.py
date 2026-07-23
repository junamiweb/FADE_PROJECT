"""Fail if BTC/ETH 1h CSVs are stale after refresh (Action health gate).

GitHub Actions can finish green with *no* data updates when Binance returns
empty from the runner — then ledgers freeze while the workflow looks healthy.

Run:
    python -m fade.pipeline.assert_data_fresh
    python -m fade.pipeline.assert_data_fresh --max-lag-hours 3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_FILES = ("btc_1h.csv", "eth_1h.csv")


def _last_bar_utc(path: Path) -> pd.Timestamp:
    df = pd.read_csv(path, usecols=["timestamp"], parse_dates=["timestamp"])
    if df.empty:
        raise ValueError(f"{path}: empty")
    ts = pd.Timestamp(df["timestamp"].max())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def check_fresh(
    files: tuple[str, ...] = DEFAULT_FILES,
    max_lag_hours: float = 3.0,
    now: datetime | None = None,
) -> dict:
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    max_lag = pd.Timedelta(hours=max_lag_hours)
    rows = []
    ok = True
    for name in files:
        path = Path(name)
        if not path.exists():
            rows.append({"file": name, "ok": False, "error": "missing"})
            ok = False
            continue
        last = _last_bar_utc(path)
        lag = pd.Timestamp(now_utc) - last
        file_ok = lag <= max_lag
        if not file_ok:
            ok = False
        rows.append({
            "file": name,
            "ok": file_ok,
            "last_bar_utc": str(last),
            "lag_hours": round(float(lag.total_seconds()) / 3600.0, 3),
            "max_lag_hours": max_lag_hours,
        })
    return {
        "ok": ok,
        "checked_utc": now_utc.isoformat(),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert OHLCV CSVs are fresh")
    parser.add_argument(
        "--max-lag-hours", type=float, default=3.0,
        help="Fail if last 1h bar older than this (default 3h)",
    )
    parser.add_argument(
        "--files", nargs="*", default=list(DEFAULT_FILES),
    )
    args = parser.parse_args()
    result = check_fresh(tuple(args.files), max_lag_hours=args.max_lag_hours)
    for row in result["files"]:
        status = "OK" if row.get("ok") else "STALE"
        if "error" in row:
            print(f"  {status}  {row['file']}: {row['error']}")
        else:
            print(
                f"  {status}  {row['file']}: last={row['last_bar_utc']}  "
                f"lag={row['lag_hours']}h (max {row['max_lag_hours']}h)"
            )
    if not result["ok"]:
        print(
            "FAIL: data stale after refresh — Action would otherwise look green "
            "while forward ledgers freeze. Check Binance reachability from runner.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("PASS: data fresh")


if __name__ == "__main__":
    main()
