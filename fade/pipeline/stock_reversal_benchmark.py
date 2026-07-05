"""Stock reversal benchmark — external sanity check for market-maturity hypothesis.

Computes the SAME raw reversion_index as scale_structure / generalization_why
(streak>=2 continuation -> rev_index = 0.5 - continuation) on 2-3 liquid equities.

Stock-specific handling (documented):
  - Session-aware streak: reset at each trading day (no overnight carry).
  - Adjusted close via yfinance auto_adjust=True (splits/dividends).
  - yfinance 1h limit: max ~730 days — documented as data constraint.

Compares:
  - SPY / AAPL today (available window)
  - BTC today (recent window matching stocks)
  - BTC 2018-2019 (full history from btc_1h.csv)

Run:
    python -m fade.pipeline.stock_reversal_benchmark
    python -m fade.pipeline.stock_reversal_benchmark --no-download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("stock_reversal")

TICKERS = ("SPY", "AAPL")
MIN_SUPPORT = 50
N_SHUFFLES = 1500


def _signed_streak_session(ret: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """Signed streak reset at each new trading day (no overnight carry)."""
    n = len(ret)
    prev_dir = np.where(ret > 0, 1, np.where(ret < 0, -1, 0))
    streak = np.zeros(n, dtype=int)
    run = 0
    prev_day = None
    for t in range(1, n):
        day = day_ids[t]
        if day != prev_day:
            run = 0
            prev_day = day
        d = prev_dir[t - 1]
        if d == 0:
            run = 0
        elif run != 0 and np.sign(run) == d:
            run += d
        else:
            run = d
        streak[t] = run
    return streak


def _reversion_index(ret: np.ndarray, streak: np.ndarray, mask: np.ndarray) -> dict:
    up = (ret > 0).astype(int)
    cont_hits = cont_n = 0
    for m in range(2, 8):
        for sgn in (m, -m):
            sel = mask & (streak == sgn)
            k = int(sel.sum())
            if k < 20:
                continue
            nxt = up[sel]
            cont_hits += int((nxt == (1 if sgn > 0 else 0)).sum())
            cont_n += k
    if cont_n < MIN_SUPPORT:
        return {"status": "low_support", "n": cont_n}
    cont = cont_hits / cont_n
    rev = 0.5 - cont
    rng = np.random.default_rng(0)
    base = float(up[mask].mean())
    null = rng.choice(up[mask], size=(N_SHUFFLES, cont_n), replace=True).mean(axis=1)
    dev = abs(cont - 0.5)
    p = (1 + int(np.sum(np.abs(null - base) >= dev))) / (1 + N_SHUFFLES)
    return {
        "status": "ok",
        "n": cont_n,
        "continuation": round(cont, 4),
        "reversion_index": round(rev, 4),
        "p_value": round(p, 4),
    }


def fetch_yahoo_hourly(ticker: str, period: str = "730d") -> pd.DataFrame:
    """Download hourly OHLCV via yfinance (adjusted close)."""
    import yfinance as yf
    raw = yf.download(ticker, interval="1h", period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Datetime": "timestamp", "Date": "timestamp",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    log.info("Saved %s (%d rows)", path, len(df))


def analyze_equity(df: pd.DataFrame, label: str, session_aware: bool = True) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    ret = df["close"].pct_change().to_numpy()
    day_ids = df["timestamp"].dt.date.to_numpy()
    if session_aware:
        streak = _signed_streak_session(ret, day_ids)
        streak_note = "session_reset_daily"
    else:
        streak = _signed_streak(ret)
        streak_note = "continuous_24h"
    mask = np.isfinite(ret)
    ri = _reversion_index(ret, streak, mask)
    span = [str(df["timestamp"].iloc[0])[:10], str(df["timestamp"].iloc[-1])[:10]]
    return {
        "label": label,
        "span": span,
        "bars": int(len(df)),
        "streak_method": streak_note,
        "adjusted_close": True,
        **ri,
    }


def analyze_btc_window(csv_path: str, start: str, end: str, label: str) -> dict:
    df = load_ohlcv(csv_path)
    sub = df.loc[start:end]
    if len(sub) < MIN_SUPPORT:
        return {"label": label, "status": "insufficient_data", "n": len(sub)}
    ret = sub["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    mask = np.isfinite(ret)
    ri = _reversion_index(ret, streak, mask)
    return {
        "label": label,
        "span": [str(sub.index[0])[:10], str(sub.index[-1])[:10]],
        "bars": len(sub),
        "streak_method": "continuous_24h_crypto",
        **ri,
    }


def run_benchmark(download: bool = True) -> dict:
    stock_results = {}
    if download:
        for t in TICKERS:
            try:
                df = fetch_yahoo_hourly(t)
                path = f"{t.lower()}_1h.csv"
                _save_csv(df, path)
                stock_results[t] = analyze_equity(df, t, session_aware=True)
            except Exception as exc:
                log.error("%s download failed: %s", t, exc)
                stock_results[t] = {"label": t, "status": "error", "error": str(exc)}
    else:
        for t in TICKERS:
            path = f"{t.lower()}_1h.csv"
            if Path(path).exists():
                df = pd.read_csv(path)
                stock_results[t] = analyze_equity(df, t, session_aware=True)
            else:
                stock_results[t] = {"label": t, "status": "missing"}

    # BTC windows
    btc_path = "btc_1h.csv"
    btc_today = analyze_btc_window(btc_path, "2024-01-01", "2026-12-31", "BTC_today")
    btc_2018 = analyze_btc_window(btc_path, "2018-01-01", "2019-12-31", "BTC_2018_2019")

    comparison = {}
    for k, v in {**stock_results, "BTC_today": btc_today, "BTC_2018_2019": btc_2018}.items():
        if v.get("status") == "ok":
            comparison[k] = {
                "reversion_index": v["reversion_index"],
                "continuation": v["continuation"],
                "p_value": v.get("p_value"),
                "n": v["n"],
                "span": v.get("span"),
            }

    # Interpretation
    spy_rev = comparison.get("SPY", {}).get("reversion_index")
    btc_now = comparison.get("BTC_today", {}).get("reversion_index")
    btc_old = comparison.get("BTC_2018_2019", {}).get("reversion_index")

    if spy_rev is not None and btc_now is not None and btc_old is not None:
        if abs(spy_rev) < 0.02 and btc_now < 0.03 and btc_old > 0.03:
            interp = "SUPPORTS market-maturity: BTC converging toward stock-like efficiency"
        elif btc_now > 0.03:
            interp = "BTC still shows meaningful reversal today — not fully matured"
        else:
            interp = "Mixed: BTC weak today but stocks also show some structure"
    else:
        interp = "incomplete comparison"

    return {
        "tickers": list(TICKERS),
        "data_limitation": "yfinance 1h max ~730 days; stocks use session-reset streak",
        "stock_results": stock_results,
        "btc_windows": {"BTC_today": btc_today, "BTC_2018_2019": btc_2018},
        "comparison": comparison,
        "interpretation": interp,
    }


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print("STOCK REVERSAL BENCHMARK (reversion_index = 0.5 - continuation)")
    print(line)
    print(f"  Limitation: {r['data_limitation']}")
    print(f"\n  {'asset':<16}{'rev_index':>10}{'cont':>8}{'p':>8}{'n':>8}{'span'}")
    for k, v in r.get("comparison", {}).items():
        print(f"  {k:<16}{v['reversion_index']:>+10.4f}{v['continuation']:>8.4f}"
              f"{v.get('p_value', 0):>8.4f}{v['n']:>8}  {v['span'][0]}->{v['span'][1]}")
    print()
    print(f"  INTERPRETATION: {r['interpretation']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock reversal benchmark")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--json-out", default="fade/output/stock_reversal_benchmark.json")
    args = parser.parse_args()
    result = run_benchmark(download=not args.no_download)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print(result)


if __name__ == "__main__":
    main()
