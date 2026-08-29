"""Funding carry forward tracker — live ledger of the funding regime.

Context: funding_carry_v1 passed its pre-registered criteria but realism_v2
showed the trade is economically marginal at the CURRENT funding regime
(~5%/yr on notional vs ~4% risk-free). The carry is only worth acting on in
hot-funding regimes (2021/2024-style, 15-30%+ annualized). This tracker
exists to (a) build an honest FORWARD record of realized carry — same
discipline as every other candidate — and (b) flag regime shifts live.

Each tick:
  1. Refreshes funding_btc.csv / funding_eth.csv incrementally (Binance
     public API, free).
  2. Appends one JSONL line per NEW 8h funding period per asset to
     fade/output/funding_carry_ledger.jsonl — recording the funding rate,
     the gate state (trailing 7d mean > 0, decided from data <= previous
     period), and the regime meter (trailing 30d funding annualized).

The ledger records raw facts; net-PnL under any fee/capital model can be
derived from it later without re-fetching history.

Run (hourly Action / manual):
    python -m fade.pipeline.funding_carry_tracker tick
    python -m fade.pipeline.funding_carry_tracker report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GATE_WINDOW = 21            # 7d x 3 periods/day, matches funding_carry_v1
REGIME_WINDOW = 90          # 30d x 3 periods/day
PERIODS_PER_YEAR = 3 * 365
HOT_REGIME_ANNUALIZED = 0.10   # trailing 30d funding above this = HOT flag

LEDGER_PATH = Path("fade/output/funding_carry_ledger.jsonl")
ASSETS = {"btc": ("BTCUSDT", Path("funding_btc.csv")),
          "eth": ("ETHUSDT", Path("funding_eth.csv"))}


def _load_funding(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    s = pd.Series(df["funding_rate"].astype(float).values, index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _refresh(symbol: str, path: Path) -> pd.Series:
    from download_derivatives import fetch_funding
    existing = _load_funding(path) if path.exists() else pd.Series(dtype=float)
    start = (existing.index.max() - pd.Timedelta(hours=1)).strftime("%Y-%m-%d") \
        if len(existing) else "2019-09-01"
    fresh = fetch_funding(symbol, start=start)
    if not fresh.empty:
        fresh_s = pd.Series(fresh["funding_rate"].values,
                            index=pd.to_datetime(fresh["timestamp"], utc=True))
        merged = pd.concat([existing, fresh_s]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        out = pd.DataFrame({"timestamp": merged.index, "funding_rate": merged.values})
        out.to_csv(path, index=False)
        return merged
    return existing


def _last_ledger_ts(asset: str) -> pd.Timestamp | None:
    if not LEDGER_PATH.exists():
        return None
    last = None
    with LEDGER_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("asset") == asset:
                last = rec["period_ts"]
    return pd.Timestamp(last) if last else None


def tick() -> dict:
    appended = {}
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        for asset, (symbol, csv_path) in ASSETS.items():
            funding = _refresh(symbol, csv_path)
            if funding.empty:
                appended[asset] = 0
                continue
            gate = funding.rolling(GATE_WINDOW, min_periods=GATE_WINDOW).mean().shift(1)
            regime = funding.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).mean() \
                .shift(1) * PERIODS_PER_YEAR

            last_ts = _last_ledger_ts(asset)
            new = funding[funding.index > last_ts] if last_ts is not None else funding.tail(1)
            n = 0
            for ts, rate in new.items():
                g = gate.get(ts)
                r = regime.get(ts)
                rec = {
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    "asset": asset,
                    "period_ts": str(ts),
                    "funding_rate": round(float(rate), 8),
                    "gate_on": bool(g > 0) if g == g and g is not None else None,
                    "regime_trailing_30d_annualized": round(float(r), 4) if r == r and r is not None else None,
                    "regime_hot": bool(r >= HOT_REGIME_ANNUALIZED) if r == r and r is not None else None,
                }
                fh.write(json.dumps(rec) + "\n")
                n += 1
            appended[asset] = n
    return {"status": "ok", "appended": appended,
            "ledger": str(LEDGER_PATH)}


def report() -> dict:
    if not LEDGER_PATH.exists():
        return {"status": "empty_ledger"}
    rows = []
    with LEDGER_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = {"status": "ok", "n_records": len(rows), "assets": {}}
    df = pd.DataFrame(rows)
    for asset, grp in df.groupby("asset"):
        grp = grp.sort_values("period_ts")
        gated = grp[grp["gate_on"] == True]  # noqa: E712
        realized = float(gated["funding_rate"].sum())
        last = grp.iloc[-1]
        out["assets"][asset] = {
            "n_periods": len(grp),
            "span": f"{grp['period_ts'].iloc[0]} -> {grp['period_ts'].iloc[-1]}",
            "gate_coverage_pct": round(float((grp["gate_on"] == True).mean()) * 100, 1),  # noqa: E712
            "realized_gated_carry_sum": round(realized, 6),
            "realized_gated_carry_annualized": (
                round(realized / max(len(grp), 1) * PERIODS_PER_YEAR, 4)
            ),
            "current_regime_30d_annualized": (
                float(last["regime_trailing_30d_annualized"])
                if pd.notna(last.get("regime_trailing_30d_annualized")) else None
            ),
            "current_regime_hot": (
                bool(last["regime_hot"]) if pd.notna(last.get("regime_hot")) else None
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding carry forward tracker")
    parser.add_argument("cmd", choices=("tick", "report"), default="tick", nargs="?")
    args = parser.parse_args()
    result = tick() if args.cmd == "tick" else report()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
