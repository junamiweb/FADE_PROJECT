"""Funding + streak reversal combo — v0.3 pre-registered holdout study.

Tests whether extreme funding regimes modulate streak>=3 contrarian hit-rate
on the NEXT 1h bar. Thresholds fitted on dev only. Holdout exploratory — NOT lockbox.

Hypothesis (from decay H1): EXTREME_NEG funding may strengthen local reversal.

Run:
    python -m fade.pipeline.funding_streak_combo
    python -m fade.pipeline.funding_streak_combo --price eth_1h.csv --funding funding_eth.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.pre_registration import load_manifest, save_manifest
from fade.pipeline.trend_structure import _signed_streak

STUDY_ID = "funding_streak_combo_1h"
HOLDOUT_FRAC = 0.30
MIN_N = 80
OUTPUT_PATH = Path("fade/output/funding_streak_combo.json")


def _load_funding(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    s = pd.Series(df["funding_rate"].astype(float).values, index=ts).sort_index()
    return s.groupby(s.index.floor("h")).last()


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if not any(s.get("study_id") == STUDY_ID for s in studies):
        studies.append({
            "study_id": STUDY_ID,
            "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
            "rule": "streak>=3 contrarian on next 1h bar",
            "funding_buckets": ["EXTREME_NEG", "NEUTRAL", "EXTREME_POS"],
            "thresholds": "dev p10/p90 on funding",
            "data_split": "holdout_70_30_exploratory",
        })
        save_manifest(m)


def run_combo(price_csv: str, funding_csv: str,
              holdout_frac: float = HOLDOUT_FRAC) -> dict:
    df = load_ohlcv(price_csv)
    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    funding = _load_funding(funding_csv).reindex(df.index, method="ffill")

    frame = pd.DataFrame({
        "streak": streak, "slen": np.abs(streak), "fwd": fwd, "fund": funding,
    }, index=df.index).dropna(subset=["fwd", "fund"])

    n = len(frame)
    split = int(n * (1 - holdout_frac))
    dev, hold = frame.iloc[:split], frame.iloc[split:]

    p10 = float(dev["fund"].quantile(0.10))
    p90 = float(dev["fund"].quantile(0.90))

    def _bucket(f: float) -> str:
        if f <= p10:
            return "EXTREME_NEG"
        if f >= p90:
            return "EXTREME_POS"
        return "NEUTRAL"

    hold = hold.copy()
    hold["bucket"] = hold["fund"].map(_bucket)

    rows = []
    for bucket in ("EXTREME_NEG", "NEUTRAL", "EXTREME_POS"):
        sub = hold[(hold["bucket"] == bucket) & (hold["slen"] >= 3)]
        k = len(sub)
        if k < MIN_N:
            rows.append({"bucket": bucket, "n": k, "status": "low"})
            continue
        pred = (sub["streak"] < 0).astype(int)
        actual = (sub["fwd"] > 0).astype(int)
        hit = float(np.mean(pred.to_numpy() == actual.to_numpy()))
        rows.append({
            "bucket": bucket, "n": k, "status": "ok",
            "hit_rate": round(hit, 4), "edge": round(hit - 0.5, 4),
        })

    ok = [r for r in rows if r.get("status") == "ok"]
    best = max(ok, key=lambda r: r["hit_rate"]) if ok else None
    neg = next((r for r in ok if r["bucket"] == "EXTREME_NEG"), None)
    neu = next((r for r in ok if r["bucket"] == "NEUTRAL"), None)

    boost = None
    if neg and neu:
        boost = round(neg["hit_rate"] - neu["hit_rate"], 4)

    return {
        "asset": Path(price_csv).stem,
        "funding": Path(funding_csv).name,
        "dev_p10": round(p10, 6),
        "dev_p90": round(p90, 6),
        "holdout_bars": len(hold),
        "buckets": rows,
        "extreme_neg_boost_vs_neutral": boost,
        "best_bucket": best["bucket"] if best else None,
        "verdict": _verdict(rows, boost),
    }


def _verdict(rows: list, boost: float | None) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return "INCONCLUSIVE — insufficient holdout coverage per bucket."
    neg = next((r for r in ok if r["bucket"] == "EXTREME_NEG"), None)
    if neg and neg["hit_rate"] >= 0.58:
        return f"STRONG in EXTREME_NEG: {neg['hit_rate']} (exploratory holdout)."
    if boost is not None and boost >= 0.02:
        return f"MODERATE boost in EXTREME_NEG (+{boost*100:.1f}pp vs neutral) — exploratory."
    return "NO clear funding modulation of streak reversal on holdout."


def run_all(pairs: list[tuple[str, str]]) -> dict:
    _ensure_preregistered()
    results = [run_combo(p, f) for p, f in pairs]
    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def _print(r: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("FUNDING + STREAK REVERSAL COMBO (holdout exploratory)")
    print(line)
    for item in r.get("results", []):
        print(f"\n  {item['asset']}  funding={item['funding']}  p10={item['dev_p10']} p90={item['dev_p90']}")
        for b in item.get("buckets", []):
            if b.get("status") != "ok":
                print(f"    {b['bucket']:<14} n={b['n']}  low")
            else:
                print(f"    {b['bucket']:<14} n={b['n']}  hit={b['hit_rate']}  edge={b['edge']:+.4f}")
        print(f"    -> {item['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding streak combo test")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="price.csv:funding.csv pairs")
    args = parser.parse_args()
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        pairs = [("btc_1h.csv", "funding_btc.csv"), ("eth_1h.csv", "funding_eth.csv")]
    r = run_all(pairs)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
