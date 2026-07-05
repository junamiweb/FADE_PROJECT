"""Conviction stability over time -- does the mechanism persist across years?

Splits history into calendar years and measures the FIXED contrarian rules
(no re-fitting) on each year's bars:
    streak>=3 contrarian hit-rate
    streak>=2 + 3-TF-agree combo (next 1h return)

If hit-rates stay above 50% across most years, the mechanism is stationary.
If they collapse in recent years, the edge may be decaying.

Run:
    python -m fade.pipeline.conviction_stability
    python -m fade.pipeline.conviction_stability eth_1h.csv
    python -m fade.pipeline.conviction_stability btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.trend_structure import _signed_streak

MIN_N = 80
MULTI_INTERVALS = ("5m", "15m", "30m", "1h")


def _asset_prefix(csv: str) -> str:
    stem = Path(csv).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _multi_files(prefix: str) -> list[str]:
    return [f"{prefix}_{iv}.csv" for iv in MULTI_INTERVALS
            if Path(f"{prefix}_{iv}.csv").exists()]


def _yearly_streak(csv: str) -> list[dict]:
    df = load_ohlcv(csv)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    years = df.index.year
    rows = []
    for yr in sorted(set(years)):
        mask = years == yr
        sl = np.abs(streak[mask]) >= 3
        k = int(sl.sum())
        if k < MIN_N:
            rows.append({"year": yr, "rule": "streak>=3", "n": k, "status": "low"})
            continue
        pred = (streak[mask][sl] < 0).astype(int)
        actual = (ret.to_numpy()[mask][sl] > 0).astype(int)
        hit = float(np.mean(pred == actual))
        rows.append({"year": yr, "rule": "streak>=3", "n": k,
                     "hit": round(hit, 4), "edge": round(hit - 0.5, 4), "status": "ok"})
    return rows


def _yearly_combo(csv_1h: str) -> list[dict]:
    prefix = _asset_prefix(csv_1h)
    files = _multi_files(prefix)
    if len(files) < 2:
        return [{"year": 0, "rule": "combo r>=2 K>=3", "n": 0,
                 "status": "low", "note": f"need >=2 TF files for {prefix}"}]

    df = load_ohlcv(csv_1h)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    fwd = ret.shift(-1)
    sigs = {Path(c).stem: _contrarian_grid(c) for c in files}
    frame = pd.DataFrame({"streak": streak, "slen": np.abs(streak), "fwd": fwd},
                         index=df.index)
    for n, s in sigs.items():
        frame[n] = s
    frame = frame.dropna()
    cols = list(sigs.keys())
    years = frame.index.year
    rows = []
    for yr in sorted(set(years)):
        sub = frame[years == yr]
        sl = sub["slen"].to_numpy() >= 2
        mat = sub[cols].to_numpy()
        pos = (mat > 0).sum(axis=1)
        neg = (mat < 0).sum(axis=1)
        mdir = np.where(pos >= 3, 1, np.where(neg >= 3, 0, -1))
        sdir = (sub["streak"].to_numpy() < 0).astype(int)
        sel = sl & (mdir >= 0) & (mdir == sdir)
        k = int(sel.sum())
        if k < MIN_N:
            rows.append({"year": yr, "rule": "combo r>=2 K>=3", "n": k, "status": "low"})
            continue
        pred = sdir[sel]
        act = (sub["fwd"].to_numpy()[sel] > 0).astype(int)
        hit = float(np.mean(pred == act))
        rows.append({"year": yr, "rule": "combo r>=2 K>=3", "n": k,
                     "hit": round(hit, 4), "edge": round(hit - 0.5, 4), "status": "ok"})
    return rows


def run_stability(csv_1h: str = "btc_1h.csv") -> dict:
    streak_rows = _yearly_streak(csv_1h)
    combo_rows = _yearly_combo(csv_1h)
    all_ok = [r for r in streak_rows + combo_rows if r.get("status") == "ok"]
    above = sum(1 for r in all_ok if r["hit"] > 0.5)
    return {
        "asset": Path(csv_1h).stem,
        "streak": streak_rows,
        "combo": combo_rows,
        "verdict": f"{above}/{len(all_ok)} year-rule pairs above 50%",
    }


def _print_one(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print(f"CONVICTION STABILITY BY YEAR ({r['asset']}, fixed rules)")
    print(line)
    for label, rows in (("STREAK>=3 (same bar)", r["streak"]),
                        ("COMBO r>=2 + 3TF (next 1h)", r["combo"])):
        print(f"\n  {label}")
        print(f"  {'year':>6}{'n':>8}{'hit':>8}{'edge':>8}")
        for x in rows:
            if x.get("status") != "ok":
                note = x.get("note", "low")
                print(f"  {x.get('year', '?'):>6}{x.get('n', 0):>8}   {note}")
                continue
            star = " *" if x["hit"] > 0.52 else ""
            print(f"  {x['year']:>6}{x['n']:>8}{x['hit']:>8}{x['edge']:>+8}{star}")
    print(line)
    print(f"  VERDICT: {r['verdict']}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE conviction stability")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv"],
                        help="1h CSV(s), e.g. btc_1h.csv eth_1h.csv")
    args = parser.parse_args()
    for csv in args.csv:
        if not Path(csv).exists():
            print(f"skip {csv} (missing)")
            continue
        _print_one(run_stability(csv))


if __name__ == "__main__":
    main()
