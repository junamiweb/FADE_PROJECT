"""Sparse PRIMARY replay — holdout validation for Phase 1 (tier >= HIGH).

Pre-registered exploratory holdout (NOT lockbox). Measures coverage and hit-rate
when PRIMARY only fires on conviction tiers elite/strong/high (+ quality proxy).

Run:
    python -m fade.pipeline.sparse_primary_replay
    python -m fade.pipeline.sparse_primary_replay btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.conviction import TIER_DEFS
from fade.core.data_loader import load_ohlcv
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.pre_registration import load_manifest, save_manifest
from fade.pipeline.trend_structure import _signed_streak
from fade.pipeline.forecast_tiers import _SPARSE_CONVICT_TIER_IDS

STUDY_ID = "sparse_primary_replay"
HOLDOUT_FRAC = 0.30
MULTI_IV = ("5m", "15m", "30m", "1h")
OUTPUT_PATH = Path("fade/output/sparse_primary_replay.json")


def _prefix(csv: str) -> str:
    stem = Path(csv).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _active_tier(slen: int, streak_dir: str, tf_agree: int, multi_dir: str,
                 aligned: bool) -> tuple[str, str] | None:
    for tid, _label, min_s, min_k, _hit, _note in TIER_DEFS:
        if min_k == 0:
            if slen < min_s or streak_dir == "FLAT":
                continue
            return tid, streak_dir
        if min_s == 0:
            if tf_agree < min_k or multi_dir == "FLAT":
                continue
            return tid, multi_dir
        if slen < min_s or tf_agree < min_k or not aligned:
            continue
        return tid, streak_dir
    return None


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if not any(s.get("study_id") == STUDY_ID for s in studies):
        studies.append({
            "study_id": STUDY_ID,
            "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
            "tiers_included": sorted(_SPARSE_CONVICT_TIER_IDS),
            "data_split": "holdout_70_30_exploratory",
            "success_criteria": {"min_hit_rate": 0.58, "note": "exploratory; forward tracker is truth"},
        })
        save_manifest(m)


def score_sparse(csv_1h: str, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    prefix = _prefix(csv_1h)
    df = load_ohlcv(csv_1h)
    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    grids = {iv: _contrarian_grid(f"{prefix}_{iv}.csv")
             for iv in MULTI_IV if Path(f"{prefix}_{iv}.csv").exists()}

    frame = pd.DataFrame({"streak": streak, "slen": np.abs(streak), "fwd": fwd},
                         index=df.index)
    for iv, s in grids.items():
        frame[iv] = s
    frame = frame.dropna(subset=["fwd"])
    cols = [c for c in MULTI_IV if c in frame.columns]

    n = len(frame)
    split = int(n * (1 - holdout_frac))
    hold = frame.iloc[split:]

    sparse_rows, legacy_rows = [], []
    for _ts, row in hold.iterrows():
        slen = int(row["slen"])
        s = int(row["streak"]) if np.isfinite(row["streak"]) else 0
        streak_dir = "FLAT" if slen < 2 else ("UP" if s < 0 else "DOWN")
        dirs = []
        for c in cols:
            v = row[c]
            if v == 0 or not np.isfinite(v):
                dirs.append(None)
            else:
                dirs.append("UP" if v > 0 else "DOWN")
        ok = [d for d in dirs if d is not None]
        up = sum(1 for d in ok if d == "UP")
        dn = sum(1 for d in ok if d == "DOWN")
        if up >= dn and up > 0:
            multi_dir, tf_agree = "UP", up
        elif dn > 0:
            multi_dir, tf_agree = "DOWN", dn
        else:
            multi_dir, tf_agree = "FLAT", 0
        aligned = streak_dir != "FLAT" and multi_dir != "FLAT" and streak_dir == multi_dir

        tier = _active_tier(slen, streak_dir, tf_agree, multi_dir, aligned)
        actual_up = int(row["fwd"] > 0)

        if tier:
            tid, direction = tier
            hit = int((direction == "UP") == actual_up)
            legacy_rows.append({"hit": hit, "tier": tid})
            if tid in _SPARSE_CONVICT_TIER_IDS:
                sparse_rows.append({"hit": hit, "tier": tid})

    if not sparse_rows:
        return {"asset": Path(csv_1h).stem, "status": "no_sparse_signals",
                "legacy_signals": len(legacy_rows)}

    sr = pd.DataFrame(sparse_rows)
    lr = pd.DataFrame(legacy_rows)
    by_tier = {}
    for tid in sorted(sr["tier"].unique()):
        sub = sr[sr["tier"] == tid]
        by_tier[tid] = {"n": len(sub), "hit": round(float(sub["hit"].mean()), 4)}

    return {
        "asset": Path(csv_1h).stem,
        "status": "ok",
        "holdout_bars": len(hold),
        "sparse_signals": len(sr),
        "sparse_coverage_pct": round(100 * len(sr) / len(hold), 2),
        "sparse_hit": round(float(sr["hit"].mean()), 4),
        "legacy_signals": len(lr),
        "legacy_hit": round(float(lr["hit"].mean()), 4) if len(lr) else None,
        "by_tier": by_tier,
    }


def run_all(csvs: list[str]) -> dict:
    _ensure_preregistered()
    results = [score_sparse(c) for c in csvs if Path(c).exists()]
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "verdict": (
            f"Sparse PRIMARY: mean hit {np.mean([r['sparse_hit'] for r in ok]):.4f} "
            f"on {sum(r['sparse_signals'] for r in ok)} signals "
            f"(coverage ~{np.mean([r['sparse_coverage_pct'] for r in ok]):.1f}% holdout)."
            if ok else "no sparse signals on holdout"
        ),
    }


def _print(r: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("SPARSE PRIMARY REPLAY (holdout, tier >= HIGH)")
    print(line)
    for x in r.get("results", []):
        if x.get("status") != "ok":
            print(f"  {x.get('asset', '?')}: {x.get('status')}")
            continue
        print(f"  {x['asset']}: sparse hit={x['sparse_hit']}  n={x['sparse_signals']}  "
              f"coverage={x['sparse_coverage_pct']}%  (legacy all tiers: {x['legacy_hit']} n={x['legacy_signals']})")
        for tid, v in x.get("by_tier", {}).items():
            print(f"    {tid}: {v['hit']}  n={v['n']}")
    print()
    print(f"  {r.get('verdict')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse PRIMARY holdout replay")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    args = parser.parse_args()
    r = run_all(args.csv)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
