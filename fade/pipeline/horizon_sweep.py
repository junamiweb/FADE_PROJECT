"""Horizon sweep — 4h / 8h vs 1h baseline (Phase step 3).

Pre-registered BEFORE run (see fade/output/pre_registration.json).
Uses holdout 70/30 ONLY — lockbox v1 BURNED, no lockbox v2 yet.

Metrics per asset × resolution:
  - reversion_index (streak>=2, holdout slice)
  - path_lean3 holdout hit-rate + p-value
  - PnL @ 5bps: raw long-short + fixed min_hold=24 (fair cross-horizon compare)

Run:
    python -m fade.pipeline.horizon_sweep
    python -m fade.pipeline.horizon_sweep --assets btc eth --intervals 1h 4h 8h
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fade.config import lean_config
from fade.core.conviction import HOLDOUT_FRAC
from fade.pipeline.holdout import holdout_test
from fade.pipeline.pnl_reality_check_v2 import (
    MIN_HOLD_GRID,
    _holdout_path_lean3,
    _min_hold_positions,
)
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.pipeline.pre_registration import load_manifest, save_manifest
from fade.pipeline.trend_structure import _signed_streak
from fade.core.data_loader import load_ohlcv
from fade.utils.logging import get_logger

log = get_logger("horizon_sweep")

STUDY_ID = "horizon_sweep_4h_8h"
DEFAULT_ASSETS = ("btc", "eth")
DEFAULT_INTERVALS = ("1h", "4h", "8h")
FIXED_MIN_HOLD = 24
FEE_BPS = 5.0
OUTPUT_PATH = Path("fade/output/horizon_sweep.json")

# Pre-registered success criteria (exploratory holdout — NOT final OOS).
SUCCESS_CRITERIA = {
    "exploratory_pnl_positive_5bps": "raw_ls or min_hold_24 positive on holdout",
    "beats_1h_same_asset": "better total_return @5bps than same-asset 1h baseline",
    "final_claim_requires": "lockbox_v2 one-shot after pre-register",
}


def _reversion_index_holdout(csv_path: str, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    n = len(ret)
    split = int(n * (1.0 - holdout_frac))
    hold = np.zeros(n, dtype=bool)
    hold[split:] = True
    up = (ret > 0).astype(int)
    cont_hits = cont_n = 0
    for m in range(2, 8):
        for sgn in (m, -m):
            sel = hold & (streak == sgn)
            k = int(sel.sum())
            if k < 30:
                continue
            nxt = up[sel]
            cont_hits += int((nxt == (1 if sgn > 0 else 0)).sum())
            cont_n += k
    if cont_n < 80:
        return {"status": "low_support", "n": cont_n}
    cont = cont_hits / cont_n
    return {
        "status": "ok",
        "n": cont_n,
        "reversion_index": round(0.5 - cont, 4),
        "continuation": round(cont, 4),
    }


def _pnl_holdout(csv_path: str, fee_bps: float = FEE_BPS,
                 holdout_frac: float = HOLDOUT_FRAC) -> dict:
    got = _holdout_path_lean3(csv_path, holdout_frac)
    if got is None:
        return {"status": "no_rules"}
    preds, _, _ = got
    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4

    raw_target = np.where(pred_up == 1, 1.0, -1.0)
    e_raw = _equity(raw_target, bar_ret, fee_rate, 0.0)
    raw_stats = {**_stats(e_raw["strat_ret"], e_raw["equity"], bpy),
                 "n_changes": e_raw["n_changes"]}

    pos_mh = _min_hold_positions(raw_target, FIXED_MIN_HOLD)
    e_mh = _equity(pos_mh, bar_ret, fee_rate, 0.0)
    mh_stats = {**_stats(e_mh["strat_ret"], e_mh["equity"], bpy),
                "n_changes": e_mh["n_changes"], "min_hold": FIXED_MIN_HOLD}

    hit = float(np.mean(pred_up == (bar_ret > 0).astype(int)))
    return {
        "status": "ok",
        "holdout_bars": len(bar_ret),
        "directional_hit": round(hit, 4),
        "raw_long_short": raw_stats,
        "min_hold_fixed": mh_stats,
    }


def ensure_preregistered() -> dict:
    """Append horizon study to manifest if not present."""
    m = load_manifest()
    studies = m.setdefault("studies", [])
    existing = next((s for s in studies if s.get("study_id") == STUDY_ID), None)
    entry = {
        "study_id": STUDY_ID,
        "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
        "assets": list(DEFAULT_ASSETS),
        "intervals": list(DEFAULT_INTERVALS),
        "metrics": ["reversion_index", "path_lean3_holdout", "pnl_5bps_raw", "pnl_5bps_minhold24"],
        "data_split": "holdout_70_30_exploratory_only",
        "lockbox_policy": "v1 BURNED; v2 reserved for winner one-shot",
        "success_criteria": SUCCESS_CRITERIA,
    }
    if not existing:
        studies.append(entry)
        save_manifest(m)
    return entry


def run_sweep(
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    intervals: tuple[str, ...] = DEFAULT_INTERVALS,
    holdout_frac: float = HOLDOUT_FRAC,
) -> dict:
    ensure_preregistered()
    rows = []
    for asset in assets:
        baseline_pnl = None
        for iv in intervals:
            csv = f"{asset}_{iv}.csv"
            row = {"asset": asset, "interval": iv, "csv": csv}
            if not Path(csv).exists():
                row["status"] = "missing"
                rows.append(row)
                continue

            ri = _reversion_index_holdout(csv, holdout_frac)
            ho = holdout_test(csv, holdout_frac=holdout_frac, config=lean_config())
            pnl = _pnl_holdout(csv, holdout_frac=holdout_frac)

            row.update({
                "status": "ok",
                "n_bars": int(len(load_ohlcv(csv))),
                "reversion_index": ri,
                "holdout": {
                    "hit_rate": ho.get("holdout_hit_rate"),
                    "lift": ho.get("holdout_lift_vs_random"),
                    "p_value": ho.get("p_value"),
                    "coverage": ho.get("coverage"),
                } if ho.get("status") == "ok" else {"status": ho.get("status")},
                "pnl_5bps": pnl,
            })

            if iv == "1h" and pnl.get("status") == "ok":
                baseline_pnl = pnl

            if baseline_pnl and iv != "1h" and pnl.get("status") == "ok":
                b_mh = baseline_pnl["min_hold_fixed"]["total_return"]
                c_mh = pnl["min_hold_fixed"]["total_return"]
                row["vs_1h_minhold24"] = round(c_mh - b_mh, 4)
                row["beats_1h_pnl"] = c_mh > b_mh and c_mh > 0

            rows.append(row)

    # Summary
    ok = [r for r in rows if r.get("status") == "ok"]
    positive_mh = [r for r in ok if r.get("pnl_5bps", {}).get("min_hold_fixed", {}).get("total_return", -1) > 0]
    beats_1h = [r for r in ok if r.get("beats_1h_pnl")]

    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fee_bps": FEE_BPS,
        "fixed_min_hold": FIXED_MIN_HOLD,
        "note": "Exploratory holdout only — NOT lockbox-validated.",
        "results": rows,
        "summary": {
            "n_ok": len(ok),
            "positive_minhold24_5bps": [
                f"{r['asset']}_{r['interval']}" for r in positive_mh
            ],
            "beats_1h_minhold24": [
                f"{r['asset']}_{r['interval']}" for r in beats_1h
            ],
        },
        "verdict": _verdict(ok, positive_mh, beats_1h),
    }


def _verdict(ok: list, positive_mh: list, beats_1h: list) -> str:
    if not ok:
        return "INCONCLUSIVE — missing data."
    pos_names = [f"{r['asset']}_{r['interval']}" for r in positive_mh]
    beat_names = [f"{r['asset']}_{r['interval']}" for r in beats_1h]
    if positive_mh:
        return (
            f"EXPLORATORY POSITIVE @5bps (min_hold=24) on holdout: {pos_names}. "
            f"Beats 1h: {beat_names or 'none'}. "
            "Requires lockbox v2 one-shot before any production claim."
        )
    return (
        "NO horizon beats 1h PnL @5bps on holdout — longer horizons do not fix fee drag. "
        "1h remains baseline for Phase."
    )


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"HORIZON SWEEP — {r.get('study_id')}  (holdout exploratory, {r.get('fee_bps')}bps)")
    print(line)
    print(f"  {r.get('note')}")
    print(f"\n  {'asset':<6}{'iv':<5}{'rev_idx':>9}{'hit':>8}{'pnl_raw':>10}{'pnl_mh24':>10}{'vs_1h':>8}")
    for row in r.get("results", []):
        if row.get("status") != "ok":
            print(f"  {row.get('asset','?'):<6}{row.get('interval','?'):<5}  {row.get('status')}")
            continue
        ri = row.get("reversion_index", {})
        rev = ri.get("reversion_index", "?") if ri.get("status") == "ok" else "?"
        hit = row.get("holdout", {}).get("hit_rate", "?")
        pnl = row.get("pnl_5bps", {})
        raw = pnl.get("raw_long_short", {}).get("total_return")
        mh = pnl.get("min_hold_fixed", {}).get("total_return")
        raw_s = f"{raw*100:.1f}%" if raw is not None else "?"
        mh_s = f"{mh*100:.1f}%" if mh is not None else "?"
        vs = row.get("vs_1h_minhold24")
        vs_s = f"{vs*100:+.1f}pp" if vs is not None else "-"
        print(f"  {row['asset']:<6}{row['interval']:<5}{str(rev):>9}{str(hit):>8}"
              f"{raw_s:>10}{mh_s:>10}{vs_s:>8}")
    print()
    print(f"  {r.get('verdict')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE horizon sweep 4h/8h vs 1h")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument("--intervals", nargs="+", default=list(DEFAULT_INTERVALS))
    args = parser.parse_args()
    result = run_sweep(tuple(args.assets), tuple(args.intervals))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print(result)


if __name__ == "__main__":
    main()
