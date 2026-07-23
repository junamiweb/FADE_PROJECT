"""Tradeable-edge scan — min move gate × min_hold @ net fees (exploratory).

North star: only keep ideas that can clear trading costs. This grids
path_lean3 direction with:
  - past |return| entry filter (no look-ahead — lag-1 abs return in bps)
  - min_hold before flip
  - optional streak>=2 conviction-style gate (signed streak from price)

Holdout 70/30 chronological. Fee default 5 bps per side (10 bps RT).

NOT forward validation. NOT production. Grid is exploratory — survivors need
pre-registration + lockbox before any claim.

Run:
    python -m fade.pipeline.tradeable_edge_scan
    python -m fade.pipeline.tradeable_edge_scan btc_1h.csv eth_1h.csv --fee-bps 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.conviction import HOLDOUT_FRAC
from fade.core.data_loader import load_ohlcv
from fade.pipeline.pnl_reality_check_v2 import (
    _holdout_path_lean3,
    _min_hold_positions,
)
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("tradeable_edge_scan")

STUDY_ID = "tradeable_edge_scan_v1"
OUTPUT = Path("fade/output/tradeable_edge_scan.json")

# Keep grid small — north star over combinatorial fishing.
MOVE_BPS_GRID = (0, 10, 20, 30, 50, 80, 100)
MIN_HOLD_GRID = (1, 6, 12, 24, 48)
STREAK_GATES = (0, 2)  # 0 = no streak gate; 2 = |streak| >= 2


def _scan_asset(
    csv_path: str,
    *,
    holdout_frac: float,
    fee_bps: float,
) -> dict:
    got = _holdout_path_lean3(csv_path, holdout_frac)
    if got is None:
        return {"status": "no_rules", "asset": Path(csv_path).stem}

    preds, split, n_rules = got
    df = load_ohlcv(csv_path)
    hold_close = df["close"].iloc[split:]
    # Past move only: abs return of the bar that just closed (lag of next-bar ret).
    past_abs_bps = (hold_close.pct_change().abs() * 1e4).reindex(preds.index)
    past_abs_bps = past_abs_bps.fillna(0.0).to_numpy()

    ret = df["close"].pct_change().to_numpy()
    streak_full = _signed_streak(ret)
    streak_hold = pd.Series(streak_full[split:], index=df.index[split:])
    streak = streak_hold.reindex(preds.index).fillna(0).to_numpy()

    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    raw_target = np.where(pred_up == 1, 1.0, -1.0)

    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4

    bh = _equity(np.ones(len(bar_ret)), bar_ret, 0.0, 0.0)
    buy_hold = _stats(bh["strat_ret"], bh["equity"], bpy)

    rows: list[dict] = []
    for move_bps in MOVE_BPS_GRID:
        move_ok = past_abs_bps >= float(move_bps)
        for streak_min in STREAK_GATES:
            if streak_min > 0:
                streak_ok = np.abs(streak) >= streak_min
            else:
                streak_ok = np.ones(len(raw_target), dtype=bool)
            gated = np.where(move_ok & streak_ok, raw_target, 0.0)
            for mh in MIN_HOLD_GRID:
                pos = _min_hold_positions(gated, mh)
                e = _equity(pos, bar_ret, fee_rate, 0.0)
                st = _stats(e["strat_ret"], e["equity"], bpy)
                active = int(np.sum(pos != 0))
                rows.append({
                    "move_bps": move_bps,
                    "streak_min": streak_min,
                    "min_hold": mh,
                    "active_bars": active,
                    "coverage": round(active / max(len(pos), 1), 4),
                    "n_changes": e["n_changes"],
                    "cost_drag": round(e["total_cost"], 4),
                    **st,
                })

    positive = [r for r in rows if r["total_return"] > 0]
    # Prefer positive return, then Sharpe, then fewer comparisons luck: require
    # minimum activity so a 2-trade miracle does not win.
    viable = [r for r in positive if r["n_changes"] >= 30 and r["active_bars"] >= 200]
    ranking = viable if viable else positive
    best = (
        max(ranking, key=lambda r: (r["total_return"], r.get("sharpe") or -999))
        if ranking else
        max(rows, key=lambda r: r["total_return"])
    )

    # Baseline references from the grid
    def _pick(move: int, streak: int, mh: int) -> dict | None:
        for r in rows:
            if r["move_bps"] == move and r["streak_min"] == streak and r["min_hold"] == mh:
                return r
        return None

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "n_rules_frozen": n_rules,
        "holdout_bars": len(bar_ret),
        "fee_bps_per_side": fee_bps,
        "fee_bps_round_trip": fee_bps * 2,
        "buy_and_hold": buy_hold,
        "n_grid_cells": len(rows),
        "n_positive_return": len(positive),
        "n_viable_positive": len(viable),
        "best_viable_or_best": best,
        "refs": {
            "ungated_mh1": _pick(0, 0, 1),
            "ungated_mh12": _pick(0, 0, 12),
            "move50_streak2_mh12": _pick(50, 2, 12),
            "move50_streak2_mh24": _pick(50, 2, 24),
            "move80_streak2_mh24": _pick(80, 2, 24),
        },
        "top5_viable": sorted(
            viable, key=lambda r: r["total_return"], reverse=True,
        )[:5],
        "all_rows": rows,
    }


def run_scan(
    csvs: list[str],
    *,
    holdout_frac: float = HOLDOUT_FRAC,
    fee_bps: float = 5.0,
) -> dict:
    assets = []
    for csv in csvs:
        log.info("scan %s @ %sbps", csv, fee_bps)
        assets.append(_scan_asset(csv, holdout_frac=holdout_frac, fee_bps=fee_bps))

    survivors = []
    for a in assets:
        if a.get("status") != "ok":
            continue
        best = a["best_viable_or_best"]
        bh = a["buy_and_hold"]["total_return"]
        if (
            a["n_viable_positive"] > 0
            and best["total_return"] > 0
            and best["total_return"] > bh
        ):
            survivors.append({
                "asset": a["asset"],
                "best": best,
                "beats_buy_hold": True,
            })
        elif a["n_viable_positive"] > 0 and best["total_return"] > 0:
            survivors.append({
                "asset": a["asset"],
                "best": best,
                "beats_buy_hold": False,
            })

    verdict = "NO_TRADEABLE_SURVIVOR"
    verdict_he = (
        "אין שורד viable עם תשואה חיובית @5bps אחרי סינון פעילות — "
        "סף־תנועה+min_hold לא מציל את path_lean3 בסריקה הזו."
    )
    if any(s.get("beats_buy_hold") for s in survivors):
        verdict = "EXPLORATORY_CANDIDATE"
        verdict_he = (
            "יש שילוב exploratory שמכה buy&hold @5bps עם כיסוי מינימלי — "
            "לא לקדם בלי pre-reg + lockbox. חשוד לניפוח grid."
        )
    elif survivors:
        verdict = "POSITIVE_BUT_WEAK"
        verdict_he = (
            "יש תשואה חיובית viable אך לא מכה buy&hold — לא מספיק לכוכב הצפון."
        )

    return {
        "study_id": STUDY_ID,
        "sandbox": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "north_star": "predict → trade → net profit",
        "protocol": {
            "split": f"chronological holdout {holdout_frac:.0%} (rules frozen on dev)",
            "signal": "path_lean3 direction",
            "move_gate": "lag-1 |return| in bps (past bar only)",
            "streak_gate": "|signed_streak| >= k (optional)",
            "min_hold": "bars before flip",
            "fee_bps_per_side": fee_bps,
            "viable_rule": "total_return>0 AND n_changes>=30 AND active_bars>=200",
            "warning_he": "grid exploratory — multiple comparisons; not forward truth",
        },
        "move_bps_grid": list(MOVE_BPS_GRID),
        "min_hold_grid": list(MIN_HOLD_GRID),
        "streak_gates": list(STREAK_GATES),
        "_full_assets": assets,
        "survivors": survivors,
        "verdict": verdict,
        "verdict_he": verdict_he,
        "next_he": [
            "אם EXPLORATORY_CANDIDATE — pre-reg פרמטרים קפואים ואז lockbox v2 חד־פעמי",
            "אם אין שורד — לקבור path_lean3 ungated כנתיב מסחר; לחפש מבנה אחר",
            "לא לשנות PRIMARY/ETH candidate לפי הסריקה הזו בלבד",
        ],
    }


def _print(r: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"TRADEABLE EDGE SCAN  fee={r['protocol']['fee_bps_per_side']}bps/side")
    print(f"{'=' * 70}")
    for a in r.get("_full_assets") or []:
        if a.get("status") != "ok":
            print(f"  {a.get('asset')}: {a.get('status')}")
            continue
        best = a["best_viable_or_best"]
        bh = a["buy_and_hold"]["total_return"] * 100
        print(f"\n  {a['asset']}  holdout_bars={a['holdout_bars']}  "
              f"positive={a['n_positive_return']}/{a['n_grid_cells']}  "
              f"viable={a['n_viable_positive']}")
        print(f"    buy&hold: {bh:+.1f}%")
        print(
            f"    best: move>={best['move_bps']}bps streak>={best['streak_min']} "
            f"mh={best['min_hold']} -> {best['total_return']*100:+.1f}% "
            f"sharpe={best.get('sharpe')} trades={best['n_changes']} "
            f"cov={best['coverage']}"
        )
        for name, ref in (a.get("refs") or {}).items():
            if not ref:
                continue
            print(
                f"    ref {name}: {ref['total_return']*100:+.1f}% "
                f"trades={ref['n_changes']} cov={ref['coverage']}"
            )
        if a.get("top5_viable"):
            print("    top viable:")
            for t in a["top5_viable"][:3]:
                print(
                    f"      move>={t['move_bps']} s>={t['streak_min']} "
                    f"mh={t['min_hold']}: {t['total_return']*100:+.1f}%"
                )
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  {r['verdict_he']}")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tradeable edge scan (exploratory)")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    result = run_scan(
        args.csv,
        holdout_frac=args.holdout_frac,
        fee_bps=args.fee_bps,
    )
    _print(result)

    # Persist without gigantic duplication: keep all_rows per asset
    out = {k: v for k, v in result.items() if k != "_full_assets"}
    out["assets"] = []
    for a in result["_full_assets"]:
        compact = dict(a)
        rows = compact.pop("all_rows", [])
        compact["n_rows"] = len(rows)
        # store only positive + refs detail already present
        compact["positive_rows"] = [r for r in rows if r["total_return"] > 0]
        out["assets"].append(compact)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
