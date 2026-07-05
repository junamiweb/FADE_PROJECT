"""Regime-gated min_hold PnL — Branch A follow-up (batch 31).

After decay_diagnosis (branch C -> default A): exploit turnover reduction by
trading path_lean3 ONLY when vol-regime matches, with minimum holding period.

Tests LOW_VR / NORMAL / HIGH_VR gates + min_hold sweep at 5 bps per side.
VR tertiles fitted on dev (first 70%) only — no look-ahead.

Run:
    python -m fade.pipeline.pnl_regime_minhold
    python -m fade.pipeline.pnl_regime_minhold btc_1h.csv --fee-bps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fade.config import lean_config
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.pnl_reality_check_v2 import (
    MIN_HOLD_GRID,
    _holdout_path_lean3,
    _min_hold_positions,
)
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.core.data_loader import load_ohlcv
from fade.core.conviction import HOLDOUT_FRAC
from fade.utils.logging import get_logger

log = get_logger("pnl_regime_minhold")

VR_REGIMES = ("LOW_VR", "NORMAL", "HIGH_VR")


def _vr_regime_series(csv_path: str, holdout_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (holdout_vr_regime labels, holdout mask aligned to preds index)."""
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    from fade.config import Config
    config = Config()
    vr = compute_vol_ratio(ret, config.vol_ratio_short_window, config.vol_ratio_long_window)
    n = len(df)
    split = int(n * (1.0 - holdout_frac))
    dev_vr = vr.iloc[:split].dropna()
    low = float(dev_vr.quantile(1 / 3))
    high = float(dev_vr.quantile(2 / 3))
    reg = assign_vr_regime(vr, low, high)
    return reg.iloc[split:].to_numpy(), split


def run_regime_minhold(
    csv_path: str = "btc_1h.csv",
    holdout_frac: float = HOLDOUT_FRAC,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> dict:
    got = _holdout_path_lean3(csv_path, holdout_frac)
    if got is None:
        return {"status": "no_rules", "asset": Path(csv_path).stem}

    preds, split, n_rules = got
    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    n_bars = len(bar_ret)

    vr_hold, _ = _vr_regime_series(csv_path, holdout_frac)
    # align VR to preds length (preds may be subset of holdout bars)
    if len(vr_hold) > n_bars:
        vr_hold = vr_hold[-n_bars:]

    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4
    slip = slippage_bps / 1e4

    raw_target = np.where(pred_up == 1, 1.0, -1.0)
    results = {}

    for regime in VR_REGIMES:
        gate = vr_hold[:n_bars] == regime
        regime_results = []
        for mh in MIN_HOLD_GRID:
            # Only signal when regime active; flat otherwise
            gated_target = np.where(gate, raw_target, 0.0)
            pos = _min_hold_positions(gated_target, mh)
            e = _equity(pos, bar_ret, fee_rate, slip)
            regime_results.append({
                "min_hold": mh,
                "active_bars": int(np.sum(gate)),
                **_stats(e["strat_ret"], e["equity"], bpy),
                "n_changes": e["n_changes"],
                "cost_drag": round(e["total_cost"], 4),
            })
        best = max(regime_results, key=lambda x: x["total_return"])
        results[regime] = {"variants": regime_results, "best": best}

    # Ungated min_hold baseline (from batch 30)
    ungated = []
    for mh in MIN_HOLD_GRID:
        pos = _min_hold_positions(raw_target, mh)
        e = _equity(pos, bar_ret, fee_rate, slip)
        ungated.append({
            "min_hold": mh,
            **_stats(e["strat_ret"], e["equity"], bpy),
            "n_changes": e["n_changes"],
        })
    best_ungated = max(ungated, key=lambda x: x["total_return"])

    global_best = best_ungated
    global_best_regime = "ungated"
    for reg, data in results.items():
        if data["best"]["total_return"] > global_best["total_return"]:
            global_best = data["best"]
            global_best_regime = reg

    any_positive = global_best["total_return"] > 0

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "fee_bps": fee_bps,
        "n_rules": n_rules,
        "holdout_bars": n_bars,
        "by_regime": results,
        "ungated_best": best_ungated,
        "global_best": {"regime": global_best_regime, **global_best},
        "any_positive_at_fee": any_positive,
        "verdict": (
            f"SURVIVES {fee_bps}bps: {global_best_regime} min_hold={global_best.get('min_hold')} "
            f"return {global_best['total_return']*100:.1f}% ({global_best['n_changes']} trades)."
            if any_positive else
            f"NO SURVIVOR at {fee_bps}bps — best {global_best_regime} "
            f"min_hold={global_best.get('min_hold')} {global_best['total_return']*100:.1f}%."
        ),
    }


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"REGIME-GATED MIN_HOLD PnL — {r.get('asset', '?').upper()} @ {r.get('fee_bps')}bps")
    print(line)
    if r.get("status") != "ok":
        print(f"  {r.get('status')}")
        print(line + "\n")
        return
    print(f"  {'regime':<10}{'min_hold':>8}{'return':>10}{'trades':>8}{'active':>8}")
    for reg in VR_REGIMES:
        best = r["by_regime"][reg]["best"]
        print(f"  {reg:<10}{best['min_hold']:>8}{best['total_return']*100:>9.1f}%"
              f"{best['n_changes']:>8}{best.get('active_bars', 0):>8}")
    u = r["ungated_best"]
    print(f"  {'ungated':<10}{u['min_hold']:>8}{u['total_return']*100:>9.1f}%"
          f"{u['n_changes']:>8}{'-':>8}")
    print(line)
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-gated min_hold PnL")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    _print(run_regime_minhold(args.csv, holdout_frac=args.holdout_frac, fee_bps=args.fee_bps))


if __name__ == "__main__":
    main()
