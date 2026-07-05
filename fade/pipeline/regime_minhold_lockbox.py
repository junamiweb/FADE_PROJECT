"""One-shot lockbox validation for regime-gated min_hold PnL (batch 33).

Batch 32 found exploratory winners on the 70/30 holdout:
  BTC: HIGH_VR + min_hold=48  -> +76.8% @ 5bps
  ETH: LOW_VR  + min_hold=12  -> +94.2% @ 5bps

This module tests those PRE-REGISTERED configs once on the sealed lockbox
(newest 18%, never used for tuning). Rules mined + VR tertiles fit ONLY on
pre-lockbox data. No parameter search on lockbox.

Run ONCE:
    python -m fade.pipeline.regime_minhold_lockbox
    python -m fade.pipeline.regime_minhold_lockbox btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config, lean_config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.predictor import collect_calibration_samples, predict_calibrated
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.backtest import walk_forward
from fade.pipeline.final_lockbox import DEFAULT_LOCKBOX_FRAC, MANIFEST_PATH, _load_manifest
from fade.pipeline.holdout import _select_stable_rules
from fade.pipeline.pnl_reality_check_v2 import _min_hold_positions
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.utils.logging import get_logger

log = get_logger("regime_minhold_lockbox")

# Pre-registered from batch 32 holdout grid (NOT tuned on lockbox).
PREREGISTERED = {
    "btc_1h": {"vr_regime": "HIGH_VR", "min_hold": 48, "holdout_return_5bps": 0.768},
    "eth_1h": {"vr_regime": "LOW_VR", "min_hold": 12, "holdout_return_5bps": 0.942},
}

OUTPUT_PATH = Path("fade/output/regime_minhold_lockbox.json")


def _lockbox_predictions(csv_path: str, lockbox_frac: float):
    """Frozen path_lean3 preds on lockbox; rules/thresholds from pre-lockbox only."""
    config = lean_config()
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    close = df["close"].reindex(atoms.index)
    n = len(atoms)
    cut = int(n * (1.0 - lockbox_frac))

    dev_atoms, lock_atoms = atoms.iloc[:cut], atoms.iloc[cut:]
    dev_fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index).iloc[:cut]

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return None

    cal = CalibrationStore(config.cache_dir / "_lockbox_pnl_cal.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}
    dev_disc = ev.discretize(dev_atoms, ev.compute_thresholds(dev_atoms, config))
    dev_events = ev.build_events(dev_disc, config, allowed=set(frozen.index))
    dev_preds = predict_calibrated(dev_events, frozen, cal, positive={})
    if not dev_preds.empty:
        samples = collect_calibration_samples(dev_preds, dev_fwd)
        if samples:
            cal.update(samples)

    thresholds = ev.compute_thresholds(dev_atoms, config)
    lock_disc = ev.discretize(lock_atoms, thresholds)
    lock_events = ev.build_events(lock_disc, config, allowed=set(frozen.index))
    preds = predict_calibrated(lock_events, frozen, cal, positive={})
    if preds.empty:
        return None

    lock_close = close.iloc[cut:]
    bar_ret = lock_close.pct_change().shift(-1)
    out = preds.join(bar_ret.rename("bar_ret")).dropna(subset=["bar_ret"])
    return out, cut, len(frozen)


def _vr_on_lockbox(csv_path: str, cut: int) -> np.ndarray:
    """VR regime labels on lockbox bars; tertiles from pre-lockbox only."""
    config = Config()
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    vr = compute_vol_ratio(ret, config.vol_ratio_short_window, config.vol_ratio_long_window)
    dev_vr = vr.iloc[:cut].dropna()
    low = float(dev_vr.quantile(1 / 3))
    high = float(dev_vr.quantile(2 / 3))
    reg = assign_vr_regime(vr, low, high)
    return reg.iloc[cut:].to_numpy(), low, high


def eval_lockbox_regime_pnl(
    csv_path: str,
    vr_regime: str,
    min_hold: int,
    lockbox_frac: float = DEFAULT_LOCKBOX_FRAC,
    fee_bps: float = 5.0,
) -> dict:
    asset = Path(csv_path).stem
    got = _lockbox_predictions(csv_path, lockbox_frac)
    if got is None:
        return {"asset": asset, "status": "no_rules"}

    preds, cut, n_rules = got
    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    n_bars = len(bar_ret)

    vr_lock, low_thr, high_thr = _vr_on_lockbox(csv_path, cut)
    if len(vr_lock) > n_bars:
        vr_lock = vr_lock[-n_bars:]

    gate = vr_lock[:n_bars] == vr_regime
    raw_target = np.where(pred_up == 1, 1.0, -1.0)
    gated_target = np.where(gate, raw_target, 0.0)
    pos = _min_hold_positions(gated_target, min_hold)

    res = asset.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4
    e = _equity(pos, bar_ret, fee_rate, 0.0)
    strat = _stats(e["strat_ret"], e["equity"], bpy)
    bh_eq = np.cumprod(1.0 + bar_ret)
    bh = _stats(bar_ret, bh_eq, bpy)

    # Directional hit on regime-active bars only
    active = gate & (pos != 0)
    dir_hit = None
    if int(active.sum()) >= 30:
        dir_hit = float(np.mean(
            (pred_up[active] == 1) == (bar_ret[active] > 0)
        ))

    manifest = _load_manifest()
    meta = next((lb for lb in manifest.get("lockboxes", [])
                 if lb.get("asset") == asset), {})

    holdout_ref = PREREGISTERED.get(asset, {})
    lock_ret = strat["total_return"]
    hold_ret = holdout_ref.get("holdout_return_5bps")

    if hold_ret is not None:
        if lock_ret > 0 and lock_ret >= hold_ret * 0.5:
            tag = "PARTIAL_HOLD — positive on lockbox but below holdout headline"
        elif lock_ret > 0:
            tag = "POSITIVE_LOCKBOX — survives fees on untouched data"
        elif lock_ret > -0.15:
            tag = "FAIL_SOFT — near breakeven, holdout winner was overfit"
        else:
            tag = "FAIL — holdout winner does NOT generalize to lockbox"
    else:
        tag = "UNREGISTERED_ASSET"

    return {
        "status": "ok",
        "asset": asset,
        "config": {"vr_regime": vr_regime, "min_hold": min_hold, "fee_bps": fee_bps},
        "lockbox_frac": lockbox_frac,
        "lockbox_sha256": meta.get("sha256", "")[:16],
        "lockbox_span": [meta.get("lockbox_start"), meta.get("lockbox_end")],
        "n_rules": n_rules,
        "lockbox_bars": n_bars,
        "active_regime_bars": int(gate.sum()),
        "directional_hit_active": round(dir_hit, 4) if dir_hit is not None else None,
        "vr_thresholds_pre_lockbox": {"low": round(low_thr, 4), "high": round(high_thr, 4)},
        "pnl": {
            **strat,
            "n_changes": e["n_changes"],
            "cost_drag": round(e["total_cost"], 4),
        },
        "buy_hold": bh,
        "holdout_reference_return_5bps": hold_ret,
        "delta_return_vs_holdout": round(lock_ret - hold_ret, 4) if hold_ret else None,
        "tag": tag,
        "verdict": (
            f"LOCKBOX @ {fee_bps}bps: {lock_ret*100:.1f}% "
            f"(holdout was {(hold_ret or 0)*100:.1f}%). {tag}."
        ),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_all(csv_paths: list[str] | None = None, fee_bps: float = 5.0) -> dict:
    manifest = _load_manifest()
    for box in manifest.get("lockboxes", []):
        if box.get("status") == "BURNED":
            log.warning(
                "Lockbox v1 BURNED — do not re-run for new configs. Use forward "
                "outcome_tracker or seal lockbox v2."
            )
            break

    paths = csv_paths or ["btc_1h.csv", "eth_1h.csv"]

    results = []
    for csv in paths:
        stem = Path(csv).stem
        if stem not in PREREGISTERED:
            log.warning("No pre-registered config for %s", stem)
            continue
        if not Path(csv).exists():
            results.append({"asset": stem, "status": "missing"})
            continue
        cfg = PREREGISTERED[stem]
        results.append(eval_lockbox_regime_pnl(
            csv, cfg["vr_regime"], cfg["min_hold"], fee_bps=fee_bps))

    any_positive = any(r.get("pnl", {}).get("total_return", 0) > 0 for r in results
                       if r.get("status") == "ok")
    all_positive = all(r.get("pnl", {}).get("total_return", 0) > 0 for r in results
                       if r.get("status") == "ok")

    return {
        "status": "ok",
        "fee_bps": fee_bps,
        "note": "Pre-registered configs from batch 32; ONE-SHOT on sealed lockbox.",
        "results": results,
        "any_positive": any_positive,
        "all_positive": all_positive,
        "overall_verdict": (
            "HOLDBOX VALIDATES regime+min_hold — at least one asset positive @5bps."
            if any_positive and all_positive else
            "MIXED — some lockbox configs positive, not unified."
            if any_positive else
            "REJECTED — holdout winners fail on true lockbox @5bps (overfit confirmed)."
        ),
    }


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print("REGIME MIN_HOLD LOCKBOX — one-shot @ 5bps (pre-registered batch 32 configs)")
    print(line)
    print(f"  {r.get('note', '')}")
    print()
    for item in r.get("results", []):
        if item.get("status") != "ok":
            print(f"  {item.get('asset', '?')}: {item.get('status')}")
            continue
        cfg = item["config"]
        pnl = item["pnl"]
        print(f"  {item['asset']}  VR={cfg['vr_regime']}  min_hold={cfg['min_hold']}")
        print(f"    lockbox span : {item.get('lockbox_span')}")
        print(f"    bars/active  : {item['lockbox_bars']}/{item['active_regime_bars']}")
        print(f"    dir_hit      : {item.get('directional_hit_active')}")
        print(f"    return @5bps : {pnl['total_return']*100:.1f}%  trades={pnl['n_changes']}")
        print(f"    buy_hold     : {item['buy_hold']['total_return']*100:.1f}%")
        print(f"    holdout ref  : {item.get('holdout_reference_return_5bps', 0)*100:.1f}%")
        print(f"    TAG          : {item['tag']}")
        print()
    print(line)
    print(f"  OVERALL: {r.get('overall_verdict')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime min_hold lockbox one-shot")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    parser.add_argument("--fee-bps", type=float, default=5.0)
    args = parser.parse_args()
    for csv in args.csv:
        if not Path(csv).exists():
            log.error("File not found: %s", csv)
            sys.exit(1)
    result = run_all(args.csv, fee_bps=args.fee_bps)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print(result)


if __name__ == "__main__":
    main()
