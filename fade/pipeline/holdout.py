"""Strict out-of-sample holdout test — the harshest check of the base edge.

The regime-weighting failure showed that a signal can look real in aggregate
yet evaporate on unseen data. This test applies the same skepticism to the
CORE rules themselves:

    1. Split the series chronologically: development (first 70%) + holdout (30%).
    2. The holdout is quarantined — it never touches threshold fitting, rule
       mining, or stability selection.
    3. Mine + select stable rules using ONLY the development slice (its own
       internal walk-forward), exactly as the live system would.
    4. FREEZE those rules (event -> direction, confidence) and their dev-fitted
       discretisation thresholds.
    5. Apply them, untouched, to the holdout and measure directional lift.
    6. Permutation test on the holdout: shuffle outcomes to build a null and get
       an empirical p-value.

If the edge survives here, it is genuinely out-of-sample. If not, everything
built on top of it is suspect.

Run:
    python -m fade.pipeline.holdout btc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.utils.logging import get_logger

log = get_logger("holdout")

P_VALUE_MAX = 0.05


def _select_stable_rules(stability: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Same promotion criteria the live system uses for positive memory."""
    if stability.empty:
        return pd.DataFrame(columns=["direction", "confidence"])
    mask = (
        (stability["folds_present"] >= config.stability_min_folds)
        & (stability["consistency"] >= config.stability_min_consistency)
        & (stability["avg_oos_hit"] > 0.5)
    )
    stable = stability[mask]
    if stable.empty:
        return pd.DataFrame(columns=["direction", "confidence"])
    out = pd.DataFrame({
        "direction": stable["direction"].astype(int),
        "confidence": stable["avg_oos_hit"].astype(float),
    })
    return out


def holdout_test(
    csv_path: str,
    holdout_frac: float = 0.30,
    n_shuffles: int = 300,
    seed: int = 0,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    rng = np.random.default_rng(seed)

    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    n = len(atoms)
    split = int(n * (1.0 - holdout_frac))
    dev_atoms, hold_atoms = atoms.iloc[:split], atoms.iloc[split:]
    dev_fwd = fwd.iloc[:split]
    hold_fwd = fwd.iloc[split:]

    # --- Development: mine + select stable rules on dev only -----------
    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)

    result: dict = {
        "asset": asset,
        "n_total": int(n),
        "n_dev": int(split),
        "n_holdout": int(n - split),
        "n_stable_rules": int(len(frozen)),
    }
    if frozen.empty:
        result["status"] = "no_rules"
        result["verdict"] = "INCONCLUSIVE - no stable rules survived development."
        return result

    # --- Freeze thresholds on dev, apply to holdout --------------------
    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)

    if preds.empty:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - frozen rules never fired on holdout."
        return result

    valid = hold_fwd.reindex(preds.index).notna()
    pred_v = preds["pred"][valid].to_numpy()
    fwd_v = hold_fwd.reindex(preds.index)[valid].to_numpy()
    pred_sc, act_sc = score_predictions(pred_v, fwd_v, config.move_threshold)
    coverage = int(len(pred_sc))
    if coverage == 0:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - no scorable holdout predictions."
        return result

    real_hit = float(np.mean(pred_sc == act_sc))
    real_lift = real_hit - 0.5

    # Momentum baseline: return_6h when present, else return_1h (lean sets).
    mom_col = "return_6h" if "return_6h" in hold_atoms.columns else (
        "return_1h" if "return_1h" in hold_atoms.columns else None)
    if mom_col is not None:
        mom_all = (hold_atoms[mom_col].reindex(preds.index)[valid] > 0).astype(int).to_numpy()
        mom_sc, act_mom = score_predictions(mom_all, fwd_v, config.move_threshold)
        momentum_hit = float(np.mean(mom_sc == act_mom)) if len(mom_sc) else float("nan")
    else:
        momentum_hit = float("nan")

    null_hits = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(act_sc)
        null_hits[i] = np.mean(pred_sc == shuffled)
    n_ge = int(np.sum(null_hits >= real_hit))
    p_value = (1 + n_ge) / (1 + n_shuffles)

    result.update({
        "status": "ok",
        "holdout_hit_rate": round(real_hit, 4),
        "holdout_lift_vs_random": round(real_lift, 4),
        "holdout_lift_vs_momentum": round(real_hit - momentum_hit, 4),
        "coverage": coverage,
        "null_mean": round(float(np.mean(null_hits)), 4),
        "null_std": round(float(np.std(null_hits)), 4),
        "p_value": round(p_value, 4),
    })
    result["verdict"] = _verdict(result)
    return result


def _verdict(r: dict) -> str:
    lift, p = r["holdout_lift_vs_random"], r["p_value"]
    if lift <= 0:
        return "FAIL - no positive edge on unseen holdout data."
    if p <= P_VALUE_MAX:
        return "PASS - base edge survives strict out-of-sample holdout."
    return "WEAK - positive but within shuffle noise (not significant on holdout)."


def _print(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print(f"FADE STRICT HOLDOUT TEST - {r['asset'].upper()}")
    print(line)
    print(f"  Split: dev={r['n_dev']} bars  |  holdout={r['n_holdout']} bars (quarantined)")
    print(f"  Stable rules frozen from dev : {r['n_stable_rules']}")
    if r.get("status") != "ok":
        print(f"\n  {r['verdict']}")
        print(line + "\n")
        return
    print(f"  Holdout coverage (scored)    : {r['coverage']}")
    print()
    print(f"  Holdout hit-rate             : {r['holdout_hit_rate']}")
    print(f"  Lift vs random               : {r['holdout_lift_vs_random']:+.4f}")
    print(f"  Lift vs momentum             : {r['holdout_lift_vs_momentum']:+.4f}")
    print(f"  Shuffle null (mean +/- std)  : {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
    print(f"  Permutation p-value          : {r['p_value']:.4f}")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv or not Path(csv).exists():
        log.error("Usage: python -m fade.pipeline.holdout path/to/ohlcv.csv")
        sys.exit(1)
    result = holdout_test(csv)
    _print(result)


if __name__ == "__main__":
    main()
