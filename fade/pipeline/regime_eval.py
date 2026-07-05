"""Rigorous test: does regime-weighting improve probability quality?

Regime-weighting rescales a prediction's confidence (distance from 50%) by how
reliable its regime was out-of-sample. This module measures — WITHOUT look-ahead
— whether that rescaling actually produces better-calibrated probabilities.

Method (honest, no leakage):
  1. Walk-forward to collect out-of-sample predictions (pred, raw_prob) + regime.
  2. Chronological 50/50 split of those OOS predictions.
  3. Regime reliability (hit-rate per regime + overall) computed on the FIRST
     half only.
  4. Apply the resulting confidence scale to the SECOND half, then compare
     Brier score and ECE: weighted vs unweighted.

Because the scale is derived only from earlier data, an improvement on the later
half is genuine. A prediction's probability ``p`` is read as P(direction is
correct); ``correct`` is 1/0. Lower Brier and ECE = better probabilities.

Run:
    python -m fade.pipeline.regime_eval btc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import REGIMES, assign_regimes
from fade.core.significant_changes import detect_significant_changes
from fade.pipeline.backtest import walk_forward
from fade.utils.logging import get_logger

log = get_logger("regime_eval")

# Fixed probability bins over [0.5, 1.0] for ECE (pre-registered, not tuned).
_ECE_EDGES = np.linspace(0.5, 1.0, 11)


def _brier(p: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((p - correct) ** 2)) if p.size else float("nan")


def _ece(p: np.ndarray, correct: np.ndarray) -> float:
    """Expected calibration error over fixed bins."""
    if p.size == 0:
        return float("nan")
    total = p.size
    err = 0.0
    for lo, hi in zip(_ECE_EDGES[:-1], _ECE_EDGES[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(correct[mask]))
        err += (mask.sum() / total) * abs(conf - acc)
    return float(err)


def _regime_scales(
    correct: np.ndarray,
    regime: np.ndarray,
    cap: float = 2.0,
) -> tuple[dict[str, float], float]:
    """Per-regime confidence scale learned from a data slice (no calibration)."""
    overall = float(np.mean(correct)) if correct.size else float("nan")
    overall_lift = overall - 0.5
    scales: dict[str, float] = {}
    for r in REGIMES:
        m = regime == r
        if not m.any() or overall_lift <= 0:
            scales[r] = 1.0
            continue
        regime_lift = float(np.mean(correct[m])) - 0.5
        scales[r] = float(np.clip(regime_lift / overall_lift, 0.0, cap))
    return scales, overall


def evaluate_regime_weighting(csv_path: str, config: Config | None = None) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    changes = detect_significant_changes(atoms, fwd, config)
    regimes = assign_regimes(changes, config.post_shock_bars)

    from fade.core.calibration import CalibrationStore
    scratch = CalibrationStore(config.cache_dir / f"_eval_{asset}.json")
    bt = walk_forward(atoms, fwd, config, calibration=scratch, regimes=regimes)
    preds = bt.oos_predictions
    if preds.empty:
        return {"asset": asset, "status": "no_predictions"}

    actual_up = (fwd > 0).astype(int).reindex(preds.index)
    reg = regimes.reindex(preds.index)
    valid = actual_up.notna() & reg.notna() & fwd.reindex(preds.index).notna()
    preds = preds[valid]
    correct = (preds["pred"].to_numpy() == actual_up[valid].to_numpy()).astype(int)
    raw = preds["raw_prob"].to_numpy(dtype=float)
    reg_arr = reg[valid].to_numpy()

    n = len(preds)
    if n < 200:
        return {"asset": asset, "status": "insufficient", "n": int(n)}

    split = n // 2
    scales, overall = _regime_scales(correct[:split], reg_arr[:split])

    # Second-half evaluation only.
    raw_h = raw[split:]
    correct_h = correct[split:]
    reg_h = reg_arr[split:]
    scale_vec = np.array([scales.get(r, 1.0) for r in reg_h])
    weighted = 0.5 + (raw_h - 0.5) * scale_vec

    result = {
        "asset": asset,
        "status": "ok",
        "n_total": int(n),
        "n_eval": int(len(raw_h)),
        "overall_hit_first_half": round(overall, 4),
        "scales": {r: round(s, 3) for r, s in scales.items()},
        "brier_unweighted": round(_brier(raw_h, correct_h), 5),
        "brier_weighted": round(_brier(weighted, correct_h), 5),
        "ece_unweighted": round(_ece(raw_h, correct_h), 5),
        "ece_weighted": round(_ece(weighted, correct_h), 5),
    }
    result["brier_delta"] = round(result["brier_weighted"] - result["brier_unweighted"], 5)
    result["ece_delta"] = round(result["ece_weighted"] - result["ece_unweighted"], 5)
    result["verdict"] = _verdict(result)
    return result


def _verdict(r: dict) -> str:
    b, e = r["brier_delta"], r["ece_delta"]
    if b < 0 and e < 0:
        return "IMPROVES - regime-weighting lowers both Brier and ECE."
    if b < 0 or e < 0:
        return "MIXED - one metric improves, the other does not."
    return "NO GAIN - regime-weighting does not improve probability quality."


def _print(r: dict) -> None:
    line = "=" * 64
    print("\n" + line)
    print(f"REGIME-WEIGHTING EVALUATION - {r.get('asset', '?').upper()}")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}  (n={r.get('n', r.get('n_total', 0))})")
        print(line + "\n")
        return
    print(f"  OOS predictions      : {r['n_total']}  (eval on last {r['n_eval']})")
    print(f"  Overall hit (1st half): {r['overall_hit_first_half']}")
    print("  Learned scales (1st half):")
    for reg, s in r["scales"].items():
        print(f"    {reg:<14} x{s}")
    print()
    print(f"  {'metric':<10}{'unweighted':>12}{'weighted':>12}{'delta':>10}")
    print(f"  {'Brier':<10}{r['brier_unweighted']:>12}{r['brier_weighted']:>12}{r['brier_delta']:>+10}")
    print(f"  {'ECE':<10}{r['ece_unweighted']:>12}{r['ece_weighted']:>12}{r['ece_delta']:>+10}")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv or not Path(csv).exists():
        log.error("Usage: python -m fade.pipeline.regime_eval path/to/ohlcv.csv")
        sys.exit(1)
    result = evaluate_regime_weighting(csv)
    _print(result)


if __name__ == "__main__":
    main()
