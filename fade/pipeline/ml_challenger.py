"""ML challenger baseline — sandbox only, NOT part of the FADE core engine.

Honest out-of-sample comparison: gradient-boosted classifier on path_lean3
atoms vs FADE's rule-based holdout (~53–54%). This module is explicitly
outside the scope-guard "no ML" invariant for core inference; it exists to
sanity-check whether a simple fitted model can match or beat interpretable rules.

Protocol (matches holdout.py split):
    1. Chronological 70% dev / 30% holdout — holdout never seen in training.
    2. Features: path_lean3 atoms (close_pos, range_pct, streak_signed) via
       lean_config(); continuous values from compute_atoms (causal, no look-ahead).
    3. Target: next-bar direction (sign of 1h forward return).
    4. Headline metrics scored ONLY on the holdout slice.

Run from repo root:
    python -m fade.pipeline.ml_challenger
    python -m fade.pipeline.ml_challenger btc_1h.csv eth_1h.csv
    python -m fade.pipeline.ml_challenger --lstm btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config, lean_config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.targets import score_predictions

try:
    from xgboost import XGBClassifier

    _BACKEND = "xgboost"
except ImportError:
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        _BACKEND = "sklearn"
    except ImportError:
        _BACKEND = None

# FADE rule-based holdout references (docs/PROJECT_STATE.md, holdout / lean_search).
FADE_BASELINE = {
    "btc_1h": 0.5464,
    "eth_1h": 0.5327,  # approximate; training_suite eth full ~53.3%
}
FADE_BASELINE_RANGE = (0.53, 0.54)

HOLDOUT_FRAC = 0.30
N_SHUFFLES = 300
P_VALUE_MAX = 0.05
DEFAULT_ASSETS = ("btc_1h.csv", "eth_1h.csv")


def _make_classifier(seed: int):
    if _BACKEND == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )
    if _BACKEND == "sklearn":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )
    raise RuntimeError(
        "ML challenger requires xgboost or scikit-learn. "
        "Install one: pip install xgboost  OR  pip install scikit-learn"
    )


def _build_dataset(
    csv_path: str | Path,
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Align causal path_lean3 features with next-bar direction labels."""
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    valid = fwd.notna()
    X = atoms.loc[valid]
    y = (fwd.loc[valid] > 0).astype(int)
    meta = {
        "asset": Path(csv_path).stem.lower(),
        "n_total": int(len(X)),
        "feature_cols": list(X.columns),
    }
    return X, y, meta


def run_challenger(
    csv_path: str | Path,
    holdout_frac: float = HOLDOUT_FRAC,
    n_shuffles: int = N_SHUFFLES,
    seed: int = 0,
    config: Config | None = None,
) -> dict:
    """Train a gradient booster on dev; score holdout direction predictions."""
    config = config or lean_config()
    X, y, meta = _build_dataset(csv_path, config)
    n = len(X)
    split = int(n * (1.0 - holdout_frac))
    if split < 50 or (n - split) < 20:
        return {
            **meta,
            "status": "too_short",
            "verdict": "INCONCLUSIVE - series too short for 70/30 split.",
        }

    X_dev, X_hold = X.iloc[:split], X.iloc[split:]
    y_dev, y_hold = y.iloc[:split], y.iloc[split:]

    clf = _make_classifier(seed)
    clf.fit(X_dev.to_numpy(), y_dev.to_numpy())
    pred_hold = clf.predict(X_hold.to_numpy())

    pred_v, act_v = score_predictions(pred_hold, y_hold.to_numpy(), config.move_threshold)
    coverage = int(len(pred_v))
    if coverage == 0:
        return {
            **meta,
            "status": "no_coverage",
            "n_dev": split,
            "n_holdout": n - split,
            "verdict": "INCONCLUSIVE - no scorable holdout predictions.",
        }

    hit = float(np.mean(pred_v == act_v))
    lift = hit - 0.5

    rng = np.random.default_rng(seed)
    null_hits = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null_hits[i] = np.mean(pred_v == rng.permutation(act_v))
    n_ge = int(np.sum(null_hits >= hit))
    p_value = (1 + n_ge) / (1 + n_shuffles)

    asset = meta["asset"]
    fade_ref = FADE_BASELINE.get(asset)
    if fade_ref is not None:
        vs_fade = round(hit - fade_ref, 4)
        fade_note = f"{fade_ref:.2%} (FADE path_lean3 holdout ref)"
    else:
        vs_fade = None
        fade_note = f"{FADE_BASELINE_RANGE[0]:.0%}–{FADE_BASELINE_RANGE[1]:.0%} (FADE typical)"

    verdict = _verdict(lift, p_value)

    return {
        **meta,
        "status": "ok",
        "backend": _BACKEND,
        "atomset": "path_lean3",
        "n_dev": split,
        "n_holdout": n - split,
        "n_predictions": coverage,
        "holdout_hit_rate": round(hit, 4),
        "holdout_lift_vs_random": round(lift, 4),
        "null_mean": round(float(np.mean(null_hits)), 4),
        "null_std": round(float(np.std(null_hits)), 4),
        "p_value": round(p_value, 4),
        "fade_baseline_ref": fade_note,
        "vs_fade_baseline": vs_fade,
        "verdict": verdict,
    }


def _verdict(lift: float, p_value: float) -> str:
    if lift <= 0:
        return "FAIL - no positive edge on holdout."
    if p_value <= P_VALUE_MAX:
        return "PASS - holdout edge significant vs shuffle null."
    return "WEAK - positive lift but within shuffle noise."


def _print_report(r: dict) -> None:
    line = "=" * 66
    asset = r.get("asset", "?").upper()
    print("\n" + line)
    print(f"FADE ML CHALLENGER (SANDBOX - NOT CORE) - {asset}")
    print(line)
    print("  Scope: challenger baseline only; core inference path untouched.")
    print(f"  Backend: {r.get('backend', '?')}  |  atoms: {r.get('atomset', 'path_lean3')}")
    if r.get("feature_cols"):
        print(f"  Features: {', '.join(r['feature_cols'])}")
    if r.get("status") != "ok":
        print(f"\n  {r.get('verdict', r.get('status'))}")
        print(line + "\n")
        return
    print(f"  Split: dev={r['n_dev']} bars  |  holdout={r['n_holdout']} bars (quarantined)")
    print(f"  Holdout predictions (all bars) : {r['n_predictions']}")
    print()
    print(f"  Holdout hit-rate                 : {r['holdout_hit_rate']:.2%}")
    print(f"  Lift vs random                   : {r['holdout_lift_vs_random']:+.4f}")
    print(f"  Shuffle null (mean +/- std)      : {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
    print(f"  Permutation p-value              : {r['p_value']:.4f}")
    print()
    print(f"  FADE rule-based holdout ref      : {r['fade_baseline_ref']}")
    if r.get("vs_fade_baseline") is not None:
        sign = "+" if r["vs_fade_baseline"] >= 0 else ""
        print(f"  ML challenger vs FADE baseline   : {sign}{r['vs_fade_baseline']:.4f}")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    if _BACKEND is None:
        print("ERROR: install xgboost or scikit-learn to run the ML challenger.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="FADE ML challenger (sandbox baseline — NOT core engine)",
    )
    parser.add_argument(
        "assets",
        nargs="*",
        default=list(DEFAULT_ASSETS),
        help="OHLCV CSV paths (default: btc_1h.csv eth_1h.csv)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="also run LSTM sequential challenger (PyTorch or Keras) and compare",
    )
    args = parser.parse_args()

    config = lean_config()
    any_ok = False
    for csv in args.assets:
        path = Path(csv)
        if not path.exists():
            print(f"\nSKIP {csv} — file not found.\n")
            continue
        result = run_challenger(path, seed=args.seed, config=config)
        _print_report(result)
        any_ok = any_ok or result.get("status") == "ok"

        if args.lstm:
            from fade.pipeline.ml_challenger_lstm import (
                lstm_backend_available,
                print_comparison,
                print_lstm_report,
                run_lstm_challenger,
            )

            if not lstm_backend_available():
                print(
                    "\nLSTM SKIP (SANDBOX - NOT CORE): no PyTorch or Keras/TensorFlow. "
                    "Install torch for sandbox: pip install torch\n"
                )
            else:
                lstm_result = run_lstm_challenger(path, seed=args.seed, config=config)
                print_lstm_report(lstm_result)
                print_comparison(result, lstm_result)
                any_ok = any_ok or lstm_result.get("status") == "ok"

    if not any_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
