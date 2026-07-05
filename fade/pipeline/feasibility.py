"""FADE feasibility test - answers ONE question:

    "Is there a real, generalising atomic structure here, or just noise?"

This is NOT a new feature. It reuses the existing walk-forward engine and runs
two classic statistical checks, then prints a hard PASS / FAIL / INCONCLUSIVE.

Checks
------
1. SHUFFLE (permutation) TEST
   Shuffle the forward-return labels so any event->outcome relationship is
   destroyed, while keeping the return distribution intact. Re-run the whole
   walk-forward many times to build a NULL distribution of "lift". If the real
   lift is not clearly above this null, the edge is indistinguishable from luck.
   -> yields an empirical p-value.

2. SEGMENT STABILITY TEST
   Split the timeline into contiguous segments (no shuffling of time) and run an
   independent walk-forward inside each. A real edge should be positive in most
   segments, not carried by one lucky period.
   -> yields a consistency fraction.

Pass criteria are fixed IN ADVANCE (pre-registration) to avoid fishing:
    p_value <= 0.05  AND  segment_consistency >= 0.60
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import generate_synthetic_ohlcv, load_ohlcv
from fade.pipeline.backtest import walk_forward
from fade.utils.logging import get_logger

log = get_logger("feasibility")

# --- pre-registered pass criteria (do not tune after seeing results) ---
P_VALUE_MAX = 0.05
CONSISTENCY_MIN = 0.60


def _mean_lift(atoms: pd.DataFrame, fwd: pd.Series, config: Config) -> float:
    """Mean lift-vs-random across walk-forward folds (NaN if no coverage)."""
    bt = walk_forward(atoms, fwd, config)
    if not bt.fold_metrics:
        return float("nan")
    lifts = [m["lift_vs_random"] for m in bt.fold_metrics
             if m["lift_vs_random"] == m["lift_vs_random"]]
    return float(np.mean(lifts)) if lifts else float("nan")


def _shuffle_forward(fwd: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Permute only the finite forward-returns in place (keeps coverage/dist)."""
    values = fwd.to_numpy(dtype=float).copy()
    finite = np.where(np.isfinite(values))[0]
    permuted = rng.permutation(values[finite])
    values[finite] = permuted
    return pd.Series(values, index=fwd.index, name=fwd.name)


def _segment_bounds(n: int, n_segments: int) -> list[tuple[int, int]]:
    span = n // n_segments
    return [(k * span, (n if k == n_segments - 1 else (k + 1) * span))
            for k in range(n_segments)]


def feasibility(
    csv_path: str | None = None,
    n_shuffles: int = 40,
    n_segments: int = 4,
    seed: int = 0,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    rng = np.random.default_rng(seed)

    if csv_path:
        df = load_ohlcv(csv_path)
        log.info("Loaded %d rows from %s", len(df), csv_path)
    else:
        df = generate_synthetic_ohlcv()
        log.info("Using %d rows of synthetic data", len(df))

    # Compute atoms + target once (minimal computation).
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    # --- 1. Real lift on true labels ---
    real_lift = _mean_lift(atoms, fwd, config)
    log.info("real mean lift vs random = %.4f", real_lift)

    # --- 2. Shuffle / permutation null ---
    null_lifts = []
    for i in range(n_shuffles):
        shuffled = _shuffle_forward(fwd, rng)
        null_lifts.append(_mean_lift(atoms, shuffled, config))
    null_arr = np.array([x for x in null_lifts if x == x])
    # Empirical p-value with +1 smoothing (never reports p=0).
    n_ge = int(np.sum(null_arr >= real_lift)) if null_arr.size else 0
    p_value = (1 + n_ge) / (1 + null_arr.size) if null_arr.size else float("nan")

    # --- 3. Segment stability (time-ordered, no shuffling) ---
    seg_lifts = []
    for start, end in _segment_bounds(len(atoms), n_segments):
        seg_atoms = atoms.iloc[start:end]
        seg_fwd = fwd.iloc[start:end]
        seg_lifts.append(_mean_lift(seg_atoms, seg_fwd, config))
    seg_arr = np.array([x for x in seg_lifts if x == x])
    consistency = float(np.mean(seg_arr > 0)) if seg_arr.size else float("nan")

    result = {
        "real_lift": real_lift,
        "null_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
        "null_std": float(np.std(null_arr)) if null_arr.size else float("nan"),
        "p_value": p_value,
        "segment_lifts": seg_lifts,
        "segment_consistency": consistency,
        "n_shuffles": int(null_arr.size),
        "n_segments": n_segments,
    }
    result["verdict"] = _verdict(result)
    _print(result)
    return result


def _verdict(r: dict) -> str:
    p, c, real = r["p_value"], r["segment_consistency"], r["real_lift"]
    if real != real or p != p or c != c:
        return "INCONCLUSIVE - insufficient coverage to decide."
    if real <= 0:
        return "FAIL - no positive edge on real labels."
    if p <= P_VALUE_MAX and c >= CONSISTENCY_MIN:
        return "PASS - edge is significant AND stable across time. Feasible."
    if p <= P_VALUE_MAX:
        return "WEAK - significant vs noise but not stable across segments."
    return "FAIL - edge is within the range of random shuffles (likely luck)."


def _print(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print("FADE FEASIBILITY TEST  -  'Is there anything real here?'")
    print(line)
    print(f"  Pre-registered pass: p<={P_VALUE_MAX}  AND  consistency>={CONSISTENCY_MIN}")
    print()
    print(f"  Real lift vs random        : {r['real_lift']:+.4f}")
    print(f"  Shuffle null (mean +/- std): {r['null_mean']:+.4f} +/- {r['null_std']:.4f}"
          f"   (n={r['n_shuffles']})")
    print(f"  Permutation p-value        : {r['p_value']:.4f}")
    print()
    segs = "  ".join(f"{x:+.3f}" if x == x else "  nan" for x in r["segment_lifts"])
    print(f"  Segment lifts ({r['n_segments']} splits)   : {segs}")
    print(f"  Segment consistency        : {r['segment_consistency']:.2f}"
          f"  ({int(round(r['segment_consistency']*r['n_segments']))}/{r['n_segments']} positive)")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if csv and not Path(csv).exists():
        log.warning("CSV %s not found; using synthetic data.", csv)
        csv = None
    feasibility(csv)


if __name__ == "__main__":
    main()
