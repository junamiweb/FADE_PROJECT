"""Walk-forward backtest (sequential splits only - no random splitting).

Uses an expanding training window. For each fold:
  1. Compute discretisation thresholds on the *training* slice only.
  2. Mine rules on training events (negative memory filtered out first).
  3. Evaluate on the immediately-following test slice.
  4. Record which rules fired and their direction (for stability tracking).

Stability across folds is the core anti-overfitting signal: a rule is only
trustworthy if it recurs in many folds with a consistent direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import events as ev
from fade.core import rules as rl
from fade.core.calibration import CalibrationStore
from fade.core.evaluator import evaluate
from fade.core.predictor import predict_calibrated
from fade.core.regimes import evaluate_by_regime
from fade.core.targets import material_direction


@dataclass
class BacktestResult:
    fold_metrics: list[dict] = field(default_factory=list)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_metrics: dict = field(default_factory=dict)
    n_events_last: int = 0
    n_rules_last: int = 0


def _fold_bounds(n: int, config: Config) -> list[tuple[int, int, int]]:
    """Return (train_end, test_start, test_end) index bounds per fold."""
    start = int(n * config.initial_train_frac)
    if start >= n:
        return []
    test_span = (n - start) // config.n_folds
    if test_span <= 0:
        return []
    bounds = []
    for k in range(config.n_folds):
        train_end = start + k * test_span
        test_start = train_end
        test_end = n if k == config.n_folds - 1 else train_end + test_span
        if test_start >= test_end:
            break
        bounds.append((train_end, test_start, test_end))
    return bounds


def walk_forward(
    atoms: pd.DataFrame,
    fwd_return: pd.Series,
    config: Config,
    blocked: set[str] | None = None,
    calibration: CalibrationStore | None = None,
    positive: dict[str, dict] | None = None,
    regimes: pd.Series | None = None,
) -> BacktestResult:
    """Run the sequential walk-forward backtest and track rule stability."""
    blocked = blocked or set()
    n = len(atoms)
    bounds = _fold_bounds(n, config)
    result = BacktestResult()

    # Per-event out-of-sample records across folds. A rule is only "stable" if
    # its *training* direction keeps predicting the *test* slice correctly.
    directions: dict[str, list[int]] = {}      # mined (train) direction per fold
    confidences: dict[str, list[float]] = {}   # train confidence per fold
    oos_correct: dict[str, list[int]] = {}     # 1 if train dir beat 0.5 on test
    oos_hit: dict[str, list[float]] = {}       # test hit-rate per fold

    for fold, (train_end, test_start, test_end) in enumerate(bounds):
        train_atoms = atoms.iloc[:train_end]
        test_atoms = atoms.iloc[test_start:test_end]
        if len(train_atoms) < config.min_support or test_atoms.empty:
            continue

        # Thresholds from train only -> no look-ahead into the test slice.
        thresholds = ev.compute_thresholds(train_atoms, config)

        train_disc = ev.discretize(train_atoms, thresholds)
        train_events = ev.build_events(train_disc, config, blocked=blocked)
        frequent = ev.frequent_events(train_events, config.min_event_frequency)
        rules = rl.mine_rules(train_events, fwd_return, config,
                              frequent=frequent, ref_time=train_atoms.index.max())

        test_disc = ev.discretize(test_atoms, thresholds)
        test_events = ev.build_events(test_disc, config, allowed=set(rules.index),
                                      blocked=blocked)
        # Momentum baseline uses the 6h atom (raw, not discretised). Some atom
        # sets (e.g. a news-only feature set) do not include it; fall back to a
        # NaN series so the momentum baseline is simply undefined, not an error.
        if "return_6h" in test_atoms.columns:
            return_6h = test_atoms["return_6h"]
        else:
            return_6h = pd.Series(np.nan, index=test_atoms.index)
        metrics = evaluate(test_events, rules, fwd_return, return_6h)
        metrics["fold"] = fold
        metrics["n_rules"] = int(len(rules))
        result.fold_metrics.append(metrics)

        if calibration is not None:
            fold_preds = predict_calibrated(
                test_events, rules, calibration, positive or {}
            )
            if not fold_preds.empty:
                result.oos_predictions = pd.concat(
                    [result.oos_predictions, fold_preds]
                )

        # Out-of-sample per-rule scoring on this fold's test slice.
        test_hits = _oos_hit_rates(test_events, rules, fwd_return, config)
        for event, row in rules.iterrows():
            directions.setdefault(event, []).append(int(row["direction"]))
            confidences.setdefault(event, []).append(float(row["confidence"]))
            hr = test_hits.get(event)
            if hr is not None:
                oos_hit.setdefault(event, []).append(hr)
                oos_correct.setdefault(event, []).append(1 if hr > 0.5 else 0)

        result.n_events_last = int(train_events["event"].nunique())
        result.n_rules_last = int(len(rules))

    result.stability = _stability_table(
        directions, confidences, oos_correct, oos_hit, len(bounds)
    )
    if regimes is not None and not result.oos_predictions.empty:
        result.regime_metrics = evaluate_by_regime(
            result.oos_predictions, fwd_return, regimes
        )
    return result


def _oos_hit_rates(
    test_events: pd.DataFrame,
    rules: pd.DataFrame,
    fwd_return: pd.Series,
    config: Config,
) -> dict[str, float]:
    """Out-of-sample hit-rate per rule on a test slice."""
    if test_events.empty or rules.empty:
        return {}
    df = test_events.copy()
    df["fwd"] = fwd_return.reindex(df.index).to_numpy()
    df = df.dropna(subset=["fwd"])
    if df.empty:
        return {}
    labels = material_direction(
        pd.Series(df["fwd"].values), config.move_threshold
    ).to_numpy()
    keep = np.isfinite(labels)
    df = df[keep].copy()
    if df.empty:
        return {}
    df["up"] = labels[keep].astype(int)
    grp = df.groupby("event")["up"]
    up_frac = grp.mean()
    support = grp.size()
    out: dict[str, float] = {}
    for event, frac in up_frac.items():
        if support[event] < config.min_support:
            continue
        direction = int(rules.loc[event, "direction"])
        out[event] = float(frac if direction == 1 else 1.0 - frac)
    return out


def _stability_table(
    directions: dict[str, list[int]],
    confidences: dict[str, list[float]],
    oos_correct: dict[str, list[int]],
    oos_hit: dict[str, list[float]],
    total_folds: int,
) -> pd.DataFrame:
    """Summarise cross-fold out-of-sample behaviour per event.

    ``folds_present`` counts folds where the rule had a testable out-of-sample
    estimate. ``consistency`` is the fraction of those folds where the rule beat
    the coin-flip (>0.5). ``stability`` = coverage * consistency and is the
    metric used to promote rules to positive memory.
    """
    rows = []
    for event, dirs in directions.items():
        correct = oos_correct.get(event, [])
        folds_present = len(correct)
        if folds_present == 0:
            continue
        arr = np.array(correct)
        consistency = float(arr.mean())
        modal_dir = int(round(np.mean(directions[event])))
        coverage_frac = folds_present / total_folds if total_folds else 0.0
        rows.append({
            "event": event,
            "folds_present": folds_present,
            "coverage_frac": coverage_frac,
            "direction": modal_dir,
            "consistency": consistency,
            "avg_oos_hit": float(np.mean(oos_hit[event])),
            "avg_confidence": float(np.mean(confidences[event])),
            "stability": coverage_frac * consistency,
        })
    if not rows:
        return pd.DataFrame(
            columns=["event", "folds_present", "coverage_frac", "direction",
                     "consistency", "avg_oos_hit", "avg_confidence", "stability"]
        ).set_index("event")
    return (
        pd.DataFrame(rows)
        .set_index("event")
        .sort_values(["stability", "avg_oos_hit"], ascending=False)
    )
