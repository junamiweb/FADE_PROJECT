"""Evaluation: turn rules into predictions and score them against baselines.

Prediction at each timestamp = confidence-weighted vote of all matching rules.
Scored by directional hit-rate on *covered* timestamps and compared against:
    - random baseline (0.5)
    - momentum baseline (sign of return_6h)

Predictive lift = model hit-rate minus the stronger baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict(events: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matching-rule votes into one prediction per timestamp.

    Returns a frame indexed by timestamp with columns:
        pred (1=up, 0=down), score (signed confidence sum), n_rules.
    Only timestamps covered by at least one rule are returned.
    """
    if rules.empty or events.empty:
        return pd.DataFrame(columns=["pred", "score", "n_rules"])

    matched = events[events["event"].isin(rules.index)].copy()
    if matched.empty:
        return pd.DataFrame(columns=["pred", "score", "n_rules"])

    conf = rules["confidence"].to_dict()
    direction = rules["direction"].to_dict()
    # Signed vote per matched event: +conf for up, -conf for down.
    matched["vote"] = matched["event"].map(
        lambda e: conf[e] * (1.0 if direction[e] == 1 else -1.0)
    )
    matched["ts"] = matched.index

    grp = matched.groupby("ts")
    out = pd.DataFrame({
        "score": grp["vote"].sum(),
        "n_rules": grp["vote"].size(),
    })
    out["pred"] = (out["score"] > 0).astype(int)
    return out[["pred", "score", "n_rules"]]


def _hit_rate(pred: pd.Series, actual_up: pd.Series) -> float:
    aligned = actual_up.reindex(pred.index)
    mask = aligned.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == aligned[mask]).mean())


def evaluate(
    events: pd.DataFrame,
    rules: pd.DataFrame,
    fwd_return: pd.Series,
    return_6h: pd.Series,
) -> dict:
    """Score model predictions vs random and momentum baselines on a test set."""
    preds = predict(events, rules)
    actual_up = (fwd_return > 0).astype(int)

    if preds.empty:
        return {
            "coverage": 0,
            "model_hit_rate": float("nan"),
            "random_hit_rate": 0.5,
            "momentum_hit_rate": float("nan"),
            "lift_vs_random": float("nan"),
            "lift_vs_momentum": float("nan"),
            "mean_signed_edge": float("nan"),
        }

    model_hr = _hit_rate(preds["pred"], actual_up)

    # Momentum baseline evaluated on the same covered timestamps for fairness.
    mom_pred = (return_6h.reindex(preds.index) > 0).astype(int)
    momentum_hr = _hit_rate(mom_pred, actual_up)

    # Directional edge: mean forward return in the predicted direction.
    covered_fwd = fwd_return.reindex(preds.index)
    signed = np.where(preds["pred"] == 1, covered_fwd, -covered_fwd)
    mean_edge = float(np.nanmean(signed))

    return {
        "coverage": int(len(preds)),
        "model_hit_rate": model_hr,
        "random_hit_rate": 0.5,
        "momentum_hit_rate": momentum_hr,
        "lift_vs_random": model_hr - 0.5,
        "lift_vs_momentum": model_hr - momentum_hr,
        "mean_signed_edge": mean_edge,
    }
