"""Rule engine.

A rule maps an event -> probability distribution of the 4h-forward outcome
(UP vs DOWN). Rules store support, confidence, decayed weight, directional
edge, mean forward return, and last-seen timestamp.

Older evidence is down-weighted via exponential decay (half-life in hours),
so the engine favours structure that persists into the recent past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core.targets import material_direction


def _decay_weights(index: pd.DatetimeIndex, ref_time: pd.Timestamp, half_life_h: float) -> np.ndarray:
    """Exponential decay weight per observation based on age in hours."""
    if half_life_h is None or half_life_h <= 0:
        return np.ones(len(index))
    age_h = (ref_time - index).total_seconds() / 3600.0
    age_h = np.clip(age_h, 0.0, None)
    return np.power(0.5, age_h / half_life_h)


def mine_rules(
    events: pd.DataFrame,
    fwd_return: pd.Series,
    config: Config,
    frequent: set[str] | None = None,
    ref_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Mine frequency-based rules from a long-format events table.

    Returns a DataFrame indexed by event key with columns:
        support, weighted_support, p_up, confidence, direction,
        mean_fwd, last_seen.
    Only events passing the frequency filter, min-support and min-confidence
    thresholds are retained (anti-overfitting).
    """
    df = events.copy()
    if frequent is not None:
        df = df[df["event"].isin(frequent)]

    df["fwd"] = fwd_return.reindex(df.index).to_numpy()
    df = df.dropna(subset=["fwd"])
    if df.empty:
        return _empty_rules()

    labels = material_direction(
        pd.Series(df["fwd"].values), config.move_threshold
    ).to_numpy()
    keep = np.isfinite(labels)
    df = df[keep].copy()
    if df.empty:
        return _empty_rules()

    ref_time = ref_time or df.index.max()
    df["up"] = labels[keep]
    df["w"] = _decay_weights(df.index, ref_time, config.decay_half_life_h)
    df["w_up"] = df["w"] * df["up"]
    df["w_fwd"] = df["w"] * df["fwd"]
    df["ts"] = df.index

    grouped = df.groupby("event", sort=False)
    agg = grouped.agg(
        support=("fwd", "size"),
        weighted_support=("w", "sum"),
        w_up=("w_up", "sum"),
        w_fwd=("w_fwd", "sum"),
        last_seen=("ts", "max"),
    )

    agg["p_up"] = agg["w_up"] / agg["weighted_support"]
    agg["confidence"] = np.maximum(agg["p_up"], 1.0 - agg["p_up"])
    agg["direction"] = np.where(agg["p_up"] >= 0.5, 1, 0).astype(int)
    agg["mean_fwd"] = agg["w_fwd"] / agg["weighted_support"]

    rules = agg[
        (agg["support"] >= config.min_support)
        & (agg["confidence"] >= config.min_confidence)
    ].copy()

    cols = ["support", "weighted_support", "p_up", "confidence",
            "direction", "mean_fwd", "last_seen"]
    return rules[cols].sort_values("confidence", ascending=False)


def _empty_rules() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["support", "weighted_support", "p_up", "confidence",
                 "direction", "mean_fwd", "last_seen"]
    )
