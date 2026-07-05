"""Calibrated predictor — each forecast carries a probability %.

Uses stable-rule OOS hit-rates when available (positive memory), falls back
to training confidence, then maps through the persistent calibration table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fade.core.calibration import CalibrationStore


def _rule_weight(
    event: str,
    rules: pd.DataFrame,
    positive: dict[str, dict],
) -> tuple[int, float]:
    """Return (direction, weight) for one rule."""
    direction = int(rules.loc[event, "direction"])
    if event in positive and "avg_oos_hit" in positive[event]:
        weight = float(positive[event]["avg_oos_hit"])
    else:
        weight = float(rules.loc[event, "confidence"])
    return direction, weight


def predict_calibrated(
    events: pd.DataFrame,
    rules: pd.DataFrame,
    calibration: CalibrationStore,
    positive: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Produce direction + raw + calibrated probability for each timestamp.

    Columns: pred, raw_prob, calibrated_prob, n_rules.
    """
    positive = positive or {}
    if rules.empty or events.empty:
        return pd.DataFrame(
            columns=["pred", "raw_prob", "calibrated_prob", "n_rules"]
        )

    matched = events[events["event"].isin(rules.index)].copy()
    if matched.empty:
        return pd.DataFrame(
            columns=["pred", "raw_prob", "calibrated_prob", "n_rules"]
        )

    rows = []
    for ts, grp in matched.groupby(matched.index):
        up_w, dn_w, up_n, dn_n = 0.0, 0.0, 0, 0
        for event in grp["event"]:
            d, w = _rule_weight(event, rules, positive)
            if d == 1:
                up_w += w
                up_n += 1
            else:
                dn_w += w
                dn_n += 1
        if up_n + dn_n == 0:
            continue
        pred = 1 if up_w >= dn_w else 0
        # Average hit-rate of rules in the winning direction (not vote share).
        raw_prob = (up_w / up_n) if pred == 1 and up_n else (dn_w / dn_n)
        cal_prob = calibration.calibrate(raw_prob)
        rows.append({
            "ts": ts,
            "pred": pred,
            "raw_prob": raw_prob,
            "calibrated_prob": cal_prob,
            "n_rules": up_n + dn_n,
        })

    if not rows:
        return pd.DataFrame(
            columns=["pred", "raw_prob", "calibrated_prob", "n_rules"]
        )

    out = pd.DataFrame(rows).set_index("ts")
    return out[["pred", "raw_prob", "calibrated_prob", "n_rules"]]


def collect_calibration_samples(
    preds: pd.DataFrame,
    fwd_return: pd.Series,
) -> list[tuple[float, int]]:
    """Build (raw_prob, correct) pairs from predictions with known outcomes."""
    if preds.empty:
        return []
    actual_up = (fwd_return > 0).astype(int)
    samples = []
    for ts, row in preds.iterrows():
        if ts not in actual_up.index or not np.isfinite(fwd_return.loc[ts]):
            continue
        correct = int(row["pred"] == actual_up.loc[ts])
        samples.append((float(row["raw_prob"]), correct))
    return samples
