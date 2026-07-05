"""Prediction targets — directional labels with optional magnitude filter."""

from __future__ import annotations

import numpy as np
import pandas as pd


def material_direction(fwd: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Binary direction label; neutral zone excluded when ``threshold > 0``.

    Returns 1 (material up), 0 (material down), or NaN (neutral / missing).
    At threshold 0 this reduces to the classic sign(fwd) label.
    """
    if threshold <= 0:
        return (fwd > 0).astype(float).where(fwd.notna())

    labels = pd.Series(np.nan, index=fwd.index, dtype=float)
    labels[fwd > threshold] = 1.0
    labels[fwd < -threshold] = 0.0
    return labels


def score_predictions(
    pred: np.ndarray,
    fwd: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, actual) pairs that are scorable under a magnitude rule.

    At threshold 0: all finite rows score (actual = sign(fwd)).
    At threshold > 0: only rows where |fwd| > threshold score; correct when
    pred matches the material direction.
    """
    mask = np.isfinite(fwd)
    if threshold > 0:
        mask &= np.abs(fwd) > threshold
    if not mask.any():
        return np.array([]), np.array([])
    actual = (fwd[mask] > 0).astype(int)
    return pred[mask], actual
