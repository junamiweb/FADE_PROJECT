"""Market regime tagging from significant-change detection.

Regimes (v0.2 P0):
    SHOCK_ACTIVE  - bar with a significant price/volume shock
    POST_SHOCK    - within N bars after a shock (rules may behave differently)
    NORMAL        - everything else

No ML. Derived only from ``significant_changes`` flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = ("NORMAL", "SHOCK_ACTIVE", "POST_SHOCK")
VR_REGIMES = ("LOW_VR", "NORMAL", "HIGH_VR")


def assign_regimes(changes: pd.DataFrame, post_shock_bars: int = 4) -> pd.Series:
    """Tag each bar with a regime label."""
    regime = pd.Series("NORMAL", index=changes.index, dtype=object)
    shocked = changes["change_type"].astype(str) != ""
    regime.loc[shocked] = "SHOCK_ACTIVE"

    shock_positions = np.flatnonzero(shocked.to_numpy())
    for i in shock_positions:
        for offset in range(1, post_shock_bars + 1):
            j = i + offset
            if j < len(regime) and regime.iloc[j] == "NORMAL":
                regime.iloc[j] = "POST_SHOCK"
    return regime


def compute_vol_ratio(
    returns: pd.Series,
    short_window: int = 24,
    long_window: int = 168,
    shift: int = 1,
) -> pd.Series:
    """Short/long rolling volatility ratio (causal).

    VR = std(returns, short) / std(returns, long). Values < 1 indicate vol
    compression (mean-reversion friendly); > 1 indicate vol expansion (momentum).
    ``shift`` excludes the current bar's return so the ratio is known before the
    bar closes (no look-ahead when gating predictions on bar *t*).
    """
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    ratio = short_vol / long_vol.replace(0.0, np.nan)
    if shift:
        ratio = ratio.shift(shift)
    return ratio.rename("vol_ratio")


def assign_vr_regime(
    vol_ratio: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    """Classify bars by volatility ratio: LOW_VR, NORMAL, HIGH_VR."""
    regime = pd.Series("NORMAL", index=vol_ratio.index, dtype=object)
    valid = vol_ratio.notna()
    regime.loc[valid & (vol_ratio <= low_threshold)] = "LOW_VR"
    regime.loc[valid & (vol_ratio >= high_threshold)] = "HIGH_VR"
    return regime


def evaluate_by_regime(
    preds: pd.DataFrame,
    fwd_return: pd.Series,
    regimes: pd.Series,
) -> dict[str, dict]:
    """Hit-rate and lift per regime on out-of-sample predictions."""
    if preds.empty:
        return {}

    actual_up = (fwd_return > 0).astype(int)
    aligned_regimes = regimes.reindex(preds.index)
    out: dict[str, dict] = {}

    for regime in REGIMES:
        mask = aligned_regimes == regime
        sub = preds[mask]
        if sub.empty:
            continue
        act = actual_up.reindex(sub.index)
        valid = act.notna()
        if valid.sum() == 0:
            continue
        hr = float((sub.loc[valid, "pred"] == act[valid]).mean())
        out[regime] = {
            "hit_rate": round(hr, 4),
            "lift_vs_random": round(hr - 0.5, 4),
            "n_predictions": int(valid.sum()),
        }
    return out


def save_regime_stats(path: str | Path, regime_metrics: dict) -> dict:
    """Persist per-regime OOS reliability for later use in forecasting.

    Stores each regime's hit-rate plus an n-weighted overall hit-rate. This is
    derived only from past out-of-sample folds, so using it to weight a new
    forecast introduces no look-ahead.
    """
    total_n = sum(m["n_predictions"] for m in regime_metrics.values())
    overall = (
        sum(m["hit_rate"] * m["n_predictions"] for m in regime_metrics.values())
        / total_n
    ) if total_n else None

    data = {
        "overall_hit_rate": round(overall, 4) if overall is not None else None,
        "regimes": regime_metrics,
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def load_regime_stats(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def regime_confidence_scale(
    regime: str,
    stats: dict,
    cap: float = 2.0,
) -> float:
    """Confidence multiplier for a regime, based on its OOS reliability.

    Scales prediction confidence around 0.5 WITHOUT changing direction:
      * regime historically better than average  -> scale > 1 (more confident)
      * regime historically worse / no edge       -> scale < 1 (pull toward 50%)
      * regime with non-positive lift              -> scale 0 (no confidence)

    Because it only rescales the distance from 0.5, it cannot flip the predicted
    direction and therefore cannot manufacture edge — it only affects the %.
    """
    if not stats:
        return 1.0
    overall = stats.get("overall_hit_rate")
    m = (stats.get("regimes") or {}).get(regime)
    if not m or overall is None:
        return 1.0
    overall_lift = overall - 0.5
    if overall_lift <= 0:
        return 1.0
    regime_lift = m["hit_rate"] - 0.5
    return float(np.clip(regime_lift / overall_lift, 0.0, cap))
