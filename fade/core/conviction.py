"""Conviction state -- read the latest bar's reversal-mechanism strength.

Uses fixed contrarian definitions validated on holdout (no fitting):
    streak contrarian: predict against a run of length >= 2
    multi-res: 5m/15m/30m/1h contrarian signals must agree

Calibrated percentages are empirical holdout hit-rates from conviction_combo.py
(batch 21). They are honest labels, not model estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.conviction_gate import MULTI_FILES, _contrarian_grid
from fade.pipeline.trend_structure import _signed_streak

HOLDOUT_FRAC = 0.30
HIGH_VR_MIN_STREAK = 3

# (tier_id, label_he, min_streak, min_agree, calibrated_hit, note_he)
# Evaluated only if streak_len >= min_streak AND tf_agree >= min_agree (0 = skip).
TIER_DEFS = [
    ("elite", "ELITE (r>=2 + 4 TF)", 2, 4, 0.626, "~1% מהזמן"),
    ("strong", "STRONG (r>=3 + 3 TF)", 3, 3, 0.596, "~4% מהזמן"),
    ("high", "HIGH (r>=2 + 3 TF)", 2, 3, 0.582, "~9% מהזמן"),
    ("multi3", "MULTI-3 TF", 0, 3, 0.583, "~11% מהזמן"),
    ("streak4", "STREAK>=4", 4, 0, 0.571, "~10% מהזמן"),
    ("streak3", "STREAK>=3", 3, 0, 0.553, "~22% מהזמן"),
    ("streak2", "STREAK>=2", 2, 0, 0.546, "~48% מהזמן"),
]


def _latest_streak(csv: str) -> tuple[int, int, str]:
    """Return (signed_streak, abs_len, contrarian_direction UP/DOWN/FLAT)."""
    df = load_ohlcv(csv)
    ret = df["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    s = int(streak[-1]) if np.isfinite(streak[-1]) else 0
    if s == 0 or abs(s) < 2:
        return s, abs(s), "FLAT"
    return s, abs(s), "UP" if s < 0 else "DOWN"  # contrarian


def _multi_files_for(primary_csv: str) -> list[str]:
    """Resolution files for multi-TF conviction, matched to asset prefix."""
    stem = Path(primary_csv).stem  # e.g. btc_1h, eth_1h
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return [f"{prefix}_{iv}.csv" for iv in ("5m", "15m", "30m", "1h")
            if Path(f"{prefix}_{iv}.csv").exists()]


def _latest_multi(root: Path, primary_csv: str) -> dict:
    """Contrarian direction per TF at the latest aligned 1h timestamp."""
    sigs = {}
    for c in _multi_files_for(str(root / Path(primary_csv).name)):
        p = root / Path(c).name if not Path(c).is_absolute() else Path(c)
        if not p.exists():
            p = Path(c)
        if not p.exists():
            continue
        s = _contrarian_grid(str(p))
        if s.empty:
            continue
        last = s.iloc[-1]
        if last == 0:
            sigs[Path(c).stem] = None
        else:
            sigs[Path(c).stem] = "UP" if last > 0 else "DOWN"
    return sigs


def _latest_vr_regime(primary_csv: str, config: Config | None = None) -> str | None:
    """Latest VR bucket; tertile thresholds fitted on dev when config cuts unset."""
    config = config or Config()
    df = load_ohlcv(primary_csv)
    ret = df["close"].pct_change()
    vr = compute_vol_ratio(
        ret,
        config.vol_ratio_short_window,
        config.vol_ratio_long_window,
    )
    frame = pd.DataFrame({"vr": vr}).dropna()
    if frame.empty:
        return None
    if config.vr_low_threshold is not None and config.vr_high_threshold is not None:
        low_thr, high_thr = config.vr_low_threshold, config.vr_high_threshold
    else:
        split = int(len(frame) * (1 - HOLDOUT_FRAC))
        dev = frame.iloc[:max(split, 1)]
        low_thr = float(dev["vr"].quantile(1.0 / 3.0))
        high_thr = float(dev["vr"].quantile(2.0 / 3.0))
    regime = assign_vr_regime(frame["vr"], low_thr, high_thr)
    return str(regime.iloc[-1])


def read_conviction_state(primary_csv: str = "btc_1h.csv",
                          config: Config | None = None) -> dict:
    config = config or Config()
    root = Path(primary_csv).parent
    signed, slen, streak_dir = _latest_streak(primary_csv)
    multi = _latest_multi(root, primary_csv)
    vr_regime = _latest_vr_regime(primary_csv, config)

    dirs = [d for d in multi.values() if d is not None]
    agree_up = sum(1 for d in dirs if d == "UP")
    agree_dn = sum(1 for d in dirs if d == "DOWN")
    if agree_up >= agree_dn and agree_up > 0:
        multi_dir, tf_agree = "UP", agree_up
    elif agree_dn > 0:
        multi_dir, tf_agree = "DOWN", agree_dn
    else:
        multi_dir, tf_agree = "FLAT", 0

    aligned = streak_dir != "FLAT" and multi_dir != "FLAT" and streak_dir == multi_dir

    active = None
    vr_filter_reason = None
    high_vr = vr_regime == "HIGH_VR"
    for tid, label, min_s, min_k, hit, note in TIER_DEFS:
        eff_min_s = max(min_s, HIGH_VR_MIN_STREAK) if high_vr and min_s > 0 else min_s
        if min_k == 0:
            if slen < eff_min_s or streak_dir == "FLAT":
                continue
            direction = streak_dir
        elif min_s == 0:
            if tf_agree < min_k or multi_dir == "FLAT":
                continue
            direction = multi_dir
        else:
            if slen < eff_min_s or tf_agree < min_k or not aligned:
                continue
            direction = streak_dir
        active = {
            "id": tid, "label": label,
            "direction": direction,
            "calibrated_pct": round(hit * 100, 1),
            "note": note,
        }
        break

    if active is None and high_vr and slen >= 2 and streak_dir != "FLAT":
        vr_filter_reason = "high_vr_filter"

    out = {
        "streak_signed": signed,
        "streak_len": slen,
        "streak_dir": streak_dir,
        "multi": multi,
        "tf_agree": tf_agree,
        "multi_dir": multi_dir,
        "aligned": aligned,
        "active_tier": active,
        "vr_regime": vr_regime,
    }
    if vr_filter_reason:
        out["vr_filter_reason"] = vr_filter_reason
    return out
