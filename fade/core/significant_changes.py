"""Significant change detection — from OHLCV only, no APIs, no ML.

Flags abnormal price moves, volume spikes, and combined shocks using the
existing atom features. This is the first step toward learning *when*
something meaningful changed in the market, not just predicting direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fade.config import Config

CHANGE_TYPES = ("PRICE_UP", "PRICE_DOWN", "VOLUME_SPIKE", "COMBINED_UP", "COMBINED_DOWN")


def detect_significant_changes(
    atoms: pd.DataFrame,
    fwd_return: pd.Series | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Label bars with significant price/volume shocks.

    Returns a DataFrame indexed like ``atoms`` with boolean flags and a
    ``change_type`` column (empty string when nothing fired).
    """
    config = config or Config()
    r1 = atoms["return_1h"]
    vol = atoms["volatility"]
    vz = atoms["volume_zscore"]

    price_up = r1 > config.price_shock_sigma * vol
    price_dn = r1 < -config.price_shock_sigma * vol
    price_shock = price_up | price_dn
    volume_shock = vz.abs() > config.volume_shock_z
    combined_up = price_up & volume_shock
    combined_dn = price_dn & volume_shock

    change_type = np.where(combined_up, "COMBINED_UP",
                  np.where(combined_dn, "COMBINED_DOWN",
                  np.where(price_up, "PRICE_UP",
                  np.where(price_dn, "PRICE_DOWN",
                  np.where(volume_shock, "VOLUME_SPIKE", "")))))

    out = pd.DataFrame({
        "price_shock": price_shock,
        "volume_shock": volume_shock,
        "combined": combined_up | combined_dn,
        "change_type": change_type,
        "return_1h": r1,
        "volume_zscore": vz,
    }, index=atoms.index)

    if fwd_return is not None:
        out["fwd_return"] = fwd_return.reindex(atoms.index)
    return out


def summarize_changes(changes: pd.DataFrame, recent_n: int = 5) -> dict:
    """Aggregate statistics for the run report."""
    flagged = changes[changes["change_type"] != ""]
    counts = flagged["change_type"].value_counts().to_dict() if not flagged.empty else {}

    # Post-shock behaviour: mean 4h forward return by shock type.
    post_shock: dict[str, float] = {}
    if "fwd_return" in changes.columns:
        for ctype in CHANGE_TYPES:
            mask = changes["change_type"] == ctype
            if mask.sum() > 0:
                post_shock[ctype] = round(float(changes.loc[mask, "fwd_return"].mean()), 5)

    recent = []
    if not flagged.empty:
        tail = flagged.tail(recent_n)
        for ts, row in tail.iterrows():
            recent.append({
                "timestamp": str(ts),
                "type": row["change_type"],
                "return_1h_pct": round(float(row["return_1h"]) * 100, 2),
                "volume_z": round(float(row["volume_zscore"]), 2),
            })

    return {
        "total_flagged": int(len(flagged)),
        "counts": counts,
        "post_shock_fwd": post_shock,
        "recent": recent,
    }
