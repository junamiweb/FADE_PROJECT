"""Cross-asset analysis — how BTC and ETH move together.

Answers three practical questions, all descriptive and honest:
  1. Contemporaneous correlation of hourly returns (are they redundant?).
  2. Rolling correlation over time (is the relationship stable?).
  3. Lead-lag: does one asset's move at t carry information about the other at
     t+1 (a cross-asset atom worth mining), or is co-movement purely same-bar?

Same-bar co-movement is not tradeable (you can't act on it after the fact); a
genuine lead-lag edge would be. We report both so the distinction stays honest.

Run:
    python -m fade.pipeline.correlation btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.utils.logging import get_logger

log = get_logger("correlation")


def run_correlation(csv_a: str, csv_b: str, roll_window: int = 720) -> dict:
    name_a, name_b = Path(csv_a).stem, Path(csv_b).stem
    a = load_ohlcv(csv_a)["close"].pct_change()
    b = load_ohlcv(csv_b)["close"].pct_change()

    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    n = len(joined)
    if n < roll_window * 2:
        roll_window = max(48, n // 4)

    contemp = float(joined["a"].corr(joined["b"]))

    roll = joined["a"].rolling(roll_window).corr(joined["b"]).dropna()
    roll_stats = {
        "window_bars": roll_window,
        "mean": round(float(roll.mean()), 4),
        "min": round(float(roll.min()), 4),
        "max": round(float(roll.max()), 4),
        "std": round(float(roll.std()), 4),
    }

    # Lead-lag: corr(a[t], b[t+lag]). lag>0 => a leads b.
    lags = range(-3, 4)
    lead_lag = {}
    for lag in lags:
        shifted = joined["b"].shift(-lag)
        c = joined["a"].corr(shifted)
        lead_lag[lag] = round(float(c), 4)

    # Same-direction frequency (how often both close the bar the same way).
    same_dir = float(np.mean(np.sign(joined["a"]) == np.sign(joined["b"])))

    # Directional lead-lag skill: sign(a[t]) predicting sign(b[t+1]).
    a_sign = np.sign(joined["a"].to_numpy())
    b_next = np.sign(joined["b"].shift(-1).to_numpy())
    mask = np.isfinite(b_next) & (a_sign != 0) & (b_next != 0)
    lead_hit = float(np.mean(a_sign[mask] == b_next[mask])) if mask.any() else float("nan")

    return {
        "asset_a": name_a,
        "asset_b": name_b,
        "n_common_bars": n,
        "contemporaneous_corr": round(contemp, 4),
        "rolling_corr": roll_stats,
        "lead_lag_corr": lead_lag,
        "same_direction_freq": round(same_dir, 4),
        "a_leads_b_next_bar_hit": round(lead_hit, 4) if lead_hit == lead_hit else None,
    }


def _print(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print(f"FADE CROSS-ASSET — {r['asset_a'].upper()} vs {r['asset_b'].upper()}")
    print(line)
    print(f"  common bars               : {r['n_common_bars']:,}")
    print(f"  contemporaneous return corr: {r['contemporaneous_corr']:+.4f}")
    rs = r["rolling_corr"]
    print(f"  rolling corr ({rs['window_bars']}b)     : "
          f"mean {rs['mean']:+.3f}  range [{rs['min']:+.3f}, {rs['max']:+.3f}]  "
          f"std {rs['std']:.3f}")
    print(f"  same-direction frequency   : {r['same_direction_freq']:.4f}")
    print()
    print("  lead-lag corr  (lag>0 => A leads B):")
    for lag, c in r["lead_lag_corr"].items():
        marker = "  <- same bar" if lag == 0 else ""
        print(f"      lag {lag:+d} : {c:+.4f}{marker}")
    ll = r["a_leads_b_next_bar_hit"]
    print()
    print(f"  sign(A_t) predicts sign(B_t+1): "
          f"{ll if ll is None else f'{ll:.4f}'} "
          f"(0.50 = no lead-lag edge)")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE cross-asset correlation")
    parser.add_argument("csv_a", nargs="?", default="btc_1h.csv")
    parser.add_argument("csv_b", nargs="?", default="eth_1h.csv")
    args = parser.parse_args()
    for c in (args.csv_a, args.csv_b):
        if not Path(c).exists():
            log.error("File not found: %s", c)
            return
    _print(run_correlation(args.csv_a, args.csv_b))


if __name__ == "__main__":
    main()
