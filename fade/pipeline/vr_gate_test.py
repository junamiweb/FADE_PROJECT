"""Volatility Ratio (VR) regime gate — reversal hit-rate by VR bucket on holdout.

VR = short_vol / long_vol (default 24h / 168h rolling std of 1h returns).
Literature: LOW_VR (vol compressing) favours mean-reversion; HIGH_VR (vol
expanding) favours momentum/trend. Tests whether streak>=3 contrarian reversal
is stronger in LOW_VR vs HIGH_VR.

Honest OOS: VR tertile thresholds fitted on dev (first 70%) only; hit rates on
holdout (last 30%). vol_ratio shifted by 1 bar — no look-ahead.

Run:
    python -m fade.pipeline.vr_gate_test
    python -m fade.pipeline.vr_gate_test btc_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import VR_REGIMES, assign_vr_regime, compute_vol_ratio
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("vr_gate_test")

MIN_SUPPORT = 30
HOLDOUT_FRAC = 0.30
STREAK_MIN = 3


def _contrarian_hits(streak: np.ndarray, up: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Reversal hit-rate: streak>=STREAK_MIN contrarian vs next bar direction."""
    sel = mask & (np.abs(streak) >= STREAK_MIN)
    k = int(sel.sum())
    if k == 0:
        return 0, 0
    hits = 0
    for i in np.flatnonzero(sel):
        s = streak[i]
        if s > 0:
            hits += int(up[i] == 0)
        else:
            hits += int(up[i] == 1)
    return hits, k


def vr_gate_test(
    csv_path: str = "btc_1h.csv",
    holdout_frac: float = HOLDOUT_FRAC,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    vr = compute_vol_ratio(
        ret,
        config.vol_ratio_short_window,
        config.vol_ratio_long_window,
    )
    up = (ret > 0).astype(int)

    frame = pd.DataFrame({
        "ret": ret, "streak": streak, "vr": vr, "up": up,
    }).dropna()
    n = len(frame)
    split = int(n * (1 - holdout_frac))
    dev, hold = frame.iloc[:split], frame.iloc[split:]

    if config.vr_low_threshold is not None and config.vr_high_threshold is not None:
        low_thr = config.vr_low_threshold
        high_thr = config.vr_high_threshold
        threshold_source = "config"
    else:
        low_thr = float(dev["vr"].quantile(1.0 / 3.0))
        high_thr = float(dev["vr"].quantile(2.0 / 3.0))
        threshold_source = "dev_tertiles"

    hold_regime = assign_vr_regime(hold["vr"], low_thr, high_thr)
    streak_arr = hold["streak"].to_numpy()
    up_arr = hold["up"].to_numpy()

    rows = []
    for regime in VR_REGIMES:
        mask = (hold_regime == regime).to_numpy()
        hits, k = _contrarian_hits(streak_arr, up_arr, mask)
        if k < MIN_SUPPORT:
            rows.append({"regime": regime, "n": k, "status": "low_support"})
            continue
        hr = hits / k
        rows.append({
            "regime": regime,
            "n": k,
            "status": "ok",
            "reversal_hit_rate": round(hr, 4),
            "lift_vs_random": round(hr - 0.5, 4),
        })

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "holdout_frac": holdout_frac,
        "streak_min": STREAK_MIN,
        "vr_windows": (config.vol_ratio_short_window, config.vol_ratio_long_window),
        "threshold_source": threshold_source,
        "vr_low_threshold": round(low_thr, 4),
        "vr_high_threshold": round(high_thr, 4),
        "n_dev": int(split),
        "n_holdout": int(n - split),
        "buckets": rows,
    }


def _print_result(r: dict) -> None:
    print("\n" + "=" * 60)
    print(f"VR GATE TEST — {r.get('asset', '?').upper()}")
    print("=" * 60)
    sw, lw = r["vr_windows"]
    print(f"  VR windows     : {sw}h / {lw}h")
    print(f"  Thresholds     : LOW<={r['vr_low_threshold']}  HIGH>={r['vr_high_threshold']}"
          f"  ({r['threshold_source']})")
    print(f"  Streak filter  : |streak|>={r['streak_min']} contrarian")
    print(f"  Holdout        : last {r['holdout_frac']*100:.0f}%  (n={r['n_holdout']})")
    print()
    for row in r["buckets"]:
        reg = row["regime"]
        if row["status"] != "ok":
            print(f"  {reg:<8} n={row['n']}  (low support)")
            continue
        print(f"  {reg:<8} n={row['n']:>5}  reversal_hit={row['reversal_hit_rate']:.4f}"
              f"  lift={row['lift_vs_random']:+.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VR regime gate holdout test")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv", help="OHLCV CSV")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        raise SystemExit(1)

    result = vr_gate_test(args.csv, holdout_frac=args.holdout_frac)
    _print_result(result)


if __name__ == "__main__":
    main()
