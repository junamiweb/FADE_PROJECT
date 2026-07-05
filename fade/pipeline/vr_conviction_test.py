"""VR x conviction tier holdout test — hit rates by VR bucket.

Measures streak>=2, streak>=3, and multi-TF (>=3 agree) contrarian reversal
hit rates split by LOW_VR / NORMAL / HIGH_VR. Also reports coverage vs hit-rate
tradeoff before/after suppressing low-streak tiers in HIGH_VR (same rule as
``read_conviction_state``).

Honest OOS: VR tertile thresholds from dev (first 70%); metrics on holdout only.

Run:
    python -m fade.pipeline.vr_conviction_test
    python -m fade.pipeline.vr_conviction_test btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import VR_REGIMES, assign_vr_regime, compute_vol_ratio
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("vr_conviction_test")

HOLDOUT_FRAC = 0.30
MIN_SUPPORT = 30


def _multi_files_for(primary_csv: str) -> list[str]:
    stem = Path(primary_csv).stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return [f"{prefix}_{iv}.csv" for iv in ("5m", "15m", "30m", "1h")
            if Path(f"{prefix}_{iv}.csv").exists()]


def _streak_hits(streak: np.ndarray, up: np.ndarray, mask: np.ndarray,
                 min_len: int) -> tuple[int, int]:
    sel = mask & (np.abs(streak) >= min_len)
    k = int(sel.sum())
    if k == 0:
        return 0, 0
    hits = 0
    for i in np.flatnonzero(sel):
        s = streak[i]
        hits += int(up[i] == 0) if s > 0 else int(up[i] == 1)
    return hits, k


def _multi_hits(frame: pd.DataFrame, cols: list[str], mask: np.ndarray,
                min_agree: int) -> tuple[int, int]:
    if not cols:
        return 0, 0
    sig = frame[cols].to_numpy()
    pos = (sig > 0).sum(axis=1)
    neg = (sig < 0).sum(axis=1)
    multi_dir = np.where(pos >= min_agree, 1, np.where(neg >= min_agree, 0, -1))
    sel = mask & (multi_dir >= 0)
    k = int(sel.sum())
    if k == 0:
        return 0, 0
    pred = multi_dir[sel]
    act = (frame["up_fwd"].to_numpy()[sel]).astype(int)
    hits = int((pred == act).sum())
    return hits, k


def _vr_thresholds(dev_vr: pd.Series, config: Config) -> tuple[float, float, str]:
    if config.vr_low_threshold is not None and config.vr_high_threshold is not None:
        return config.vr_low_threshold, config.vr_high_threshold, "config"
    return (
        float(dev_vr.quantile(1.0 / 3.0)),
        float(dev_vr.quantile(2.0 / 3.0)),
        "dev_tertiles",
    )


def vr_conviction_test(
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
    up_fwd = (ret.shift(-1) > 0).astype(int)

    multi_cols: dict[str, pd.Series] = {}
    for c in _multi_files_for(csv_path):
        multi_cols[Path(c).stem] = _contrarian_grid(c)

    frame = pd.DataFrame({
        "ret": ret, "streak": streak, "vr": vr, "up": up, "up_fwd": up_fwd,
    }, index=df.index)
    for name, s in multi_cols.items():
        frame[name] = s.reindex(frame.index)
    frame = frame.dropna(subset=["vr", "streak", "up"])
    cols = list(multi_cols.keys())

    n = len(frame)
    split = int(n * (1 - holdout_frac))
    dev, hold = frame.iloc[:split], frame.iloc[split:]
    low_thr, high_thr, threshold_source = _vr_thresholds(dev["vr"], config)
    hold_regime = assign_vr_regime(hold["vr"], low_thr, high_thr)

    streak_arr = hold["streak"].to_numpy()
    up_arr = hold["up"].to_numpy()
    regime_arr = hold_regime.to_numpy()

    tier_defs = [
        ("streak2", "streak", 2),
        ("streak3", "streak", 3),
        ("multi3", "multi", 3),
    ]

    by_bucket = []
    for regime in VR_REGIMES:
        mask = regime_arr == regime
        row: dict = {"regime": regime, "tiers": {}}
        for tid, kind, thr in tier_defs:
            if kind == "streak":
                hits, k = _streak_hits(streak_arr, up_arr, mask, thr)
            else:
                hits, k = _multi_hits(hold, cols, mask, thr)
            if k < MIN_SUPPORT:
                row["tiers"][tid] = {"n": k, "status": "low_support"}
            else:
                hr = hits / k
                row["tiers"][tid] = {
                    "n": k, "status": "ok",
                    "hit_rate": round(hr, 4),
                    "lift_vs_random": round(hr - 0.5, 4),
                }
        by_bucket.append(row)

    # Before/after HIGH_VR filter on streak>=2 (main conviction coverage tier).
    all_mask = np.ones(len(hold), dtype=bool)
    high_mask = regime_arr == "HIGH_VR"
    before_hits, before_n = _streak_hits(streak_arr, up_arr, all_mask, 2)
    after_mask = all_mask & ~high_mask
    after_hits, after_n = _streak_hits(streak_arr, up_arr, after_mask, 2)
    downgraded_hits, downgraded_n = _streak_hits(streak_arr, up_arr, high_mask, 3)

    def _rate(h: int, k: int) -> float | None:
        return round(h / k, 4) if k else None

    filter_summary = {
        "before": {
            "tier": "streak>=2",
            "n": before_n,
            "hit_rate": _rate(before_hits, before_n),
            "coverage_frac": round(before_n / len(hold), 4) if len(hold) else 0,
        },
        "after_high_vr_suppress": {
            "tier": "streak>=2 (exclude HIGH_VR)",
            "n": after_n,
            "hit_rate": _rate(after_hits, after_n),
            "coverage_frac": round(after_n / len(hold), 4) if len(hold) else 0,
        },
        "high_vr_downgrade": {
            "tier": "streak>=3 in HIGH_VR only",
            "n": downgraded_n,
            "hit_rate": _rate(downgraded_hits, downgraded_n),
            "coverage_frac": round(downgraded_n / len(hold), 4) if len(hold) else 0,
        },
    }

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "holdout_frac": holdout_frac,
        "n_dev": split,
        "n_holdout": n - split,
        "threshold_source": threshold_source,
        "vr_low_threshold": round(low_thr, 4),
        "vr_high_threshold": round(high_thr, 4),
        "multi_tf_files": cols,
        "buckets": by_bucket,
        "filter": filter_summary,
    }


def _print_result(r: dict) -> None:
    print("\n" + "=" * 68)
    print(f"VR x CONVICTION — {r.get('asset', '?').upper()}")
    print("=" * 68)
    print(f"  Thresholds : LOW<={r['vr_low_threshold']}  HIGH>={r['vr_high_threshold']}"
          f"  ({r['threshold_source']})")
    print(f"  Holdout    : n={r['n_holdout']}  multi-TF={r.get('multi_tf_files', [])}")
    print()
    print(f"  {'VR':<8} {'tier':<10} {'n':>6} {'hit':>8} {'lift':>8}")
    for bucket in r["buckets"]:
        reg = bucket["regime"]
        for tid, info in bucket["tiers"].items():
            if info.get("status") != "ok":
                print(f"  {reg:<8} {tid:<10} {info.get('n', 0):>6}  (low support)")
                continue
            print(f"  {reg:<8} {tid:<10} {info['n']:>6} {info['hit_rate']:>8.4f}"
                  f" {info['lift_vs_random']:>+8.4f}")
    print()
    print("  HIGH_VR filter tradeoff (streak>=2 baseline):")
    for key, row in r["filter"].items():
        hr = row.get("hit_rate")
        hr_s = f"{hr:.4f}" if hr is not None else "n/a"
        print(f"    {key:<28} n={row['n']:>5}  hit={hr_s}  cover={row['coverage_frac']:.4f}")
    print("=" * 68 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VR x conviction holdout test")
    parser.add_argument("csvs", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    args = parser.parse_args()

    for csv in args.csvs:
        if not Path(csv).exists():
            log.error("File not found: %s", csv)
            continue
        _print_result(vr_conviction_test(csv, holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
