"""Minute-resolution holdout test on high-volatility windows.

Motivation: the daily-vs-hourly holdout showed the atomic edge lives at FINE
timescales (hourly PASS, daily FAIL). This zooms further in — to 1-minute bars
sampled around the most volatile moments — to ask whether structure is even
stronger there.

The data (``btc_1m_vol.csv``) is a set of disjoint 1-minute windows tagged with
a ``segment`` id. Returns are computed WITHIN each segment only, so no atom or
forward-return ever crosses the time gaps between windows.

Split is by segment (chronological): earliest 70% of windows = development,
latest 30% = quarantined holdout. Rules are mined/selected on development,
frozen, and applied to the holdout with a permutation test — identical logic to
the standard strict holdout.

Run:
    python -m fade.pipeline.minute_vol            # uses btc_1m_vol.csv
    python -m fade.pipeline.minute_vol btc_1m_vol.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.evaluator import predict
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import P_VALUE_MAX, _select_stable_rules, _verdict
from fade.utils.logging import get_logger

log = get_logger("minute_vol")


def _segment_atoms(df_seg: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Atoms + forward return computed strictly inside one contiguous segment."""
    a = atoms_mod.compute_atoms(df_seg, config)
    f = atoms_mod.forward_return(df_seg, config.forward_horizon).reindex(a.index)
    valid = f.notna()
    return a[valid], f[valid]


def minute_vol_test(
    csv_path: str = "btc_1m_vol.csv",
    dev_frac: float = 0.70,
    n_shuffles: int = 300,
    seed: int = 0,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"])

    seg_ids = sorted(raw["segment"].unique())
    n_dev = max(1, int(len(seg_ids) * dev_frac))
    dev_ids = set(seg_ids[:n_dev])

    dev_a, dev_f, hold_a, hold_f = [], [], [], []
    for sid, g in raw.groupby("segment"):
        g = g.set_index("timestamp").sort_index()
        a, f = _segment_atoms(g, config)
        if a.empty:
            continue
        if sid in dev_ids:
            dev_a.append(a); dev_f.append(f)
        else:
            hold_a.append(a); hold_f.append(f)

    result: dict = {
        "asset": "btc_1m_vol",
        "n_segments": len(seg_ids),
        "n_dev_segments": len(dev_ids),
        "n_bars_total": int(len(raw)),
    }
    if not dev_a or not hold_a:
        result["status"] = "insufficient_segments"
        result["verdict"] = "INCONCLUSIVE - not enough segments to split."
        return result

    dev_atoms = pd.concat(dev_a).sort_index()
    dev_fwd = pd.concat(dev_f).sort_index()
    hold_atoms = pd.concat(hold_a).sort_index()
    hold_fwd = pd.concat(hold_f).sort_index()
    result["n_dev_bars"] = int(len(dev_atoms))
    result["n_holdout_bars"] = int(len(hold_atoms))

    # --- Development: mine + select stable rules -----------------------
    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    result["n_stable_rules"] = int(len(frozen))
    if frozen.empty:
        result["status"] = "no_rules"
        result["verdict"] = "INCONCLUSIVE - no stable rules survived development."
        return result

    # --- Freeze thresholds on dev, apply to holdout --------------------
    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)
    if preds.empty:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - frozen rules never fired on holdout."
        return result

    actual_up = (hold_fwd > 0).astype(int).reindex(preds.index)
    valid = actual_up.notna()
    pred_v = preds["pred"][valid].to_numpy()
    act_v = actual_up[valid].to_numpy()
    coverage = int(valid.sum())
    if coverage == 0:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - no scorable holdout predictions."
        return result

    real_hit = float(np.mean(pred_v == act_v))
    null_hits = np.array([np.mean(pred_v == rng.permutation(act_v))
                          for _ in range(n_shuffles)])
    p_value = (1 + int(np.sum(null_hits >= real_hit))) / (1 + n_shuffles)

    result.update({
        "status": "ok",
        "coverage": coverage,
        "holdout_hit_rate": round(real_hit, 4),
        "holdout_lift_vs_random": round(real_hit - 0.5, 4),
        "null_mean": round(float(np.mean(null_hits)), 4),
        "null_std": round(float(np.std(null_hits)), 4),
        "p_value": round(p_value, 4),
    })
    result["verdict"] = _verdict(result)
    return result


def _print(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print("FADE MINUTE-VOLATILITY HOLDOUT TEST - BTC (1m around shocks)")
    print(line)
    print(f"  Segments: {r.get('n_segments')}  "
          f"(dev {r.get('n_dev_segments')} / holdout {r.get('n_segments', 0) - r.get('n_dev_segments', 0)})")
    print(f"  Bars total: {r.get('n_bars_total'):,}")
    if r.get("status") != "ok":
        print(f"  dev bars={r.get('n_dev_bars', 0):,}  holdout bars={r.get('n_holdout_bars', 0):,}"
              f"  stable rules={r.get('n_stable_rules', 0)}")
        print(f"\n  {r['verdict']}")
        print(line + "\n")
        return
    print(f"  dev bars={r['n_dev_bars']:,}  holdout bars={r['n_holdout_bars']:,}")
    print(f"  Stable rules frozen : {r['n_stable_rules']}")
    print(f"  Holdout coverage    : {r['coverage']:,}")
    print()
    print(f"  Holdout hit-rate    : {r['holdout_hit_rate']}")
    print(f"  Lift vs random      : {r['holdout_lift_vs_random']:+.4f}")
    print(f"  Shuffle null        : {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
    print(f"  Permutation p-value : {r['p_value']:.4f}")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else "btc_1m_vol.csv"
    if not Path(csv).exists():
        log.error("File not found: %s  (run: python download_history.py minutevol)", csv)
        sys.exit(1)
    _print(minute_vol_test(csv))


if __name__ == "__main__":
    main()
