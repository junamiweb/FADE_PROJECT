"""Historical replay — track lift over expanding time windows.

Splits history into sequential windows and runs walk-forward on each
expanding slice. Answers: does edge persist (or improve) as more data
accumulates?

Run:
    python -m fade.pipeline.replay btc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import assign_regimes
from fade.core.significant_changes import detect_significant_changes
from fade.memory import MemoryStore
from fade.pipeline.backtest import walk_forward
from fade.utils.logging import get_logger

log = get_logger("replay")


def run_replay(csv_path: str, config: Config | None = None) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    changes = detect_significant_changes(atoms, fwd, config)
    regimes = assign_regimes(changes, config.post_shock_bars)

    memory = MemoryStore(config.memory_dir, asset=asset)
    blocked = memory.blocked_events()

    n = len(atoms)
    n_windows = config.replay_windows
    min_start = int(n * config.initial_train_frac)
    if min_start >= n:
        return {"asset": asset, "windows": [], "message": "insufficient data"}

    span = (n - min_start) // n_windows
    if span <= 0:
        return {"asset": asset, "windows": [], "message": "insufficient data"}

    windows = []
    for k in range(1, n_windows + 1):
        end = min_start + k * span
        if k == n_windows:
            end = n
        slice_atoms = atoms.iloc[:end]
        slice_fwd = fwd.iloc[:end]
        slice_regimes = regimes.iloc[:end]
        if len(slice_atoms) < config.min_support * 2:
            continue

        bt = walk_forward(
            slice_atoms, slice_fwd, config,
            blocked=blocked, regimes=slice_regimes,
        )
        fm = pd.DataFrame(bt.fold_metrics) if bt.fold_metrics else pd.DataFrame()
        lift = float(fm["lift_vs_random"].mean()) if not fm.empty else float("nan")
        windows.append({
            "window": k,
            "bars": end,
            "lift_vs_random": round(lift, 4) if lift == lift else None,
            "hit_rate": round(float(fm["model_hit_rate"].mean()), 4) if not fm.empty else None,
            "n_rules": bt.n_rules_last,
        })

    lifts = [w["lift_vs_random"] for w in windows if w["lift_vs_random"] is not None]
    trend = None
    if len(lifts) >= 2:
        trend = round(float(lifts[-1] - lifts[0]), 4)

    return {"asset": asset, "windows": windows, "lift_trend": trend}


def _print_replay(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"FADE REPLAY - {result.get('asset', '?').upper()}")
    print("=" * 60)
    if not result.get("windows"):
        print(f"  {result.get('message', 'no windows')}")
        print("=" * 60 + "\n")
        return

    print(f"  {'win':>4}  {'bars':>6}  {'hit-rate':>9}  {'lift':>8}  {'rules':>6}")
    for w in result["windows"]:
        hr = f"{w['hit_rate']:.3f}" if w["hit_rate"] is not None else "  n/a"
        lift = f"{w['lift_vs_random']:+.3f}" if w["lift_vs_random"] is not None else "   n/a"
        print(f"  {w['window']:>4}  {w['bars']:>6}  {hr:>9}  {lift:>8}  {w['n_rules']:>6}")

    trend = result.get("lift_trend")
    if trend is not None:
        label = "improving" if trend > 0 else "flat/declining"
        print(f"\n  Lift trend (last - first): {trend:+.4f} ({label})")
    print("=" * 60 + "\n")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv or not Path(csv).exists():
        log.error("Usage: python -m fade.pipeline.replay path/to/ohlcv.csv")
        sys.exit(1)
    result = run_replay(csv)
    _print_replay(result)


if __name__ == "__main__":
    main()
