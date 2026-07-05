"""Multi-resolution ensemble — does cross-timeframe agreement improve accuracy?

Three resolutions each independently PASS the strict holdout (15m, 30m, 1h).
The vision (chatGPT.txt, "cross-regime validation") suggests combining them:
when several timeframes agree on direction, confidence should be higher.

To make the timeframes comparable, every resolution predicts the SAME target —
the direction of price a fixed wall-clock horizon ahead (default 60 minutes) —
using its own atom granularity:
    1h  -> 1 bar ahead
    30m -> 2 bars ahead
    15m -> 4 bars ahead

Method (strict, no look-ahead):
  1. Per resolution: chronological 70/30 split, mine + freeze stable rules on the
     development slice only, apply to the quarantined holdout.
  2. Each prediction is stamped with its DECISION TIME = bar_open + one interval
     (the moment the bar closes and the signal becomes actionable). All three
     resolutions then predict the identical forward window, so they are aligned.
  3. Build the ensemble on the 1h decision grid: for each 1h decision time, take
     the most recent available prediction from each resolution (merge_asof
     backward, tolerance = one target horizon).
  4. Compare, on the SAME rows and outcome, single-resolution vs:
       * unanimous agreement (all three same direction)
       * majority vote
     with a permutation test on each.

If agreement lifts hit-rate above the best single resolution, multi-timeframe
confirmation adds real value.

Run:
    python -m fade.pipeline.multi_res
    python -m fade.pipeline.multi_res --target-min 60
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("multi_res")

INTERVAL_MIN = {"15m": 15, "30m": 30, "1h": 60}
FILES = {"15m": "btc_15m.csv", "30m": "btc_30m.csv", "1h": "btc_1h.csv"}
BASE_RES = "1h"


def _resolution_preds(
    res: str,
    target_min: int,
    holdout_frac: float,
    config: Config,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Frozen holdout predictions for one resolution, stamped by decision time.

    Returns (preds_df[decision_time, pred], outcome_series indexed by decision
    time) or None if no stable rules survived development.
    """
    interval_min = INTERVAL_MIN[res]
    horizon = max(1, round(target_min / interval_min))
    cfg = dataclasses.replace(config, forward_horizon=horizon)

    df = load_ohlcv(FILES[res])
    atoms = atoms_mod.compute_atoms(df, cfg)
    fwd = atoms_mod.forward_return(df, horizon).reindex(atoms.index)

    n = len(atoms)
    split = int(n * (1.0 - holdout_frac))
    dev_atoms, dev_fwd = atoms.iloc[:split], fwd.iloc[:split]
    hold_atoms = atoms.iloc[split:]
    hold_fwd = fwd.iloc[split:]

    dev_bt = walk_forward(dev_atoms, dev_fwd, cfg)
    frozen = _select_stable_rules(dev_bt.stability, cfg)
    if frozen.empty:
        return None

    thresholds = ev.compute_thresholds(dev_atoms, cfg)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, cfg, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)
    if preds.empty:
        return None

    decision = preds.index + pd.Timedelta(minutes=interval_min)
    out = pd.DataFrame({"decision_time": decision, "pred": preds["pred"].to_numpy()})
    out = out.sort_values("decision_time").reset_index(drop=True)

    actual_up = (hold_fwd > 0).astype(int)
    actual_up.index = actual_up.index + pd.Timedelta(minutes=interval_min)
    actual_up = actual_up[hold_fwd.reindex(hold_fwd.index).notna().to_numpy()]
    return out, actual_up


def _score(pred: np.ndarray, actual: np.ndarray, n_shuffles: int, seed: int) -> dict:
    if pred.size == 0:
        return {"coverage": 0, "hit_rate": float("nan"),
                "lift": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    hit = float(np.mean(pred == actual))
    null = np.array([np.mean(pred == rng.permutation(actual)) for _ in range(n_shuffles)])
    p = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)
    return {
        "coverage": int(pred.size),
        "hit_rate": round(hit, 4),
        "lift": round(hit - 0.5, 4),
        "null_mean": round(float(np.mean(null)), 4),
        "p_value": round(p, 4),
    }


def run_ensemble(
    target_min: int = 60,
    holdout_frac: float = 0.30,
    n_shuffles: int = 300,
    seed: int = 0,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    per_res: dict[str, pd.DataFrame] = {}
    base_outcome: pd.Series | None = None

    for res in ("15m", "30m", "1h"):
        if not Path(FILES[res]).exists():
            log.warning("missing %s, skipping", FILES[res])
            continue
        got = _resolution_preds(res, target_min, holdout_frac, config)
        if got is None:
            log.warning("no stable rules for %s", res)
            continue
        preds, outcome = got
        per_res[res] = preds
        if res == BASE_RES:
            base_outcome = outcome

    if BASE_RES not in per_res or base_outcome is None:
        return {"status": "no_base", "target_min": target_min}

    # Common decision grid = base (1h) decision times.
    grid = per_res[BASE_RES].rename(columns={"pred": "pred_1h"}).copy()
    tol = pd.Timedelta(minutes=target_min)
    for res in ("15m", "30m"):
        if res not in per_res:
            continue
        p = per_res[res].rename(columns={"pred": f"pred_{res}"})
        grid = pd.merge_asof(grid, p, on="decision_time",
                             direction="backward", tolerance=tol)

    out_df = base_outcome.rename("actual").reset_index()
    out_df.columns = ["decision_time", "actual"]
    grid = grid.merge(out_df, on="decision_time", how="inner").dropna(subset=["actual"])
    grid["actual"] = grid["actual"].astype(int)

    pred_cols = [c for c in ("pred_15m", "pred_30m", "pred_1h") if c in grid.columns]

    results: dict = {"status": "ok", "target_min": target_min,
                     "grid_rows": int(len(grid)), "resolutions": {}}

    # Single-resolution baselines on the shared grid.
    for c in pred_cols:
        sub = grid.dropna(subset=[c])
        results["resolutions"][c.replace("pred_", "")] = _score(
            sub[c].to_numpy().astype(int), sub["actual"].to_numpy(), n_shuffles, seed)

    # Ensemble requires all three present.
    full = grid.dropna(subset=pred_cols)
    if len(pred_cols) >= 2 and not full.empty:
        votes = full[pred_cols].astype(int)
        agree_mask = votes.nunique(axis=1) == 1
        unan = full[agree_mask]
        results["unanimous"] = _score(
            unan[pred_cols[0]].to_numpy().astype(int),
            unan["actual"].to_numpy(), n_shuffles, seed)
        results["unanimous"]["agreement_rate"] = round(len(unan) / len(full), 4)

        maj_pred = (votes.sum(axis=1) > (len(pred_cols) / 2)).astype(int)
        results["majority"] = _score(
            maj_pred.to_numpy(), full["actual"].to_numpy(), n_shuffles, seed)

    return results


def _print(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print(f"FADE MULTI-RESOLUTION ENSEMBLE  (target = {r.get('target_min')} min ahead)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  Shared 1h decision grid rows: {r['grid_rows']:,}")
    print()
    print(f"  {'model':<16}{'cover':>9}{'hit':>9}{'lift':>9}{'null':>9}{'p':>9}")
    for name, m in r["resolutions"].items():
        print(f"  {'single '+name:<16}{m['coverage']:>9}{m['hit_rate']:>9}"
              f"{m['lift']:>+9}{m['null_mean']:>9}{m['p_value']:>9}")
    if "unanimous" in r:
        m = r["unanimous"]
        print(f"  {'ALL agree':<16}{m['coverage']:>9}{m['hit_rate']:>9}"
              f"{m['lift']:>+9}{m['null_mean']:>9}{m['p_value']:>9}"
              f"   (agree {m['agreement_rate']*100:.0f}%)")
        mm = r["majority"]
        print(f"  {'majority':<16}{mm['coverage']:>9}{mm['hit_rate']:>9}"
              f"{mm['lift']:>+9}{mm['null_mean']:>9}{mm['p_value']:>9}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE multi-resolution ensemble")
    parser.add_argument("--target-min", type=int, default=60)
    args = parser.parse_args()
    _print(run_ensemble(target_min=args.target_min))


if __name__ == "__main__":
    main()
