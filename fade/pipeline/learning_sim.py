"""Simulated live learning — hide the future, reveal it slowly, measure improvement.

This is the honest version of a "self-improving loop": instead of waiting for
real time, we take full history, hide the tail from the tool, and feed it back
one chunk at a time. At each checkpoint the tool may mine rules ONLY from data
revealed so far, then predicts the next (still-hidden) chunk. We know the real
outcome — the tool does not — so we score every prediction immediately.

Honesty guarantee (no look-ahead):
    At checkpoint i the tool sees bars [0 .. cut_i]. It mines + selects stable
    rules and fits discretisation thresholds on that revealed slice only, then
    predicts bars (cut_i .. cut_i + step]. Those outcomes are the "future" it
    never trained on. Calibration updates only AFTER each chunk is scored.

What it answers (the vision's "improves with more attempts"):
    Does accuracy in later chunks exceed earlier ones as revealed history grows?
    A rising learning curve = genuine self-improvement, not overfitting.

Run:
    python -m fade.pipeline.learning_sim btc_1h.csv
    python -m fade.pipeline.learning_sim btc_1h.csv --checkpoints 20
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.core.predictor import collect_calibration_samples, predict_calibrated
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("learning_sim")


def run_learning_sim(
    csv_path: str,
    seed_frac: float = 0.40,
    n_checkpoints: int = 15,
    range_start: float = 0.0,
    range_end: float = 1.0,
    train_mode: str = "expanding",
    train_window: int | None = None,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    # Restrict to a sub-window of history so different jobs cover different eras.
    full = len(atoms)
    lo = int(full * max(0.0, range_start))
    hi = int(full * min(1.0, range_end))
    atoms = atoms.iloc[lo:hi]
    fwd = fwd.iloc[lo:hi]

    n = len(atoms)
    seed_end = int(n * seed_frac)
    if seed_end >= n - 10:
        return {"asset": asset, "status": "insufficient_data"}

    step = (n - seed_end) // n_checkpoints
    if step <= 0:
        return {"asset": asset, "status": "insufficient_data"}

    window = train_window or seed_end

    # Fresh scratch calibration that learns incrementally as chunks are scored.
    cal = CalibrationStore(config.cache_dir / f"_learnsim_{asset}.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}

    checkpoints = []
    cum_correct = 0
    cum_total = 0

    for i in range(n_checkpoints):
        cut = seed_end + i * step
        pred_end = n if i == n_checkpoints - 1 else cut + step

        if train_mode == "fixed":
            train_start = max(0, cut - window)
            revealed_atoms = atoms.iloc[train_start:cut]
            revealed_fwd = fwd.iloc[train_start:cut]
        else:
            revealed_atoms = atoms.iloc[:cut]
            revealed_fwd = fwd.iloc[:cut]
        if len(revealed_atoms) < config.min_support * 2:
            continue

        # --- Mine + select stable rules on REVEALED data only ---
        bt = walk_forward(revealed_atoms, revealed_fwd, config)
        frozen = _select_stable_rules(bt.stability, config)
        if frozen.empty:
            checkpoints.append({"checkpoint": i + 1, "revealed_bars": cut,
                                "status": "no_rules"})
            continue

        # Freeze thresholds on revealed; apply to the hidden next chunk.
        thresholds = ev.compute_thresholds(revealed_atoms, config)
        future_atoms = atoms.iloc[cut:pred_end]
        future_fwd = fwd.iloc[cut:pred_end]
        disc = ev.discretize(future_atoms, thresholds)
        events = ev.build_events(disc, config, allowed=set(frozen.index))

        # Calibrated prediction using calibration learned from PAST chunks only.
        preds = predict_calibrated(events, frozen, cal, positive={})
        if preds.empty:
            checkpoints.append({"checkpoint": i + 1, "revealed_bars": cut,
                                "status": "no_coverage", "n_rules": int(len(frozen))})
            continue

        fwd_v = future_fwd.reindex(preds.index).to_numpy()
        pred_sc, act_sc = score_predictions(
            preds["pred"].to_numpy(), fwd_v, config.move_threshold)
        if len(pred_sc) == 0:
            checkpoints.append({"checkpoint": i + 1, "revealed_bars": cut,
                                "status": "no_scorable"})
            continue

        chunk_hit = float(np.mean(pred_sc == act_sc))
        cum_correct += int(np.sum(pred_sc == act_sc))
        cum_total += len(pred_sc)

        # Learn: update calibration with this chunk's realised outcomes.
        samples = collect_calibration_samples(preds, future_fwd)
        if samples:
            cal.update(samples)

        checkpoints.append({
            "checkpoint": i + 1,
            "revealed_bars": cut,
            "n_rules": int(len(frozen)),
            "coverage": len(pred_sc),
            "chunk_hit_rate": round(chunk_hit, 4),
            "cum_hit_rate": round(cum_correct / cum_total, 4),
        })
        log.info("cp %d/%d revealed=%d rules=%d chunk_hit=%.4f cum=%.4f",
                 i + 1, n_checkpoints, cut, len(frozen), chunk_hit,
                 cum_correct / cum_total)

    scored = [c for c in checkpoints if "chunk_hit_rate" in c]
    result = {"asset": asset, "status": "ok" if scored else "no_scored",
              "train_mode": train_mode,
              "train_window": int(window) if train_mode == "fixed" else None,
              "range": [round(range_start, 3), round(range_end, 3)],
              "n_bars": int(n), "checkpoints": checkpoints, "n_scored": len(scored)}
    if len(scored) >= 4:
        third = max(1, len(scored) // 3)
        early = np.mean([c["chunk_hit_rate"] for c in scored[:third]])
        late = np.mean([c["chunk_hit_rate"] for c in scored[-third:]])
        result["early_hit"] = round(float(early), 4)
        result["late_hit"] = round(float(late), 4)
        result["improvement"] = round(float(late - early), 4)
        result["final_cum_hit"] = scored[-1]["cum_hit_rate"]
    return result


def _print(r: dict) -> None:
    line = "=" * 64
    print("\n" + line)
    print(f"FADE LEARNING SIMULATION - {r.get('asset', '?').upper()}")
    mode = r.get("train_mode", "expanding")
    if mode == "fixed":
        print(f"  train: fixed window ({r.get('train_window', '?')} bars)")
    print("  (history hidden, revealed slowly; tool predicts the unseen future)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  {'cp':>3}  {'revealed':>9}  {'rules':>6}  {'cover':>7}  "
          f"{'chunk_hit':>10}  {'cum_hit':>8}")
    for c in r["checkpoints"]:
        if "chunk_hit_rate" not in c:
            print(f"  {c['checkpoint']:>3}  {c.get('revealed_bars', 0):>9}  "
                  f"{c.get('status', '?')}")
            continue
        print(f"  {c['checkpoint']:>3}  {c['revealed_bars']:>9}  {c['n_rules']:>6}  "
              f"{c['coverage']:>7}  {c['chunk_hit_rate']:>10}  {c['cum_hit_rate']:>8}")
    print(line)
    if "improvement" in r:
        trend = ("IMPROVES with more history" if r["improvement"] > 0.005
                 else "flat" if r["improvement"] > -0.005
                 else "DEGRADES with more history")
        print(f"  early chunks hit : {r['early_hit']}")
        print(f"  late chunks hit  : {r['late_hit']}")
        print(f"  self-improvement : {r['improvement']:+.4f}  ({trend})")
        print(f"  final cumulative : {r['final_cum_hit']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE simulated live learning")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--checkpoints", type=int, default=15)
    parser.add_argument("--seed-frac", type=float, default=0.40)
    parser.add_argument("--range-start", type=float, default=0.0)
    parser.add_argument("--range-end", type=float, default=1.0)
    parser.add_argument("--train-mode", choices=("expanding", "fixed"), default="expanding")
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--move-threshold", type=float, default=None)
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    cfg = Config()
    if args.move_threshold is not None:
        cfg = dataclasses.replace(cfg, move_threshold=args.move_threshold)
    _print(run_learning_sim(args.csv, seed_frac=args.seed_frac,
                            n_checkpoints=args.checkpoints,
                            range_start=args.range_start,
                            range_end=args.range_end,
                            train_mode=args.train_mode,
                            train_window=args.train_window,
                            config=cfg))


if __name__ == "__main__":
    main()
