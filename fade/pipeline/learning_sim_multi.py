"""Multi-resolution progressive learning — unanimous agreement over hidden future.

Applies the validated multi-res formula (15m + 30m + 1h must agree) inside the
honest learning-simulation framework: at each checkpoint all three resolutions
mine rules only on data revealed so far, predict the still-hidden next chunk,
then we score unanimous vs single-resolution on the shared 1h decision grid.

Answers: does cross-timeframe agreement keep its lift (+0.047 in holdout) when
rules are re-mined progressively — not just on a frozen one-shot holdout?

Run:
    python -m fade.pipeline.learning_sim_multi
    python -m fade.pipeline.learning_sim_multi --checkpoints 8
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
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("learning_sim_multi")

INTERVAL_MIN = {"15m": 15, "30m": 30, "1h": 60}
FILES = {"15m": "btc_15m.csv", "30m": "btc_30m.csv", "1h": "btc_1h.csv"}


def _chunk_preds(
    res: str,
    train_end: pd.Timestamp,
    pred_end: pd.Timestamp,
    target_min: int,
    config: Config,
) -> pd.DataFrame | None:
    """Mine on data <= train_end, predict (train_end, pred_end], stamp decision time."""
    interval_min = INTERVAL_MIN[res]
    horizon = max(1, round(target_min / interval_min))
    cfg = dataclasses.replace(config, forward_horizon=horizon)

    df = load_ohlcv(FILES[res])
    atoms = atoms_mod.compute_atoms(df, cfg)
    fwd = atoms_mod.forward_return(df, horizon).reindex(atoms.index)

    train_mask = atoms.index <= train_end
    pred_mask = (atoms.index > train_end) & (atoms.index <= pred_end)
    train_atoms, train_fwd = atoms.loc[train_mask], fwd.loc[train_mask]
    pred_atoms, pred_fwd = atoms.loc[pred_mask], fwd.loc[pred_mask]

    if len(train_atoms) < cfg.min_support * 2 or pred_atoms.empty:
        return None

    bt = walk_forward(train_atoms, train_fwd, cfg)
    frozen = _select_stable_rules(bt.stability, cfg)
    if frozen.empty:
        return None

    thresholds = ev.compute_thresholds(train_atoms, cfg)
    disc = ev.discretize(pred_atoms, thresholds)
    events = ev.build_events(disc, cfg, allowed=set(frozen.index))
    preds = predict(events, frozen)
    if preds.empty:
        return None

    decision = preds.index + pd.Timedelta(minutes=interval_min)
    out = pd.DataFrame({
        "decision_time": decision,
        "pred": preds["pred"].to_numpy(),
        "fwd": pred_fwd.reindex(preds.index).to_numpy(),
    })
    return out.dropna(subset=["fwd"])


def run_multi_learning(
    target_min: int = 60,
    seed_frac: float = 0.40,
    n_checkpoints: int = 8,
    config: Config | None = None,
) -> dict:
    config = config or Config()
    if not Path(FILES["1h"]).exists():
        return {"status": "missing_1h"}

    base = load_ohlcv(FILES["1h"])
    n = len(base)
    seed_end = int(n * seed_frac)
    step = (n - seed_end) // n_checkpoints
    if step <= 0:
        return {"status": "insufficient_data"}

    checkpoints = []
    cum_1h = {"correct": 0, "total": 0}
    cum_uni = {"correct": 0, "total": 0}

    for i in range(n_checkpoints):
        cut_idx = seed_end + i * step
        pred_idx = n - 1 if i == n_checkpoints - 1 else cut_idx + step
        train_end = base.index[cut_idx - 1]
        pred_end = base.index[pred_idx]

        per_res: dict[str, pd.DataFrame] = {}
        for res in ("15m", "30m", "1h"):
            if not Path(FILES[res]).exists():
                continue
            got = _chunk_preds(res, train_end, pred_end, target_min, config)
            if got is not None:
                per_res[res] = got

        if "1h" not in per_res:
            checkpoints.append({"checkpoint": i + 1, "status": "no_1h"})
            continue

        grid = per_res["1h"].rename(columns={"pred": "pred_1h", "fwd": "fwd_1h"}).copy()
        tol = pd.Timedelta(minutes=target_min)
        for res in ("15m", "30m"):
            if res not in per_res:
                continue
            p = per_res[res][["decision_time", "pred"]].rename(columns={"pred": f"pred_{res}"})
            grid = pd.merge_asof(grid.sort_values("decision_time"),
                                 p.sort_values("decision_time"),
                                 on="decision_time", direction="backward", tolerance=tol)

        pred_cols = [c for c in ("pred_15m", "pred_30m", "pred_1h") if c in grid.columns]
        grid = grid.dropna(subset=["fwd_1h"])
        if grid.empty:
            checkpoints.append({"checkpoint": i + 1, "status": "no_scorable"})
            continue

        # 1h baseline on this chunk.
        p1, a1 = score_predictions(
            grid["pred_1h"].to_numpy().astype(int),
            grid["fwd_1h"].to_numpy(), config.move_threshold)
        hit_1h = float(np.mean(p1 == a1)) if len(p1) else float("nan")

        # Unanimous (all present resolutions agree).
        full = grid.dropna(subset=pred_cols)
        hit_uni = float("nan")
        n_uni = 0
        if len(pred_cols) >= 2 and not full.empty:
            votes = full[pred_cols].astype(int)
            agree = votes.nunique(axis=1) == 1
            unan = full[agree]
            if not unan.empty:
                pu, au = score_predictions(
                    unan[pred_cols[0]].to_numpy(),
                    unan["fwd_1h"].to_numpy(), config.move_threshold)
                if len(pu):
                    hit_uni = float(np.mean(pu == au))
                    n_uni = len(pu)
                    cum_uni["correct"] += int(np.sum(pu == au))
                    cum_uni["total"] += len(pu)

        if len(p1):
            cum_1h["correct"] += int(np.sum(p1 == a1))
            cum_1h["total"] += len(p1)

        checkpoints.append({
            "checkpoint": i + 1,
            "train_end": str(train_end)[:10],
            "hit_1h": round(hit_1h, 4) if hit_1h == hit_1h else None,
            "hit_unanimous": round(hit_uni, 4) if hit_uni == hit_uni else None,
            "cover_1h": len(p1),
            "cover_unanimous": n_uni,
            "agreement_rate": round(n_uni / len(full), 4) if len(full) and n_uni else None,
        })
        log.info("cp %d hit_1h=%.4f hit_uni=%s agree=%s",
                 i + 1, hit_1h if hit_1h == hit_1h else 0,
                 f"{hit_uni:.4f}" if hit_uni == hit_uni else "n/a",
                 checkpoints[-1].get("agreement_rate"))

    scored = [c for c in checkpoints if c.get("hit_1h") is not None]
    result = {"status": "ok" if scored else "no_scored", "checkpoints": checkpoints}
    if cum_1h["total"]:
        result["cum_hit_1h"] = round(cum_1h["correct"] / cum_1h["total"], 4)
    if cum_uni["total"]:
        result["cum_hit_unanimous"] = round(cum_uni["correct"] / cum_uni["total"], 4)
        result["agreement_overall"] = round(
            sum(c.get("cover_unanimous", 0) for c in scored) /
            max(1, sum(c.get("cover_1h", 0) for c in scored)), 4)
    return result


def _print(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print("FADE MULTI-RES PROGRESSIVE LEARNING")
    print("  (three resolutions mine on revealed past, predict hidden future)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  {'cp':>3}  {'train_end':>10}  {'1h_hit':>8}  {'uni_hit':>8}  "
          f"{'agree%':>7}  {'cover':>7}")
    for c in r["checkpoints"]:
        if c.get("hit_1h") is None:
            print(f"  {c['checkpoint']:>3}  {c.get('status', '?')}")
            continue
        uni = f"{c['hit_unanimous']:.4f}" if c.get("hit_unanimous") is not None else "     n/a"
        agr = f"{c['agreement_rate']*100:.0f}%" if c.get("agreement_rate") else "  n/a"
        print(f"  {c['checkpoint']:>3}  {c['train_end']:>10}  {c['hit_1h']:>8.4f}  "
              f"{uni:>8}  {agr:>7}  {c['cover_unanimous']:>7}")
    print(line)
    if "cum_hit_1h" in r and "cum_hit_unanimous" in r:
        lift = r["cum_hit_unanimous"] - r["cum_hit_1h"]
        print(f"  cumulative 1h alone     : {r['cum_hit_1h']}")
        print(f"  cumulative unanimous    : {r['cum_hit_unanimous']}  "
              f"(lift {lift:+.4f}, agree {r.get('agreement_overall', 0)*100:.0f}%)")
    elif "cum_hit_1h" in r:
        print(f"  cumulative 1h alone     : {r['cum_hit_1h']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE multi-res progressive learning")
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--seed-frac", type=float, default=0.40)
    parser.add_argument("--target-min", type=int, default=60)
    args = parser.parse_args()
    _print(run_multi_learning(n_checkpoints=args.checkpoints,
                              seed_frac=args.seed_frac,
                              target_min=args.target_min))


if __name__ == "__main__":
    main()
