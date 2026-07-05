"""Grid search for the best FADE formula — with honest overfit protection.

Searches combinations of:
    * resolution   (data file: 15m / 30m / 1h)
    * horizon      (bars ahead: 1 / 2 / 3 ...)
    * atom set     (core5 / plus7 / full9)

THE MULTIPLE-COMPARISONS TRAP: if we scored many configs on one holdout and
picked the best, the winner would look good partly by luck. To avoid this we
split each series chronologically into THREE parts:

    [-------- development 55% --------][-- validation 25% --][-- test 20% --]

  1. DEVELOPMENT — mine + select stable rules (walk-forward inside dev only).
  2. VALIDATION  — freeze rules, score every config here, RANK, pick the winner.
                   All the "peeking" and selection happens on validation.
  3. TEST        — touched ONCE, only for the single winning config, with a
                   permutation test. This number is the unbiased estimate.

Because the test slice never influences selection, the winner's test result is
honest. We also report the validation-to-test drop: a large drop is the
signature of overfitting the search itself.

Run:
    python -m fade.pipeline.grid_search
    python -m fade.pipeline.grid_search --horizons 1,2 --resolutions 15m,30m,1h
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import ATOM_SETS, Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("grid_search")

FILES = {"15m": "btc_15m.csv", "30m": "btc_30m.csv", "1h": "btc_1h.csv"}


def _skill_on_slice(frozen, thresholds, atoms_slice, fwd_slice, config,
                    n_shuffles=0, seed=0) -> dict:
    """Apply frozen rules to a slice; return hit-rate, drift-adjusted skill, p."""
    disc = ev.discretize(atoms_slice, thresholds)
    events = ev.build_events(disc, config, allowed=set(frozen.index))
    preds = predict(events, frozen)
    if preds.empty:
        return {"coverage": 0, "hit_rate": float("nan"),
                "skill": float("nan"), "p_value": float("nan")}
    actual = (fwd_slice > 0).astype(int).reindex(preds.index)
    valid = actual.notna() & fwd_slice.reindex(preds.index).notna()
    pv = preds["pred"][valid].to_numpy()
    av = actual[valid].to_numpy()
    if pv.size == 0:
        return {"coverage": 0, "hit_rate": float("nan"),
                "skill": float("nan"), "p_value": float("nan")}
    hit = float(np.mean(pv == av))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(pv == rng.permutation(av)) for _ in range(n_shuffles)]) \
        if n_shuffles else np.array([0.5])
    null_mean = float(np.mean(null))
    p = ((1 + int(np.sum(null >= hit))) / (1 + n_shuffles)) if n_shuffles else float("nan")
    return {
        "coverage": int(pv.size),
        "hit_rate": round(hit, 4),
        "null_mean": round(null_mean, 4),
        "skill": round(hit - null_mean, 4),
        "p_value": round(p, 4) if p == p else None,
    }


def _prep(res: str, horizon: int, atom_set: str, config: Config):
    """Load a resolution, build the config, and split atoms/fwd into 3 parts."""
    cfg = dataclasses.replace(config, forward_horizon=horizon,
                              atom_columns=ATOM_SETS[atom_set])
    df = load_ohlcv(FILES[res])
    atoms = atoms_mod.compute_atoms(df, cfg)
    fwd = atoms_mod.forward_return(df, horizon).reindex(atoms.index)
    n = len(atoms)
    d_end = int(n * 0.55)
    v_end = int(n * 0.80)
    return cfg, atoms, fwd, d_end, v_end


def evaluate_config(res: str, horizon: int, atom_set: str,
                    config: Config, score_test: bool = False,
                    n_shuffles: int = 300, seed: int = 0) -> dict:
    """Mine on dev, score on validation (always) and test (only if requested)."""
    cfg, atoms, fwd, d_end, v_end = _prep(res, horizon, atom_set, config)
    dev_atoms, dev_fwd = atoms.iloc[:d_end], fwd.iloc[:d_end]

    dev_bt = walk_forward(dev_atoms, dev_fwd, cfg)
    frozen = _select_stable_rules(dev_bt.stability, cfg)
    base = {"resolution": res, "horizon": horizon, "atom_set": atom_set,
            "n_rules": int(len(frozen))}
    if frozen.empty:
        base["status"] = "no_rules"
        return base

    thresholds = ev.compute_thresholds(dev_atoms, cfg)
    val = _skill_on_slice(frozen, thresholds,
                          atoms.iloc[d_end:v_end], fwd.iloc[d_end:v_end], cfg)
    base["status"] = "ok"
    base["val_skill"] = val["skill"]
    base["val_hit"] = val["hit_rate"]
    base["val_cov"] = val["coverage"]
    if score_test:
        test = _skill_on_slice(frozen, thresholds,
                               atoms.iloc[v_end:], fwd.iloc[v_end:], cfg,
                               n_shuffles=n_shuffles, seed=seed)
        base["test_skill"] = test["skill"]
        base["test_hit"] = test["hit_rate"]
        base["test_null"] = test["null_mean"]
        base["test_cov"] = test["coverage"]
        base["test_p"] = test["p_value"]
    return base


def run_grid(resolutions=("15m", "30m", "1h"), horizons=(1, 2, 3),
             atom_sets=("core5", "plus7", "full9"),
             config: Config | None = None, n_shuffles: int = 300,
             seed: int = 0) -> dict:
    config = config or Config()
    rows = []
    for res in resolutions:
        if not Path(FILES[res]).exists():
            log.warning("missing %s, skipping", FILES[res])
            continue
        for h in horizons:
            for aset in atom_sets:
                r = evaluate_config(res, h, aset, config,
                                    score_test=False, seed=seed)
                rows.append(r)
                log.info("val | %s h=%d %s -> skill=%s rules=%s",
                         res, h, aset, r.get("val_skill"), r.get("n_rules"))

    ok = [r for r in rows if r.get("status") == "ok" and r.get("val_skill") == r.get("val_skill")]
    if not ok:
        return {"status": "no_valid_configs", "rows": rows}

    ok.sort(key=lambda r: r["val_skill"], reverse=True)
    winner = ok[0]

    # ONE honest test-set evaluation, only for the winner.
    final = evaluate_config(winner["resolution"], winner["horizon"],
                            winner["atom_set"], config,
                            score_test=True, n_shuffles=n_shuffles, seed=seed)
    return {"status": "ok", "n_configs": len(ok),
            "ranked": ok, "winner": final}


def _print(res: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print("FADE GRID SEARCH  (dev=mine | validation=select | test=final, touched once)")
    print(line)
    if res.get("status") != "ok":
        print(f"  status: {res.get('status')}")
        print(line + "\n")
        return
    print(f"  Configs scored on validation: {res['n_configs']}")
    print()
    print(f"  {'rank':>4}  {'res':>4}  {'h':>2}  {'atoms':>6}  {'rules':>6}  "
          f"{'val_skill':>10}  {'val_hit':>8}  {'val_cov':>8}")
    for i, r in enumerate(res["ranked"][:12], 1):
        print(f"  {i:>4}  {r['resolution']:>4}  {r['horizon']:>2}  {r['atom_set']:>6}  "
              f"{r['n_rules']:>6}  {r['val_skill']:>+10}  {r['val_hit']:>8}  {r['val_cov']:>8}")

    w = res["winner"]
    print("\n" + "-" * 78)
    print("  WINNER (selected on validation) — unbiased TEST-set result:")
    print(f"    formula     : resolution={w['resolution']}  horizon={w['horizon']}  "
          f"atoms={w['atom_set']}  ({w['n_rules']} rules)")
    print(f"    validation  : skill={w['val_skill']:+.4f}  hit={w['val_hit']}")
    if "test_skill" in w:
        drop = w["val_skill"] - w["test_skill"]
        print(f"    TEST (once) : skill={w['test_skill']:+.4f}  hit={w['test_hit']}  "
              f"null={w['test_null']}  cov={w['test_cov']}  p={w['test_p']}")
        print(f"    val->test drop: {drop:+.4f}  "
              f"({'stable' if abs(drop) < 0.01 else 'watch for search overfit'})")
        verdict = ("PASS - winning formula holds out of sample."
                   if (w.get("test_p") is not None and w["test_p"] <= 0.05
                       and w["test_skill"] > 0)
                   else "WEAK - winner did not stay significant on the test slice.")
        print(f"    VERDICT     : {verdict}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE grid search")
    parser.add_argument("--resolutions", type=str, default="15m,30m,1h")
    parser.add_argument("--horizons", type=str, default="1,2,3")
    parser.add_argument("--atom-sets", type=str, default="core5,plus7,full9")
    args = parser.parse_args()
    _print(run_grid(
        resolutions=tuple(args.resolutions.split(",")),
        horizons=tuple(int(x) for x in args.horizons.split(",")),
        atom_sets=tuple(args.atom_sets.split(",")),
    ))


if __name__ == "__main__":
    main()
