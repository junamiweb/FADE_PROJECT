"""Does the lean 'path_min' atom set generalize beyond BTC 1h?

path_min = (return_1h, volatility, volume_zscore, streak_signed) scored 54.05%
out-of-sample directional hit on btc_1h.csv (p=0.002), beating core5 (53.26%).
This module re-runs the SAME strict 70/30 quarantined holdout (frozen dev rules
+ permutation p-value) for core5, path_min, and core5_path across several assets
and resolutions, then prints one comparison table and a VERDICT on whether the
edge holds off its home turf.

Mechanics: it calls fade.pipeline.holdout.holdout_test unchanged. That function
computes a momentum baseline from return_6h, which path_min does NOT contain, so
it raises KeyError for return_6h-free sets. When that happens we fall back to a
faithful local mirror of holdout_test (identical quarantine + permutation logic;
only the extra momentum-baseline metric is skipped, guarding the missing column
exactly the way walk_forward already does). No core file is modified and no
look-ahead is introduced -- the holdout slice stays quarantined either way.

Run:
    python -m fade.pipeline.path_min_generalization
    python -m fade.pipeline.path_min_generalization --shuffles 500
    python -m fade.pipeline.path_min_generalization --assets btc_1h.csv eth_1h.csv
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
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import P_VALUE_MAX, _select_stable_rules, holdout_test

# Reference first (btc_1h is path_min's home turf), then the generalization set.
DEFAULT_ASSETS = ["btc_1h.csv", "eth_1h.csv", "btc_15m.csv", "btc_30m.csv", "btc_5m.csv"]
DEFAULT_SETS = ["core5", "path_min", "core5_path"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _local_holdout(
    csv_path: str,
    config: Config,
    holdout_frac: float = 0.30,
    n_shuffles: int = 300,
    seed: int = 0,
) -> dict:
    """Faithful mirror of holdout.holdout_test, minus the return_6h momentum
    baseline. Used only when holdout_test raises because the atom set has no
    return_6h. Same chronological split, same frozen dev rules, same permutation
    p-value -- the holdout slice never touches fitting, so no look-ahead.
    """
    asset = Path(csv_path).stem.lower()
    rng = np.random.default_rng(seed)

    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    n = len(atoms)
    split = int(n * (1.0 - holdout_frac))
    dev_atoms, hold_atoms = atoms.iloc[:split], atoms.iloc[split:]
    dev_fwd = fwd.iloc[:split]
    hold_fwd = fwd.iloc[split:]

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)

    result: dict = {
        "asset": asset,
        "n_total": int(n),
        "n_dev": int(split),
        "n_holdout": int(n - split),
        "n_stable_rules": int(len(frozen)),
    }
    if frozen.empty:
        result["status"] = "no_rules"
        result["verdict"] = "INCONCLUSIVE - no stable rules survived development."
        return result

    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)

    if preds.empty:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - frozen rules never fired on holdout."
        return result

    valid = hold_fwd.reindex(preds.index).notna()
    pred_v = preds["pred"][valid].to_numpy()
    fwd_v = hold_fwd.reindex(preds.index)[valid].to_numpy()
    pred_sc, act_sc = score_predictions(pred_v, fwd_v, config.move_threshold)
    coverage = int(len(pred_sc))
    if coverage == 0:
        result["status"] = "no_coverage"
        result["verdict"] = "INCONCLUSIVE - no scorable holdout predictions."
        return result

    real_hit = float(np.mean(pred_sc == act_sc))
    real_lift = real_hit - 0.5

    null_hits = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(act_sc)
        null_hits[i] = np.mean(pred_sc == shuffled)
    n_ge = int(np.sum(null_hits >= real_hit))
    p_value = (1 + n_ge) / (1 + n_shuffles)

    result.update({
        "status": "ok",
        "holdout_hit_rate": round(real_hit, 4),
        "holdout_lift_vs_random": round(real_lift, 4),
        "coverage": coverage,
        "null_mean": round(float(np.mean(null_hits)), 4),
        "null_std": round(float(np.std(null_hits)), 4),
        "p_value": round(p_value, 4),
    })
    lift, p = real_lift, p_value
    if lift <= 0:
        result["verdict"] = "FAIL - no positive edge on unseen holdout data."
    elif p <= P_VALUE_MAX:
        result["verdict"] = "PASS - base edge survives strict out-of-sample holdout."
    else:
        result["verdict"] = "WEAK - positive but within shuffle noise."
    return result


def run_one(csv_path: str, set_name: str, n_shuffles: int, holdout_frac: float) -> dict:
    """Run the strict holdout for one (asset, atom-set). Prefer the shared
    holdout_test; on failure (e.g. return_6h-free set) fall back to the local
    mirror and record how the number was produced.
    """
    cfg = dataclasses.replace(Config(), atom_columns=ATOM_SETS[set_name])
    try:
        r = holdout_test(csv_path, holdout_frac=holdout_frac, n_shuffles=n_shuffles, config=cfg)
        r["method"] = "holdout_test"
        r["set"] = set_name
        return r
    except Exception as exc:  # noqa: BLE001 - report, never modify core
        r = _local_holdout(csv_path, cfg, holdout_frac=holdout_frac, n_shuffles=n_shuffles)
        r["method"] = "local(no-mom):" + type(exc).__name__
        r["set"] = set_name
        return r


def _cell(r: dict, key: str, fmt: str) -> str:
    val = r.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return format(val, fmt)


def _print_table(rows: list[dict]) -> None:
    header = f"{'asset':<10} {'set':<11} {'n_rules':>7} {'coverage':>8} {'hit':>7} {'lift':>8} {'p_value':>8}  {'method':<22} status"
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for r in rows:
        print(
            f"{r.get('asset', '?'):<10} "
            f"{r.get('set', '?'):<11} "
            f"{_cell(r, 'n_stable_rules', 'd'):>7} "
            f"{_cell(r, 'coverage', 'd'):>8} "
            f"{_cell(r, 'holdout_hit_rate', '.4f'):>7} "
            f"{_cell(r, 'holdout_lift_vs_random', '+.4f'):>8} "
            f"{_cell(r, 'p_value', '.4f'):>8}  "
            f"{r.get('method', '?'):<22} "
            f"{r.get('status', 'error')}"
        )
    print(line)


def _verdict(rows: list[dict], assets: list[str]) -> None:
    by_asset: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_asset.setdefault(r["asset"], {})[r["set"]] = r

    beats = 0            # path_min hit > core5 hit (both scorable)
    compared = 0         # assets where both path_min and core5 scored
    pm_pass = 0          # path_min significant positive edge
    pm_scored = 0        # assets where path_min produced a hit rate
    details: list[str] = []

    for asset in [Path(a).stem.lower() for a in assets]:
        sets = by_asset.get(asset, {})
        pm = sets.get("path_min")
        c5 = sets.get("core5")
        pm_ok = pm is not None and pm.get("status") == "ok"
        c5_ok = c5 is not None and c5.get("status") == "ok"
        if pm_ok:
            pm_scored += 1
            if pm["holdout_lift_vs_random"] > 0 and pm["p_value"] <= P_VALUE_MAX:
                pm_pass += 1
        if pm_ok and c5_ok:
            compared += 1
            pm_hit = pm["holdout_hit_rate"]
            c5_hit = c5["holdout_hit_rate"]
            won = pm_hit > c5_hit
            beats += 1 if won else 0
            tag = "path_min WINS" if won else "core5 wins/ties"
            details.append(
                f"  {asset:<10} path_min={pm_hit:.4f} (p={pm['p_value']:.3f})  "
                f"core5={c5_hit:.4f} (p={c5['p_value']:.3f})  -> {tag}"
            )
        else:
            reason = []
            if not pm_ok:
                reason.append(f"path_min={pm.get('status', 'error') if pm else 'missing'}")
            if not c5_ok:
                reason.append(f"core5={c5.get('status', 'error') if c5 else 'missing'}")
            details.append(f"  {asset:<10} not compared ({', '.join(reason)})")

    line = "=" * 78
    print("\n" + line)
    print("VERDICT - does path_min's edge generalize beyond btc_1h?")
    print(line)
    for d in details:
        print(d)
    print("-" * 78)
    print(f"  path_min beats core5 on {beats}/{compared} head-to-head assets.")
    print(f"  path_min shows a significant positive edge (lift>0, p<={P_VALUE_MAX}) "
          f"on {pm_pass}/{pm_scored} scorable assets.")

    if compared == 0:
        print("  RESULT: INCONCLUSIVE - no asset let both sets be scored.")
    elif beats == compared and pm_pass == pm_scored and pm_scored > 0:
        print("  RESULT: GENERALIZES - path_min beats core5 everywhere and keeps a "
              "significant edge.")
    elif beats > compared / 2.0:
        print("  RESULT: PARTIAL - path_min wins on the majority of assets but not "
              "uniformly; the ~54% btc_1h edge is asset/resolution dependent.")
    else:
        print("  RESULT: DOES NOT GENERALIZE - path_min's btc_1h advantage over core5 "
              "does not hold on most other assets/resolutions.")
    print(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Test path_min generalization vs core5 across assets.")
    ap.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS, help="CSV filenames in repo root.")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help="Atom set names from ATOM_SETS.")
    ap.add_argument("--shuffles", type=int, default=300, help="Permutation shuffles for the p-value.")
    ap.add_argument("--holdout-frac", type=float, default=0.30, help="Quarantined holdout fraction.")
    args = ap.parse_args()

    rows: list[dict] = []
    used_assets: list[str] = []
    for asset in args.assets:
        path = asset if Path(asset).exists() else str(REPO_ROOT / asset)
        if not Path(path).exists():
            print(f"[skip] {asset} not found (looked in cwd and {REPO_ROOT}).")
            continue
        used_assets.append(asset)
        for set_name in args.sets:
            if set_name not in ATOM_SETS:
                print(f"[skip] unknown atom set '{set_name}'.")
                continue
            print(f"[run ] {Path(asset).stem:<10} {set_name} ...", flush=True)
            rows.append(run_one(path, set_name, args.shuffles, args.holdout_frac))

    if not rows:
        print("No results - no valid assets/sets were run.")
        return

    print()
    _print_table(rows)
    _verdict(rows, used_assets)


if __name__ == "__main__":
    main()
