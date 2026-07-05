"""FADE v0.1 main learning loop and report.

Each run:
  1. Load data (CSV if given, else reproducible synthetic BTC-like 1H data).
  2. Compute atoms (cached).
  3. Build events.
  4. Filter against negative memory (checked BEFORE rule generation).
  5. Walk-forward backtest -> mine + evaluate rules, track stability.
  6. Update positive memory (stable rules) and negative memory (failures).
  7. Print the report.

Run:  python -m fade.pipeline.main [path/to/ohlcv.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import generate_synthetic_ohlcv, load_ohlcv
from fade.core.calibration import CalibrationStore
from fade.core.predictor import collect_calibration_samples
from fade.core.regimes import assign_regimes, save_regime_stats
from fade.core.significant_changes import detect_significant_changes, summarize_changes
from fade.memory import MemoryStore
from fade.pipeline.backtest import walk_forward
from fade.pipeline.forecast import run_forecast
from fade.utils.cache import DiskCache, hash_key
from fade.utils.logging import get_logger

log = get_logger("main")


def run(csv_path: str | None = None, config: Config | None = None) -> dict:
    config = config or Config()
    cache = DiskCache(config.cache_dir)
    asset = Path(csv_path).stem.lower() if csv_path else "synthetic"
    memory = MemoryStore(config.memory_dir, asset=asset)

    # --- 1. Load data --------------------------------------------------
    if csv_path:
        df = load_ohlcv(csv_path)
        log.info("Loaded %d rows from %s", len(df), csv_path)
    else:
        df = generate_synthetic_ohlcv()
        log.info("No CSV provided - using %d rows of synthetic data", len(df))

    # Per-asset calibration: different markets get separate reliability tables
    calibration = CalibrationStore(config.memory_dir / f"calibration_{asset}.json")
    log.info("Asset '%s' — memory + calibration per-asset", asset)

    # --- 2. Compute atoms (cached; deterministic given data + config) --
    key = hash_key("pool", df, config.return_6h_window, config.volatility_window,
                   config.volume_window, config.trend_window)
    pool = cache.memoize(key, lambda: atoms_mod.compute_atom_pool(df, config))
    atoms = pool[list(config.atom_columns)].dropna()
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    # --- 2b. Significant change detection (uses full pool for shock atoms) -
    changes = detect_significant_changes(pool.reindex(atoms.index), fwd, config)
    change_summary = summarize_changes(changes)
    regimes = assign_regimes(changes, config.post_shock_bars)

    # --- 3./4. Events + negative-memory filter -------------------------
    blocked = memory.blocked_events()
    full_disc = ev.discretize(atoms, ev.compute_thresholds(atoms, config))
    all_events = ev.build_events(full_disc, config, blocked=blocked)
    frequent = ev.frequent_events(all_events, config.min_event_frequency)
    n_events = int(all_events["event"].nunique())

    # --- 5. Walk-forward backtest --------------------------------------
    bt = walk_forward(
        atoms, fwd, config, blocked=blocked,
        calibration=calibration, positive=memory.positive,
        regimes=regimes,
    )
    stability = bt.stability

    # --- 5b. Update calibration from OOS predictions ---------------------
    cal_samples = collect_calibration_samples(bt.oos_predictions, fwd)
    cal_metrics = calibration.update(cal_samples)
    calibration.save()

    # Persist per-regime OOS reliability for regime-weighted forecasting.
    if bt.regime_metrics:
        save_regime_stats(
            config.memory_dir / f"regime_stats_{asset}.json", bt.regime_metrics
        )

    # --- 6. Update memory ----------------------------------------------
    stable_mask = (
        (stability["folds_present"] >= config.stability_min_folds)
        & (stability["consistency"] >= config.stability_min_consistency)
        & (stability["avg_oos_hit"] > 0.5)
    ) if not stability.empty else pd.Series(dtype=bool)

    stable = stability[stable_mask] if not stability.empty else stability
    unstable = stability[~stable_mask] if not stability.empty else stability

    for event, row in stable.iterrows():
        memory.upsert_positive(event, {
            "direction": int(row["direction"]),
            "consistency": round(float(row["consistency"]), 4),
            "avg_oos_hit": round(float(row["avg_oos_hit"]), 4),
            "avg_confidence": round(float(row["avg_confidence"]), 4),
            "folds_present": int(row["folds_present"]),
            "stability": round(float(row["stability"]), 4),
        })
    memory.prune_positive(set(stable.index))

    # Patterns that recurred but were directionally unstable = anti-patterns.
    rejected = 0
    for event, row in unstable.iterrows():
        if row["folds_present"] >= config.stability_min_folds:
            memory.record_failure(event, "unstable_direction", config.negative_min_failures)
            rejected += 1
    memory.save()

    # --- 6b. Latest forecast -------------------------------------------
    latest_forecast = None
    if csv_path:
        fc = run_forecast(csv_path, config)
        if fc.get("status") == "ok":
            latest_forecast = {
                "timestamp": fc["timestamp"],
                "direction": fc["direction"],
                "raw_prob_pct": fc["raw_prob_pct"],
                "calibrated_prob_pct": fc["calibrated_prob_pct"],
                "n_rules": fc["n_rules"],
            }

    # --- 7. Report -----------------------------------------------------
    report = _build_report(config, n_events, frequent, bt, stability, stable,
                           rejected, memory, cal_metrics, calibration,
                           latest_forecast, change_summary, asset, bt.regime_metrics)
    _print_report(report, stability, bt)
    return report


def _build_report(config, n_events, frequent, bt, stability, stable, rejected, memory,
                  cal_metrics, calibration, latest_forecast, change_summary, asset,
                  regime_metrics) -> dict:
    fold_metrics = pd.DataFrame(bt.fold_metrics) if bt.fold_metrics else pd.DataFrame()
    model_hr = float(fold_metrics["model_hit_rate"].mean()) if not fold_metrics.empty else float("nan")
    mom_hr = float(fold_metrics["momentum_hit_rate"].mean()) if not fold_metrics.empty else float("nan")
    coverage = int(fold_metrics["coverage"].sum()) if not fold_metrics.empty else 0

    cal_summary = calibration.summary()
    prev_ece = None
    if len(cal_summary["history"]) >= 2:
        prev_ece = cal_summary["history"][-2].get("ece")

    return {
        "n_atoms": len(config.atom_columns),
        "n_events": n_events,
        "n_frequent_events": len(frequent),
        "n_rules": bt.n_rules_last,
        "n_stable_rules": int(len(stable)),
        "n_rejected_patterns": rejected,
        "n_blacklisted": len([e for e, v in memory.negative.items() if v.get("blacklisted")]),
        "avg_model_hit_rate": model_hr,
        "avg_momentum_hit_rate": mom_hr,
        "random_hit_rate": 0.5,
        "lift_vs_random": model_hr - 0.5 if model_hr == model_hr else float("nan"),
        "lift_vs_momentum": model_hr - mom_hr if model_hr == model_hr else float("nan"),
        "test_coverage": coverage,
        "calibration_runs": cal_summary["runs"],
        "calibration_samples": cal_metrics.get("samples", 0),
        "calibration_ece": cal_metrics.get("ece"),
        "calibration_brier": cal_metrics.get("brier"),
        "calibration_ece_delta": (
            cal_metrics.get("ece") - prev_ece
            if prev_ece is not None and cal_metrics.get("ece") == cal_metrics.get("ece")
            else None
        ),
        "latest_forecast": latest_forecast,
        "asset": asset,
        "significant_changes": change_summary,
        "regime_metrics": regime_metrics,
    }


def _print_report(report: dict, stability: pd.DataFrame, bt) -> None:
    line = "=" * 68
    print("\n" + line)
    print("FADE v0.1  -  Financial Atom Discovery Engine  -  RUN REPORT")
    print(line)
    print(f"  Atoms computed            : {report['n_atoms']}")
    print(f"  Unique events             : {report['n_events']}")
    print(f"  Frequent events (kept)    : {report['n_frequent_events']}")
    print(f"  Rules mined (last fold)   : {report['n_rules']}")
    print(f"  Stable rules -> positive  : {report['n_stable_rules']}")
    print(f"  Rejected patterns         : {report['n_rejected_patterns']}")
    print(f"  Blacklisted (negative)    : {report['n_blacklisted']}")

    print("\n  --- Top rules by stability (out-of-sample) ---")
    if stability is not None and not stability.empty:
        top = stability.head(10)
        print(f"  {'event':<48}{'dir':>4}{'stab':>7}{'oosHit':>8}")
        for event, row in top.iterrows():
            d = "UP" if row["direction"] == 1 else "DN"
            ev_short = event if len(event) <= 46 else event[:43] + "..."
            print(f"  {ev_short:<48}{d:>4}{row['stability']:>7.2f}{row['avg_oos_hit']:>8.3f}")
    else:
        print("  (no rules survived the frequency / support filters)")

    print("\n  --- Backtest vs baselines (mean over folds) ---")
    print(f"  {'model hit-rate':<26}: {report['avg_model_hit_rate']:.4f}")
    print(f"  {'momentum hit-rate':<26}: {report['avg_momentum_hit_rate']:.4f}")
    print(f"  {'random hit-rate':<26}: {report['random_hit_rate']:.4f}")
    print(f"  {'test coverage (preds)':<26}: {report['test_coverage']}")

    print("\n  --- Predictive lift ---")
    print(f"  {'lift vs random':<26}: {report['lift_vs_random']:+.4f}")
    print(f"  {'lift vs momentum':<26}: {report['lift_vs_momentum']:+.4f}")

    rm = report.get("regime_metrics") or {}
    if rm:
        print("\n  --- Performance by regime (OOS) ---")
        for regime, m in rm.items():
            print(f"  {regime:<16} hit={m['hit_rate']:.3f}  "
                  f"lift={m['lift_vs_random']:+.3f}  n={m['n_predictions']}")

    print("\n  --- Calibrated probability (improves each run) ---")
    ece = report["calibration_ece"]
    brier = report["calibration_brier"]
    ece_s = f"{ece:.4f}" if ece == ece else "n/a"
    brier_s = f"{brier:.4f}" if brier == brier else "n/a"
    print(f"  {'calibration runs':<26}: {report['calibration_runs']}")
    print(f"  {'samples this run':<26}: {report['calibration_samples']}")
    print(f"  {'ECE (lower=better)':<26}: {ece_s}")
    print(f"  {'Brier (lower=better)':<26}: {brier_s}")
    delta = report["calibration_ece_delta"]
    if delta is not None and delta == delta:
        trend = "improved" if delta < 0 else "worsened"
        print(f"  {'ECE vs prev run':<26}: {delta:+.4f} ({trend})")

    fc = report.get("latest_forecast")
    if fc:
        print("\n  --- Latest forecast ---")
        print(f"  {fc['timestamp']}  {fc['direction']}  "
              f"raw={fc['raw_prob_pct']}%  calibrated={fc['calibrated_prob_pct']}%  "
              f"({fc['n_rules']} rules)")

    sc = report.get("significant_changes")
    if sc:
        print(f"\n  --- Significant changes ({report.get('asset', '?')}) ---")
        print(f"  {'total flagged':<26}: {sc['total_flagged']}")
        for ctype, n in sorted(sc.get("counts", {}).items()):
            fwd = sc.get("post_shock_fwd", {}).get(ctype)
            fwd_s = f"  fwd4h={fwd:+.4f}" if fwd is not None else ""
            print(f"  {ctype:<26}: {n}{fwd_s}")
        if sc.get("recent"):
            print(f"  {'recent shocks':<26}:")
            for r in sc["recent"]:
                print(f"    {r['timestamp']}  {r['type']:<14}  "
                      f"ret={r['return_1h_pct']:+.2f}%  vol_z={r['volume_z']:+.1f}")

    verdict = _verdict(report)
    print("\n  --- Summary ---")
    print(f"  {verdict}")
    print(line + "\n")


def _verdict(report: dict) -> str:
    lift_r = report["lift_vs_random"]
    stable = report["n_stable_rules"]
    if lift_r != lift_r:  # NaN
        return "No coverage: no atomic structure passed the thresholds."
    if stable > 0 and lift_r > 0.01:
        return (f"Signal found: {stable} stable atomic patterns generalise "
                f"across folds (lift vs random {lift_r:+.3f}).")
    if lift_r > 0.0:
        return ("Weak/unstable signal: positive lift but few patterns survive "
                "stability filtering. Treat as inconclusive.")
    return "No generalising structure: performance is at or below random."


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if csv and not Path(csv).exists():
        log.warning("CSV %s not found; falling back to synthetic data.", csv)
        csv = None
    run(csv)


if __name__ == "__main__":
    main()
