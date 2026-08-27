"""Forecast CLI — one calibrated prediction for the latest bar.

Uses stable rules from positive memory (run ``main`` first to refresh).
Does not re-run walk-forward; fast read-only inference.

Run:
    python -m fade.pipeline.forecast btc.csv
    python -m fade.pipeline.forecast btc.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fade.config import Config, lean_config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import (
    assign_regimes,
    assign_vr_regime,
    compute_vol_ratio,
    load_regime_stats,
    regime_confidence_scale,
)
from fade.core import dl_model
from fade.core import news_features
from fade.core.significant_changes import detect_significant_changes
from fade.core.predictor import _rule_weight, predict_calibrated
from fade.memory import MemoryStore
from fade.utils.logging import get_logger

log = get_logger("forecast")


def run_forecast(csv_path: str, config: Config | None = None) -> dict:
    """Produce a calibrated forecast for the most recent bar."""
    config = config or lean_config()
    df = load_ohlcv(csv_path)
    asset = Path(csv_path).stem.lower()

    memory = MemoryStore(config.memory_dir, asset=asset)
    calibration = CalibrationStore(config.memory_dir / f"calibration_{asset}.json")
    regime_stats = load_regime_stats(config.memory_dir / f"regime_stats_{asset}.json")
    positive = memory.positive
    blocked = memory.blocked_events()

    if not positive:
        return {
            "asset": asset,
            "status": "no_rules",
            "message": "No stable rules in positive memory. Run: python -m fade.pipeline.main",
        }

    pool = atoms_mod.compute_atom_pool(df, config)
    atoms = pool[list(config.atom_columns)].dropna()
    changes = detect_significant_changes(pool.reindex(atoms.index), config=config)
    regimes = assign_regimes(changes, config.post_shock_bars)
    current_regime = str(regimes.iloc[-1])
    returns = pool["return_1h"].reindex(atoms.index)
    vr = compute_vol_ratio(
        returns,
        config.vol_ratio_short_window,
        config.vol_ratio_long_window,
    )
    if config.vr_low_threshold is not None and config.vr_high_threshold is not None:
        vr_regimes = assign_vr_regime(vr, config.vr_low_threshold, config.vr_high_threshold)
        current_vr_regime = str(vr_regimes.iloc[-1])
    else:
        current_vr_regime = None
    thresholds = ev.compute_thresholds(atoms, config)
    last_ts = atoms.index[-1]
    disc = ev.discretize(atoms.iloc[[-1]], thresholds)

    atom_states = {col: disc[col].iloc[0] for col in config.atom_columns}
    events = ev.build_events(
        disc, config, allowed=set(positive.keys()), blocked=blocked,
    )

    if events.empty:
        return {
            "asset": asset,
            "status": "no_match",
            "timestamp": str(last_ts),
            "regime": current_regime,
            "vr_regime": current_vr_regime,
            "atom_states": atom_states,
            "message": "No stable rules match the current atom state.",
        }

    rules = pd.DataFrame(positive).T
    rules["direction"] = rules["direction"].astype(int)
    rules["confidence"] = rules["avg_oos_hit"].astype(float)

    preds = predict_calibrated(events, rules, calibration, positive)
    if preds.empty:
        return {
            "asset": asset,
            "status": "no_prediction",
            "timestamp": str(last_ts),
            "atom_states": atom_states,
        }

    row = preds.iloc[-1]

    # --- DL model forecast (if available) ---
    dl_result = None
    if dl_model.dl_backend_available():
        try:
            dl_result = dl_model.forecast_latest(
                csv_path=csv_path,
                config=config,
                news_csv=news_features.resolve_news_csv(asset, config.root.parent),
            )
        except Exception:
            log.exception("DL forecast failed - continuing without it")
            dl_result = None

    # Regime-weighted confidence: rescale distance from 0.5 by how reliable the
    # current regime was out-of-sample. Direction is untouched (no fake edge).
    # DISABLED by default — failed the chronological split test (regime reliability
    # is not stable over time). See Config.regime_weighting_enabled.
    raw_prob = float(row["raw_prob"])
    scale = (
        regime_confidence_scale(current_regime, regime_stats)
        if config.regime_weighting_enabled
        else 1.0
    )
    adj_raw = 0.5 + (raw_prob - 0.5) * scale
    adj_cal = calibration.calibrate(adj_raw)

    matched_events = events["event"].unique().tolist()
    rules_used = []
    for event in matched_events:
        if event not in rules.index:
            continue
        d, w = _rule_weight(event, rules, positive)
        rules_used.append({
            "event": event,
            "direction": "UP" if d == 1 else "DOWN",
            "weight": round(w, 4),
        })
    rules_used.sort(key=lambda r: r["weight"], reverse=True)

    return {
        "asset": asset,
        "status": "ok",
        "timestamp": str(preds.index[-1]),
        "horizon_h": config.forward_horizon,
        "regime": current_regime,
        "vr_regime": current_vr_regime,
        "regime_scale": round(scale, 3),
        "direction": "UP" if row["pred"] == 1 else "DOWN",
        "raw_prob_pct": round(raw_prob * 100, 1),
        "regime_adjusted_prob_pct": round(adj_raw * 100, 1),
        "calibrated_prob_pct": round(adj_cal * 100, 1),
        "calibrated_unweighted_pct": round(float(row["calibrated_prob"]) * 100, 1),
        "n_rules": int(row["n_rules"]),
        "atom_states": atom_states,
        "rules_used": rules_used,
        "dl_forecast": dl_result,

    }


def _print_forecast(result: dict) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"\nFADE FORECAST - {result.get('asset', '?')} - {status}")
        if msg := result.get("message"):
            print(f"  {msg}")
        if regime := result.get("regime"):
            print(f"  Regime        : {regime}")
        if atoms := result.get("atom_states"):
            print(f"  atoms: {atoms}")
        print()
        return

    print("\n" + "=" * 60)
    print(f"FADE FORECAST - {result['asset'].upper()}")
    print("=" * 60)
    print(f"  Timestamp     : {result['timestamp']}")
    print(f"  Horizon       : {result['horizon_h']}h forward")
    print(f"  Regime        : {result.get('regime', 'n/a')}")
    print(f"  Direction     : {result['direction']}")
    print(f"  Raw prob      : {result['raw_prob_pct']}%")
    scale = result.get("regime_scale")
    if scale is not None and abs(scale - 1.0) > 1e-6:
        print(f"  Regime weight : x{scale}  (raw -> {result['regime_adjusted_prob_pct']}%)")
    if dl := result.get("dl_forecast"):
        if dl.get("status") == "ok":
            print(f"  DL Prediction : {dl['direction']} ({dl['raw_prob_pct']}%)")
        else:
            print(f"  DL Prediction : {dl.get('status')} ({dl.get('verdict')})")

    print(f"  Calibrated    : {result['calibrated_prob_pct']}%")
    print(f"  Rules matched : {result['n_rules']}")

    print("\n  Atom states:")
    for atom, state in result["atom_states"].items():
        print(f"    {atom:<16} {state}")

    print("\n  Rules contributing:")
    for r in result["rules_used"][:10]:
        ev_short = r["event"] if len(r["event"]) <= 44 else r["event"][:41] + "..."
        print(f"    {ev_short:<46} {r['direction']:>4}  w={r['weight']:.3f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE calibrated forecast")
    parser.add_argument("csv", help="Path to OHLCV CSV (e.g. btc.csv)")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)

    result = run_forecast(args.csv)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_forecast(result)

    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
