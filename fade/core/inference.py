"""Fast read-only inference on the latest bar (no walk-forward)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.predictor import predict_calibrated, _rule_weight
from fade.memory import MemoryStore


def infer_latest(
    csv_path: str,
    config: Config | None = None,
) -> dict:
    """Direction + calibrated % for the most recent bar using frozen positive memory."""
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    memory = MemoryStore(config.memory_dir, asset=asset)
    calibration = CalibrationStore(config.memory_dir / f"calibration_{asset}.json")
    positive = memory.positive
    blocked = memory.blocked_events()

    if not positive:
        return {"asset": asset, "status": "no_rules"}

    df = load_ohlcv(csv_path)
    pool = atoms_mod.compute_atom_pool(df, config)
    atoms = pool[list(config.atom_columns)].dropna()
    thresholds = ev.compute_thresholds(atoms, config)
    disc = ev.discretize(atoms.iloc[[-1]], thresholds)
    events = ev.build_events(
        disc, config, allowed=set(positive.keys()), blocked=blocked,
    )
    if events.empty:
        return {
            "asset": asset,
            "status": "no_match",
            "timestamp": str(atoms.index[-1]),
            "atom_states": {c: disc[c].iloc[0] for c in config.atom_columns},
        }

    rules = pd.DataFrame(positive).T
    rules["direction"] = rules["direction"].astype(int)
    rules["confidence"] = rules["avg_oos_hit"].astype(float)
    preds = predict_calibrated(events, rules, calibration, positive)
    if preds.empty:
        return {"asset": asset, "status": "no_prediction", "timestamp": str(atoms.index[-1])}

    row = preds.iloc[-1]
    rules_used = []
    for event in events["event"].unique():
        if event not in rules.index:
            continue
        d, w = _rule_weight(event, rules, positive)
        rules_used.append({"event": event, "direction": "UP" if d == 1 else "DOWN", "weight": round(w, 4)})
    rules_used.sort(key=lambda r: r["weight"], reverse=True)

    return {
        "asset": asset,
        "status": "ok",
        "timestamp": str(preds.index[-1]),
        "direction": "UP" if row["pred"] == 1 else "DOWN",
        "pred": int(row["pred"]),
        "raw_prob_pct": round(float(row["raw_prob"]) * 100, 1),
        "calibrated_prob_pct": round(float(row["calibrated_prob"]) * 100, 1),
        "n_rules": int(row["n_rules"]),
        "atom_states": {c: disc[c].iloc[0] for c in config.atom_columns},
        "rules_used": rules_used,
    }
