"""Final lockbox — untouched newest data for a one-shot unbiased OOS estimate.

Problem: ~29 batches reused the same 70/30 holdout to pick atom sets, inflating
reported p-values (multiple-comparisons). This module seals the newest 15–20%
of each series (never used in any prior batch decision) with a checksum, mines
rules ONLY on data before the lockbox cut, and scores path_lean3 + conviction
PRIMARY once — no iteration, no tuning.

Run ONCE per asset, then treat the lockbox as burned:
    python -m fade.pipeline.final_lockbox
    python -m fade.pipeline.final_lockbox btc_1h.csv --lockbox-frac 0.18
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import lean_config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.conviction import TIER_DEFS
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.holdout import _select_stable_rules
from fade.pipeline.pnl_reality_check_v2 import (
    MULTI_IV,
    _primary_direction,
    _tier_at_row,
)
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("final_lockbox")

DEFAULT_LOCKBOX_FRAC = 0.18
MANIFEST_PATH = Path("fade/output/lockbox_manifest.json")
REFERENCE_HIT = 0.546  # path_lean3 btc_1h holdout from batch 17 (multiple-comp risk)


def _prefix(csv: str) -> str:
    stem = Path(csv).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _lockbox_hash(df: pd.DataFrame) -> str:
    """Deterministic checksum of the lockbox OHLCV slice."""
    payload = df[["open", "high", "low", "close", "volume"]].to_csv(index=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def seal_lockbox(csv_path: str, lockbox_frac: float = DEFAULT_LOCKBOX_FRAC) -> dict:
    """Compute and persist lockbox metadata (idempotent if already sealed)."""
    df = load_ohlcv(csv_path)
    n = len(df)
    cut = int(n * (1.0 - lockbox_frac))
    pre = df.iloc[:cut]
    lock = df.iloc[cut:]
    asset = Path(csv_path).stem

    meta = {
        "asset": asset,
        "csv_path": str(csv_path),
        "lockbox_frac": lockbox_frac,
        "n_total": n,
        "n_pre_lockbox": cut,
        "n_lockbox": len(lock),
        "cut_timestamp": str(pre.index[-1]) if len(pre) else None,
        "lockbox_start": str(lock.index[0]) if len(lock) else None,
        "lockbox_end": str(lock.index[-1]) if len(lock) else None,
        "sha256": _lockbox_hash(lock),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Never use for tuning after one-shot eval. Prior batches used 70/30 holdout only.",
    }
    return meta


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"version": 1, "lockboxes": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _path_lean3_lockbox(csv_path: str, lockbox_frac: float, n_shuffles: int = 300,
                        seed: int = 0) -> dict:
    """One-shot path_lean3 on sealed lockbox."""
    config = lean_config()
    rng = np.random.default_rng(seed)
    df = load_ohlcv(csv_path)
    n = len(df)
    cut = int(n * (1.0 - lockbox_frac))

    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)

    dev_atoms = atoms.iloc[:cut]
    lock_atoms = atoms.iloc[cut:]
    dev_fwd = fwd.iloc[:cut]
    lock_fwd = fwd.iloc[cut:]

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return {"status": "no_rules"}

    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(lock_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)
    if preds.empty:
        return {"status": "no_coverage"}

    valid = lock_fwd.reindex(preds.index).notna()
    pred_v = preds["pred"][valid].to_numpy()
    fwd_v = lock_fwd.reindex(preds.index)[valid].to_numpy()
    pred_sc, act_sc = score_predictions(pred_v, fwd_v, config.move_threshold)
    if len(pred_sc) == 0:
        return {"status": "no_coverage"}

    hit = float(np.mean(pred_sc == act_sc))
    null_hits = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null_hits[i] = np.mean(pred_sc == rng.permutation(act_sc))
    p_value = (1 + int(np.sum(null_hits >= hit))) / (1 + n_shuffles)

    return {
        "status": "ok",
        "hit_rate": round(hit, 4),
        "lift": round(hit - 0.5, 4),
        "coverage": int(len(pred_sc)),
        "p_value": round(p_value, 4),
        "n_rules": int(len(frozen)),
    }


def _conviction_lockbox(csv_1h: str, lockbox_frac: float) -> dict:
    """One-shot conviction + PRIMARY on lockbox bars."""
    prefix = _prefix(csv_1h)
    df = load_ohlcv(csv_1h)
    n = len(df)
    cut = int(n * (1.0 - lockbox_frac))

    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    frame = pd.DataFrame({"streak": streak, "slen": np.abs(streak), "fwd": fwd},
                         index=df.index)
    for iv in MULTI_IV:
        p = f"{prefix}_{iv}.csv"
        if Path(p).exists():
            frame[iv] = _contrarian_grid(p)
    frame = frame.dropna(subset=["fwd"])
    cols = [c for c in MULTI_IV if c in frame.columns]

    lock = frame.iloc[int(len(frame) * (1.0 - lockbox_frac)):]

    conv_rows, prim_rows = [], []
    for _ts, row in lock.iterrows():
        actual_up = int(row["fwd"] > 0)
        tid, conv_dir = _tier_at_row(row, cols)
        if conv_dir:
            conv_rows.append(int((conv_dir == "UP") == actual_up))
        prim_dir = _primary_direction(row, cols, conv_dir if tid else None)
        if prim_dir:
            prim_rows.append(int((prim_dir == "UP") == actual_up))

    out = {"status": "ok", "lockbox_bars": len(lock)}
    if conv_rows:
        out["conviction_hit"] = round(float(np.mean(conv_rows)), 4)
        out["conviction_n"] = len(conv_rows)
    else:
        out["conviction_hit"] = None
        out["conviction_n"] = 0
    if prim_rows:
        out["primary_hit"] = round(float(np.mean(prim_rows)), 4)
        out["primary_n"] = len(prim_rows)
    else:
        out["primary_hit"] = None
        out["primary_n"] = 0
    return out


def run_lockbox_eval(
    csv_path: str = "btc_1h.csv",
    lockbox_frac: float = DEFAULT_LOCKBOX_FRAC,
    seal_only: bool = False,
) -> dict:
    meta = seal_lockbox(csv_path, lockbox_frac)

    manifest = _load_manifest()
    existing = next((lb for lb in manifest.get("lockboxes", [])
                     if lb.get("asset") == meta["asset"]), None)
    if existing and existing.get("sha256") != meta["sha256"]:
        log.warning("Lockbox hash changed for %s — data was updated", meta["asset"])
    if not existing:
        manifest.setdefault("lockboxes", []).append(meta)
        _save_manifest(manifest)

    if seal_only:
        return {"status": "sealed", "meta": meta}

    path_r = _path_lean3_lockbox(csv_path, lockbox_frac)
    conv_r = _conviction_lockbox(csv_path, lockbox_frac)

    hit = path_r.get("hit_rate")
    delta_ref = round(hit - REFERENCE_HIT, 4) if hit is not None else None
    if hit is not None:
        if hit < REFERENCE_HIT - 0.01:
            oos_tag = "SUSPICIOUSLY_LOW — likely multiple-comparisons inflation in prior 54.6%"
        elif hit >= REFERENCE_HIT - 0.005:
            oos_tag = "CONSISTENT with prior holdout (within 0.5pp)"
        else:
            oos_tag = "MODESTLY_BELOW prior holdout — some inflation possible"
    else:
        oos_tag = "INCONCLUSIVE"

    return {
        "status": "ok",
        "meta": meta,
        "path_lean3": path_r,
        "conviction": conv_r,
        "reference_holdout_hit": REFERENCE_HIT,
        "delta_vs_reference": delta_ref,
        "true_unbiased_oos_tag": oos_tag,
        "verdict": (
            f"TRUE UNBIASED OOS: path_lean3 {hit} on {path_r.get('coverage', 0)} bars "
            f"(p={path_r.get('p_value')}). {oos_tag}."
            if hit is not None else path_r.get("status", "failed")
        ),
    }


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print("FADE FINAL LOCKBOX — one-shot unbiased OOS")
    print(line)
    if r.get("status") == "sealed":
        m = r["meta"]
        print(f"  Sealed {m['asset']}: {m['n_lockbox']} bars")
        print(f"  Range: {m['lockbox_start']} -> {m['lockbox_end']}")
        print(f"  SHA256: {m['sha256'][:16]}...")
        print(line + "\n")
        return

    m = r["meta"]
    print(f"  Asset     : {m['asset']}")
    print(f"  Lockbox   : {m['n_lockbox']:,} bars ({m['lockbox_frac']*100:.0f}% newest)")
    print(f"  Range     : {m['lockbox_start']} -> {m['lockbox_end']}")
    print(f"  Cut date  : {m['cut_timestamp']}  SHA256: {m['sha256'][:16]}...")
    print()

    pl = r.get("path_lean3", {})
    if pl.get("status") == "ok":
        print(f"  path_lean3 (TRUE OOS):")
        print(f"    hit-rate  : {pl['hit_rate']}  lift {pl['lift']:+.4f}")
        print(f"    coverage  : {pl['coverage']:,}  p-value {pl['p_value']}")
        print(f"    vs prior  : {r['reference_holdout_hit']} (70/30 holdout, batch 17)")
        print(f"    delta     : {r['delta_vs_reference']:+.4f}")
    else:
        print(f"  path_lean3: {pl.get('status')}")

    cv = r.get("conviction", {})
    print(f"\n  conviction stack (lockbox):")
    print(f"    conviction : {cv.get('conviction_hit')}  n={cv.get('conviction_n')}")
    print(f"    PRIMARY    : {cv.get('primary_hit')}  n={cv.get('primary_n')}")

    print()
    print(f"  TAG: {r['true_unbiased_oos_tag']}")
    print(line)
    print(f"  {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE final lockbox one-shot eval")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--lockbox-frac", type=float, default=DEFAULT_LOCKBOX_FRAC)
    parser.add_argument("--seal-only", action="store_true")
    parser.add_argument("--also", nargs="*", default=[], help="Additional CSVs (e.g. eth_1h.csv)")
    args = parser.parse_args()

    paths = [args.csv] + list(args.also)
    for csv in paths:
        if not Path(csv).exists():
            log.error("File not found: %s", csv)
            sys.exit(1)
        _print(run_lockbox_eval(csv, lockbox_frac=args.lockbox_frac, seal_only=args.seal_only))


if __name__ == "__main__":
    main()
