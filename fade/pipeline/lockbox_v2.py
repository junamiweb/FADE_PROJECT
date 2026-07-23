"""Lockbox v2 — blind seal first, one-shot reveal only on explicit burn.

Protocol (north star — real success only):
  1. Pre-register candidate (already in pre_registration.json).
  2. ``seal`` — bind a future/unseen OHLCV window by SHA256. Prints NO PnL.
  3. Wait until enough bars accumulate (refresh / Action).
  4. ``seal`` again — freezes the window when ``min_bars`` met.
  5. ``reveal`` — one-shot eval; burns the seal. Requires explicit flag.

Blind window starts AFTER ``--after`` (default: now at first seal), so prior
holdout peeks cannot contaminate the test.

Run:
    python -m fade.pipeline.lockbox_v2 status
    python -m fade.pipeline.lockbox_v2 seal eth_1h.csv --track eth_path_lean3_minhold48
    python -m fade.pipeline.lockbox_v2 seal eth_1h.csv --track eth_path_lean3_minhold48
    python -m fade.pipeline.lockbox_v2 reveal --track eth_path_lean3_minhold48 --i-understand-burn
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

from fade.core.data_loader import load_ohlcv
from fade.pipeline.pnl_reality_check_v2 import (
    _holdout_path_lean3,
    _min_hold_positions,
)
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.pipeline.pre_registration import get_candidate, load_manifest, save_manifest
from fade.utils.logging import get_logger

log = get_logger("lockbox_v2")

MANIFEST_PATH = Path("fade/output/lockbox_v2_manifest.json")
DEFAULT_TRACK = "eth_path_lean3_minhold48"
DEFAULT_MIN_BARS = 500  # ~21 days of 1h bars


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _load() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version": 2,
        "protocol_he": (
            "seal בלי תוצאות → צבירה → seal מקפיא hash → reveal חד-פעמי שורף"
        ),
        "seals": [],
    }


def _save(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")


def _slice_hash(df: pd.DataFrame) -> str:
    payload = df[["open", "high", "low", "close", "volume"]].to_csv(index=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _find_seal(m: dict, track_id: str, asset: str) -> dict | None:
    """Active seal only (WAITING / SEALED). Burned seals are history."""
    active = None
    for s in m.get("seals", []):
        if s.get("track_id") == track_id and s.get("asset") == asset:
            if s.get("status") in ("WAITING", "SEALED"):
                active = s
    return active


def cmd_status(track_id: str | None) -> dict:
    m = _load()
    seals = m.get("seals", [])
    if track_id:
        seals = [s for s in seals if s.get("track_id") == track_id]
    return {"status": "ok", "seals": seals, "manifest": str(MANIFEST_PATH)}


def cmd_seal(
    csv_path: str,
    *,
    track_id: str,
    after: str | None,
    min_bars: int,
) -> dict:
    cand = get_candidate(track_id)
    if cand is None:
        return {
            "status": "error",
            "error": f"track_id not pre-registered: {track_id}",
            "hint": "register in fade/output/pre_registration.json first",
        }

    df = load_ohlcv(csv_path)
    asset = Path(csv_path).stem
    m = _load()
    seal = _find_seal(m, track_id, asset)

    if seal and seal.get("status") == "SEALED":
        return {
            "status": "already_sealed",
            "seal": {
                k: seal[k] for k in (
                    "track_id", "asset", "status", "n_lockbox",
                    "lockbox_start", "lockbox_end", "sha256", "sealed_at_utc",
                )
            },
            "note_he": "כבר חתום — ממתין ל-reveal (אוטונומי או ידני)",
        }

    if seal is None:
        after_ts = _parse_ts(after) if after else pd.Timestamp(_utc_now())
        seal = {
            "track_id": track_id,
            "asset": asset,
            "csv_path": str(csv_path),
            "status": "WAITING",
            "blind_after_utc": str(after_ts),
            "min_bars": int(min_bars),
            "candidate_config": cand.get("config"),
            "created_at_utc": _utc_now().isoformat(),
            "note_he": (
                "חלון עיוור מתחיל אחרי blind_after — בלי PnL עד reveal"
            ),
        }
        m.setdefault("seals", []).append(seal)
    else:
        after_ts = _parse_ts(seal["blind_after_utc"])

    window = df[df.index > after_ts]
    n = len(window)
    seal["n_available"] = n
    seal["available_start"] = str(window.index[0]) if n else None
    seal["available_end"] = str(window.index[-1]) if n else None
    seal["last_seal_check_utc"] = _utc_now().isoformat()

    if n < int(seal["min_bars"]):
        seal["status"] = "WAITING"
        seal["bars_still_needed"] = int(seal["min_bars"]) - n
        _save(m)
        _sync_prereg_lockbox_v2(m)
        return {
            "status": "WAITING",
            "track_id": track_id,
            "asset": asset,
            "blind_after_utc": seal["blind_after_utc"],
            "n_available": n,
            "min_bars": seal["min_bars"],
            "bars_still_needed": seal["bars_still_needed"],
            "available_end": seal["available_end"],
            "metrics_revealed": False,
            "note_he": (
                f"עדיין לא מספיק נרות עיוורים ({n}/{seal['min_bars']}). "
                "המשך refresh / Action — בלי לראות תוצאות."
            ),
        }

    # Freeze exact window now (do not keep growing after SEALED).
    sha = _slice_hash(window)
    seal.update({
        "status": "SEALED",
        "n_lockbox": n,
        "lockbox_start": str(window.index[0]),
        "lockbox_end": str(window.index[-1]),
        "sha256": sha,
        "sealed_at_utc": _utc_now().isoformat(),
        "bars_still_needed": 0,
    })
    _save(m)
    _sync_prereg_lockbox_v2(m)
    return {
        "status": "SEALED",
        "track_id": track_id,
        "asset": asset,
        "n_lockbox": n,
        "lockbox_start": seal["lockbox_start"],
        "lockbox_end": seal["lockbox_end"],
        "sha256": sha,
        "metrics_revealed": False,
        "note_he": (
            "נחתם. אסור לכוון פרמטרים. "
            "כדי לגלות תוצאה: reveal --i-understand-burn"
        ),
    }


def _sync_prereg_lockbox_v2(m: dict) -> None:
    pr = load_manifest()
    seals = m.get("seals", [])
    active = [s for s in seals if s.get("status") in ("WAITING", "SEALED")]
    pr.setdefault("lockboxes", {})["v2"] = {
        "status": active[0]["status"] if active else "RESERVED",
        "manifest_file": str(MANIFEST_PATH),
        "seals_summary": [
            {
                "track_id": s.get("track_id"),
                "asset": s.get("asset"),
                "status": s.get("status"),
                "n_available": s.get("n_available"),
                "n_lockbox": s.get("n_lockbox"),
                "sha256_prefix": (s.get("sha256") or "")[:16] or None,
            }
            for s in seals
        ],
        "updated_utc": _utc_now().isoformat(),
    }
    save_manifest(pr)


def _verify_hash(seal: dict) -> tuple[bool, str]:
    df = load_ohlcv(seal["csv_path"])
    after = _parse_ts(seal["blind_after_utc"])
    end = _parse_ts(seal["lockbox_end"])
    window = df[(df.index > after) & (df.index <= end)]
    if len(window) != int(seal["n_lockbox"]):
        return False, f"bar count changed: {len(window)} vs {seal['n_lockbox']}"
    sha = _slice_hash(window)
    if sha != seal.get("sha256"):
        return False, "sha256 mismatch — data altered after seal"
    return True, sha


def _eval_minhold_pnl(
    csv_path: str,
    *,
    lock_start: pd.Timestamp,
    lock_end: pd.Timestamp,
    min_hold: int,
    fee_bps: float,
) -> dict:
    """Mine on all bars before lock_start; score PnL only inside sealed window."""
    df = load_ohlcv(csv_path)
    # Use holdout helper on full series then restrict — mine on pre-lock only:
    # _holdout_path_lean3 uses last holdout_frac; we need custom cut.
    from fade.config import lean_config
    from fade.core import atoms as atoms_mod
    from fade.core import events as ev
    from fade.core.calibration import CalibrationStore
    from fade.core.predictor import collect_calibration_samples, predict_calibrated
    from fade.pipeline.backtest import walk_forward
    from fade.pipeline.holdout import _select_stable_rules

    config = lean_config()
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    close = df["close"].reindex(atoms.index)

    pre_mask = atoms.index <= lock_start
    lock_mask = (atoms.index > lock_start) & (atoms.index <= lock_end)
    dev_atoms = atoms.loc[pre_mask]
    lock_atoms = atoms.loc[lock_mask]
    if len(dev_atoms) < 500 or len(lock_atoms) < 50:
        return {"status": "insufficient_bars", "n_dev": len(dev_atoms), "n_lock": len(lock_atoms)}

    dev_fwd = fwd.loc[dev_atoms.index]
    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return {"status": "no_rules"}

    cal = CalibrationStore(config.cache_dir / "_lockbox_v2_cal.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}
    thresholds = ev.compute_thresholds(dev_atoms, config)
    dev_disc = ev.discretize(dev_atoms, thresholds)
    dev_events = ev.build_events(dev_disc, config, allowed=set(frozen.index))
    dev_preds = predict_calibrated(dev_events, frozen, cal, positive={})
    if not dev_preds.empty:
        samples = collect_calibration_samples(dev_preds, dev_fwd)
        if samples:
            cal.update(samples)

    lock_disc = ev.discretize(lock_atoms, thresholds)
    lock_events = ev.build_events(lock_disc, config, allowed=set(frozen.index))
    preds = predict_calibrated(lock_events, frozen, cal, positive={})
    if preds.empty:
        return {"status": "no_preds"}

    lock_close = close.loc[lock_atoms.index]
    bar_ret = lock_close.pct_change().shift(-1)
    out = preds.join(bar_ret.rename("bar_ret")).dropna(subset=["bar_ret"])
    if out.empty:
        return {"status": "no_scored_bars"}

    bar_ret_a = out["bar_ret"].to_numpy()
    raw = np.where(out["pred"].to_numpy().astype(int) == 1, 1.0, -1.0)
    pos = _min_hold_positions(raw, min_hold)
    fee_rate = fee_bps / 1e4
    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    e = _equity(pos, bar_ret_a, fee_rate, 0.0)
    st = _stats(e["strat_ret"], e["equity"], bpy)
    bh = _equity(np.ones(len(bar_ret_a)), bar_ret_a, 0.0, 0.0)
    bh_st = _stats(bh["strat_ret"], bh["equity"], bpy)

    passed = (
        st["total_return"] > 0
        and st["total_return"] > bh_st["total_return"]
        and e["n_changes"] >= 30
        and (st.get("sharpe") or 0) > 0
    )
    return {
        "status": "ok",
        "n_rules": len(frozen),
        "n_scored": len(out),
        "min_hold": min_hold,
        "fee_bps_per_side": fee_bps,
        "strategy": {**st, "n_changes": e["n_changes"], "cost_drag": round(e["total_cost"], 4)},
        "buy_and_hold": bh_st,
        "success_contract_pass": passed,
    }


def cmd_reveal(
    track_id: str,
    *,
    confirm_burn: bool,
    asset: str | None = None,
    autonomous: bool = False,
) -> dict:
    if not confirm_burn and not autonomous:
        return {
            "status": "refused",
            "error": "pass --i-understand-burn to reveal and burn the seal",
            "metrics_revealed": False,
        }

    m = _load()
    seals = [s for s in m.get("seals", []) if s.get("track_id") == track_id]
    if asset:
        seals = [s for s in seals if s.get("asset") == asset]
    if not seals:
        return {"status": "error", "error": "no seal for track/asset"}
    # Prefer SEALED seal for this asset; else last matching
    sealed = [s for s in seals if s.get("status") == "SEALED"]
    seal = sealed[-1] if sealed else seals[-1]
    if seal.get("status") != "SEALED":
        return {
            "status": "error",
            "error": f"seal status is {seal.get('status')}, need SEALED",
            "hint": "run seal until SEALED",
            "asset": seal.get("asset"),
        }

    ok, info = _verify_hash(seal)
    if not ok:
        return {"status": "error", "error": f"integrity fail: {info}"}

    cand = get_candidate(track_id) or {}
    cfg = cand.get("config") or {}
    min_hold = int(cfg.get("min_hold_bars") or 48)
    fee_bps = float((cand.get("validation") or {}).get("fee_bps") or 5.0)

    result = _eval_minhold_pnl(
        seal["csv_path"],
        lock_start=_parse_ts(seal["blind_after_utc"]),
        lock_end=_parse_ts(seal["lockbox_end"]),
        min_hold=min_hold,
        fee_bps=fee_bps,
    )

    seal["status"] = "BURNED"
    seal["burned_at_utc"] = _utc_now().isoformat()
    seal["burn_result_summary"] = {
        "success_contract_pass": result.get("success_contract_pass"),
        "strategy_total_return": (result.get("strategy") or {}).get("total_return"),
        "buy_hold_total_return": (result.get("buy_and_hold") or {}).get("total_return"),
        "n_changes": (result.get("strategy") or {}).get("n_changes"),
    }
    out_path = Path(f"fade/output/lockbox_v2_reveal_{seal['asset']}.json")
    seal["reveal_result_path"] = str(out_path)
    seal["autonomous_reveal"] = bool(autonomous)
    _save(m)
    _sync_prereg_lockbox_v2(m)

    out = {
        "status": "BURNED",
        "track_id": track_id,
        "autonomous": bool(autonomous),
        "seal": {
            "asset": seal["asset"],
            "n_lockbox": seal["n_lockbox"],
            "lockbox_start": seal["lockbox_start"],
            "lockbox_end": seal["lockbox_end"],
            "sha256": seal["sha256"],
        },
        "result": result,
        "metrics_revealed": True,
        "note_he": "החלון שרוף — אסור בדיקה נוספת על אותו חלון.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _print(obj: dict) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE lockbox v2 blind seal / reveal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show seals (no metrics)")
    p_status.add_argument("--track", default=None)

    p_seal = sub.add_parser("seal", help="Seal or update waiting blind window (no PnL)")
    p_seal.add_argument("csv", nargs="?", default="eth_1h.csv")
    p_seal.add_argument("--track", default=DEFAULT_TRACK)
    p_seal.add_argument(
        "--after", default=None,
        help="Blind window starts after this UTC time (default: now on first seal)",
    )
    p_seal.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS)

    p_reveal = sub.add_parser("reveal", help="One-shot eval + burn (explicit)")
    p_reveal.add_argument("--track", default=DEFAULT_TRACK)
    p_reveal.add_argument("--asset", default=None, help="Asset stem, e.g. eth_1h")
    p_reveal.add_argument(
        "--i-understand-burn", action="store_true",
        help="Required unless --autonomous",
    )
    p_reveal.add_argument(
        "--autonomous", action="store_true",
        help="Allow burn without interactive flag (Action / blind_autonomy)",
    )

    args = parser.parse_args()
    if args.cmd == "status":
        _print(cmd_status(args.track))
    elif args.cmd == "seal":
        if not Path(args.csv).exists():
            print(json.dumps({"status": "error", "error": f"missing {args.csv}"}))
            sys.exit(1)
        _print(cmd_seal(
            args.csv,
            track_id=args.track,
            after=args.after,
            min_bars=args.min_bars,
        ))
    elif args.cmd == "reveal":
        out = cmd_reveal(
            args.track,
            confirm_burn=args.i_understand_burn,
            asset=args.asset,
            autonomous=args.autonomous,
        )
        _print(out)
        if out.get("status") in ("refused", "error"):
            sys.exit(1)


if __name__ == "__main__":
    main()
