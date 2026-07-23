"""Thesis-hold — enter on signal, exit on TP / SL / invalidation (new method).

Motivation (north star): path_lean3 dies on fees because it churns every bar.
Fixed min_hold helps but is a blunt timer. Thesis-hold keeps a directional
thesis until the market pays a fee-multiple target, hits a stop, times out,
or the signal flips after a short lock.

Fixed small config set (not a fishing grid). Exploratory holdout only —
NOT forward validation, NOT production.

Run:
    python -m fade.pipeline.thesis_hold_holdout
    python -m fade.pipeline.thesis_hold_holdout btc_1h.csv eth_1h.csv --fee-bps 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fade.core.conviction import HOLDOUT_FRAC
from fade.pipeline.pnl_reality_check_v2 import (
    _holdout_path_lean3,
    _min_hold_positions,
)
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.utils.logging import get_logger

log = get_logger("thesis_hold")

STUDY_ID = "thesis_hold_v1"
OUTPUT = Path("fade/output/thesis_hold_holdout.json")

# Theory-first configs. TP ≥ ~2× round-trip fee (10 bps @ 5+5).
# v2 adds cooldown + enter-on-flip after v1 proved always-in churn.
CONFIGS = (
    {
        "name": "thesis_v1",
        "tp_bps": 30.0,
        "sl_bps": 45.0,
        "max_hold": 48,
        "min_hold": 6,
        "cooldown": 0,
        "entry_on_flip": False,
        "conf_min": None,
    },
    {
        "name": "thesis_v2_cooldown",
        "tp_bps": 30.0,
        "sl_bps": 45.0,
        "max_hold": 48,
        "min_hold": 6,
        "cooldown": 12,
        "entry_on_flip": True,
        "conf_min": None,
    },
    {
        "name": "thesis_v2_conf_cool",
        "tp_bps": 30.0,
        "sl_bps": 45.0,
        "max_hold": 48,
        "min_hold": 6,
        "cooldown": 12,
        "entry_on_flip": True,
        "conf_min": 0.55,
    },
)


def _thesis_positions(
    signal: np.ndarray,
    bar_ret: np.ndarray,
    *,
    tp: float,
    sl: float,
    max_hold: int,
    min_hold: int,
    cooldown: int = 0,
    entry_on_flip: bool = False,
) -> tuple[np.ndarray, dict]:
    """Build position series with TP/SL/timeout/invalidation exits.

    ``signal`` is desired direction in {-1, 0, +1} at bar t (known at t).
    ``bar_ret[t]`` is the return from t -> t+1 (realized after holding through t).
    Cumulative PnL for the open thesis is tracked in return space (pre-fee);
    fees are applied later via ``_equity`` on position changes.
    """
    n = len(signal)
    pos = np.zeros(n, dtype=float)
    cur = 0.0
    bars_held = 0
    cum = 0.0
    cool_left = 0
    prev_sig = 0.0
    exits = {"tp": 0, "sl": 0, "timeout": 0, "invalidate": 0, "entries": 0}

    for i in range(n):
        sig = float(signal[i])
        flipped = sig != 0.0 and sig != prev_sig

        if cur == 0.0:
            if cool_left > 0:
                cool_left -= 1
                pos[i] = 0.0
                prev_sig = sig
                continue
            can_enter = sig != 0.0 and (flipped if entry_on_flip else True)
            if can_enter:
                cur = sig
                bars_held = 0
                cum = 0.0
                exits["entries"] += 1
            pos[i] = cur
            if cur != 0.0:
                cum += cur * bar_ret[i]
                bars_held += 1
            prev_sig = sig
            continue

        pos[i] = cur
        cum += cur * bar_ret[i]
        bars_held += 1

        exit_now = False
        reason = None
        if bars_held >= min_hold:
            if cum >= tp:
                exit_now, reason = True, "tp"
            elif cum <= -sl:
                exit_now, reason = True, "sl"
        if not exit_now and bars_held >= max_hold:
            exit_now, reason = True, "timeout"
        if not exit_now and bars_held >= min_hold:
            if sig != 0.0 and sig != cur:
                exit_now, reason = True, "invalidate"

        if exit_now:
            exits[reason] += 1
            cur = 0.0
            bars_held = 0
            cum = 0.0
            cool_left = max(0, int(cooldown))

        prev_sig = sig

    return pos, exits


def _signal_from_preds(preds, conf_min: float | None) -> np.ndarray:
    pred_up = preds["pred"].to_numpy().astype(int)
    raw = np.where(pred_up == 1, 1.0, -1.0)
    if conf_min is None or "calibrated_prob" not in preds.columns:
        return raw
    prob = preds["calibrated_prob"].to_numpy()
    # Distance from 0.5 as conviction; trade only if far enough.
    edge = np.abs(prob - 0.5)
    need = abs(conf_min - 0.5) if conf_min >= 0.5 else conf_min
    return np.where(edge >= need, raw, 0.0)


def run_asset(csv_path: str, fee_bps: float, holdout_frac: float) -> dict:
    got = _holdout_path_lean3(csv_path, holdout_frac)
    if got is None:
        return {"status": "no_rules", "asset": Path(csv_path).stem}

    preds, _split, n_rules = got
    bar_ret = preds["bar_ret"].to_numpy()
    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4

    bh = _equity(np.ones(len(bar_ret)), bar_ret, 0.0, 0.0)
    buy_hold = _stats(bh["strat_ret"], bh["equity"], bpy)

    # Baselines
    raw = np.where(preds["pred"].to_numpy().astype(int) == 1, 1.0, -1.0)
    base = {}
    for name, positions in (
        ("long_short_every_bar", raw),
        ("min_hold_48", _min_hold_positions(raw, 48)),
    ):
        e = _equity(positions, bar_ret, fee_rate, 0.0)
        base[name] = {
            **_stats(e["strat_ret"], e["equity"], bpy),
            "n_changes": e["n_changes"],
            "cost_drag": round(e["total_cost"], 4),
            "active_bars": int(np.sum(positions != 0)),
        }

    methods = []
    for cfg in CONFIGS:
        sig = _signal_from_preds(preds, cfg["conf_min"])
        tp = cfg["tp_bps"] / 1e4
        sl = cfg["sl_bps"] / 1e4
        pos, exits = _thesis_positions(
            sig, bar_ret,
            tp=tp, sl=sl,
            max_hold=cfg["max_hold"],
            min_hold=cfg["min_hold"],
            cooldown=int(cfg.get("cooldown") or 0),
            entry_on_flip=bool(cfg.get("entry_on_flip")),
        )
        e = _equity(pos, bar_ret, fee_rate, 0.0)
        methods.append({
            "config": cfg,
            "exits": exits,
            "active_bars": int(np.sum(pos != 0)),
            "coverage": round(float(np.mean(pos != 0)), 4),
            **_stats(e["strat_ret"], e["equity"], bpy),
            "n_changes": e["n_changes"],
            "cost_drag": round(e["total_cost"], 4),
            "beats_buy_hold": bool(
                e["equity"][-1] - 1.0 > buy_hold["total_return"]
            ),
            "beats_min_hold_48": bool(
                e["equity"][-1] - 1.0 > base["min_hold_48"]["total_return"]
            ),
        })

    best = max(methods, key=lambda m: m["total_return"])
    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "n_rules": n_rules,
        "holdout_bars": len(bar_ret),
        "fee_bps_per_side": fee_bps,
        "buy_and_hold": buy_hold,
        "baselines": base,
        "methods": methods,
        "best_method": best["config"]["name"],
        "best_total_return": best["total_return"],
    }


def run(csvs: list[str], fee_bps: float = 5.0, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    assets = [run_asset(c, fee_bps, holdout_frac) for c in csvs]

    any_beat_bh = False
    any_beat_mh = False
    for a in assets:
        if a.get("status") != "ok":
            continue
        for m in a["methods"]:
            if m["beats_buy_hold"] and m["n_changes"] >= 20:
                any_beat_bh = True
            if m["beats_min_hold_48"] and m["n_changes"] >= 20:
                any_beat_mh = True

    if any_beat_bh:
        verdict = "PROMISING_EXPLORATORY"
        verdict_he = (
            "Thesis-hold מכה buy&hold ב-holdout לפחות על נכס אחד — "
            "עדיין exploratory; צריך pre-reg + lockbox לפני כל טענה."
        )
    elif any_beat_mh:
        verdict = "BETTER_THAN_TIMER"
        verdict_he = (
            "משתפר מול min_hold=48 אך לא מול buy&hold — שיפור טכני, לא מספיק לכוכב הצפון."
        )
    else:
        verdict = "REJECT_FOR_NOW"
        verdict_he = (
            "Thesis-hold לא מכה את הבסיסים החשובים @5bps — לא לקדם; לקבור או לשנות תזה."
        )

    return {
        "study_id": STUDY_ID,
        "sandbox": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method_he": (
            "כניסה לפי כיוון path_lean3; יציאה ביעד רווח / סטופ / פג־תוקף / אות הפוך — "
            "במקום היפוך כל נר או טיימר בלבד."
        ),
        "north_star": "predict → trade → net profit",
        "protocol": {
            "split": f"holdout last {holdout_frac:.0%}, rules frozen on dev",
            "fee_bps_per_side": fee_bps,
            "n_configs": len(CONFIGS),
            "warning_he": "exploratory — not lockbox, not forward",
        },
        "configs": list(CONFIGS),
        "assets": assets,
        "verdict": verdict,
        "verdict_he": verdict_he,
    }


def _print(r: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"THESIS-HOLD HOLDOUT  fee={r['protocol']['fee_bps_per_side']}bps/side")
    print(f"{'=' * 70}")
    print(f"  {r['method_he']}")
    for a in r["assets"]:
        if a.get("status") != "ok":
            print(f"\n  {a.get('asset')}: {a.get('status')}")
            continue
        bh = a["buy_and_hold"]["total_return"] * 100
        print(f"\n  {a['asset']}  bars={a['holdout_bars']}  buy&hold={bh:+.1f}%")
        for name, b in a["baselines"].items():
            print(
                f"    baseline {name}: {b['total_return']*100:+.1f}% "
                f"trades={b['n_changes']} drag={b['cost_drag']}"
            )
        for m in a["methods"]:
            flag = []
            if m["beats_buy_hold"]:
                flag.append("vsBH")
            if m["beats_min_hold_48"]:
                flag.append("vsMH48")
            tags = ",".join(flag) if flag else "-"
            ex = m["exits"]
            print(
                f"    {m['config']['name']}: {m['total_return']*100:+.1f}% "
                f"sharpe={m.get('sharpe')} trades={m['n_changes']} "
                f"cov={m['coverage']} [{tags}] "
                f"exits tp={ex['tp']} sl={ex['sl']} to={ex['timeout']} inv={ex['invalidate']}"
            )
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  {r['verdict_he']}")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-hold holdout (exploratory)")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    result = run(args.csv, fee_bps=args.fee_bps, holdout_frac=args.holdout_frac)
    _print(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
