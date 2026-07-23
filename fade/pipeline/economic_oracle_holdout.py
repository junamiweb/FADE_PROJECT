"""Economic oracle — trade only when past same-state edge clears fees.

Success contract (hard — north star):
  PASS only if on chronological holdout @ 5 bps/side ALL hold:
    1) total_return > 0
    2) total_return > buy_and_hold
    3) n_trades >= 30
    4) sharpe > 0
  Otherwise FAIL. No vanity hit-rate wins.

Method (no ML):
  Discrete causal state = (streak_bucket, VR_regime).
  At bar t, look only at prior bars in the same state; take mean next-bar
  return. Enter long/short only if that mean clears ``k × round-trip fee``.
  Optional min_hold to stop microscopic churn.

Exploratory holdout. Lockbox only if PASS — and only after separate seal.

Run:
    python -m fade.pipeline.economic_oracle_holdout
    python -m fade.pipeline.economic_oracle_holdout btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import lean_config
from fade.core.conviction import HOLDOUT_FRAC
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.pnl_reality_check_v2 import _min_hold_positions
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("economic_oracle")

STUDY_ID = "economic_oracle_v2"
OUTPUT = Path("fade/output/economic_oracle_holdout.json")

# v1 failed honestly: next-bar state means (~0-6 bps) never clear 10 bps RT fees.
# v2 estimates mean FORWARD RETURN over H bars (same horizon we hold).
# Primary: H=12, must clear 2× RT fees. Sensitivity is secondary.
PRIMARY = {
    "name": "oracle_H12_k2_n80",
    "horizon": 12,
    "k_fee": 2.0,
    "min_n": 80,
}
SENSITIVITY = (
    {"name": "sens_H24_k2", "horizon": 24, "k_fee": 2.0, "min_n": 80},
    {"name": "sens_H12_k15", "horizon": 12, "k_fee": 1.5, "min_n": 80},
    {"name": "sens_H48_k2", "horizon": 48, "k_fee": 2.0, "min_n": 80},
)


def _streak_bucket(streak: np.ndarray) -> np.ndarray:
    """Causal streak buckets: strong down / mild / strong up."""
    out = np.empty(len(streak), dtype=object)
    out[:] = "mild"
    out[streak <= -3] = "strong_down"
    out[streak >= 3] = "strong_up"
    return out


def _build_frame(csv_path: str, holdout_frac: float, horizon: int) -> tuple[pd.DataFrame, int]:
    cfg = lean_config()
    df = load_ohlcv(csv_path)
    close = df["close"]
    ret = close.pct_change()
    bar_ret = ret.shift(-1)  # 1-bar return for equity simulation
    # Multi-bar forward return used for economic decision (known only after H bars).
    fwd_h = close.shift(-horizon) / close - 1.0
    streak = _signed_streak(ret.fillna(0.0).to_numpy())
    vr = compute_vol_ratio(
        ret, cfg.vol_ratio_short_window, cfg.vol_ratio_long_window, shift=1,
    )
    n = len(df)
    split = int(n * (1.0 - holdout_frac))
    dev_vr = vr.iloc[:split].dropna()
    low = float(dev_vr.quantile(1 / 3))
    high = float(dev_vr.quantile(2 / 3))
    regime = assign_vr_regime(vr, low, high)

    frame = pd.DataFrame({
        "bar_ret": bar_ret,
        "fwd_h": fwd_h,
        "streak_bucket": _streak_bucket(streak),
        "vr": regime.to_numpy(),
    }, index=df.index)
    frame["state"] = (
        frame["streak_bucket"].astype(str) + "|" + frame["vr"].astype(str)
    )
    return frame, split


def _oracle_signal(
    frame: pd.DataFrame,
    split: int,
    *,
    horizon: int,
    k_fee: float,
    min_n: int,
    fee_rt: float,
) -> np.ndarray:
    """Expanding-state mean H-bar return; trade only if |mu| >= k * fee_rt."""
    n = len(frame)
    signal = np.zeros(n, dtype=float)
    fwd = frame["fwd_h"].to_numpy()
    states = frame["state"].to_numpy()
    completed: dict[str, list[float]] = defaultdict(list)

    for i in range(n):
        s = str(states[i])
        past = completed[s]
        if i >= split and len(past) >= min_n:
            mu = float(np.mean(past))
            thresh = k_fee * fee_rt
            if mu >= thresh:
                signal[i] = 1.0
            elif mu <= -thresh:
                signal[i] = -1.0
        # Outcome of bar i-horizon becomes known at bar i (fwd_h[i-horizon]).
        j = i - horizon
        if j >= 0:
            s_j = str(states[j])
            r_j = fwd[j]
            if np.isfinite(r_j) and "nan" not in s_j:
                completed[s_j].append(float(r_j))

    return signal


def _passes_contract(stats: dict, buy_hold: float, n_changes: int) -> dict:
    checks = {
        "total_return_positive": stats["total_return"] > 0,
        "beats_buy_and_hold": stats["total_return"] > buy_hold,
        "enough_trades": n_changes >= 30,
        "sharpe_positive": (stats.get("sharpe") or 0) > 0,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_asset(csv_path: str, fee_bps: float, holdout_frac: float) -> dict:
    fee_rt = 2.0 * fee_bps / 1e4
    fee_rate = fee_bps / 1e4
    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)

    configs = (PRIMARY,) + SENSITIVITY
    methods = []
    buy_hold = None
    holdout_bars = None

    for cfg in configs:
        frame, split = _build_frame(csv_path, holdout_frac, cfg["horizon"])
        hold = frame.iloc[split:].dropna(subset=["bar_ret"])
        bar_ret_h = hold["bar_ret"].to_numpy()
        if buy_hold is None:
            bh_e = _equity(np.ones(len(bar_ret_h)), bar_ret_h, 0.0, 0.0)
            buy_hold = _stats(bh_e["strat_ret"], bh_e["equity"], bpy)
            holdout_bars = len(bar_ret_h)

        full_sig = _oracle_signal(
            frame, split,
            horizon=cfg["horizon"],
            k_fee=cfg["k_fee"],
            min_n=cfg["min_n"],
            fee_rt=fee_rt,
        )
        sig_h = (
            pd.Series(full_sig, index=frame.index)
            .reindex(hold.index).fillna(0.0).to_numpy()
        )
        # Hold for the same horizon the edge was estimated on.
        pos = _min_hold_positions(sig_h, cfg["horizon"])
        e = _equity(pos, bar_ret_h, fee_rate, 0.0)
        st = _stats(e["strat_ret"], e["equity"], bpy)
        contract = _passes_contract(st, buy_hold["total_return"], e["n_changes"])
        methods.append({
            "config": cfg,
            "primary": cfg["name"] == PRIMARY["name"],
            **st,
            "n_changes": e["n_changes"],
            "cost_drag": round(e["total_cost"], 4),
            "active_bars": int(np.sum(pos != 0)),
            "coverage": round(float(np.mean(pos != 0)), 4),
            "signal_bars": int(np.sum(sig_h != 0)),
            "contract": contract,
        })

    primary = next(m for m in methods if m["primary"])
    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "holdout_bars": holdout_bars,
        "fee_bps_per_side": fee_bps,
        "fee_rt": fee_rt,
        "buy_and_hold": buy_hold,
        "primary": primary,
        "sensitivity": [m for m in methods if not m["primary"]],
        "primary_pass": primary["contract"]["pass"],
    }


def run(csvs: list[str], fee_bps: float = 5.0, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    assets = []
    for c in csvs:
        log.info("oracle %s", c)
        assets.append(run_asset(c, fee_bps, holdout_frac))

    primary_passes = [a["asset"] for a in assets if a.get("primary_pass")]
    if len(primary_passes) == len(assets) and assets:
        verdict = "PASS_HOLDOUT"
        verdict_he = (
            "האורקל עבר את חוזה ההצלחה על כל הנכסים ב-holdout — "
            "השלב הבא היחיד: חותמת lockbox v2 חד-פעמית (בלי שינוי פרמטרים)."
        )
        next_he = "seal lockbox v2 + one-shot eval of frozen oracle_k2_n80_mh12"
    elif primary_passes:
        verdict = "PARTIAL"
        verdict_he = (
            f"עבר רק על: {', '.join(primary_passes)}. "
            "לא מספיק להצלחה מלאה — אין קידום ל-lockbox עד מעבר מלא או נכס יחיד pre-reg."
        )
        next_he = "either harden until both pass, or pre-reg single-asset path explicitly"
    else:
        verdict = "FAIL"
        verdict_he = (
            "FAIL אמיתי: האורקל לא עמד בחוזה (רווח נטו > buy&hold + עסקאות + sharpe). "
            "לא מציגים hit-rate כניצחון."
        )
        next_he = "change state definition or abandon; do not loosen contract"

    return {
        "study_id": STUDY_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "north_star": "predict → trade → net profit (real only)",
        "success_contract": {
            "fee_bps_per_side": fee_bps,
            "require": [
                "total_return > 0",
                "total_return > buy_and_hold",
                "n_trades >= 30",
                "sharpe > 0",
            ],
            "primary_config": PRIMARY,
            "note_he": "הצלחה = עמידה בחוזה. אין הצלחה מובטחת בשוק — יש חובת יושר והצלחה רק אם אמיתית.",
        },
        "method_he": (
            "אורקל כלכלי v2: באותו מצב (רצף+VR) מהעבר — סחור רק אם ממוצע התשואה "
            f"ל-{PRIMARY['horizon']} נרות מנקה לפחות פי {PRIMARY['k_fee']} מעלות העסקה הלוך-חזור; "
            "מחזיקים באותו אופק."
        ),
        "v1_note_he": (
            "v1 נכשל ביושר: ממוצע נר-הבא במצבים (~0-6 bps) לא מנקה 10 bps עמלות — "
            "לכן אפס עסקאות. v2 עובר לאופק ארוך יותר בלי להרפות את החוזה."
        ),
        "assets": assets,
        "primary_pass_assets": primary_passes,
        "verdict": verdict,
        "verdict_he": verdict_he,
        "next_he": next_he,
    }


def _print(r: dict) -> None:
    print(f"\n{'=' * 70}")
    print("ECONOMIC ORACLE — SUCCESS CONTRACT @ 5bps")
    print(f"{'=' * 70}")
    print(f"  {r['method_he']}")
    print(f"  contract: {r['success_contract']['require']}")
    for a in r["assets"]:
        bh = a["buy_and_hold"]["total_return"] * 100
        p = a["primary"]
        status = "PASS" if p["contract"]["pass"] else "FAIL"
        print(f"\n  {a['asset']}  BH={bh:+.1f}%  PRIMARY -> {status}")
        print(
            f"    {p['config']['name']}: ret={p['total_return']*100:+.1f}% "
            f"sharpe={p.get('sharpe')} trades={p['n_changes']} "
            f"cov={p['coverage']} signals={p['signal_bars']}"
        )
        print(f"    checks={p['contract']['checks']}")
        for s in a["sensitivity"]:
            st = "PASS" if s["contract"]["pass"] else "fail"
            print(
                f"    sens {s['config']['name']}: {s['total_return']*100:+.1f}% "
                f"trades={s['n_changes']} [{st}]"
            )
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  {r['verdict_he']}")
    print(f"  NEXT: {r['next_he']}")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
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
