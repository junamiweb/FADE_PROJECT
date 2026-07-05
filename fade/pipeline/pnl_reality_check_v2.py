"""PnL reality check v2 — re-test trading costs on the CURRENT engine (post batch 11–29).

Batch 10 showed gross edge (+96%) collapses to -99% at 5 bps because path_lean3
rules flip almost every bar. Since then accuracy improved (~53% -> ~54.6%) and a
conviction / multi-res stack was built, but neither was re-tested against fees.

Tests on quarantined 70/30 holdout (btc_1h default):
  1. raw_path_lean3   — frozen path_lean3 rules, long-short every bar (batch 10 baseline).
  2. conviction_streak2 — trade only when streak>=2 contrarian fires (flat otherwise).
  3. conviction_combo   — streak>=2 AND 3-TF agree (aligned combo gate).
  4. primary_policy     — PRIMARY stack: conviction tiers + frequent-wins conflict rule.
  5. min_hold sweep     — raw path_lean3 with minimum holding period before flip.

Fee sensitivity: 1, 5, 10 bps per side (default slippage 0 for clean comparison).

Run:
    python -m fade.pipeline.pnl_reality_check_v2
    python -m fade.pipeline.pnl_reality_check_v2 btc_1h.csv --fee-bps 5 10 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import lean_config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.conviction import TIER_DEFS, HOLDOUT_FRAC
from fade.core.data_loader import load_ohlcv
from fade.core.predictor import collect_calibration_samples, predict_calibrated
from fade.pipeline.backtest import walk_forward
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.holdout import _select_stable_rules
from fade.pipeline.pnl_sim import BARS_PER_YEAR, _equity, _stats
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("pnl_reality_v2")

MULTI_IV = ("5m", "15m", "30m", "1h")
MIN_HOLD_GRID = (1, 2, 3, 4, 6, 8, 12, 24, 48)


def _prefix(csv: str) -> str:
    stem = Path(csv).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _holdout_path_lean3(csv_path: str, holdout_frac: float):
    """Frozen path_lean3 calibrated predictions on holdout."""
    config = lean_config()
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    close = df["close"].reindex(atoms.index)

    n = len(atoms)
    split = int(n * (1.0 - holdout_frac))
    dev_atoms, hold_atoms = atoms.iloc[:split], atoms.iloc[split:]
    dev_fwd = fwd.iloc[:split]

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return None

    cal = CalibrationStore(config.cache_dir / "_pnl_v2_cal.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}
    dev_disc = ev.discretize(dev_atoms, ev.compute_thresholds(dev_atoms, config))
    dev_events = ev.build_events(dev_disc, config, allowed=set(frozen.index))
    dev_preds = predict_calibrated(dev_events, frozen, cal, positive={})
    if not dev_preds.empty:
        samples = collect_calibration_samples(dev_preds, dev_fwd)
        if samples:
            cal.update(samples)

    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict_calibrated(hold_events, frozen, cal, positive={})
    if preds.empty:
        return None

    hold_close = close.iloc[split:]
    bar_ret = hold_close.pct_change().shift(-1)
    out = preds.join(bar_ret.rename("bar_ret")).dropna(subset=["bar_ret"])
    return out, split, len(frozen)


def _conviction_frame(csv_1h: str) -> pd.DataFrame | None:
    """Per-bar conviction / multi-res state aligned to 1h index."""
    prefix = _prefix(csv_1h)
    df = load_ohlcv(csv_1h)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    frame = pd.DataFrame({"streak": streak, "slen": np.abs(streak)}, index=df.index)
    for iv in MULTI_IV:
        p = f"{prefix}_{iv}.csv"
        if Path(p).exists():
            frame[iv] = _contrarian_grid(p)
    return frame


def _tier_at_row(row, cols: list[str]) -> tuple[str | None, str | None]:
    """Return (tier_id, direction UP/DOWN) for highest active tier, or (None, None)."""
    s = int(row["streak"]) if np.isfinite(row["streak"]) else 0
    slen = int(row["slen"])
    streak_dir = "FLAT" if slen < 2 else ("UP" if s < 0 else "DOWN")

    dirs = []
    for c in cols:
        v = row.get(c, np.nan)
        if v == 0 or not np.isfinite(v):
            dirs.append(None)
        else:
            dirs.append("UP" if v > 0 else "DOWN")
    ok = [d for d in dirs if d is not None]
    up = sum(1 for d in ok if d == "UP")
    dn = sum(1 for d in ok if d == "DOWN")
    if up >= dn and up > 0:
        multi_dir, tf_agree = "UP", up
    elif dn > 0:
        multi_dir, tf_agree = "DOWN", dn
    else:
        multi_dir, tf_agree = "FLAT", 0
    aligned = streak_dir != "FLAT" and multi_dir != "FLAT" and streak_dir == multi_dir

    for tid, _label, min_s, min_k, _hit, _note in TIER_DEFS:
        if min_k == 0:
            if slen < min_s or streak_dir == "FLAT":
                continue
            return tid, streak_dir
        if min_s == 0:
            if tf_agree < min_k or multi_dir == "FLAT":
                continue
            return tid, multi_dir
        if slen < min_s or tf_agree < min_k or not aligned:
            continue
        return tid, streak_dir
    return None, None


def _primary_direction(row, cols: list[str], conv_dir: str | None) -> str | None:
    """PRIMARY policy: frequent (15+30+1h unanimous) wins on conflict."""
    if conv_dir is None:
        return None
    core = []
    for c in ("15m", "30m", "1h"):
        if c not in cols:
            continue
        v = row.get(c, np.nan)
        if v == 0 or not np.isfinite(v):
            continue
        core.append("UP" if v > 0 else "DOWN")
    freq_dir = None
    if len(core) == 3 and len(set(core)) == 1:
        freq_dir = core[0]
    if freq_dir is not None and freq_dir != conv_dir:
        return freq_dir
    return conv_dir


def _dir_to_pos(direction: str | None) -> float:
    if direction == "UP":
        return 1.0
    if direction == "DOWN":
        return -1.0
    return 0.0


def _min_hold_positions(target: np.ndarray, min_bars: int) -> np.ndarray:
    """Hold position at least min_bars before allowing a flip."""
    pos = np.zeros(len(target))
    cur = 0.0
    bars_in_pos = 0
    for i in range(len(target)):
        want = target[i]
        if want == cur:
            pos[i] = cur
            if cur != 0:
                bars_in_pos += 1
        elif bars_in_pos >= min_bars or cur == 0:
            cur = want
            pos[i] = cur
            bars_in_pos = 1 if cur != 0 else 0
        else:
            pos[i] = cur
            bars_in_pos += 1
    return pos


def _simulate_variant(name: str, positions: np.ndarray, bar_ret: np.ndarray,
                      fee_rate: float, slippage: float, bpy: float) -> dict:
    e = _equity(positions, bar_ret, fee_rate, slippage)
    active = int(np.sum(positions != 0))
    return {
        "variant": name,
        "active_bars": active,
        **_stats(e["strat_ret"], e["equity"], bpy),
        "n_changes": e["n_changes"],
        "cost_drag": round(e["total_cost"], 4),
    }


def run_reality_check(
    csv_path: str = "btc_1h.csv",
    holdout_frac: float = HOLDOUT_FRAC,
    fee_bps_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    slippage_bps: float = 0.0,
) -> dict:
    got = _holdout_path_lean3(csv_path, holdout_frac)
    if got is None:
        return {"status": "no_rules", "asset": Path(csv_path).stem}

    preds, split, n_rules = got
    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    n_bars = len(bar_ret)

    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    hit_raw = float(np.mean(pred_up == (bar_ret > 0).astype(int)))

    # Conviction signals on same holdout timestamps
    conv_frame = _conviction_frame(csv_path)
    cols = [c for c in MULTI_IV if conv_frame is not None and c in conv_frame.columns]
    hold_idx = preds.index

    streak2_pos = np.zeros(n_bars)
    combo_pos = np.zeros(n_bars)
    primary_pos = np.zeros(n_bars)

    if conv_frame is not None:
        hold_conv = conv_frame.reindex(hold_idx).fillna(0)
        for i, (ts, row) in enumerate(hold_conv.iterrows()):
            tid, conv_dir = _tier_at_row(row, cols)
            # streak>=2 only (lowest conviction tier with min_k=0, min_s=2)
            slen = int(row["slen"])
            s = int(row["streak"]) if np.isfinite(row["streak"]) else 0
            s2_dir = None
            if slen >= 2 and s != 0:
                s2_dir = "UP" if s < 0 else "DOWN"
            streak2_pos[i] = _dir_to_pos(s2_dir)

            # combo: streak>=2 + 3 TF aligned
            combo_dir = None
            if slen >= 2 and s != 0:
                dirs = []
                for c in cols:
                    v = row.get(c, np.nan)
                    if v == 0 or not np.isfinite(v):
                        continue
                    dirs.append("UP" if v > 0 else "DOWN")
                up = sum(1 for d in dirs if d == "UP")
                dn = sum(1 for d in dirs if d == "DOWN")
                if up >= 3:
                    mdir = "UP"
                elif dn >= 3:
                    mdir = "DOWN"
                else:
                    mdir = "FLAT"
                streak_d = "UP" if s < 0 else "DOWN"
                if mdir == streak_d:
                    combo_dir = streak_d
            combo_pos[i] = _dir_to_pos(combo_dir)

            # PRIMARY: any conviction tier + conflict policy
            prim_dir = _primary_direction(row, cols, conv_dir if tid else None)
            primary_pos[i] = _dir_to_pos(prim_dir)

    raw_target = np.where(pred_up == 1, 1.0, -1.0)
    raw_ls = raw_target.copy()

    fee_results = {}
    for fee_bps in fee_bps_list:
        fee_rate = fee_bps / 1e4
        slip = slippage_bps / 1e4
        bh_eq = np.cumprod(1.0 + bar_ret)
        bh = _stats(bar_ret, bh_eq, bpy)

        variants = [
            _simulate_variant("raw_path_lean3", raw_ls, bar_ret, fee_rate, slip, bpy),
            _simulate_variant("conviction_streak2", streak2_pos, bar_ret, fee_rate, slip, bpy),
            _simulate_variant("conviction_combo_r2_k3", combo_pos, bar_ret, fee_rate, slip, bpy),
            _simulate_variant("primary_policy", primary_pos, bar_ret, fee_rate, slip, bpy),
        ]
        for mh in MIN_HOLD_GRID:
            pos = _min_hold_positions(raw_target, mh)
            variants.append(
                _simulate_variant(f"min_hold_{mh}bars", pos, bar_ret, fee_rate, slip, bpy)
            )

        fee_results[f"{fee_bps}bps"] = {
            "buy_hold": bh,
            "variants": {v["variant"]: v for v in variants},
            "any_positive": any(v["total_return"] > 0 for v in variants),
            "best": max(variants, key=lambda v: v["total_return"]),
        }

    # Summary across fees
    survives_5bps = fee_results.get("5.0bps", {}).get("any_positive", False)
    best_5 = fee_results.get("5.0bps", {}).get("best", {})
    min_hold_positive = [
        mh for mh in MIN_HOLD_GRID
        if fee_results.get("5.0bps", {}).get("variants", {})
        .get(f"min_hold_{mh}bars", {}).get("total_return", -1) > 0
    ]

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "holdout_bars": n_bars,
        "n_rules": n_rules,
        "span": [str(hold_idx.min())[:10], str(hold_idx.max())[:10]],
        "raw_directional_hit": round(hit_raw, 4),
        "fee_results": fee_results,
        "survives_5bps_any_variant": survives_5bps,
        "best_at_5bps": best_5,
        "min_hold_positive_at_5bps": min_hold_positive,
        "verdict": _verdict(survives_5bps, best_5, min_hold_positive),
    }


def _verdict(survives: bool, best: dict, min_hold_pos: list) -> str:
    if survives:
        return (
            f"SOME VARIANT SURVIVES 5bps — best: {best.get('variant')} "
            f"return {best.get('total_return', 0)*100:.1f}% "
            f"({best.get('n_changes')} trades)."
        )
    mh = f"min_hold needs {min_hold_pos[0]}+ bars" if min_hold_pos else "no min_hold helps"
    return (
        f"COSTS EAT THE EDGE at 5bps — best {best.get('variant', '?')} "
        f"{best.get('total_return', 0)*100:.1f}%. Directional edge (~54%) is real but "
        f"not tradeable at hourly retail fees. {mh}."
    )


def _print(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"FADE PnL REALITY CHECK v2 — {r.get('asset', '?').upper()}")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return

    print(f"  holdout: {r['holdout_bars']:,} bars  {r['span'][0]} -> {r['span'][1]}")
    print(f"  path_lean3 rules: {r['n_rules']}  raw hit: {r['raw_directional_hit']}")
    print()

    for fee_key, fr in r["fee_results"].items():
        print(f"  --- {fee_key} per side ---")
        bh = fr["buy_hold"]
        print(f"  {'strategy':<26}{'return':>9}{'sharpe':>8}{'trades':>8}{'active':>8}")
        print(f"  {'buy_hold':<26}{bh['total_return']*100:>8.1f}%"
              f"{bh['sharpe']:>8}{'-':>8}{'-':>8}")
        for name in ("raw_path_lean3", "conviction_streak2", "conviction_combo_r2_k3",
                     "primary_policy"):
            v = fr["variants"][name]
            print(f"  {name:<26}{v['total_return']*100:>8.1f}%"
                  f"{v['sharpe']:>8}{v['n_changes']:>8}{v['active_bars']:>8}")
        # min_hold summary: show best and first positive
        mh_vars = [(k, v) for k, v in fr["variants"].items() if k.startswith("min_hold_")]
        best_mh = max(mh_vars, key=lambda x: x[1]["total_return"])
        print(f"  best min_hold: {best_mh[0]} -> {best_mh[1]['total_return']*100:.1f}% "
              f"({best_mh[1]['n_changes']} trades)")
        print()

    print(line)
    print(f"  VERDICT: {r['verdict']}")
    if r.get("min_hold_positive_at_5bps"):
        print(f"  Min-hold positive at 5bps: {r['min_hold_positive_at_5bps']}")
    else:
        print("  Min-hold sweep: NONE positive at 5bps")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE PnL reality check v2")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--fee-bps", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    _print(run_reality_check(args.csv, holdout_frac=args.holdout_frac,
                             fee_bps_list=tuple(args.fee_bps),
                             slippage_bps=args.slippage_bps))


if __name__ == "__main__":
    main()
