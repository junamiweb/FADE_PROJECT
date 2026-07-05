"""Honest PnL simulation — does the validated edge survive real trading costs?

We have a directional edge (~53% at 1h, ~56% on multi-res agreement). Accuracy
is not money: at hourly frequency, exchange fees + slippage can eat a thin edge
alive. This simulates actual trading on the quarantined holdout with realistic
costs and compares against the honest baseline: buy-and-hold.

Protocol (no look-ahead):
    1. Chronological 70/30 split; mine + freeze stable rules on dev only.
    2. On the holdout, each bar's calibrated prediction sets the next-bar
       position. Position is held one bar (forward_horizon=1), then re-decided.
    3. Costs: taker fee per side (bps) charged on |change in position| each bar,
       plus optional slippage. Flat (0) on bars with no signal.

Variants:
    long_only     — long when pred=up, flat otherwise.
    long_short    — long/short with pred.
    conf_gated    — long_short but only when calibrated prob >= --conf (fewer
                    trades -> less fee drag).

Metrics: total & annualised return, per-bar Sharpe (annualised), max drawdown,
number of position changes, total fee drag, directional hit-rate. All compared
to buy-and-hold over the identical holdout window.

Run:
    python -m fade.pipeline.pnl_sim btc_1h.csv
    python -m fade.pipeline.pnl_sim btc_1h.csv --fee-bps 4 --conf 0.55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.predictor import predict_calibrated
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("pnl_sim")

BARS_PER_YEAR = {"1h": 24 * 365, "30m": 48 * 365, "15m": 96 * 365,
                 "4h": 6 * 365, "8h": 3 * 365, "1d": 365}


def _holdout_predictions(csv_path: str, config: Config, holdout_frac: float):
    """Frozen-rule calibrated predictions on the quarantined holdout."""
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

    # Calibration learned on dev (so probabilities are honest on holdout).
    cal = CalibrationStore(config.cache_dir / "_pnl_cal.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}
    dev_disc = ev.discretize(dev_atoms, ev.compute_thresholds(dev_atoms, config))
    dev_events = ev.build_events(dev_disc, config, allowed=set(frozen.index))
    dev_preds = predict_calibrated(dev_events, frozen, cal, positive={})
    if not dev_preds.empty:
        from fade.core.predictor import collect_calibration_samples
        samples = collect_calibration_samples(dev_preds, dev_fwd)
        if samples:
            cal.update(samples)

    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict_calibrated(hold_events, frozen, cal, positive={})
    if preds.empty:
        return None

    # Per-bar realised return (close-to-close over one horizon) on holdout bars.
    hold_close = close.iloc[split:]
    bar_ret = hold_close.pct_change().shift(-1)  # return from this bar to next
    out = preds.join(bar_ret.rename("bar_ret")).dropna(subset=["bar_ret"])
    return out, hold_close, len(frozen)


def _equity(positions: np.ndarray, bar_ret: np.ndarray, fee_rate: float,
            slippage: float) -> dict:
    """Simulate an equity curve from a position series and per-bar returns."""
    pos = positions.astype(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev)
    costs = turnover * (fee_rate + slippage)
    strat_ret = pos * bar_ret - costs
    equity = np.cumprod(1.0 + strat_ret)
    return {"strat_ret": strat_ret, "equity": equity,
            "n_changes": int(np.sum(turnover > 0)),
            "total_cost": float(np.sum(costs))}


def _stats(strat_ret: np.ndarray, equity: np.ndarray, bars_per_year: float) -> dict:
    n = len(strat_ret)
    total = float(equity[-1] - 1.0)
    ann = float(equity[-1] ** (bars_per_year / max(n, 1)) - 1.0) if n else 0.0
    mu, sd = float(np.mean(strat_ret)), float(np.std(strat_ret))
    sharpe = float(mu / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = float(np.min(equity / peak - 1.0)) if n else 0.0
    return {"total_return": round(total, 4), "annual_return": round(ann, 4),
            "sharpe": round(sharpe, 3), "max_drawdown": round(dd, 4)}


def run_pnl(csv_path: str, holdout_frac: float = 0.30, fee_bps: float = 4.0,
            slippage_bps: float = 1.0, conf: float = 0.55,
            config: Config | None = None) -> dict:
    config = config or Config()
    res = Path(csv_path).stem.split("_")[-1]
    bpy = BARS_PER_YEAR.get(res, 24 * 365)
    fee_rate = fee_bps / 1e4
    slippage = slippage_bps / 1e4

    got = _holdout_predictions(csv_path, config, holdout_frac)
    if got is None:
        return {"status": "no_rules", "asset": Path(csv_path).stem}
    preds, hold_close, n_rules = got

    bar_ret = preds["bar_ret"].to_numpy()
    pred_up = preds["pred"].to_numpy().astype(int)
    cal_prob = preds["calibrated_prob"].to_numpy()
    actual_up = (bar_ret > 0).astype(int)
    hit = float(np.mean(pred_up == actual_up))

    # Buy-and-hold over the same covered bars.
    bh_equity = np.cumprod(1.0 + bar_ret)
    bh = _stats(bar_ret, bh_equity, bpy)

    variants = {}
    # long-only: +1 when up else flat
    pos_lo = np.where(pred_up == 1, 1.0, 0.0)
    e = _equity(pos_lo, bar_ret, fee_rate, slippage)
    variants["long_only"] = {**_stats(e["strat_ret"], e["equity"], bpy),
                             "n_changes": e["n_changes"],
                             "cost_drag": round(e["total_cost"], 4)}
    # long-short: +1 up, -1 down
    pos_ls = np.where(pred_up == 1, 1.0, -1.0)
    e = _equity(pos_ls, bar_ret, fee_rate, slippage)
    variants["long_short"] = {**_stats(e["strat_ret"], e["equity"], bpy),
                              "n_changes": e["n_changes"],
                              "cost_drag": round(e["total_cost"], 4)}
    # confidence-gated long-short: act only when calibrated prob >= conf
    gate = cal_prob >= conf
    pos_cg = np.where(gate, np.where(pred_up == 1, 1.0, -1.0), 0.0)
    e = _equity(pos_cg, bar_ret, fee_rate, slippage)
    variants["conf_gated"] = {**_stats(e["strat_ret"], e["equity"], bpy),
                              "n_changes": e["n_changes"],
                              "cost_drag": round(e["total_cost"], 4),
                              "active_bars": int(np.sum(gate)),
                              "conf": conf}
    # hold-until-flip: keep the last directional position across bars, only
    # trade when the predicted direction actually changes (minimal turnover —
    # the honest way to trade a persistent directional bias).
    target = np.where(pred_up == 1, 1.0, -1.0)
    pos_hf = np.empty_like(target)
    cur = 0.0
    for i in range(len(target)):
        cur = target[i]
        pos_hf[i] = cur
    e = _equity(pos_hf, bar_ret, fee_rate, slippage)
    variants["hold_until_flip"] = {**_stats(e["strat_ret"], e["equity"], bpy),
                                   "n_changes": e["n_changes"],
                                   "cost_drag": round(e["total_cost"], 4)}
    # daily rebalance: sample the signal once per day (24 bars) -> far fewer
    # trades. Uses only past info (the signal at the start of each day).
    step = max(1, int(BARS_PER_YEAR.get(res, 24 * 365) / 365))
    pos_daily = pos_ls.copy()
    hold = 0.0
    for i in range(len(pos_daily)):
        if i % step == 0:
            hold = pos_ls[i]
        pos_daily[i] = hold
    e = _equity(pos_daily, bar_ret, fee_rate, slippage)
    variants["daily_rebalance"] = {**_stats(e["strat_ret"], e["equity"], bpy),
                                   "n_changes": e["n_changes"],
                                   "cost_drag": round(e["total_cost"], 4)}

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "resolution": res,
        "n_rules": n_rules,
        "holdout_bars": int(len(preds)),
        "span": [str(preds.index.min())[:10], str(preds.index.max())[:10]],
        "fee_bps": fee_bps, "slippage_bps": slippage_bps,
        "directional_hit": round(hit, 4),
        "buy_hold": bh,
        "variants": variants,
        "verdict": _verdict(variants, bh),
    }


def _verdict(variants: dict, bh: dict) -> str:
    best = max(variants.values(), key=lambda v: v["total_return"])
    best_name = [k for k, v in variants.items() if v is best][0]
    beats_bh = best["total_return"] > bh["total_return"]
    positive = best["total_return"] > 0
    if positive and beats_bh:
        return (f"BEATS BUY-HOLD - {best_name} returns {best['total_return']*100:.1f}% "
                f"vs {bh['total_return']*100:.1f}% (Sharpe {best['sharpe']}).")
    if positive:
        return (f"PROFITABLE BUT TRAILS BUY-HOLD - best {best_name} "
                f"{best['total_return']*100:.1f}% vs hold {bh['total_return']*100:.1f}%.")
    return (f"COSTS EAT THE EDGE - best variant {best_name} is "
            f"{best['total_return']*100:.1f}% after fees. Directional edge not tradeable at this frequency.")


def _print(r: dict) -> None:
    line = "=" * 74
    print("\n" + line)
    print(f"FADE PnL SIMULATION - {r.get('asset', '?').upper()}  (holdout, after costs)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  holdout: {r['holdout_bars']:,} bars  {r['span'][0]} -> {r['span'][1]}   "
          f"rules={r['n_rules']}")
    print(f"  costs  : fee {r['fee_bps']} bps/side + slippage {r['slippage_bps']} bps   "
          f"directional hit {r['directional_hit']}")
    print()
    bh = r["buy_hold"]
    print(f"  {'strategy':<14}{'return':>10}{'annual':>10}{'sharpe':>9}"
          f"{'maxDD':>9}{'trades':>8}{'costDrag':>10}")
    print(f"  {'buy_and_hold':<14}{bh['total_return']*100:>9.1f}%{bh['annual_return']*100:>9.1f}%"
          f"{bh['sharpe']:>9}{bh['max_drawdown']*100:>8.1f}%{'-':>8}{'-':>10}")
    for name, v in r["variants"].items():
        print(f"  {name:<14}{v['total_return']*100:>9.1f}%{v['annual_return']*100:>9.1f}%"
              f"{v['sharpe']:>9}{v['max_drawdown']*100:>8.1f}%{v['n_changes']:>8}"
              f"{v['cost_drag']*100:>9.1f}%")
    print(line)
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE PnL simulation with costs")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    _print(run_pnl(args.csv, holdout_frac=args.holdout_frac, fee_bps=args.fee_bps,
                   slippage_bps=args.slippage_bps, conf=args.conf))


if __name__ == "__main__":
    main()
