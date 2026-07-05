"""Cross-scale structure — micro vs macro, and how the scales relate.

Three analyses, all on the streak / mean-reversion mechanism found earlier:

1. REVERSION LADDER (micro -> macro): for each resolution, the holdout
   continuation rate after a streak (>=2 same-direction bars) and a linear slope
   of continuation vs streak length. Shows HOW the intraday inefficiency scales
   from seconds up to daily.

2. MICRO x MACRO interaction: attach each micro bar's concurrent MACRO trend
   (sign of the last CLOSED daily return) and ask whether the micro streak
   reversal is stronger with or against the macro trend. This is the micro<->macro
   "correlation": does the big-picture trend modulate the small-picture reversal?
   No look-ahead: the daily return is only attached once it is fully closed
   (available_at = day_end), via merge_asof backward.

3. SIGNAL CORRELATION across timeframes: the contrarian signal s(t)=-sign(streak)
   from each resolution, resampled onto a shared hourly grid, then pairwise
   correlated. Returns across scales are ~identical, but the reversal SIGNALS may
   be much less correlated -> each timeframe's memory carries semi-independent
   information (relevant for combining them).

Run:
    python -m fade.pipeline.scale_structure
    python -m fade.pipeline.scale_structure --micro btc_1h.csv --macro btc_daily.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.trend_structure import _signed_streak

LADDER_FILES = ["btc_1s.csv", "btc_1m_vol.csv", "btc_5m.csv", "btc_10m.csv",
                "btc_15m.csv", "btc_30m.csv", "btc_1h.csv", "btc_daily.csv"]
CORR_FILES = ["btc_5m.csv", "btc_15m.csv", "btc_30m.csv", "btc_1h.csv"]
MIN_SUPPORT = 50


# ------------------------------------------------------------------ ladder ----
def reversion_ladder(files: list[str], holdout_frac: float = 0.30,
                     n_shuffles: int = 1500, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for c in files:
        if not Path(c).exists():
            continue
        df = load_ohlcv(c)
        ret = df["close"].pct_change().to_numpy()
        streak = _signed_streak(ret)
        up = (ret > 0).astype(int)
        finite = np.isfinite(ret)
        n = len(ret)
        split = int(n * (1 - holdout_frac))
        hold = np.zeros(n, dtype=bool)
        hold[split:] = True
        hold &= finite
        base = float(up[hold].mean())

        # aggregate continuation after any streak >= 2
        cont_hits = cont_n = 0
        by_len = {}
        for m in range(2, 8):
            for sgn in (m, -m):
                sel = hold & (streak == sgn)
                k = int(sel.sum())
                if k < MIN_SUPPORT:
                    continue
                nxt_up = up[sel]
                cont = nxt_up.mean() if sgn > 0 else 1 - nxt_up.mean()
                by_len.setdefault(m, []).append(cont)
                cont_hits += int((nxt_up == (1 if sgn > 0 else 0)).sum())
                cont_n += k
        if cont_n < MIN_SUPPORT:
            out.append({"asset": Path(c).stem, "status": "low_support"})
            continue
        cont_rate = cont_hits / cont_n
        # slope of mean continuation vs streak length
        xs = sorted(by_len)
        ys = [float(np.mean(by_len[m])) for m in xs]
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
        # permutation on aggregate continuation vs base (base=0.5 reference for continuation)
        null = np.empty(n_shuffles)
        up_hold = up[hold]
        for i in range(n_shuffles):
            samp = rng.choice(up_hold, size=cont_n, replace=False)
            null[i] = samp.mean()
        dev = abs(cont_rate - 0.5)
        # compare deviation-from-0.5 of continuation vs deviation of random up-samples from base
        p = (1 + int(np.sum(np.abs(null - base) >= dev))) / (1 + n_shuffles)
        out.append({
            "asset": Path(c).stem, "status": "ok",
            "n_holdout": int(hold.sum()),
            "streak_n": cont_n,
            "continuation": round(cont_rate, 4),
            "reversion_index": round(0.5 - cont_rate, 4),
            "slope_per_len": round(slope, 4),
            "p_value": round(p, 4),
        })
    return out


# ----------------------------------------------------- micro x macro ----------
def micro_macro(micro_csv: str, macro_csv: str, holdout_frac: float = 0.30,
                n_shuffles: int = 2000, seed: int = 0) -> dict:
    if not (Path(micro_csv).exists() and Path(macro_csv).exists()):
        return {"status": "missing_files"}
    micro = load_ohlcv(micro_csv)
    macro = load_ohlcv(macro_csv)

    mret = micro["close"].pct_change()
    streak = _signed_streak(mret.to_numpy())
    micro_df = pd.DataFrame({
        "ts": micro.index,
        "ret": mret.to_numpy(),
        "streak": streak,
    }).dropna()

    # macro daily return, made available only at day end (no look-ahead)
    daily_ret = macro["close"].pct_change()
    macro_df = pd.DataFrame({
        "available_at": macro.index + pd.Timedelta(days=1),
        "macro_ret": daily_ret.to_numpy(),
    }).dropna().sort_values("available_at")

    merged = pd.merge_asof(
        micro_df.sort_values("ts"), macro_df,
        left_on="ts", right_on="available_at", direction="backward",
    ).dropna(subset=["macro_ret"])

    n = len(merged)
    if n < 500:
        return {"status": "insufficient_data", "n": int(n)}
    split = int(n * (1 - holdout_frac))
    hold = merged.iloc[split:].copy()

    hold["up"] = (hold["ret"] > 0).astype(int)
    hold["micro_dir"] = np.sign(hold["streak"]).astype(int)
    hold["macro_dir"] = np.sign(hold["macro_ret"]).astype(int)
    hold = hold[(hold["micro_dir"] != 0) & (hold["macro_dir"] != 0)
                & (np.abs(hold["streak"]) >= 2)]

    base_up = float(hold["up"].mean())
    rng = np.random.default_rng(seed)
    cells = []
    for md in (1, -1):
        for xd in (1, -1):
            sub = hold[(hold["micro_dir"] == md) & (hold["macro_dir"] == xd)]
            k = len(sub)
            if k < MIN_SUPPORT:
                cells.append({"micro": md, "macro": xd, "n": k, "status": "low"})
                continue
            up_rate = float(sub["up"].mean())
            # reversal strength = how far next-up deviates AGAINST the micro streak
            reversal = (1 - up_rate) if md > 0 else up_rate
            null = np.empty(n_shuffles)
            allup = hold["up"].to_numpy()
            for i in range(n_shuffles):
                null[i] = rng.choice(allup, size=k, replace=False).mean()
            dev = abs(up_rate - base_up)
            p = (1 + int(np.sum(np.abs(null - base_up) >= dev))) / (1 + n_shuffles)
            cells.append({
                "micro": "up-streak" if md > 0 else "down-streak",
                "macro": "macro-up" if xd > 0 else "macro-down",
                "n": k, "next_up": round(up_rate, 4),
                "reversal_strength": round(reversal, 4),
                "p_value": round(p, 4), "status": "ok",
            })
    return {"status": "ok", "micro": Path(micro_csv).stem,
            "macro": Path(macro_csv).stem, "n_holdout": int(len(hold)),
            "base_up": round(base_up, 4), "cells": cells}


# ------------------------------------------------ signal correlation ----------
def signal_correlation(files: list[str], grid: str = "1h") -> dict:
    sigs = {}
    rets = {}
    for c in files:
        if not Path(c).exists():
            continue
        df = load_ohlcv(c)
        r = df["close"].pct_change()
        streak = _signed_streak(r.to_numpy())
        contr = pd.Series(-np.sign(streak), index=df.index)  # contrarian signal
        # last value within each grid bucket = prevailing signal
        sigs[Path(c).stem] = contr.resample(grid).last()
        rets[Path(c).stem] = r.resample(grid).sum()
    if len(sigs) < 2:
        return {"status": "insufficient_files"}
    sig_df = pd.DataFrame(sigs).dropna()
    ret_df = pd.DataFrame(rets).dropna()
    return {
        "status": "ok", "grid": grid, "n": int(len(sig_df)),
        "signal_corr": sig_df.corr().round(3).to_dict(),
        "return_corr": ret_df.corr().round(3).to_dict(),
        "assets": list(sigs),
    }


# ------------------------------------------------------------- print ----------
def _print_ladder(rows: list[dict]) -> None:
    line = "=" * 70
    print("\n" + line)
    print("1) REVERSION LADDER  (micro -> macro; holdout, streak>=2)")
    print(line)
    print(f"  {'scale':<14}{'hold_n':>9}{'streak_n':>9}{'continue':>10}"
          f"{'rev_index':>10}{'slope':>9}{'p':>8}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['asset']:<14}{'':>9}   {r.get('status')}")
            continue
        star = " *" if r["p_value"] <= 0.05 else ""
        print(f"  {r['asset']:<14}{r['n_holdout']:>9}{r['streak_n']:>9}"
              f"{r['continuation']:>10}{r['reversion_index']:>+10}"
              f"{r['slope_per_len']:>+9}{r['p_value']:>8}{star}")
    print("  rev_index>0 = mean-reversion; slope<0 = reversal grows with streak")


def _print_mm(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print(f"2) MICRO x MACRO  ({r.get('micro','?')} bars x {r.get('macro','?')} trend)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        return
    print(f"  holdout n={r['n_holdout']}  base up-rate={r['base_up']}")
    print(f"  {'micro streak':<12}{'macro':<12}{'n':>8}{'next_up':>9}"
          f"{'reversal':>10}{'p':>8}")
    for c in r["cells"]:
        if c.get("status") != "ok":
            print(f"  {str(c['micro']):<12}{str(c['macro']):<12}{c['n']:>8}   low")
            continue
        star = " *" if c["p_value"] <= 0.05 else ""
        print(f"  {c['micro']:<12}{c['macro']:<12}{c['n']:>8}{c['next_up']:>9}"
              f"{c['reversal_strength']:>10}{c['p_value']:>8}{star}")


def _print_corr(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print(f"3) CROSS-TIMEFRAME SIGNAL CORRELATION  (grid={r.get('grid')}, "
          f"n={r.get('n')})")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        return
    assets = r["assets"]
    print("  contrarian SIGNAL correlation:")
    print("            " + "".join(f"{a.split('_')[-1]:>9}" for a in assets))
    for a in assets:
        row = r["signal_corr"][a]
        print(f"  {a.split('_')[-1]:<9}" + "".join(f"{row[b]:>9}" for b in assets))
    print("  realized RETURN correlation:")
    print("            " + "".join(f"{a.split('_')[-1]:>9}" for a in assets))
    for a in assets:
        row = r["return_corr"][a]
        print(f"  {a.split('_')[-1]:<9}" + "".join(f"{row[b]:>9}" for b in assets))


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE cross-scale structure")
    parser.add_argument("--micro", default="btc_1h.csv")
    parser.add_argument("--macro", default="btc_daily.csv")
    args = parser.parse_args()

    ladder = reversion_ladder(LADDER_FILES)
    _print_ladder(ladder)
    _print_mm(micro_macro(args.micro, args.macro))
    _print_corr(signal_correlation(CORR_FILES))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
