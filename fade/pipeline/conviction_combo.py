"""Combined conviction + calibration + ETH -- all three follow-ups in one module.

1. COMBINED GATE (BTC 1h): streak length >= L AND multi-TF agreement >= K must
   BOTH fire before predicting the next 1h return. Tests whether the two axes are
   orthogonal enough to stack.

2. CALIBRATION TABLE: each conviction tier's empirical holdout hit-rate IS the
   calibrated probability (reliability). Reports tier, n, hit, and deviation from
   0.5 so forecasts can quote honest % confidence.

3. ETH: streak-length gate on eth_1h (multi-res unavailable -- only 1h file exists).

Honest: fixed contrarian rule, quarantined holdout, permutation p-values.
ASCII-only output for Windows console.

Run:
    python -m fade.pipeline.conviction_combo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.conviction_gate import (
    HOLDOUT_FRAC, MIN_COVER, MULTI_FILES, _contrarian_grid, _perm_p,
    streak_length_gate,
)
from fade.pipeline.trend_structure import _signed_streak


def _combined_gate(holdout_frac: float = HOLDOUT_FRAC,
                   n_shuffles: int = 2000, seed: int = 0) -> dict:
    """Streak>=L AND multi-TF agree>=K on 1h grid, predict next 1h return."""
    df = load_ohlcv("btc_1h.csv")
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    fwd = ret.shift(-1)

    sigs = {Path(c).stem: _contrarian_grid(c) for c in MULTI_FILES if Path(c).exists()}
    frame = pd.DataFrame({"streak": streak, "streak_len": np.abs(streak), "ret_fwd": fwd},
                         index=df.index)
    for name, s in sigs.items():
        frame[name] = s
    frame = frame.dropna()
    n = len(frame)
    split = int(n * (1 - holdout_frac))
    hold = frame.iloc[split:]
    cols = list(sigs.keys())
    actual_up = (hold["ret_fwd"] > 0).astype(int).to_numpy()
    rng = np.random.default_rng(seed)

    rows = []
    for L in (2, 3, 4, 5):
        for K in (2, 3, 4):
            sl = hold["streak_len"].to_numpy() >= L
            sig_mat = hold[cols].to_numpy()
            pos = (sig_mat > 0).sum(axis=1)
            neg = (sig_mat < 0).sum(axis=1)
            multi_dir = np.where(pos >= K, 1, np.where(neg >= K, 0, -1))
            # streak contrarian direction
            streak_dir = (hold["streak"].to_numpy() < 0).astype(int)
            agree = (multi_dir >= 0) & sl & (multi_dir == streak_dir)
            k = int(agree.sum())
            if k < MIN_COVER:
                rows.append({"L": L, "K": K, "coverage": k, "status": "low"})
                continue
            pred = streak_dir[agree]
            act = actual_up[agree]
            hit = float(np.mean(pred == act))
            p = _perm_p(pred, act, hit, n_shuffles, rng)
            rows.append({
                "L": L, "K": K, "coverage": k,
                "cover_frac": round(k / len(hold), 4),
                "hit": round(hit, 4), "edge": round(hit - 0.5, 4),
                "p_value": round(p, 4), "status": "ok",
            })
    return {"status": "ok", "n_holdout": int(len(hold)), "rows": rows}


def _calibration_table(holdout_frac: float = HOLDOUT_FRAC) -> list[dict]:
    """Empirical hit-rates per tier = calibrated confidence labels."""
    tiers = []
    # streak-only tiers (btc 1h, same-bar for streak gate consistency)
    sg = streak_length_gate("btc_1h.csv", holdout_frac=holdout_frac)
    for x in sg.get("rows", []):
        if x.get("status") != "ok":
            continue
        tiers.append({
            "tier": f"streak>={x['min_len']}",
            "asset": "btc_1h", "n": x["coverage"],
            "calibrated_pct": round(x["hit"] * 100, 1),
            "edge": x["edge"],
        })
    # combined tiers
    cg = _combined_gate(holdout_frac=holdout_frac)
    for x in cg.get("rows", []):
        if x.get("status") != "ok":
            continue
        tiers.append({
            "tier": f"combo L>={x['L']} K>={x['K']}",
            "asset": "btc_1h", "n": x["coverage"],
            "calibrated_pct": round(x["hit"] * 100, 1),
            "edge": x["edge"],
        })
    # eth streak tiers
    esg = streak_length_gate("eth_1h.csv", holdout_frac=holdout_frac)
    for x in esg.get("rows", []):
        if x.get("status") != "ok":
            continue
        tiers.append({
            "tier": f"streak>={x['min_len']}",
            "asset": "eth_1h", "n": x["coverage"],
            "calibrated_pct": round(x["hit"] * 100, 1),
            "edge": x["edge"],
        })
    return tiers


def _print_combined(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print("1) COMBINED GATE (streak>=L AND multi-TF>=K, next 1h, BTC holdout)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}"); return
    print(f"  holdout n={r['n_holdout']}")
    print(f"  {'L':>4}{'K':>4}{'coverage':>10}{'cover%':>9}{'hit':>8}{'edge':>8}{'p':>8}")
    best = None
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['L']:>4}{x['K']:>4}{x['coverage']:>10}   low"); continue
        star = " *" if x["p_value"] <= 0.05 else ""
        print(f"  {x['L']:>4}{x['K']:>4}{x['coverage']:>10}{x['cover_frac']:>9}"
              f"{x['hit']:>8}{x['edge']:>+8}{x['p_value']:>8}{star}")
        if best is None or x["hit"] > best["hit"]:
            best = x
    if best:
        print(f"  BEST: L>={best['L']} K>={best['K']} -> {best['hit']:.1%} "
              f"(n={best['coverage']})")


def _print_cal(tiers: list[dict]) -> None:
    line = "=" * 70
    print("\n" + line)
    print("2) CALIBRATION TABLE (empirical hit = honest confidence %)")
    print(line)
    print(f"  {'tier':<22}{'asset':<10}{'n':>8}{'cal%':>8}{'edge':>8}")
    for t in sorted(tiers, key=lambda x: -x["calibrated_pct"]):
        print(f"  {t['tier']:<22}{t['asset']:<10}{t['n']:>8}"
              f"{t['calibrated_pct']:>7.1f}%{t['edge']:>+8.4f}")


def _print_eth() -> None:
    line = "=" * 70
    print("\n" + line)
    print("3) ETH STREAK GATE (eth_1h only -- no multi-res files for ETH)")
    print(line)
    r = streak_length_gate("eth_1h.csv")
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}"); return
    print(f"  holdout bars: {r['holdout_bars']}")
    print(f"  {'run>=L':>7}{'coverage':>10}{'hit':>8}{'edge':>8}{'p':>8}")
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['min_len']:>7}{x['coverage']:>10}   low"); continue
        star = " *" if x["p_value"] <= 0.05 else ""
        print(f"  {x['min_len']:>7}{x['coverage']:>10}{x['hit']:>8}"
              f"{x['edge']:>+8}{x['p_value']:>8}{star}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE conviction combo")
    parser.parse_args()
    comb = _combined_gate()
    _print_combined(comb)
    _print_cal(_calibration_table())
    _print_eth()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
