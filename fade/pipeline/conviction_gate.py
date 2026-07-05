"""Conviction gating -- trade coverage for accuracy on the reversal mechanism.

generalization_why showed the edge is real everywhere but thin unless the local
mechanism is strong. Instead of predicting every bar, this GATES predictions on
conviction and maps the precision/coverage frontier on the strict holdout:

1. STREAK-LENGTH GATE: the pure contrarian rule "predict against a run of length
   >= L". As L rises, coverage falls but (hypothesis) accuracy rises -- a clean
   read of how much signal concentrates in long runs.

2. MULTI-RESOLUTION AGREEMENT GATE: align contrarian streak signals from several
   timeframes on the 1h grid and predict the next 1h return only when >= K of
   them agree. Revisits the earlier unanimous ~62% (n=265) with an explicit
   coverage/accuracy curve so we can see if it is real or sparse-noise.

Honest protocol: streak at bar t uses bars strictly before t; the target is the
return at t (gate 1) / next 1h return (gate 2). Everything is measured on the
quarantined last 30% -- no dev fitting is required because the contrarian rule is
a fixed definition (against the run), not a mined parameter. Permutation p-values
against a random-direction null of the same coverage.

Run:
    python -m fade.pipeline.conviction_gate
    python -m fade.pipeline.conviction_gate --asset btc_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.trend_structure import _signed_streak

HOLDOUT_FRAC = 0.30
MULTI_FILES = ["btc_5m.csv", "btc_15m.csv", "btc_30m.csv", "btc_1h.csv"]
MIN_COVER = 40


def _perm_p(pred: np.ndarray, actual_up: np.ndarray, hit: float,
            n_shuffles: int, rng) -> float:
    """Null = random +/-1 predictions of same size vs the actual outcomes."""
    k = len(actual_up)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        d = rng.choice([0, 1], size=k)
        null[i] = np.mean(d == actual_up)
    return (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)


def streak_length_gate(csv_path: str, holdout_frac: float = HOLDOUT_FRAC,
                       n_shuffles: int = 2000, seed: int = 0) -> dict:
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    up = (ret > 0).astype(int)
    n = len(ret)
    split = int(n * (1 - holdout_frac))
    hold = np.zeros(n, dtype=bool)
    hold[split:] = True
    hold &= np.isfinite(ret)

    rng = np.random.default_rng(seed)
    total_hold = int(hold.sum())
    rows = []
    for L in range(1, 9):
        sel = hold & (np.abs(streak) >= L)
        k = int(sel.sum())
        if k < MIN_COVER:
            rows.append({"min_len": L, "coverage": k, "status": "low"})
            continue
        # contrarian: predict opposite of the run's sign
        pred = (streak[sel] < 0).astype(int)  # down-run -> predict up(1)
        actual = up[sel]
        hit = float(np.mean(pred == actual))
        p = _perm_p(pred, actual, hit, n_shuffles, rng)
        rows.append({"min_len": L, "coverage": k,
                     "cover_frac": round(k / total_hold, 4),
                     "hit": round(hit, 4), "edge": round(hit - 0.5, 4),
                     "p_value": round(p, 4), "status": "ok"})
    return {"status": "ok", "asset": Path(csv_path).stem,
            "holdout_bars": total_hold, "rows": rows}


def _contrarian_grid(csv: str, grid: str = "1h") -> pd.Series:
    df = load_ohlcv(csv)
    r = df["close"].pct_change()
    streak = _signed_streak(r.to_numpy())
    sig = pd.Series(0.0, index=df.index)
    mask = np.abs(streak) >= 2
    sig.iloc[mask] = -np.sign(streak[mask])
    return sig.resample(grid).last()


def multi_res_gate(files: list[str] | None = None, holdout_frac: float = HOLDOUT_FRAC,
                   n_shuffles: int = 2000, seed: int = 0) -> dict:
    files = files or MULTI_FILES
    sigs = {}
    for c in files:
        if Path(c).exists():
            sigs[Path(c).stem] = _contrarian_grid(c)
    if len(sigs) < 2:
        return {"status": "insufficient_files"}
    target = load_ohlcv("btc_1h.csv")["close"].pct_change().shift(-1)
    df = pd.DataFrame(sigs)
    df["ret"] = target
    df = df.dropna()
    n = len(df)
    split = int(n * (1 - holdout_frac))
    hold = df.iloc[split:]
    cols = list(sigs)
    actual_up = (hold["ret"] > 0).astype(int).to_numpy()
    rng = np.random.default_rng(seed)

    rows = []
    n_tf = len(cols)
    for K in range(1, n_tf + 1):
        # net signed agreement; require >= K timeframes agreeing on one side
        sig_mat = hold[cols].to_numpy()
        pos = (sig_mat > 0).sum(axis=1)
        neg = (sig_mat < 0).sum(axis=1)
        direction = np.where(pos >= K, 1, np.where(neg >= K, 0, -1))
        sel = direction >= 0
        k = int(sel.sum())
        if k < MIN_COVER:
            rows.append({"min_agree": K, "coverage": k, "status": "low"})
            continue
        hit = float(np.mean(direction[sel] == actual_up[sel]))
        p = _perm_p(direction[sel], actual_up[sel], hit, n_shuffles, rng)
        rows.append({"min_agree": K, "coverage": k,
                     "cover_frac": round(k / len(hold), 4),
                     "hit": round(hit, 4), "edge": round(hit - 0.5, 4),
                     "p_value": round(p, 4), "status": "ok"})
    return {"status": "ok", "n_holdout": int(len(hold)),
            "timeframes": cols, "rows": rows}


def _print_len(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print(f"1) STREAK-LENGTH GATE - {r.get('asset','?').upper()} "
          f"(contrarian, holdout)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}"); return
    print(f"  holdout bars: {r['holdout_bars']}")
    print(f"  {'run>=L':>7}{'coverage':>10}{'cover%':>9}{'hit':>8}{'edge':>8}{'p':>8}")
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['min_len']:>7}{x['coverage']:>10}   low"); continue
        star = " *" if x["p_value"] <= 0.05 else ""
        print(f"  {x['min_len']:>7}{x['coverage']:>10}{x['cover_frac']:>9}"
              f"{x['hit']:>8}{x['edge']:>+8}{x['p_value']:>8}{star}")


def _print_multi(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print("2) MULTI-RESOLUTION AGREEMENT GATE (predict next 1h, holdout)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}"); return
    print(f"  timeframes: {', '.join(r['timeframes'])}   holdout n={r['n_holdout']}")
    print(f"  {'agree>=K':>9}{'coverage':>10}{'cover%':>9}{'hit':>8}{'edge':>8}{'p':>8}")
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['min_agree']:>9}{x['coverage']:>10}   low"); continue
        star = " *" if x["p_value"] <= 0.05 else ""
        print(f"  {x['min_agree']:>9}{x['coverage']:>10}{x['cover_frac']:>9}"
              f"{x['hit']:>8}{x['edge']:>+8}{x['p_value']:>8}{star}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE conviction gating")
    parser.add_argument("--asset", default="btc_1h.csv")
    args = parser.parse_args()
    _print_len(streak_length_gate(args.asset))
    _print_multi(multi_res_gate())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
