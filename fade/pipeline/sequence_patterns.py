"""Sequence (n-gram) prediction mechanism — do recent state SEQUENCES predict?

The core FADE engine is memoryless: it scores a single bar's atom-combo at time
T. It never asks whether the ORDERED SEQUENCE of recent bars ("up, up, down...")
carries predictive structure. This module adds that missing mechanism — a
temporal n-gram predictor — to hunt for behavioural patterns and trends.

State alphabet (compact, to avoid combinatorial overfitting):
    sign  : U (return>0) / D (return<=0)                      -> 2^k patterns
    mag3  : + (ret> +thr) / 0 (|ret|<=thr) / - (ret< -thr)    -> 3^k patterns

Mechanism: at each bar t, form the k-gram from bars [t-k .. t-1] (strictly past,
no look-ahead) and predict the direction of the return AT t. On the dev split we
measure each k-gram's next-up rate and freeze a direction; on the quarantined
holdout we score it, with a permutation p-value and Bonferroni correction across
all patterns of that k (honest multiple-comparison control).

This answers: is there momentum ("UUU -> U") or reversal ("UUU -> D") structure
that a single-bar model cannot see?

Run:
    python -m fade.pipeline.sequence_patterns btc_1h.csv
    python -m fade.pipeline.sequence_patterns btc_1h.csv --alphabet mag3 --k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv

MIN_DEV_SUPPORT = 30
MIN_HOLD_SUPPORT = 15


def _symbols(ret: np.ndarray, alphabet: str, thr: float) -> np.ndarray:
    if alphabet == "sign":
        return np.where(ret > 0, "U", "D").astype(object)
    # mag3
    s = np.full(len(ret), "0", dtype=object)
    s[ret > thr] = "+"
    s[ret < -thr] = "-"
    return s


def _kgrams(symbols: np.ndarray, k: int) -> list[str | None]:
    """k-gram ending at t-1 for each index t (None until enough history)."""
    out: list[str | None] = [None] * len(symbols)
    for t in range(k, len(symbols)):
        out[t] = "".join(symbols[t - k:t])
    return out


def run_sequences(csv_path: str, alphabet: str = "sign", k: int = 3,
                  holdout_frac: float = 0.30, thr: float = 0.003,
                  n_shuffles: int = 2000, seed: int = 0) -> dict:
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change().to_numpy()
    # target: direction of return at t; predictor: k-gram of [t-k..t-1]
    symbols = _symbols(ret, alphabet, thr)
    grams = _kgrams(symbols, k)

    idx = np.array([t for t in range(len(ret)) if grams[t] is not None
                    and np.isfinite(ret[t])])
    g = np.array([grams[t] for t in idx], dtype=object)
    up = (ret[idx] > 0).astype(int)

    n = len(idx)
    if n < 500:
        return {"status": "insufficient_data", "n": int(n)}
    split = int(n * (1.0 - holdout_frac))
    g_dev, up_dev = g[:split], up[:split]
    g_hold, up_hold = g[split:], up[split:]
    base_up_hold = float(up_hold.mean())

    # Freeze a direction per k-gram from dev.
    frozen: dict[str, int] = {}
    dev_rate: dict[str, float] = {}
    for pat in set(g_dev):
        mask = g_dev == pat
        if mask.sum() < MIN_DEV_SUPPORT:
            continue
        r = float(up_dev[mask].mean())
        dev_rate[pat] = r
        frozen[pat] = 1 if r >= 0.5 else 0

    rng = np.random.default_rng(seed)
    rows = []
    for pat, direction in frozen.items():
        hmask = g_hold == pat
        hn = int(hmask.sum())
        if hn < MIN_HOLD_SUPPORT:
            continue
        actual = up_hold[hmask]
        hit = float(np.mean(actual == direction))
        # permutation: random holdout bars of same size
        null = np.empty(n_shuffles)
        for i in range(n_shuffles):
            samp = rng.choice(up_hold, size=hn, replace=False)
            null[i] = np.mean(samp == direction)
        p = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)
        rows.append({
            "pattern": pat,
            "dir": "up" if direction == 1 else "down",
            "dev_rate": round(dev_rate[pat], 4),
            "hold_n": hn,
            "hold_hit": round(hit, 4),
            "edge": round(hit - (base_up_hold if direction == 1 else 1 - base_up_hold), 4),
            "p_value": round(p, 4),
        })

    rows.sort(key=lambda r: r["p_value"])
    n_tested = len(rows)
    bonf = 0.05 / n_tested if n_tested else float("nan")
    survive = [r for r in rows if r["p_value"] <= bonf]

    # Aggregate: does the frozen n-gram model beat base rate on the whole holdout?
    agg_hit = agg_n = 0
    for pat, direction in frozen.items():
        hmask = g_hold == pat
        if hmask.sum() == 0:
            continue
        agg_hit += int(np.sum(up_hold[hmask] == direction))
        agg_n += int(hmask.sum())
    agg_rate = round(agg_hit / agg_n, 4) if agg_n else None

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "alphabet": alphabet, "k": k,
        "n": int(n), "n_dev": int(split), "n_holdout": int(n - split),
        "holdout_base_up": round(base_up_hold, 4),
        "n_patterns_tested": n_tested,
        "bonferroni_alpha": round(bonf, 5) if bonf == bonf else None,
        "n_survive_bonferroni": len(survive),
        "aggregate_holdout_hit": agg_rate,
        "aggregate_edge_vs_base": round(agg_rate - max(base_up_hold, 1 - base_up_hold), 4)
                                   if agg_rate else None,
        "patterns": rows,
        "verdict": _verdict(rows, survive, agg_rate, base_up_hold),
    }


def _verdict(rows, survive, agg_rate, base_up) -> str:
    if not rows:
        return "INCONCLUSIVE - no pattern had enough support."
    if survive:
        names = ", ".join(f"{r['pattern']}->{r['dir']}" for r in survive)
        return f"SEQUENCE SIGNAL - {names} survive Bonferroni on unseen holdout."
    raw = [r for r in rows if r["p_value"] <= 0.05]
    if raw:
        return ("MILD - some patterns hit raw p<=0.05 but none survive Bonferroni; "
                "sequence structure is weak.")
    return "NO SEQUENCE SIGNAL - recent state order does not predict the next bar."


def _print(r: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print(f"FADE SEQUENCE PATTERNS - {r.get('asset','?').upper()}  "
          f"(alphabet={r.get('alphabet')}, k={r.get('k')})")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  bars: {r['n']:,} (dev {r['n_dev']} / holdout {r['n_holdout']})   "
          f"holdout base-up {r['holdout_base_up']}")
    print(f"  patterns tested: {r['n_patterns_tested']}   "
          f"Bonferroni alpha: {r['bonferroni_alpha']}   "
          f"survive: {r['n_survive_bonferroni']}")
    print(f"  aggregate n-gram holdout hit: {r['aggregate_holdout_hit']}  "
          f"(edge vs base {r['aggregate_edge_vs_base']})")
    print()
    print(f"  {'pattern':<10}{'dir':>5}{'dev':>8}{'hold_n':>8}{'hit':>8}{'edge':>8}{'p':>8}")
    for p in r["patterns"][:15]:
        print(f"  {p['pattern']:<10}{p['dir']:>5}{p['dev_rate']:>8}{p['hold_n']:>8}"
              f"{p['hold_hit']:>8}{p['edge']:>+8}{p['p_value']:>8}")
    print(line)
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE sequence (n-gram) patterns")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--alphabet", choices=("sign", "mag3"), default="sign")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--thr", type=float, default=0.003)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()
    if not Path(args.csv).exists():
        print(f"File not found: {args.csv}")
        sys.exit(1)
    _print(run_sequences(args.csv, alphabet=args.alphabet, k=args.k,
                         thr=args.thr, holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
