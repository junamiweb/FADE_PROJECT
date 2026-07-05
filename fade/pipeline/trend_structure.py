"""Trend structure map — momentum vs mean-reversion by streak length & timescale.

The sequence test found that hourly BTC REVERSES after streaks. This maps that
structure cleanly: after a run of N consecutive same-direction bars, what is the
probability the next bar CONTINUES the run vs reverses? Sweeping N and running it
across resolutions (15m / 30m / 1h / daily) reveals WHERE and HOW STRONGLY the
market mean-reverts or trends.

Honest protocol: streak state at bar t uses only bars < t; outcome is bar t. We
report the holdout (last 30%) continuation rate with a permutation p-value vs the
base rate, so a "reversal" claim is tested on unseen data — not curve-fit.

Reading it:
    continuation < 0.5  -> MEAN-REVERSION (streak tends to flip)
    continuation > 0.5  -> MOMENTUM (streak tends to persist)

Run:
    python -m fade.pipeline.trend_structure                 # all default files
    python -m fade.pipeline.trend_structure btc_1h.csv --max-streak 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fade.core.data_loader import load_ohlcv

DEFAULT_FILES = ["btc_15m.csv", "btc_30m.csv", "btc_1h.csv", "btc_daily.csv"]
MIN_SUPPORT = 30


def _signed_streak(ret: np.ndarray) -> np.ndarray:
    """Signed run length ending at t-1 (history strictly before t).

    +m: m consecutive up bars just before t. -m: m consecutive down bars.
    """
    n = len(ret)
    prev_dir = np.where(ret > 0, 1, np.where(ret < 0, -1, 0))
    streak_before = np.zeros(n, dtype=int)
    run = 0
    for t in range(1, n):
        d = prev_dir[t - 1]
        if d == 0:
            run = 0
        elif run != 0 and np.sign(run) == d:
            run += d
        else:
            run = d
        streak_before[t] = run
    return streak_before


def run_trend_structure(csv_path: str, max_streak: int = 5,
                        holdout_frac: float = 0.30, n_shuffles: int = 2000,
                        seed: int = 0) -> dict:
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    up = (ret > 0).astype(int)
    finite = np.isfinite(ret)

    n = len(ret)
    split = int(n * (1.0 - holdout_frac))
    hold_mask = np.zeros(n, dtype=bool)
    hold_mask[split:] = True
    hold_mask &= finite

    base_up = float(up[hold_mask].mean())
    rng = np.random.default_rng(seed)
    up_hold_all = up[hold_mask]

    rows = []
    for m in range(1, max_streak + 1):
        for sign, label in ((m, f"+{m} up"), (-m, f"-{m} down")):
            # exact streak == m (cap the top bucket as ">=")
            if m == max_streak:
                sel = hold_mask & (np.abs(streak) >= m) & (np.sign(streak) == np.sign(sign))
                lbl = f"{'+' if sign > 0 else '-'}{m}+ {'up' if sign>0 else 'down'}"
            else:
                sel = hold_mask & (streak == sign)
                lbl = label
            cnt = int(sel.sum())
            if cnt < MIN_SUPPORT:
                rows.append({"streak": lbl, "n": cnt, "status": "low_support"})
                continue
            nxt_up = up[sel]
            cont_rate = float(nxt_up.mean()) if sign > 0 else float(1 - nxt_up.mean())
            # continuation = P(next same direction as streak)
            up_rate = float(nxt_up.mean())
            # permutation vs base
            null = np.empty(n_shuffles)
            for i in range(n_shuffles):
                samp = rng.choice(up_hold_all, size=cnt, replace=False)
                null[i] = samp.mean()
            # two-sided-ish: significance of deviation of up_rate from base
            dev = abs(up_rate - base_up)
            p = (1 + int(np.sum(np.abs(null - base_up) >= dev))) / (1 + n_shuffles)
            rows.append({
                "streak": lbl, "n": cnt,
                "next_up_rate": round(up_rate, 4),
                "continuation_rate": round(cont_rate, 4),
                "behaviour": "reversion" if cont_rate < 0.5 else "momentum",
                "p_value": round(p, 4),
                "status": "ok",
            })

    return {
        "status": "ok",
        "asset": Path(csv_path).stem,
        "holdout_base_up": round(base_up, 4),
        "rows": rows,
    }


def _print(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print(f"FADE TREND STRUCTURE - {r.get('asset','?').upper()}  "
          f"(streak -> next; holdout)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  holdout base up-rate: {r['holdout_base_up']}")
    print()
    print(f"  {'streak':<12}{'n':>7}{'next_up':>9}{'continue':>10}{'behaviour':>12}{'p':>8}")
    for x in r["rows"]:
        if x["status"] != "ok":
            print(f"  {x['streak']:<12}{x['n']:>7}   {x['status']}")
            continue
        star = " *" if x["p_value"] <= 0.05 else ""
        print(f"  {x['streak']:<12}{x['n']:>7}{x['next_up_rate']:>9}"
              f"{x['continuation_rate']:>10}{x['behaviour']:>12}{x['p_value']:>8}{star}")
    print(line)
    print("  continuation<0.5 = mean-reversion (flips) | >0.5 = momentum (persists)")
    print("  * = deviation from base significant (p<=0.05)")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE trend structure map")
    parser.add_argument("csv", nargs="*", default=None)
    parser.add_argument("--max-streak", type=int, default=5)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()
    files = args.csv if args.csv else DEFAULT_FILES
    for c in files:
        if not Path(c).exists():
            print(f"(skip missing {c})")
            continue
        _print(run_trend_structure(c, max_streak=args.max_streak,
                                   holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
