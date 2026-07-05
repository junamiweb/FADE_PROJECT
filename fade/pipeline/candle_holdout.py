"""Holdout A/B: path_lean3 baseline vs path_candles (lean3 + candle flags).

Strict 70/30 quarantined holdout on btc_1h — same harness as holdout.py.
Reports whether named candlestick pattern atoms improve OOS hit rate or dilute
the lean baseline (honest negative if no gain).

Run:
    python -m fade.pipeline.candle_holdout
    python -m fade.pipeline.candle_holdout --csv btc_1h.csv
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from fade.config import ATOM_SETS, Config
from fade.pipeline.holdout import P_VALUE_MAX, holdout_test

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = "path_lean3"
CANDIDATE = "path_candles"


def run(csv_path: str, n_shuffles: int = 300, seed: int = 0) -> dict:
    """Compare baseline vs candle-augmented atom set on strict holdout."""
    results: dict[str, dict] = {}
    for name in (BASELINE, CANDIDATE):
        cfg = dataclasses.replace(Config(), atom_columns=ATOM_SETS[name])
        results[name] = holdout_test(
            csv_path, holdout_frac=0.30, n_shuffles=n_shuffles, seed=seed, config=cfg
        )
    return results


def _fmt_hit(r: dict) -> str:
    hit = r.get("holdout_hit_rate")
    return f"{hit:.4f}" if isinstance(hit, float) else "  -  "


def _print_comparison(results: dict[str, dict]) -> None:
    line = "=" * 72
    asset = results[BASELINE].get("asset", "?")
    print("\n" + line)
    print(f"CANDLE PATTERN HOLDOUT — {asset.upper()} (70/30 strict)")
    print(line)
    header = f"{'set':<14}  {'hit':>7}  {'lift':>7}  {'p':>7}  {'cover':>7}  {'rules':>5}  status"
    print(header)
    print("-" * 72)
    for name in (BASELINE, CANDIDATE):
        r = results[name]
        if r.get("status") != "ok":
            print(f"{name:<14}  {'-':>7}  {'-':>7}  {'-':>7}  {'-':>7}  {'-':>5}  {r.get('status')}")
            continue
        print(
            f"{name:<14}  {r['holdout_hit_rate']:>7.4f}  "
            f"{r['holdout_lift_vs_random']:>+7.4f}  {r['p_value']:>7.4f}  "
            f"{r['coverage']:>7}  {r['n_stable_rules']:>5}  {r.get('verdict', '')[:24]}"
        )
    print(line)

    base = results[BASELINE]
    cand = results[CANDIDATE]
    print("\nVERDICT")
    print(line)
    if base.get("status") != "ok" or cand.get("status") != "ok":
        print("  INCONCLUSIVE — one or both runs did not produce scorable holdout results.")
        print(line + "\n")
        return

    delta = cand["holdout_hit_rate"] - base["holdout_hit_rate"]
    print(f"  Baseline ({BASELINE})     : {_fmt_hit(base)}  (p={base['p_value']:.4f})")
    print(f"  With candles ({CANDIDATE}): {_fmt_hit(cand)}  (p={cand['p_value']:.4f})")
    print(f"  Delta (candles - lean3)   : {delta:+.4f}")

    cand_real = (
        cand.get("holdout_lift_vs_random", 0) > 0
        and cand.get("p_value", 1) <= P_VALUE_MAX
    )
    if delta > 0.001 and cand_real:
        print("  RECOMMENDATION            : INTEGRATE — candle flags improve holdout hit rate.")
    elif delta > 0:
        print("  RECOMMENDATION            : REJECT — marginal gain, not worth vocabulary bloat.")
    else:
        print("  RECOMMENDATION            : REJECT — candle flags do not beat path_lean3 baseline.")
    print(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Holdout A/B: path_lean3 vs path_candles on btc_1h."
    )
    ap.add_argument("--csv", default=str(REPO_ROOT / "btc_1h.csv"), help="OHLCV csv path")
    ap.add_argument("--shuffles", type=int, default=300, help="Permutation count")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: csv not found: {args.csv}")
        raise SystemExit(1)

    results = run(args.csv, n_shuffles=args.shuffles, seed=args.seed)
    _print_comparison(results)


if __name__ == "__main__":
    main()
