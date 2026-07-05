"""Magnitude threshold sweep — does predicting *material* moves beat direction-only?

Instead of scoring any up/down tick, rules can be mined and evaluated on moves
that exceed a minimum size (e.g. 0.3% in one bar). This separates genuine
directional skill from riding slow drift.

For each threshold the full strict holdout is re-run (re-mine on dev with the
new target, freeze, score on holdout). Reports hit-rate, drift-adjusted skill,
coverage fraction, and p-value.

Run:
    python -m fade.pipeline.magnitude_sweep btc_1h.csv
    python -m fade.pipeline.magnitude_sweep btc_15m.csv --thresholds 0,0.001,0.003,0.005
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from fade.config import Config
from fade.pipeline.holdout import holdout_test
from fade.utils.logging import get_logger

log = get_logger("magnitude_sweep")

# Default sweep in fractional return units (0.1% steps up to 1%).
DEFAULT_THRESHOLDS = (0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01)


def sweep(csv_path: str, thresholds=DEFAULT_THRESHOLDS,
          config: Config | None = None, n_shuffles: int = 300,
          seed: int = 0) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem
    rows = []
    for thr in thresholds:
        cfg = dataclasses.replace(config, move_threshold=thr)
        r = holdout_test(csv_path, config=cfg, n_shuffles=n_shuffles, seed=seed)
        row = {
            "threshold": thr,
            "threshold_pct": round(thr * 100, 3),
            "status": r.get("status"),
            "n_rules": r.get("n_stable_rules"),
            "coverage": r.get("coverage"),
            "hit_rate": r.get("holdout_hit_rate"),
            "lift": r.get("holdout_lift_vs_random"),
            "null_mean": r.get("null_mean"),
            "skill": (round(r["holdout_hit_rate"] - r["null_mean"], 4)
                      if r.get("holdout_hit_rate") is not None
                      and r.get("null_mean") is not None else None),
            "p_value": r.get("p_value"),
            "verdict": r.get("verdict"),
        }
        if r.get("status") == "ok" and r.get("coverage") and r.get("n_total"):
            row["coverage_frac"] = round(r["coverage"] / (r["n_holdout"] or 1), 4)
        rows.append(row)
        log.info("thr=%.4f hit=%s skill=%s cov=%s p=%s",
                 thr, row.get("hit_rate"), row.get("skill"),
                 row.get("coverage"), row.get("p_value"))

    ok = [r for r in rows if r.get("skill") is not None and r["skill"] == r["skill"]]
    best = max(ok, key=lambda r: r["skill"]) if ok else None
    return {"asset": asset, "rows": rows, "best": best}


def _print(res: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"FADE MAGNITUDE SWEEP - {res['asset'].upper()}  (strict holdout per threshold)")
    print(line)
    print(f"  {'thr%':>6}  {'rules':>6}  {'cover':>8}  {'cov%':>6}  "
          f"{'hit':>7}  {'skill':>8}  {'null':>7}  {'p':>8}")
    for r in res["rows"]:
        if r.get("status") != "ok":
            print(f"  {r['threshold_pct']:>5.2f}%  {r.get('status')}")
            continue
        covp = f"{r.get('coverage_frac', 0)*100:.0f}%" if r.get("coverage_frac") else "n/a"
        print(f"  {r['threshold_pct']:>5.2f}%  {r['n_rules']:>6}  {r['coverage']:>8}  "
              f"{covp:>6}  {r['hit_rate']:>7}  {r['skill']:>+8}  "
              f"{r['null_mean']:>7}  {r['p_value']:>8}")
    print(line)
    if res.get("best"):
        b = res["best"]
        print(f"  BEST skill: threshold={b['threshold_pct']}%  "
              f"skill={b['skill']:+.4f}  hit={b['hit_rate']}  p={b['p_value']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE magnitude threshold sweep")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="comma-separated fractions, e.g. 0,0.001,0.003")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    thresholds = (tuple(float(x) for x in args.thresholds.split(","))
                  if args.thresholds else DEFAULT_THRESHOLDS)
    _print(sweep(args.csv, thresholds))


if __name__ == "__main__":
    main()
