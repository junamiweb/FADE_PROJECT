"""Training suite — run the honest learning simulation across many tables/ranges.

The point is breadth of evidence, not cherry-picking: we run FADE's simulated
live learning over several assets, resolutions, and time windows, then look at
the DISTRIBUTION of out-of-sample results. A signal that only shows up in one
lucky slice is noise; one that recurs across assets and eras is real.

Every job obeys the scope guard (no look-ahead): each learning_sim checkpoint
mines only on revealed data and scores the still-hidden future.

Run:
    python -m fade.pipeline.training_suite
    python -m fade.pipeline.training_suite --checkpoints 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fade.config import Config, lean_config
from fade.pipeline.learning_sim import run_learning_sim
from fade.utils.logging import get_logger

log = get_logger("suite")

# (label, csv, range_start, range_end). Different tables + different eras.
DEFAULT_JOBS = [
    ("btc_1h  full", "btc_1h.csv", 0.00, 1.00),
    ("btc_1h  early", "btc_1h.csv", 0.00, 0.50),
    ("btc_1h  late", "btc_1h.csv", 0.50, 1.00),
    ("eth_1h  full", "eth_1h.csv", 0.00, 1.00),
    ("eth_1h  early", "eth_1h.csv", 0.00, 0.50),
    ("eth_1h  late", "eth_1h.csv", 0.50, 1.00),
    ("btc_15m full", "btc_15m.csv", 0.00, 1.00),
    ("btc_30m full", "btc_30m.csv", 0.00, 1.00),
    ("btc_5m  full", "btc_5m.csv", 0.00, 1.00),
    ("eth_15m full", "eth_15m.csv", 0.00, 1.00),
    ("eth_30m full", "eth_30m.csv", 0.00, 1.00),
]


def run_suite(jobs=None, n_checkpoints: int = 8, config: Config | None = None) -> dict:
    config = config or lean_config()
    jobs = jobs or DEFAULT_JOBS
    results = []
    for label, csv, rs, re in jobs:
        if not Path(csv).exists():
            log.warning("skip %s (missing %s)", label, csv)
            results.append({"label": label, "status": "missing_file"})
            continue
        log.info("RUN %-14s %s  range=[%.2f,%.2f]", label, csv, rs, re)
        try:
            r = run_learning_sim(csv, n_checkpoints=n_checkpoints,
                                 range_start=rs, range_end=re, config=config)
        except Exception as exc:  # keep the suite going; record the failure
            log.error("job %s failed: %s", label, exc)
            results.append({"label": label, "status": f"error: {exc}"})
            continue
        r["label"] = label
        results.append(r)
        cum = r.get("final_cum_hit")
        imp = r.get("improvement")
        log.info("  -> cum_hit=%s improvement=%s", cum, imp)

    ok = [r for r in results if r.get("status") == "ok" and "final_cum_hit" in r]
    summary = {"n_jobs": len(results), "n_ok": len(ok), "results": results}
    if ok:
        cums = np.array([r["final_cum_hit"] for r in ok])
        imps = np.array([r["improvement"] for r in ok if "improvement" in r])
        summary["cum_hit_mean"] = round(float(cums.mean()), 4)
        summary["cum_hit_min"] = round(float(cums.min()), 4)
        summary["cum_hit_max"] = round(float(cums.max()), 4)
        summary["frac_above_0.5"] = round(float(np.mean(cums > 0.5)), 3)
        if imps.size:
            summary["improvement_mean"] = round(float(imps.mean()), 4)
            summary["frac_improving"] = round(float(np.mean(imps > 0)), 3)

    out_path = config.output_dir / "training_suite.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    summary["_saved"] = str(out_path)
    return summary


def _print(s: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("FADE TRAINING SUITE  (out-of-sample learning across tables & eras)")
    print(line)
    print(f"  {'job':<16}{'bars':>8}{'cum_hit':>9}{'improve':>9}{'scored':>8}  status")
    for r in s["results"]:
        if r.get("status") != "ok":
            print(f"  {r.get('label', '?'):<16}{'':>8}{'':>9}{'':>9}{'':>8}  "
                  f"{r.get('status')}")
            continue
        print(f"  {r['label']:<16}{r.get('n_bars', 0):>8}"
              f"{r.get('final_cum_hit', float('nan')):>9.4f}"
              f"{r.get('improvement', float('nan')):>+9.4f}"
              f"{r.get('n_scored', 0):>8}  ok")
    print(line)
    if "cum_hit_mean" in s:
        print(f"  cum-hit  mean={s['cum_hit_mean']}  "
              f"[{s['cum_hit_min']}, {s['cum_hit_max']}]   "
              f"frac>0.5 = {s['frac_above_0.5']}")
        if "improvement_mean" in s:
            print(f"  improvement mean={s['improvement_mean']}  "
                  f"frac improving = {s['frac_improving']}")
        edge = s["cum_hit_mean"] - 0.5
        verdict = ("consistent positive edge" if s["frac_above_0.5"] >= 0.75 and edge > 0.01
                   else "mixed / marginal" if edge > 0
                   else "no edge")
        print(f"  VERDICT: {verdict}")
    print(f"  saved: {s.get('_saved')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE training suite")
    parser.add_argument("--checkpoints", type=int, default=8)
    args = parser.parse_args()
    _print(run_suite(n_checkpoints=args.checkpoints))


if __name__ == "__main__":
    main()
