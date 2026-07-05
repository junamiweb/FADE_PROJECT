"""Lean atom-set search - find the optimal orthogonal 3-4 atom set.

FADE's ceiling was ~53% with bloated, collinear atom sets. A lean 4-atom set
path_min = (return_1h, volatility, volume_zscore, streak_signed) reached 54.05%
out-of-sample on btc_1h.csv: dropping collinear atoms stops the strong signed-
streak (path) signal from being diluted by generic conjunction rules.

This module searches a small, bounded family of LEAN, ORTHOGONAL atom sets to
try to beat path_min on strict out-of-sample hit rate. Design rules:

  * streak_signed (the orthogonal PATH dimension) is in EVERY candidate.
  * Pick AT MOST ONE atom from each semantic cluster to stay orthogonal:
        MOMENTUM  : return_1h, close_pos, return_6h
        VOLATILITY: volatility, range_pct
        VOLUME    : volume_zscore, volume_trend
  * 3-atom sets: {momentum, volatility, streak} and {momentum, volume, streak}
  * 4-atom sets: {momentum, volatility, volume, streak}
  * Plus the known baselines path_min and core5 for reference.

Each candidate is scored with fade.pipeline.holdout.holdout_test (strict 70/30
quarantined holdout + permutation p-value). holdout.py computes a return_6h
momentum baseline, so sets WITHOUT return_6h raise a KeyError there; we wrap the
call in try/except and record status="error" rather than touching core files.

Run:
    python -m fade.pipeline.lean_search
    python -m fade.pipeline.lean_search --csv btc_1h.csv --min-coverage 3000
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from fade.config import ATOM_COLUMNS, Config
from fade.pipeline.holdout import P_VALUE_MAX, holdout_test

PATH_ATOM = "streak_signed"
MOMENTUM = ("return_1h", "close_pos", "return_6h")
VOLATILITY = ("volatility", "range_pct")
VOLUME = ("volume_zscore", "volume_trend")

PATH_MIN = ("return_1h", "volatility", "volume_zscore", "streak_signed")


def build_candidates() -> list[tuple[str, tuple[str, ...]]]:
    """Bounded, de-duplicated family of lean orthogonal atom sets.

    Returns a list of (label, atom_tuple). Labels are descriptive; the atom
    tuples are what actually drive the config.
    """
    seen: set[frozenset] = set()
    out: list[tuple[str, tuple[str, ...]]] = []

    def add(label: str, atoms: tuple[str, ...]) -> None:
        key = frozenset(atoms)
        if key in seen:
            return
        seen.add(key)
        out.append((label, atoms))

    # Reference baselines first.
    add("path_min", PATH_MIN)
    add("core5", ATOM_COLUMNS)

    # 3-atom: momentum + volatility + streak
    for m in MOMENTUM:
        for v in VOLATILITY:
            add(f"3:{m}+{v}", (m, v, PATH_ATOM))

    # 3-atom: momentum + volume + streak
    for m in MOMENTUM:
        for vol in VOLUME:
            add(f"3:{m}+{vol}", (m, vol, PATH_ATOM))

    # 4-atom: momentum + volatility + volume + streak
    for m in MOMENTUM:
        for v in VOLATILITY:
            for vol in VOLUME:
                add(f"4:{m}+{v}+{vol}", (m, v, vol, PATH_ATOM))

    return out


def evaluate(csv_path: str, atoms: tuple[str, ...]) -> dict:
    """Run the strict holdout for one atom set; never raises.

    Sets lacking return_6h make holdout.py's momentum baseline raise KeyError;
    that is caught and reported as status="error" (core files stay untouched).
    """
    config = dataclasses.replace(Config(), atom_columns=atoms)
    try:
        return holdout_test(csv_path, config=config)
    except Exception as exc:  # noqa: BLE001 - deliberately record, don't crash
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _sort_key(row: dict) -> tuple:
    """Rank: real+significant sets first, then by hit rate."""
    r = row["result"]
    ok = r.get("status") == "ok"
    hit = r.get("holdout_hit_rate", 0.0) or 0.0
    real = ok and r.get("coverage", 0) >= row["min_coverage"] and r.get("p_value", 1.0) <= P_VALUE_MAX
    return (1 if real else 0, hit)


def run(csv_path: str, min_coverage: int) -> list[dict]:
    rows: list[dict] = []
    candidates = build_candidates()
    print(f"Evaluating {len(candidates)} lean candidate sets on {csv_path} ...\n")
    for label, atoms in candidates:
        r = evaluate(csv_path, atoms)
        rows.append({"label": label, "atoms": atoms, "result": r, "min_coverage": min_coverage})
        status = r.get("status", "?")
        hit = r.get("holdout_hit_rate")
        hit_s = f"{hit:.4f}" if isinstance(hit, float) else "  -   "
        print(f"  [{status:>10}] hit={hit_s}  {label}  {atoms}")
    rows.sort(key=_sort_key, reverse=True)
    return rows


def _print_table(rows: list[dict], min_coverage: int) -> None:
    line = "=" * 96
    print("\n" + line)
    print("LEAN ATOM-SET SEARCH - RANKED (strict OOS holdout on btc_1h)")
    print(line)
    header = f"{'rank':>4}  {'hit':>7}  {'lift':>7}  {'p':>7}  {'cover':>7}  {'rules':>5}  {'real':>4}  label / atoms"
    print(header)
    print("-" * 96)
    for i, row in enumerate(rows, 1):
        r = row["result"]
        if r.get("status") != "ok":
            note = r.get("error", r.get("verdict", r.get("status", "?")))
            print(f"{i:>4}  {'-':>7}  {'-':>7}  {'-':>7}  {'-':>7}  {'-':>5}  {'-':>4}  "
                  f"{row['label']} [{r.get('status')}] {str(note)[:40]}")
            continue
        hit = r.get("holdout_hit_rate", 0.0)
        lift = r.get("holdout_lift_vs_random", 0.0)
        p = r.get("p_value", 1.0)
        cover = r.get("coverage", 0)
        rules = r.get("n_stable_rules", 0)
        real = "yes" if (cover >= min_coverage and p <= P_VALUE_MAX and lift > 0) else "no"
        print(f"{i:>4}  {hit:>7.4f}  {lift:>+7.4f}  {p:>7.4f}  {cover:>7}  {rules:>5}  {real:>4}  "
              f"{row['label']}")
    print(line)


def _print_verdict(rows: list[dict], min_coverage: int) -> None:
    line = "=" * 96
    real_rows = [
        row for row in rows
        if row["result"].get("status") == "ok"
        and row["result"].get("coverage", 0) >= min_coverage
        and row["result"].get("p_value", 1.0) <= P_VALUE_MAX
        and row["result"].get("holdout_lift_vs_random", 0.0) > 0
    ]
    print("\nVERDICT")
    print(line)
    if not real_rows:
        print("  No candidate produced a real (significant, well-covered) positive edge.")
        print(line + "\n")
        return

    best = real_rows[0]
    r = best["result"]
    print(f"  Best lean set : {best['label']}")
    print(f"  Atoms         : {best['atoms']}")
    print(f"  OOS hit-rate  : {r['holdout_hit_rate']:.4f}  (lift {r['holdout_lift_vs_random']:+.4f})")
    print(f"  p-value       : {r['p_value']:.4f}   coverage={r['coverage']}   rules={r['n_stable_rules']}")

    pm = next((row for row in rows if row["label"] == "path_min"), None)
    pm_hit = pm["result"].get("holdout_hit_rate") if pm else None
    baseline = pm_hit if isinstance(pm_hit, float) else 0.5405
    if r["holdout_hit_rate"] > baseline:
        print(f"  RESULT        : BEATS path_min baseline ({baseline:.4f}) "
              f"by {r['holdout_hit_rate'] - baseline:+.4f}.")
    else:
        print(f"  RESULT        : does NOT beat path_min baseline ({baseline:.4f}). "
              f"path_min remains best.")
    print(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Search lean orthogonal atom sets for max OOS hit rate.")
    ap.add_argument("--csv", default="btc_1h.csv", help="OHLCV csv path (default: btc_1h.csv)")
    ap.add_argument("--min-coverage", type=int, default=3000,
                    help="Min holdout coverage for a set to count as 'real' (default: 3000)")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: csv not found: {args.csv}")
        raise SystemExit(1)

    rows = run(args.csv, args.min_coverage)
    _print_table(rows, args.min_coverage)
    _print_verdict(rows, args.min_coverage)


if __name__ == "__main__":
    main()
