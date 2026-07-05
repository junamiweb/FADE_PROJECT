"""Generalization audit — multiple-comparison correction + temporal decay.

1. Run ALL tried atom sets on a unified 70/30 holdout (btc_1h) and apply
   Bonferroni + Holm correction to raw p-values.
2. Compare conviction edge in 2025–2026 vs earlier years (decay check).

Run:
    python -m fade.pipeline.generalization_audit
    python -m fade.pipeline.generalization_audit --csv btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import ATOM_SETS, Config
from fade.core.data_loader import load_ohlcv
from fade.pipeline.conviction_stability import _yearly_combo, _yearly_streak
from fade.pipeline.holdout import P_VALUE_MAX, holdout_test

# All atom sets explicitly tried across batches 15–29.
AUDITED_SETS = (
    "core5",
    "core5_path",
    "path_min",
    "path_lean3",
    "path_big",
    "path_both",
    "path_candles",
)

RECENT_YEARS = (2025, 2026)


def _holm_adjusted(p_values: list[float]) -> list[float]:
    """Holm step-down correction (sorted ascending)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adj = np.empty(m)
    for i in range(m):
        adj[i] = min(1.0, sorted_p[i] * (m - i))
    # enforce monotonicity
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i - 1])
    out = np.empty(m)
    out[order] = adj
    return out.tolist()


def run_multiple_comparison(csv_path: str = "btc_1h.csv",
                          holdout_frac: float = 0.30) -> dict:
    results = []
    for name in AUDITED_SETS:
        if name not in ATOM_SETS:
            continue
        cfg = dataclasses.replace(Config(), atom_columns=ATOM_SETS[name])
        try:
            r = holdout_test(csv_path, holdout_frac=holdout_frac, config=cfg)
        except Exception as exc:
            results.append({"atom_set": name, "status": "error", "error": str(exc)})
            continue
        results.append({
            "atom_set": name,
            "status": r.get("status"),
            "hit_rate": r.get("holdout_hit_rate"),
            "lift": r.get("holdout_lift_vs_random"),
            "p_raw": r.get("p_value"),
            "coverage": r.get("coverage"),
            "verdict": r.get("verdict"),
        })

    ok = [r for r in results if r.get("p_raw") is not None]
    m = len(ok)
    p_raws = [r["p_raw"] for r in ok]
    bonf = [min(1.0, p * m) for p in p_raws]
    holm = _holm_adjusted(p_raws)

    for r, pb, ph in zip(ok, bonf, holm):
        r["p_bonferroni"] = round(pb, 4)
        r["p_holm"] = round(ph, 4)
        r["sig_bonferroni"] = pb <= P_VALUE_MAX
        r["sig_holm"] = ph <= P_VALUE_MAX

    survivors_bonf = [r["atom_set"] for r in ok if r["sig_bonferroni"]]
    survivors_holm = [r["atom_set"] for r in ok if r["sig_holm"]]

    return {
        "csv": csv_path,
        "n_tests": m,
        "results": results,
        "survivors_bonferroni": survivors_bonf,
        "survivors_holm": survivors_holm,
        "verdict": (
            f"After Holm: {survivors_holm or 'NONE'} significant at p<={P_VALUE_MAX}. "
            f"Raw-only winner path_lean3 p={next((r['p_raw'] for r in ok if r['atom_set']=='path_lean3'), '?')} "
            f"-> Bonferroni {next((r['p_bonferroni'] for r in ok if r['atom_set']=='path_lean3'), '?')}."
        ),
    }


def run_decay_check(csv_1h: str = "btc_1h.csv") -> dict:
    """Compare 2025–2026 vs pre-2025 conviction metrics."""
    streak_rows = _yearly_streak(csv_1h)
    combo_rows = _yearly_combo(csv_1h)

    def _bucket(rows: list[dict], rule: str) -> dict:
        ok = [r for r in rows if r.get("rule") == rule and r.get("status") == "ok"]
        recent = [r for r in ok if r["year"] in RECENT_YEARS]
        earlier = [r for r in ok if r["year"] not in RECENT_YEARS]

        def _agg(group):
            if not group:
                return {"n_years": 0, "weighted_hit": None, "total_n": 0}
            total_n = sum(r["n"] for r in group)
            wh = sum(r["hit"] * r["n"] for r in group) / total_n
            return {
                "n_years": len(group),
                "years": [r["year"] for r in group],
                "weighted_hit": round(wh, 4),
                "edge": round(wh - 0.5, 4),
                "total_n": total_n,
            }

        recent_a = _agg(recent)
        earlier_a = _agg(earlier)
        delta = None
        if recent_a["weighted_hit"] is not None and earlier_a["weighted_hit"] is not None:
            delta = round(recent_a["weighted_hit"] - earlier_a["weighted_hit"], 4)

        return {
            "rule": rule,
            "recent_2025_2026": recent_a,
            "pre_2025": earlier_a,
            "delta_recent_minus_earlier": delta,
            "decay_flag": delta is not None and delta < -0.01,
        }

    streak_b = _bucket(streak_rows, "streak>=3")
    combo_b = _bucket(combo_rows, "combo r>=2 K>=3")

    any_decay = streak_b["decay_flag"] or combo_b["decay_flag"]
    return {
        "asset": Path(csv_1h).stem,
        "streak_ge3": streak_b,
        "combo_r2_k3": combo_b,
        "yearly_detail": {"streak": streak_rows, "combo": combo_rows},
        "verdict": (
            "DECAY DETECTED in 2025–2026 vs earlier — crypto microstructure may have eroded edge."
            if any_decay else
            "NO clear decay — 2025–2026 edge comparable to earlier years (within 1pp)."
        ),
    }


def _print_mc(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"MULTIPLE-COMPARISON AUDIT — {r['csv']}")
    print(line)
    print(f"  {'atom_set':<14}{'hit':>8}{'lift':>8}{'p_raw':>8}{'p_bonf':>8}{'p_holm':>8}{'sig':>6}")
    for row in r["results"]:
        if row.get("p_raw") is None:
            print(f"  {row['atom_set']:<14}  {row.get('status', '?')}")
            continue
        sig = "Y" if row.get("sig_holm") else "N"
        print(f"  {row['atom_set']:<14}{row.get('hit_rate', 0):>8.4f}"
              f"{row.get('lift', 0):>+8.4f}{row['p_raw']:>8.4f}"
              f"{row['p_bonferroni']:>8.4f}{row['p_holm']:>8.4f}{sig:>6}")
    print()
    print(f"  Holm survivors: {r['survivors_holm'] or 'NONE'}")
    print(f"  {r['verdict']}")
    print(line + "\n")


def _print_decay(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print(f"TEMPORAL DECAY CHECK — {r['asset']}")
    print(line)
    for key in ("streak_ge3", "combo_r2_k3"):
        b = r[key]
        print(f"  {b['rule']}:")
        rec = b["recent_2025_2026"]
        ear = b["pre_2025"]
        print(f"    2025–2026: hit={rec['weighted_hit']}  n={rec['total_n']}  years={rec['years']}")
        print(f"    pre-2025 : hit={ear['weighted_hit']}  n={ear['total_n']}")
        print(f"    delta    : {b['delta_recent_minus_earlier']:+}  decay={b['decay_flag']}")
    print()
    print(f"  {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE generalization audit")
    parser.add_argument("--csv", nargs="+", default=["btc_1h.csv"])
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()

    for csv in args.csv:
        if not Path(csv).exists():
            print(f"Missing: {csv}")
            continue
        _print_mc(run_multiple_comparison(csv, holdout_frac=args.holdout_frac))
        if csv.endswith("_1h.csv") or "_1h" in Path(csv).stem:
            _print_decay(run_decay_check(csv))


if __name__ == "__main__":
    main()
