"""Sequence-length (memory-depth) sweep — how far back does the past matter?

Builds on sequence_patterns.py. For each resolution and each sequence length k,
it freezes per-k-gram directions on the dev split and measures the FROZEN n-gram
model's aggregate hit rate on the quarantined holdout, plus how many individual
patterns survive Bonferroni. Sweeping k shows the market's effective "memory
depth": the point where a longer look-back stops adding predictive power (and
starts overfitting to rare, sparsely-supported grams).

Answers "patterns of different time sequences": sequence LENGTH x resolution.

Honest protocol identical to sequence_patterns: k-gram uses bars strictly before
t, target at t, dev-frozen direction, 70/30 holdout, per-pattern permutation +
Bonferroni. Aggregate edge is measured only on unseen holdout bars.

Run:
    python -m fade.pipeline.sequence_sweep                       # default files
    python -m fade.pipeline.sequence_sweep btc_1h.csv --kmax 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fade.pipeline.sequence_patterns import run_sequences

DEFAULT_FILES = ["btc_5m.csv", "btc_15m.csv", "btc_30m.csv", "btc_1h.csv"]


def run_sweep(csv_path: str, kmin: int = 2, kmax: int = 8,
              alphabet: str = "sign", n_shuffles: int = 1500) -> dict:
    rows = []
    for k in range(kmin, kmax + 1):
        r = run_sequences(csv_path, alphabet=alphabet, k=k,
                          n_shuffles=n_shuffles)
        if r.get("status") != "ok":
            rows.append({"k": k, "status": r.get("status")})
            continue
        rows.append({
            "k": k,
            "n_patterns": r["n_patterns_tested"],
            "n_possible": (2 if alphabet == "sign" else 3) ** k,
            "agg_hit": r["aggregate_holdout_hit"],
            "agg_edge": r["aggregate_edge_vs_base"],
            "n_survive": r["n_survive_bonferroni"],
            "best": r["patterns"][0]["pattern"] if r["patterns"] else None,
            "best_hit": r["patterns"][0]["hold_hit"] if r["patterns"] else None,
            "best_edge": r["patterns"][0]["edge"] if r["patterns"] else None,
            "status": "ok",
        })
    ok = [x for x in rows if x.get("status") == "ok"]
    peak = max(ok, key=lambda x: (x["agg_edge"] or -9)) if ok else None
    return {
        "status": "ok" if ok else "insufficient_data",
        "asset": Path(csv_path).stem,
        "alphabet": alphabet,
        "rows": rows,
        "peak_k": peak["k"] if peak else None,
        "peak_edge": peak["agg_edge"] if peak else None,
    }


def _print(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print(f"FADE SEQUENCE-LENGTH SWEEP - {r.get('asset','?').upper()}  "
          f"(alphabet={r.get('alphabet')})")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  {'k':>3}{'patterns':>10}{'possible':>10}{'agg_hit':>9}"
          f"{'agg_edge':>10}{'survive':>9}{'best':>10}{'best_hit':>9}")
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['k']:>3}   {x.get('status')}")
            continue
        print(f"  {x['k']:>3}{x['n_patterns']:>10}{x['n_possible']:>10}"
              f"{x['agg_hit']:>9}{x['agg_edge']:>+10}{x['n_survive']:>9}"
              f"{str(x['best']):>10}{x['best_hit']:>9}")
    print(line)
    print(f"  PEAK: k={r['peak_k']} (aggregate edge {r['peak_edge']:+})  "
          f"-> effective memory depth")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE sequence-length sweep")
    parser.add_argument("csv", nargs="*", default=None)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--alphabet", choices=("sign", "mag3"), default="sign")
    args = parser.parse_args()
    files = args.csv if args.csv else DEFAULT_FILES
    for c in files:
        if not Path(c).exists():
            print(f"(skip missing {c})")
            continue
        _print(run_sweep(c, kmin=args.kmin, kmax=args.kmax, alphabet=args.alphabet))


if __name__ == "__main__":
    main()
