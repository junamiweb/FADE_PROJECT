"""Specificity-weighted aggregation — does un-diluting strong rules break 53%?

DIAGNOSIS (batch 15): a specific strong rule (streak run>=2 -> reversal, 54.6%
OOS) gets averaged back to ~53% because `evaluator.predict` sums confidence*dir
over ALL matching rules equally. A timestamp covered by one sharp rule + ten
generic conjunctions is dominated by the generic crowd.

FIX under test: weight each rule's vote by its SPECIFICITY, so sharp rules speak
louder than vague ones. We compare, on the same strict 70/30 holdout with frozen
dev rules (identical to holdout.py), several weighting schemes:

    equal        : current engine (vote = confidence * dir)                [baseline]
    size         : * (n_atoms)              — more atoms = more specific
    size2        : * (n_atoms^2)            — stronger specificity emphasis
    rarity       : * 1/sqrt(support)        — rarer rule = more specific
    edge         : * (confidence - 0.5)     — weight by demonstrated edge size
    size_edge    : * n_atoms * (confidence-0.5)
    argmax_spec  : IGNORE the crowd — use ONLY the single most specific rule
                   (max n_atoms, tie-broken by confidence) at each timestamp

No look-ahead: everything (rules, thresholds, specificity weights) is derived
from the development slice; the holdout is scored once. streak_signed is causal.

Run:
    python -m fade.pipeline.specificity_test              # btc_1h, core5_path
    python -m fade.pipeline.specificity_test --atomset path_min
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config, ATOM_SETS
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import load_ohlcv
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules

MIN_COVER = 50


def _rule_meta(rules: pd.DataFrame) -> pd.DataFrame:
    """Attach specificity metadata to frozen rules (from their event keys)."""
    meta = rules.copy()
    meta["n_atoms"] = [k.count("|") + 1 for k in meta.index]
    return meta


def _weighted_predict(events: pd.DataFrame, rules: pd.DataFrame,
                      scheme: str) -> pd.DataFrame:
    """Aggregate votes with a specificity weight per rule."""
    if rules.empty or events.empty:
        return pd.DataFrame(columns=["pred", "score"])
    matched = events[events["event"].isin(rules.index)].copy()
    if matched.empty:
        return pd.DataFrame(columns=["pred", "score"])

    conf = rules["confidence"].to_dict()
    direction = rules["direction"].to_dict()
    n_atoms = rules["n_atoms"].to_dict()
    support = rules["support"].to_dict() if "support" in rules else {}

    def _w(e: str) -> float:
        c = conf[e]
        na = n_atoms[e]
        sup = support.get(e, 1.0)
        edge = max(c - 0.5, 1e-6)
        if scheme == "equal":
            return c
        if scheme == "size":
            return c * na
        if scheme == "size2":
            return c * (na ** 2)
        if scheme == "rarity":
            return c / np.sqrt(max(sup, 1.0))
        if scheme == "edge":
            return edge
        if scheme == "size_edge":
            return na * edge
        return c  # fallback

    matched["dir"] = matched["event"].map(lambda e: 1.0 if direction[e] == 1 else -1.0)
    matched["vote"] = matched["event"].map(_w) * matched["dir"]
    matched["ts"] = matched.index
    grp = matched.groupby("ts")
    out = pd.DataFrame({"score": grp["vote"].sum()})
    out["pred"] = (out["score"] > 0).astype(int)
    return out[["pred", "score"]]


def _argmax_spec_predict(events: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Use only the single MOST specific rule per timestamp (n_atoms, then conf)."""
    if rules.empty or events.empty:
        return pd.DataFrame(columns=["pred", "score"])
    matched = events[events["event"].isin(rules.index)].copy()
    if matched.empty:
        return pd.DataFrame(columns=["pred", "score"])
    na = rules["n_atoms"].to_dict()
    conf = rules["confidence"].to_dict()
    direction = rules["direction"].to_dict()
    matched["ts"] = matched.index
    matched["na"] = matched["event"].map(na)
    matched["conf"] = matched["event"].map(conf)
    # rank: highest n_atoms, then highest confidence
    matched = matched.sort_values(["na", "conf"], ascending=False)
    best = matched.groupby("ts", sort=False).first()
    best["pred"] = best["event"].map(lambda e: direction[e])
    best["score"] = best["conf"] * np.where(best["pred"] == 1, 1, -1)
    return best[["pred", "score"]]


def _mechanism_gate_predict(events: pd.DataFrame, rules: pd.DataFrame,
                            marker: str = "streak_signed=") -> pd.DataFrame:
    """Path-rule track: where any marker (streak) rule fires, decide using ONLY
    marker rules; elsewhere fall back to the equal-weighted crowd. Tests whether
    letting the path mechanism speak alone (un-blended) recovers its edge."""
    if rules.empty or events.empty:
        return pd.DataFrame(columns=["pred", "score"])
    matched = events[events["event"].isin(rules.index)].copy()
    if matched.empty:
        return pd.DataFrame(columns=["pred", "score"])
    conf = rules["confidence"].to_dict()
    direction = rules["direction"].to_dict()
    matched["ts"] = matched.index
    matched["is_marker"] = matched["event"].str.contains(marker, regex=False)
    matched["vote"] = matched["event"].map(
        lambda e: conf[e] * (1.0 if direction[e] == 1 else -1.0))

    out = {}
    for ts, g in matched.groupby("ts", sort=False):
        gm = g[g["is_marker"]]
        use = gm if not gm.empty else g
        score = float(use["vote"].sum())
        out[ts] = 1 if score > 0 else 0
    res = pd.DataFrame({"pred": pd.Series(out)})
    res["score"] = 0.0
    return res[["pred", "score"]]


def run_specificity(csv_path: str = "btc_1h.csv", atomset: str = "core5_path",
                    holdout_frac: float = 0.30, seed: int = 0,
                    n_shuffles: int = 500) -> dict:
    cfg = dataclasses.replace(Config(), atom_columns=ATOM_SETS[atomset])
    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, cfg)
    fwd = atoms_mod.forward_return(df, cfg.forward_horizon).reindex(atoms.index)
    n = len(atoms)
    split = int(n * (1 - holdout_frac))
    dev_a, hold_a = atoms.iloc[:split], atoms.iloc[split:]
    dev_f, hold_f = fwd.iloc[:split], fwd.iloc[split:]

    bt = walk_forward(dev_a, dev_f, cfg)
    frozen = _select_stable_rules(bt.stability, cfg)
    if frozen.empty:
        return {"status": "no_rules", "asset": Path(csv_path).stem, "atomset": atomset}
    # attach support from the dev stability table where available
    if "support" not in frozen.columns and "avg_support" in bt.stability.columns:
        frozen = frozen.join(bt.stability["avg_support"].rename("support"))
    frozen = _rule_meta(frozen)

    thr = ev.compute_thresholds(dev_a, cfg)
    disc = ev.discretize(hold_a, thr)
    all_events = ev.build_events(disc, cfg, allowed=set(frozen.index))

    rng = np.random.default_rng(seed)
    schemes = ["equal", "size", "size2", "rarity", "edge", "size_edge",
               "argmax_spec", "mechanism_gate"]
    has_marker = any("streak_signed=" in k for k in frozen.index)
    rows = []
    for sc in schemes:
        if sc == "mechanism_gate":
            if not has_marker:
                rows.append({"scheme": sc, "status": "no_marker"})
                continue
            preds = _mechanism_gate_predict(all_events, frozen)
        elif sc == "argmax_spec":
            preds = _argmax_spec_predict(all_events, frozen)
        else:
            preds = _weighted_predict(all_events, frozen, sc)
        if preds.empty:
            rows.append({"scheme": sc, "status": "no_coverage"})
            continue
        valid = hold_f.reindex(preds.index).notna()
        pv = preds["pred"][valid].to_numpy()
        fv = hold_f.reindex(preds.index)[valid].to_numpy()
        ps, act = score_predictions(pv, fv, cfg.move_threshold)
        if len(ps) < MIN_COVER:
            rows.append({"scheme": sc, "status": "low_cover", "coverage": int(len(ps))})
            continue
        hit = float(np.mean(ps == act))
        null = np.empty(n_shuffles)
        for i in range(n_shuffles):
            null[i] = np.mean(ps == rng.permutation(act))
        p = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)
        rows.append({"scheme": sc, "status": "ok", "coverage": int(len(ps)),
                     "hit": round(hit, 4), "lift": round(hit - 0.5, 4),
                     "p_value": round(p, 4)})
    base = next((r["hit"] for r in rows if r.get("scheme") == "equal" and r.get("status") == "ok"), None)
    best = max((r for r in rows if r.get("status") == "ok"), key=lambda r: r["hit"], default=None)
    return {"status": "ok", "asset": Path(csv_path).stem, "atomset": atomset,
            "n_rules": int(len(frozen)), "baseline_hit": base,
            "best_scheme": best["scheme"] if best else None,
            "best_hit": best["hit"] if best else None, "rows": rows}


def _print(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print(f"FADE SPECIFICITY-WEIGHTED AGGREGATION - {r.get('asset','?').upper()} "
          f"[{r.get('atomset')}]")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  frozen rules: {r['n_rules']}   baseline(equal) hit: {r['baseline_hit']}")
    print(f"  {'scheme':<14}{'coverage':>10}{'hit':>9}{'lift':>9}{'p':>8}")
    for x in r["rows"]:
        if x.get("status") != "ok":
            print(f"  {x['scheme']:<14}   {x.get('status')} "
                  f"{x.get('coverage','') }")
            continue
        star = " *" if x["p_value"] <= 0.05 else ""
        delta = ""
        if r["baseline_hit"] is not None and x["scheme"] != "equal":
            delta = f"  (d{x['hit'] - r['baseline_hit']:+.4f})"
        print(f"  {x['scheme']:<14}{x['coverage']:>10}{x['hit']:>9}"
              f"{x['lift']:>+9}{x['p_value']:>8}{star}{delta}")
    print(line)
    print(f"  BEST: {r['best_scheme']} @ {r['best_hit']}  "
          f"(baseline {r['baseline_hit']})")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE specificity-weighted test")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--atomset", default="core5_path", choices=list(ATOM_SETS))
    args = parser.parse_args()
    _print(run_specificity(args.csv, atomset=args.atomset))


if __name__ == "__main__":
    main()
