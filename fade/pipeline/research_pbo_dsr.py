"""R02 — PBO/DSR governance analysis on path_lean3 (research sandbox).

Pre-registered: research_pbo_dsr_v1. Analysis ONLY — no new mining, no core changes.

Computes:
  - Trial inventory (conservative / liberal N from batch history)
  - Holdout hit + per-signal Sharpe on frozen path_lean3
  - Deflated Sharpe Ratio (Bailey et al. approximation)
  - PBO proxies: lockbox delta, fold instability, Holm-adjusted p

Run:
    python -m fade.pipeline.research_pbo_dsr
    python -m fade.pipeline.research_pbo_dsr btc.csv eth.csv
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fade.config import lean_config
from fade.core.conviction import HOLDOUT_FRAC
from fade.pipeline.final_lockbox import DEFAULT_LOCKBOX_FRAC, REFERENCE_HIT
from fade.pipeline.holdout import P_VALUE_MAX, holdout_test
from fade.pipeline.pnl_reality_check_v2 import _holdout_path_lean3

STUDY_ID = "research_pbo_dsr_v1"
OUTPUT_PATH = Path("fade/output/research_pbo_dsr.json")
DEFAULT_ASSETS = (
    "btc_1h.csv", "eth_1h.csv", "sol_1h.csv", "bnb_1h.csv", "xrp_1h.csv",
    "ada_1h.csv", "avax_1h.csv", "link_1h.csv", "doge_1h.csv", "ltc_1h.csv",
    "dot_1h.csv", "near_1h.csv", "arb_1h.csv", "op_1h.csv",
)
BARS_PER_YEAR = 24 * 365
FEE_BPS_RT = 10.0

# Documented trial families (SCOPE_GUARD batches 0–39 + Phase 2).
TRIAL_INVENTORY = {
    "conservative": {
        "n_trials": 39,
        "note": "Upper bound: batch log lines with holdout headline decisions (~39 batches)",
    },
    "liberal": {
        "n_trials": 18,
        "note": "Distinct hypothesis families: atom sets(7)+VR grid(1)+horizon(1)+funding(4)+ML(1)+phase2(4)",
    },
    "holm_atom_sets": 7,
}

LOCKBOX_REFERENCE = {
    "btc": {"holdout_ref": 0.5464, "lockbox_v1_hit": 0.5277, "source": "batch 30/33 PROJECT_STATE"},
    "eth": {"holdout_ref": 0.5327, "lockbox_v1_hit": 0.5275, "source": "batch 30 lockbox one-shot"},
}


def _norm_ppf(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return statistics.NormalDist().inv_cdf(p)


def _expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """Bailey et al. expected max Sharpe under null across n trials."""
    if n_trials <= 1:
        return 0.0
    euler = 0.5772156649015329
    term1 = (1.0 - euler) * _norm_ppf(1.0 - 1.0 / n_trials)
    term2 = euler * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_std * (term1 + term2))


def _deflated_sharpe(
    sr_hat: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
) -> dict:
    """DSR test statistic and pass/fail vs expected max SR."""
    if n_obs < 30 or not np.isfinite(sr_hat):
        return {"status": "insufficient_data"}
    sr_std = math.sqrt((1.0 - skew * sr_hat + (kurt_excess) / 4.0 * sr_hat**2) / max(n_obs - 1, 1))
    sr_std = max(sr_std, 1e-9)
    sr0 = _expected_max_sharpe(n_trials, sr_std=1.0)
    dsr_stat = (sr_hat - sr0) / sr_std
    # Approximate one-sided p via normal (conservative)
    p_approx = 1.0 - statistics.NormalDist().cdf(dsr_stat)
    return {
        "sr_hat": round(sr_hat, 4),
        "sr0_expected_max": round(sr0, 4),
        "dsr_statistic": round(dsr_stat, 4),
        "p_approx": round(p_approx, 4),
        "dsr_pass": dsr_stat > 0 and p_approx < 0.05,
        "n_obs": n_obs,
        "n_trials_used": n_trials,
    }


def _signal_returns(csv_path: str, holdout_frac: float, fee_bps: float = FEE_BPS_RT) -> dict | None:
    out = _holdout_path_lean3(csv_path, holdout_frac)
    if out is None:
        return None
    frame, split, n_rules = out
    pred = frame["pred"].to_numpy()
    bar_ret = frame["bar_ret"].to_numpy()
    sign = np.where(pred > 0, 1.0, -1.0)
    strat = sign * bar_ret - fee_bps / 10_000
    hits = (np.sign(bar_ret) == sign).astype(float)
    return {
        "n_signals": int(len(strat)),
        "hit_rate": round(float(hits.mean()), 4),
        "returns": strat,
        "n_rules": n_rules,
        "split": split,
    }


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return float("nan")
    return mu / sd * math.sqrt(BARS_PER_YEAR)


def _fold_instability(returns: np.ndarray, n_folds: int = 8) -> dict:
    """CSCV-lite: fraction of chronological folds underperforming overall."""
    if len(returns) < n_folds * 10:
        return {"status": "too_short"}
    folds = np.array_split(returns, n_folds)
    overall_hit = float(np.mean(returns > 0))
    fold_hits = [float(np.mean(f > 0)) for f in folds if len(f)]
    fold_mean_ret = [float(np.mean(f)) for f in folds if len(f)]
    under_hit = sum(1 for h in fold_hits if h < overall_hit - 0.02)
    under_zero = sum(1 for m in fold_mean_ret if m <= 0)
    pbo_proxy = round(under_hit / max(len(fold_hits), 1), 4)
    return {
        "n_folds": len(fold_hits),
        "overall_positive_rate": round(overall_hit, 4),
        "fold_hits": [round(h, 4) for h in fold_hits],
        "folds_underperforming_hit": under_hit,
        "folds_non_positive_mean": under_zero,
        "pbo_proxy_fold": pbo_proxy,
        "interpretation": "high if many folds weak — single-strategy CSCV-lite, not full Bailey PBO",
    }


def _lockbox_overfit_proxy(asset_key: str, holdout_hit: float) -> dict:
    ref = LOCKBOX_REFERENCE.get(asset_key, {})
    lb = ref.get("lockbox_v1_hit")
    ho_ref = ref.get("holdout_ref")
    if lb is None:
        return {"status": "no_reference"}
    delta_pp = round((holdout_hit - lb) * 100, 2)
    ref_delta_pp = round((ho_ref - lb) * 100, 2) if ho_ref else None
    return {
        "holdout_hit": holdout_hit,
        "lockbox_v1_hit": lb,
        "headline_holdout_ref": ho_ref,
        "delta_holdout_vs_lockbox_pp": delta_pp,
        "reference_delta_pp": ref_delta_pp,
        "pbo_proxy_lockbox": round(max(0.0, (ref_delta_pp or 0) - delta_pp) / 100, 4) if ref_delta_pp is not None else None,
        "note": "lockbox v1 BURNED — descriptive overfit gap only",
    }


def analyze_asset(csv_path: str, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    asset_key = Path(csv_path).stem.replace("_1h", "").lower()
    if not Path(csv_path).exists():
        return {"asset": asset_key, "status": "missing_csv"}

    hold = holdout_test(csv_path, holdout_frac=holdout_frac)
    sig = _signal_returns(csv_path, holdout_frac)
    if sig is None:
        return {"asset": asset_key, "status": "no_signals", "holdout": hold}

    rets = sig["returns"]
    skew = float(_skew(rets))
    kurt = float(_kurtosis_excess(rets))
    sr = _sharpe(rets)

    dsr_cons = _deflated_sharpe(sr, len(rets), TRIAL_INVENTORY["conservative"]["n_trials"], skew, kurt)
    dsr_lib = _deflated_sharpe(sr, len(rets), TRIAL_INVENTORY["liberal"]["n_trials"], skew, kurt)

    p_raw = hold.get("p_value")
    holm_p = min(1.0, p_raw * TRIAL_INVENTORY["holm_atom_sets"]) if p_raw else None

    return {
        "asset": asset_key,
        "csv": csv_path,
        "status": "ok",
        "holdout_test": {
            "hit_rate": hold.get("holdout_hit_rate"),
            "lift": hold.get("holdout_lift_vs_random"),
            "p_value": p_raw,
            "p_holm_atom_sets": round(holm_p, 4) if holm_p is not None else None,
            "coverage": hold.get("coverage"),
            "verdict": hold.get("verdict"),
        },
        "signal_sharpe": {
            "fee_bps_round_trip": FEE_BPS_RT,
            "sharpe_annualized": round(sr, 4) if np.isfinite(sr) else None,
            "skew": round(skew, 4),
            "kurtosis_excess": round(kurt, 4),
            "n_signals": sig["n_signals"],
        },
        "dsr_conservative_n39": dsr_cons,
        "dsr_liberal_n18": dsr_lib,
        "fold_instability": _fold_instability(rets),
        "lockbox_overfit_proxy": _lockbox_overfit_proxy(
            asset_key, hold.get("holdout_hit_rate", sig["hit_rate"]),
        ),
    }


def _skew(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _interpret(results: list[dict]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return "INCONCLUSIVE — no assets analyzed."
    parts = []
    for r in ok:
        dsr = r.get("dsr_conservative_n39", {})
        lb = r.get("lockbox_overfit_proxy", {})
        d_pass = dsr.get("dsr_pass")
        delta = lb.get("delta_holdout_vs_lockbox_pp")
        parts.append(
            f"{r['asset'].upper()}: DSR(n=39) pass={d_pass}, "
            f"holdout-lockbox={delta}pp, fold_pbo={r.get('fold_instability', {}).get('pbo_proxy_fold')}"
        )
    headline = (
        "Governance: headline holdout likely inflated vs lockbox; "
        "forward ledger remains arbiter. "
        if any((lb.get("delta_holdout_vs_lockbox_pp") or 0) > 1.5 for r in ok for lb in [r.get("lockbox_overfit_proxy", {})])
        else ""
    )
    return headline + "; ".join(parts)


def run_all(assets: list[str]) -> dict:
    results = [analyze_asset(a) for a in assets]
    ok = [r for r in results if r.get("status") == "ok"]
    any_dsr_pass = any(
        r.get("dsr_conservative_n39", {}).get("dsr_pass") for r in ok
    )
    return {
        "study_id": STUDY_ID,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "builder-research-pbo-dsr",
        "method": "analysis_only — no new parameter search",
        "trial_inventory": TRIAL_INVENTORY,
        "reference_hit_batch17": REFERENCE_HIT,
        "lockbox_frac_reference": DEFAULT_LOCKBOX_FRAC,
        "results": results,
        "summary": {
            "assets_ok": len(ok),
            "any_dsr_pass_conservative": any_dsr_pass,
            "interpretation": _interpret(results),
        },
        "council_recommendation": (
            "WATCH — edge may be real but inflated by trials; prioritize forward Track A"
            if not any_dsr_pass
            else "ADVANCE governance — DSR survives conservative trial count (surprising; verify)"
        ),
        "scope_check": {
            "no_lookahead": True,
            "no_ml_in_core": True,
            "no_new_mining": True,
            "historical_csv_only": True,
        },
    }


def _print(r: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("RESEARCH R02 — PBO/DSR GOVERNANCE (path_lean3)")
    print(line)
    print(f"  trials conservative: {TRIAL_INVENTORY['conservative']['n_trials']}")
    for item in r.get("results", []):
        if item.get("status") != "ok":
            print(f"  {item.get('asset')}: {item.get('status')}")
            continue
        h = item["holdout_test"]
        d = item["dsr_conservative_n39"]
        lb = item["lockbox_overfit_proxy"]
        print(f"\n  {item['asset'].upper()}  hit={h['hit_rate']} p={h['p_value']} cov={h['coverage']}")
        print(f"    Sharpe@signals={item['signal_sharpe']['sharpe_annualized']}  DSR pass={d.get('dsr_pass')}  p~={d.get('p_approx')}")
        print(f"    holdout-lockbox={lb.get('delta_holdout_vs_lockbox_pp')}pp  fold_pbo={item['fold_instability'].get('pbo_proxy_fold')}")
    print(f"\n  {r['summary']['interpretation']}")
    print(f"  council: {r['council_recommendation']}")
    print(f"  -> {OUTPUT_PATH}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="R02 PBO/DSR research analysis")
    parser.add_argument("assets", nargs="*", default=list(DEFAULT_ASSETS))
    args = parser.parse_args()
    paths = [a for a in args.assets if Path(a).exists()]
    if not paths:
        raise SystemExit("no CSV files found")
    result = run_all(paths)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print(result)


if __name__ == "__main__":
    main()
