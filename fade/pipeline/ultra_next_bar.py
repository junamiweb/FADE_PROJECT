"""Ultra-sparse next-bar forecast — 90% forward target, auto-refined rule.

Active rule loaded from fade/output/ultra_active_rule.json (updated by ultra_rule_search).
Holdout is exploratory; live forward ledger is validation truth.

Run:
    python -m fade.pipeline.ultra_next_bar refine
    python -m fade.pipeline.ultra_next_bar holdout
    python -m fade.pipeline.ultra_next_bar evaluate
    python -m fade.pipeline.ultra_next_bar replay
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config, lean_config
from fade.core.conviction import read_conviction_state
from fade.core.data_loader import load_ohlcv
from fade.pipeline.pre_registration import get_candidate, load_manifest, register_candidate, save_manifest
from fade.pipeline.ultra_rule_search import (
    HOLDOUT_FRAC,
    OUTPUT as RULE_SEARCH_OUTPUT,
    RULES,
    TARGET_HIT,
    _build_frame,
    _load_funding_hourly,
    _mask,
    load_active_rule,
    run_refinement_cycle,
    run_search,
)

TRACK_ID = "ultra_cross_elite_v1"
STUDY_ID = "ultra_next_bar_90_target_v1"
LEDGER = Path("fade/output/ultra_primary_outcomes.jsonl")
OUTPUT_HOLDOUT = Path("fade/output/ultra_next_bar_holdout.json")
OUTPUT_REPLAY = Path("fade/output/ultra_historical_replay.json")
MIN_FORWARD_SIGNALS = 50


def _rule_thresholds(rule_id: str) -> dict:
    if "bigmove10" in rule_id:
        return {"bigmove": 0.01}
    if "bigmove08" in rule_id:
        return {"bigmove": 0.008}
    if "bigmove05" in rule_id or "bigmove" in rule_id:
        return {"bigmove": 0.005}
    return {}


def ensure_preregistered(
    holdout_n: int | None = None,
    holdout_hit: float | None = None,
    rule_id: str | None = None,
) -> dict:
    active = load_active_rule()
    rule_id = rule_id or active.get("rule_id", "cross_elite_bigmove05")
    existing = get_candidate(TRACK_ID)
    if existing:
        existing.setdefault("config", {})["rule"] = rule_id
        existing["config"]["auto_refine"] = True
        return existing

    entry = {
        "track_id": TRACK_ID,
        "status": "candidate_not_validated",
        "asset": "btc_1h+eth_1h",
        "config": {
            "rule": rule_id,
            "auto_refine": True,
            "abstain_policy": "default_no_signal",
        },
        "selection_history": {
            "where_selected": "Frozen catalog ultra_rule_search.py; auto-upgrade on improvement",
            "rule_search_artifact": str(RULE_SEARCH_OUTPUT).replace("\\", "/"),
            "holdout_exploratory": {
                "n": holdout_n,
                "hit_rate": holdout_hit,
                "rule": rule_id,
            },
        },
        "validation": {
            "method": "live_outcome_tracker_next_bar",
            "ledger": str(LEDGER).replace("\\", "/"),
            "primary_metric": "next_bar_hit_rate",
            "min_signals": MIN_FORWARD_SIGNALS,
            "min_hit_rate": TARGET_HIT,
            "live_only": True,
            "start_collecting_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    return register_candidate(entry)


def _frequent_from_conv(conv: dict) -> bool:
    multi = conv.get("multi") or {}
    vals = [v for k, v in multi.items() if any(x in k for x in ("15m", "30m", "1h")) and v]
    if len(vals) < 3:
        return False
    return len(set(vals)) == 1


def _series_from_csv(csv: str, config: Config | None = None) -> pd.Series:
    config = config or lean_config()
    conv = read_conviction_state(csv, config)
    df = load_ohlcv(csv)
    ret = df["close"].pct_change()
    abs_ret = float(abs(ret.iloc[-1])) if len(ret) else 0.0
    tier = (conv.get("active_tier") or {}).get("id", "")
    direction = (conv.get("active_tier") or {}).get("direction", "FLAT")
    d = 1 if direction == "UP" else (-1 if direction == "DOWN" else 0)
    row = {
        "tid": tier or "",
        "dir": d,
        "slen": int(conv.get("streak_len") or 0),
        "tf": int(conv.get("tf_agree") or 0),
        "aligned": bool(conv.get("aligned")),
        "absret": abs_ret,
        "vr": conv.get("vr_regime"),
        "freq": _frequent_from_conv(conv),
        "fwd": np.nan,
    }
    prefix = Path(csv).stem.split("_")[0]
    fund = _load_funding_hourly(prefix)
    if fund is not None and len(fund):
        split_f = int(len(fund) * (1 - HOLDOUT_FRAC))
        q90 = float(fund.iloc[:max(split_f, 1)].abs().quantile(0.9))
        fv = fund.reindex(df.index).iloc[-1]
        row["fund"] = float(fv) if np.isfinite(fv) else 0.0
        row["fund_extreme"] = abs(row["fund"]) >= q90
    return pd.Series(row)


def evaluate_ultra_signal(
    btc_csv: str = "btc_1h.csv",
    eth_csv: str = "eth_1h.csv",
    config: Config | None = None,
    rule_id: str | None = None,
) -> dict:
    config = config or lean_config()
    active = load_active_rule()
    rule_id = rule_id or active.get("rule_id", "cross_elite_bigmove05")
    holdout_hit = active.get("holdout_hit")

    if not Path(btc_csv).exists() or not Path(eth_csv).exists():
        return {"status": "missing_file", "track_id": TRACK_ID}

    bar_ts = str(load_ohlcv(btc_csv).index[-1])
    b = _series_from_csv(btc_csv, config)
    e = _series_from_csv(eth_csv, config)

    base = {
        "track_id": TRACK_ID,
        "study_id": STUDY_ID,
        "rule": rule_id,
        "bar_ts": bar_ts,
        "btc_tier": b.get("tid"),
        "eth_tier": e.get("tid"),
        "btc_abs_return_pct": round(float(b.get("absret", 0)) * 100, 4),
        "target_hit_rate": TARGET_HIT,
        "holdout_reference_hit": holdout_hit,
        "active_rule_updated": active.get("updated_utc"),
    }

    try:
        bdf = pd.DataFrame([b])
        edf = pd.DataFrame([e])
        passes = bool(_mask(rule_id, bdf, edf).iloc[0])
    except KeyError:
        passes = False

    if not passes:
        return {
            **base,
            "status": "no_signal",
            "reason": f"rule_abstain ({rule_id})",
        }

    direction = "UP" if int(b["dir"]) > 0 else "DOWN"
    ref_pct = round(float(holdout_hit or 0) * 100, 1) if holdout_hit else None
    return {
        **base,
        "status": "ok",
        "direction": direction,
        "confidence_pct": ref_pct,
        "historical_hit": (
            f"holdout reference ~{ref_pct}% (rule={rule_id}, exploratory)"
            if ref_pct
            else f"rule={rule_id}"
        ),
        "coverage_note": RULES.get(rule_id, rule_id),
    }


def evaluate_cross_elite(
    btc_csv: str = "btc_1h.csv",
    eth_csv: str = "eth_1h.csv",
    config: Config | None = None,
) -> dict:
    return evaluate_ultra_signal(btc_csv, eth_csv, config=config)


def run_holdout(
    btc_csv: str = "btc_1h.csv",
    eth_csv: str = "eth_1h.csv",
    holdout_frac: float = HOLDOUT_FRAC,
    rule_id: str | None = None,
) -> dict:
    active = load_active_rule()
    rule_id = rule_id or active.get("rule_id", "cross_elite_bigmove05")
    if not RULE_SEARCH_OUTPUT.exists():
        run_search("1h")

    fund = _load_funding_hourly("btc")
    fund_q90 = None
    if fund is not None:
        split_f = int(len(fund) * (1 - holdout_frac))
        fund_q90 = float(fund.iloc[:max(split_f, 1)].abs().quantile(0.9))

    btc = _build_frame(btc_csv, fund_q90=fund_q90)
    eth = _build_frame(eth_csv)
    common = btc.index.intersection(eth.index)
    btc = btc.loc[common]
    eth = eth.loc[common]
    split = int(len(common) * (1 - holdout_frac))
    hold_b, hold_e = btc.iloc[split:], eth.iloc[split:]

    mask = _mask(rule_id, hold_b, hold_e)
    sub = hold_b.loc[mask]
    hits = ((sub["dir"] > 0) == (sub["fwd"] > 0)).astype(int) if len(sub) else []
    n = int(len(hits))
    hit_rate = round(float(np.mean(hits)), 4) if n else None
    coverage = round(100 * n / len(hold_b), 4) if len(hold_b) else 0.0

    payload = {
        "study_id": STUDY_ID,
        "rule": rule_id,
        "rule_description": RULES.get(rule_id, rule_id),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_bars": len(hold_b),
        "signals": n,
        "coverage_pct": coverage,
        "hit_rate": hit_rate,
        "target_hit_rate": TARGET_HIT,
        "target_met_on_holdout": bool(hit_rate is not None and hit_rate >= TARGET_HIT),
        "verdict": (
            f"REJECT holdout: {hit_rate:.1%} < {TARGET_HIT:.0%} (n={n}). "
            "Refinement cycle continues."
            if hit_rate is not None and hit_rate < TARGET_HIT
            else (f"PASS holdout {hit_rate:.1%} n={n}" if n else "INSUFFICIENT_N")
        ),
    }
    ensure_preregistered(holdout_n=n, holdout_hit=hit_rate, rule_id=rule_id)
    m = load_manifest()
    for c in m.get("candidates", []):
        if c.get("track_id") == TRACK_ID:
            c.setdefault("config", {})["rule"] = rule_id
            c.setdefault("selection_history", {})["holdout_exploratory"] = {
                "n": n, "hit_rate": hit_rate, "coverage_pct": coverage, "rule": rule_id,
            }
    save_manifest(m)
    OUTPUT_HOLDOUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_historical_replay(
    btc_csv: str = "btc_1h.csv",
    eth_csv: str = "eth_1h.csv",
    holdout_frac: float = HOLDOUT_FRAC,
    rule_id: str | None = None,
) -> dict:
    active = load_active_rule()
    rule_id = rule_id or active.get("rule_id", "cross_elite_bigmove05")
    fund = _load_funding_hourly("btc")
    fund_q90 = None
    if fund is not None:
        split_f = int(len(fund) * (1 - holdout_frac))
        fund_q90 = float(fund.iloc[:max(split_f, 1)].abs().quantile(0.9))
    btc = _build_frame(btc_csv, fund_q90=fund_q90)
    eth = _build_frame(eth_csv)
    common = btc.index.intersection(eth.index)
    btc = btc.loc[common]
    eth = eth.loc[common]
    split = int(len(common) * (1 - holdout_frac))
    hold_b, hold_e = btc.iloc[split:], eth.iloc[split:]

    mask = _mask(rule_id, hold_b, hold_e)
    trades = []
    for ts in hold_b.index[mask]:
        row = hold_b.loc[ts]
        d = int(row["dir"])
        fwd = float(row["fwd"])
        trades.append({
            "bar_ts": str(ts),
            "direction": "UP" if d > 0 else "DOWN",
            "outcome": "hit" if (d > 0) == (fwd > 0) else "miss",
            "forward_return_pct": round(fwd * 100, 4),
        })
    n = len(trades)
    hits = sum(1 for t in trades if t["outcome"] == "hit")
    hr = round(hits / n, 4) if n else None
    payload = {
        "rule": rule_id,
        "mode": "historical_holdout_replay",
        "signals": n,
        "hit_rate": hr,
        "target_hit_rate": TARGET_HIT,
        "target_met": bool(hr is not None and hr >= TARGET_HIT),
        "trades": trades,
    }
    OUTPUT_REPLAY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_refine() -> dict:
    """One refinement cycle: search all horizons, upgrade rule, refresh holdout."""
    cycle = run_refinement_cycle()
    active = load_active_rule()
    holdout = run_holdout(rule_id=active.get("rule_id"))
    replay = run_historical_replay(rule_id=active.get("rule_id"))
    return {"refinement": cycle, "holdout": holdout, "replay": replay, "active_rule": active}


def _print_eval(r: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print(f"ULTRA — {r.get('rule', '?')}")
    print(line)
    if r.get("status") == "ok":
        print(f"  SIGNAL : {r['direction']}")
        print(f"  bar    : {r.get('bar_ts')}")
        print(f"  ref    : {r.get('historical_hit')}")
        print(f"  target : {TARGET_HIT:.0%} live forward ({MIN_FORWARD_SIGNALS}+ signals)")
    else:
        print(f"  NO SIGNAL: {r.get('reason')}")
        print(f"  tiers  : btc={r.get('btc_tier')} eth={r.get('eth_tier')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ULTRA next-bar (90% target, auto-refine)")
    parser.add_argument(
        "command",
        choices=("preregister", "refine", "holdout", "evaluate", "replay"),
        default="evaluate",
        nargs="?",
    )
    args = parser.parse_args()
    if args.command == "preregister":
        print(json.dumps(ensure_preregistered(), ensure_ascii=False, indent=2))
        return
    if args.command == "refine":
        out = run_refine()
        print(json.dumps({
            "active_rule": out["active_rule"],
            "holdout_hit": out["holdout"].get("hit_rate"),
            "holdout_n": out["holdout"].get("signals"),
            "lockbox_verdict": out.get("refinement", {}).get("lockbox", {}).get("verdict"),
            "verdict": out["holdout"].get("verdict"),
        }, ensure_ascii=False, indent=2))
        return
    if args.command == "holdout":
        print(json.dumps(run_holdout(), ensure_ascii=False, indent=2))
        return
    if args.command == "replay":
        print(json.dumps(run_historical_replay(), ensure_ascii=False, indent=2))
        return
    _print_eval(evaluate_ultra_signal())


if __name__ == "__main__":
    main()
