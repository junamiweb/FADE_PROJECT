"""Search frozen ULTRA rules on holdout — pick best defensible next-bar hit rate.

Pre-registered catalog only; no post-hoc tuning. Writes fade/output/ultra_rule_search.json.
Updates fade/output/ultra_active_rule.json when a better rule is found.

Run:
    python -m fade.pipeline.ultra_rule_search
    python -m fade.pipeline.ultra_rule_search --horizon 15m
    python -m fade.pipeline.ultra_rule_search --all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import lean_config
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.sparse_primary_replay import MULTI_IV, _active_tier, _prefix
from fade.pipeline.trend_structure import _signed_streak

HOLDOUT_FRAC = 0.30
TARGET_HIT = 0.90
MIN_N_REPORT = 10
OUTPUT = Path("fade/output/ultra_rule_search.json")
OUTPUT_1H = Path("fade/output/ultra_rule_search_1h.json")
OUTPUT_15M = Path("fade/output/ultra_rule_search_15m.json")
OUTPUT_LOCKBOX = Path("fade/output/ultra_lockbox_eval.json")
ACTIVE_RULE_PATH = Path("fade/output/ultra_active_rule.json")
REFINEMENT_LOG = Path("fade/output/ultra_refinement_log.jsonl")
LOCKBOX_FRAC = 0.18

# Frozen rule catalog — evaluated once each on holdout (no tuning).
RULES: dict[str, str] = {
    "cross_elite_v1": "btc_elite & eth_elite & same_dir",
    "cross_elite_s4": "cross_elite & btc_slen>=4 & eth_slen>=4",
    "cross_elite_s5": "cross_elite & btc_slen>=5 & eth_slen>=5",
    "cross_elite_lowvr": "cross_elite & both LOW_VR",
    "cross_elite_highvr": "cross_elite & both HIGH_VR",
    "cross_elite_freq": "cross_elite & both frequent 15/30/1h",
    "both_s4_4tf": "both slen>=4 & 4TF aligned & same_dir",
    "both_s4_4tf_lowvr": "both_s4_4tf & btc LOW_VR",
    "btc_s4_eth_agree": "btc slen>=4 4TF & eth agrees dir",
    "cross_elite_bigmove05": "cross_elite & |btc_ret|>=0.5%",
    "cross_elite_bigmove08": "cross_elite & |btc_ret|>=0.8%",
    "cross_elite_bigmove10": "cross_elite & |btc_ret|>=1%",
    "cross_elite_s4_bigmove05": "cross_elite_s4 & |btc_ret|>=0.5%",
    "cross_elite_bigmove05_s4": "cross_elite_bigmove05 & btc_slen>=4",
    "btc_elite_bigmove05": "btc elite only & |btc_ret|>=0.5%",
    "btc_elite_bigmove10": "btc elite only & |btc_ret|>=1%",
    "btc_elite_s4_bigmove05": "btc elite s4 4TF & |btc_ret|>=0.5%",
    "cross_elite_bigmove05_fund_q90": "cross_elite_bigmove05 & |btc_funding|>=dev_q90",
    "cross_elite_bigmove10_s4": "cross_elite_bigmove10 & btc_slen>=4",
    "cross_elite_bigmove05_eth_s4": "cross_elite_bigmove05 & eth_slen>=4",
    # Phase-2 frozen composites (pre-specified before eval)
    "cross_elite_highvr_bigmove05": "cross_elite_highvr & |btc_ret|>=0.5%",
    "cross_elite_bigmove05_both_s4": "cross_elite_bigmove05 & btc_slen>=4 & eth_slen>=4",
    "cross_elite_bigmove08_both_s4": "cross_elite_bigmove08 & btc_slen>=4 & eth_slen>=4",
    "cross_elite_bigmove08_eth_s4": "cross_elite_bigmove08 & eth_slen>=4",
    "cross_elite_bigmove10_both_s4": "cross_elite_bigmove10 & btc_slen>=4 & eth_slen>=4",
    "both_s4_4tf_bigmove05": "both_s4_4tf & |btc_ret|>=0.5%",
    "cross_elite_bigmove05_lowvr": "cross_elite_lowvr & |btc_ret|>=0.5%",
    "cross_elite_fund_q90": "cross_elite & |btc_funding|>=dev_q90",
    "cross_elite_bigmove08_s4": "cross_elite_bigmove08 & btc_slen>=4",
}


def _load_funding_hourly(prefix: str) -> pd.Series | None:
    path = Path(f"funding_{prefix}.csv")
    if not path.exists():
        return None
    f = pd.read_csv(path)
    f["timestamp"] = pd.to_datetime(f["timestamp"], format="mixed", utc=True)
    f = f.set_index("timestamp").sort_index()
    return f["funding_rate"].resample("1h").last().ffill()


def _vr_regime_series(csv: str) -> pd.Series:
    cfg = lean_config()
    df = load_ohlcv(csv)
    ret = df["close"].pct_change()
    vr = compute_vol_ratio(
        ret, cfg.vol_ratio_short_window, cfg.vol_ratio_long_window,
    )
    frame = pd.DataFrame({"vr": vr}, index=df.index).dropna()
    split = max(int(len(frame) * (1 - HOLDOUT_FRAC)), 1)
    dev = frame.iloc[:split]
    low = float(dev["vr"].quantile(1.0 / 3.0))
    high = float(dev["vr"].quantile(2.0 / 3.0))
    return assign_vr_regime(frame["vr"], low, high)


def _attach_tier_columns(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    tid = np.empty(len(frame), dtype=object)
    direction = np.zeros(len(frame), dtype=int)
    tf_agree = np.zeros(len(frame), dtype=int)
    aligned = np.zeros(len(frame), dtype=bool)
    freq = np.zeros(len(frame), dtype=bool)

    for i, (_ts, row) in enumerate(frame.iterrows()):
        slen = int(row["slen"])
        s = int(row["streak"]) if np.isfinite(row["streak"]) else 0
        streak_dir = "FLAT" if slen < 2 else ("UP" if s < 0 else "DOWN")
        dirs = []
        for c in cols:
            v = row[c]
            if v == 0 or not np.isfinite(v):
                dirs.append(None)
            else:
                dirs.append("UP" if v > 0 else "DOWN")
        ok = [d for d in dirs if d is not None]
        up = sum(1 for d in ok if d == "UP")
        dn = sum(1 for d in ok if d == "DOWN")
        if up >= dn and up > 0:
            multi_dir, ta = "UP", up
        elif dn > 0:
            multi_dir, ta = "DOWN", dn
        else:
            multi_dir, ta = "FLAT", 0
        al = streak_dir != "FLAT" and multi_dir != "FLAT" and streak_dir == multi_dir
        tier = _active_tier(slen, streak_dir, ta, multi_dir, al)
        if tier:
            t_id, d_str = tier
            tid[i] = t_id
            direction[i] = 1 if d_str == "UP" else -1
        else:
            tid[i] = ""
        tf_agree[i] = ta
        aligned[i] = al
        sub = [row[c] for c in ("15m", "30m", "1h") if c in cols]
        freq[i] = (
            len(sub) == 3
            and all(np.isfinite(x) and x != 0 for x in sub)
            and len(set("UP" if x > 0 else "DOWN" for x in sub)) == 1
        )

    frame["tid"] = tid
    frame["dir"] = direction
    frame["tf"] = tf_agree
    frame["aligned"] = aligned
    frame["freq"] = freq
    frame["absret"] = frame["ret"].abs()
    return frame


def _build_frame(csv_1h: str, fund_q90: float | None = None) -> pd.DataFrame:
    prefix = _prefix(csv_1h)
    df = load_ohlcv(csv_1h)
    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    grids = {
        iv: _contrarian_grid(f"{prefix}_{iv}.csv")
        for iv in MULTI_IV
        if Path(f"{prefix}_{iv}.csv").exists()
    }
    frame = pd.DataFrame(
        {"streak": streak, "slen": np.abs(streak), "fwd": fwd, "ret": ret},
        index=df.index,
    )
    for iv, s in grids.items():
        frame[iv] = s.reindex(frame.index)
    frame["vr"] = _vr_regime_series(csv_1h).reindex(frame.index)
    fund = _load_funding_hourly(prefix)
    if fund is not None:
        frame["fund"] = fund.reindex(frame.index)
        if fund_q90 is not None:
            frame["fund_extreme"] = frame["fund"].abs() >= fund_q90
    frame = frame.dropna(subset=["fwd"])
    cols = [c for c in MULTI_IV if c in frame.columns]
    return _attach_tier_columns(frame, cols)


def _build_frame_15m(prefix: str) -> pd.DataFrame | None:
    """15m next-bar with multi-TF grids aligned to 15m index."""
    p15 = Path(f"{prefix}_15m.csv")
    if not p15.exists():
        return None
    df = load_ohlcv(str(p15))
    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    frame = pd.DataFrame(
        {"streak": streak, "slen": np.abs(streak), "fwd": fwd, "ret": ret},
        index=df.index,
    )
    for iv in MULTI_IV:
        gp = Path(f"{prefix}_{iv}.csv")
        if not gp.exists():
            continue
        s = _contrarian_grid(str(gp))
        frame[iv] = s.reindex(frame.index, method="ffill")
    frame = frame.dropna(subset=["fwd"])
    cols = [c for c in MULTI_IV if c in frame.columns]
    if not cols:
        return None
    # VR from 1h series forward-filled to 15m
    h1 = Path(f"{prefix}_1h.csv")
    if h1.exists():
        frame["vr"] = _vr_regime_series(str(h1)).reindex(frame.index, method="ffill")
    else:
        frame["vr"] = "UNK"
    return _attach_tier_columns(frame, cols)


def _cross_elite(b: pd.DataFrame, e: pd.DataFrame) -> pd.Series:
    return (
        (b["tid"] == "elite")
        & (e["tid"] == "elite")
        & (b["dir"] == e["dir"])
        & (b["dir"] != 0)
    )


def _mask(rule_id: str, b: pd.DataFrame, e: pd.DataFrame) -> pd.Series:
    cross = _cross_elite(b, e)
    if rule_id == "cross_elite_v1":
        return cross
    if rule_id == "cross_elite_s4":
        return cross & (b["slen"] >= 4) & (e["slen"] >= 4)
    if rule_id == "cross_elite_s5":
        return cross & (b["slen"] >= 5) & (e["slen"] >= 5)
    if rule_id == "cross_elite_lowvr":
        return cross & (b["vr"] == "LOW_VR") & (e["vr"] == "LOW_VR")
    if rule_id == "cross_elite_highvr":
        return cross & (b["vr"] == "HIGH_VR") & (e["vr"] == "HIGH_VR")
    if rule_id == "cross_elite_freq":
        return cross & b["freq"] & e["freq"]
    if rule_id == "both_s4_4tf":
        return (
            (b["slen"] >= 4) & (e["slen"] >= 4)
            & (b["tf"] == 4) & (e["tf"] == 4)
            & b["aligned"] & e["aligned"]
            & (b["dir"] == e["dir"]) & (b["dir"] != 0)
        )
    if rule_id == "both_s4_4tf_lowvr":
        return _mask("both_s4_4tf", b, e) & (b["vr"] == "LOW_VR")
    if rule_id == "btc_s4_eth_agree":
        return (
            (b["slen"] >= 4) & (b["tf"] == 4) & b["aligned"]
            & (e["dir"] == b["dir"]) & (b["dir"] != 0)
        )
    if rule_id == "cross_elite_bigmove05":
        return cross & (b["absret"] >= 0.005)
    if rule_id == "cross_elite_bigmove08":
        return cross & (b["absret"] >= 0.008)
    if rule_id == "cross_elite_bigmove10":
        return cross & (b["absret"] >= 0.01)
    if rule_id == "cross_elite_s4_bigmove05":
        return _mask("cross_elite_s4", b, e) & (b["absret"] >= 0.005)
    if rule_id == "cross_elite_bigmove05_s4":
        return _mask("cross_elite_bigmove05", b, e) & (b["slen"] >= 4)
    if rule_id == "btc_elite_bigmove05":
        return (b["tid"] == "elite") & (b["dir"] != 0) & (b["absret"] >= 0.005)
    if rule_id == "btc_elite_bigmove10":
        return (b["tid"] == "elite") & (b["dir"] != 0) & (b["absret"] >= 0.01)
    if rule_id == "btc_elite_s4_bigmove05":
        return (
            (b["tid"] == "elite") & (b["slen"] >= 4) & (b["tf"] == 4) & b["aligned"]
            & (b["dir"] != 0) & (b["absret"] >= 0.005)
        )
    if rule_id == "cross_elite_bigmove05_fund_q90":
        m = _mask("cross_elite_bigmove05", b, e)
        if "fund_extreme" in b.columns:
            return m & b["fund_extreme"].fillna(False)
        return m & False
    if rule_id == "cross_elite_bigmove10_s4":
        return _mask("cross_elite_bigmove10", b, e) & (b["slen"] >= 4)
    if rule_id == "cross_elite_bigmove05_eth_s4":
        return _mask("cross_elite_bigmove05", b, e) & (e["slen"] >= 4)
    if rule_id == "cross_elite_highvr_bigmove05":
        return _mask("cross_elite_highvr", b, e) & (b["absret"] >= 0.005)
    if rule_id == "cross_elite_bigmove05_both_s4":
        return _mask("cross_elite_bigmove05", b, e) & (b["slen"] >= 4) & (e["slen"] >= 4)
    if rule_id == "cross_elite_bigmove08_both_s4":
        return _mask("cross_elite_bigmove08", b, e) & (b["slen"] >= 4) & (e["slen"] >= 4)
    if rule_id == "cross_elite_bigmove08_eth_s4":
        return _mask("cross_elite_bigmove08", b, e) & (e["slen"] >= 4)
    if rule_id == "cross_elite_bigmove10_both_s4":
        return _mask("cross_elite_bigmove10", b, e) & (b["slen"] >= 4) & (e["slen"] >= 4)
    if rule_id == "both_s4_4tf_bigmove05":
        return _mask("both_s4_4tf", b, e) & (b["absret"] >= 0.005)
    if rule_id == "cross_elite_bigmove05_lowvr":
        return _mask("cross_elite_lowvr", b, e) & (b["absret"] >= 0.005)
    if rule_id == "cross_elite_fund_q90":
        m = cross
        if "fund_extreme" in b.columns:
            return m & b["fund_extreme"].fillna(False)
        return m & False
    if rule_id == "cross_elite_bigmove08_s4":
        return _mask("cross_elite_bigmove08", b, e) & (b["slen"] >= 4)
    raise KeyError(rule_id)


def _score_rules(
    b: pd.DataFrame,
    e: pd.DataFrame,
    hold_len: int,
) -> list[dict]:
    results = []
    for rule_id, desc in RULES.items():
        m = _mask(rule_id, b, e)
        sub = b.loc[m]
        if sub.empty:
            results.append({
                "rule_id": rule_id, "description": desc,
                "n": 0, "hit_rate": None, "meets_90": False,
            })
            continue
        hits = ((sub["dir"] > 0) == (sub["fwd"] > 0)).astype(int)
        n = int(len(hits))
        hr = round(float(hits.mean()), 4)
        results.append({
            "rule_id": rule_id,
            "description": desc,
            "n": n,
            "coverage_pct": round(100 * n / hold_len, 4) if hold_len else 0,
            "hit_rate": hr,
            "meets_90": bool(hr >= TARGET_HIT and n >= MIN_N_REPORT),
        })
    results.sort(key=lambda x: (-(x["hit_rate"] or 0), -(x["n"] or 0)))
    return results


def run_search(horizon: str = "1h") -> dict:
    if horizon == "15m":
        btc = _build_frame_15m("btc")
        eth = _build_frame_15m("eth")
        if btc is None or eth is None:
            return {"horizon": "15m", "status": "missing_15m_data", "results": []}
    else:
        fund = _load_funding_hourly("btc")
        fund_q90 = None
        if fund is not None:
            split_f = int(len(fund) * (1 - HOLDOUT_FRAC))
            fund_q90 = float(fund.iloc[:max(split_f, 1)].abs().quantile(0.9))
        btc = _build_frame("btc_1h.csv", fund_q90=fund_q90)
        eth = _build_frame("eth_1h.csv")

    idx = btc.index.intersection(eth.index)
    btc, eth = btc.loc[idx], eth.loc[idx]
    split = int(len(idx) * (1 - HOLDOUT_FRAC))
    hold = idx[split:]
    b, e = btc.loc[hold], eth.loc[hold]

    results = _score_rules(b, e, len(hold))
    best_90 = next((r for r in results if r.get("meets_90")), None)
    best_overall = results[0] if results else None

    payload = {
        "horizon": horizon,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_split": f"holdout_{int((1-HOLDOUT_FRAC)*100)}_{int(HOLDOUT_FRAC*100)}",
        "holdout_bars": len(hold),
        "target_hit_rate": TARGET_HIT,
        "min_n_for_90_claim": MIN_N_REPORT,
        "results": results,
        "best_meeting_90": best_90,
        "best_overall": best_overall,
        "recommendation": (
            best_90["rule_id"] if best_90
            else best_overall["rule_id"] if best_overall and best_overall.get("n")
            else "cross_elite_bigmove05"
        ),
        "verdict": (
            f"PASS holdout 90%: {best_90['rule_id']} hit={best_90['hit_rate']} n={best_90['n']}"
            if best_90
            else (
                f"No rule meets {TARGET_HIT:.0%} with n>={MIN_N_REPORT} on holdout. "
                f"Best: {best_overall['rule_id']} hit={best_overall['hit_rate']} n={best_overall['n']}."
                if best_overall and best_overall.get("n")
                else "INSUFFICIENT_DATA"
            )
        ),
    }
    out_path = OUTPUT_1H if horizon == "1h" else OUTPUT_15M
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Combined snapshot (both horizons when available)
    combined = {"updated_utc": payload["evaluated_at_utc"]}
    if OUTPUT_1H.exists():
        combined["1h"] = json.loads(OUTPUT_1H.read_text(encoding="utf-8"))
    if OUTPUT_15M.exists():
        combined["15m"] = json.loads(OUTPUT_15M.read_text(encoding="utf-8"))
    if horizon == "1h":
        combined["1h"] = payload
    else:
        combined["15m"] = payload
    OUTPUT.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_active_rule() -> dict:
    if ACTIVE_RULE_PATH.exists():
        return json.loads(ACTIVE_RULE_PATH.read_text(encoding="utf-8"))
    return {
        "rule_id": "cross_elite_bigmove05",
        "horizon": "1h",
        "holdout_hit": 0.7188,
        "holdout_n": 32,
        "updated_utc": None,
    }


def _best_defensible(search: dict, min_n: int = 5) -> dict | None:
    """Best rule with enough samples — ignore tiny-n luck."""
    for r in search.get("results", []):
        if r.get("meets_90"):
            return r
    for r in search.get("results", []):
        n = int(r.get("n") or 0)
        if n >= min_n and r.get("hit_rate") is not None:
            return r
    return search.get("best_overall")


def maybe_upgrade_active_rule(search: dict) -> dict:
    """Promote rule if it beats current on hit_rate (min n>=5) or meets 90%."""
    active = load_active_rule()
    best = _best_defensible(search, min_n=5)
    if not best or not best.get("n"):
        return active

    cur_hr = float(active.get("holdout_hit") or 0)
    cur_n = int(active.get("holdout_n") or 0)
    new_hr = float(best["hit_rate"])
    new_n = int(best["n"])

    meets_90 = bool(best.get("meets_90"))
    improved = (new_hr > cur_hr and new_n >= 5) or (new_hr == cur_hr and new_n > cur_n)
    if not meets_90 and not improved:
        return active

    entry = {
        "rule_id": best["rule_id"],
        "horizon": search.get("horizon", "1h"),
        "holdout_hit": new_hr,
        "holdout_n": new_n,
        "meets_90_holdout": meets_90,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "description": best.get("description"),
    }
    ACTIVE_RULE_PATH.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    log_line = {
        "ts": entry["updated_utc"],
        "action": "upgrade" if improved or meets_90 else "noop",
        "from_rule": active.get("rule_id"),
        "to_rule": entry["rule_id"],
        "holdout_hit": new_hr,
        "holdout_n": new_n,
        "meets_90": meets_90,
        "horizon": entry["horizon"],
    }
    REFINEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REFINEMENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_line, ensure_ascii=False) + "\n")
    return entry


def run_lockbox_eval(horizon: str = "1h", lockbox_frac: float = LOCKBOX_FRAC) -> dict:
    """One-shot OOS on sealed newest slice — no tuning."""
    if horizon == "15m":
        btc = _build_frame_15m("btc")
        eth = _build_frame_15m("eth")
        if btc is None or eth is None:
            return {"horizon": "15m", "status": "missing_15m_data", "results": []}
    else:
        fund = _load_funding_hourly("btc")
        fund_q90 = None
        if fund is not None:
            split_f = int(len(fund) * (1 - HOLDOUT_FRAC))
            fund_q90 = float(fund.iloc[:max(split_f, 1)].abs().quantile(0.9))
        btc = _build_frame("btc_1h.csv", fund_q90=fund_q90)
        eth = _build_frame("eth_1h.csv")

    idx = btc.index.intersection(eth.index)
    btc, eth = btc.loc[idx], eth.loc[idx]
    cut = int(len(idx) * (1.0 - lockbox_frac))
    lock_b, lock_e = btc.iloc[cut:], eth.iloc[cut:]

    results = _score_rules(lock_b, lock_e, len(lock_b))
    best_90 = next((r for r in results if r.get("meets_90")), None)
    best_overall = results[0] if results else None

    payload = {
        "horizon": horizon,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_split": f"lockbox_{int(lockbox_frac * 100)}_one_shot",
        "lockbox_bars": len(lock_b),
        "lockbox_start": str(lock_b.index[0]) if len(lock_b) else None,
        "target_hit_rate": TARGET_HIT,
        "min_n_for_90_claim": MIN_N_REPORT,
        "results": results,
        "best_meeting_90": best_90,
        "best_overall": best_overall,
        "verdict": (
            f"LOCKBOX PASS 90%: {best_90['rule_id']} hit={best_90['hit_rate']} n={best_90['n']}"
            if best_90
            else (
                f"No lockbox rule meets {TARGET_HIT:.0%} with n>={MIN_N_REPORT}. "
                f"Best: {best_overall['rule_id']} hit={best_overall['hit_rate']} n={best_overall['n']}."
                if best_overall and best_overall.get("n")
                else "INSUFFICIENT_DATA"
            )
        ),
        "note": "One-shot OOS — do not re-tune on lockbox after this eval.",
    }
    OUTPUT_LOCKBOX.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_refinement_cycle() -> dict:
    """Full refinement: search 1h + 15m, lockbox check, upgrade active rule if improved."""
    s1h = run_search("1h")
    a1h = maybe_upgrade_active_rule(s1h)
    s15 = run_search("15m")
    a15 = maybe_upgrade_active_rule(s15)
    lockbox = run_lockbox_eval("1h")
    return {
        "active_rule": load_active_rule(),
        "search_1h": {"verdict": s1h.get("verdict"), "best": s1h.get("best_overall")},
        "search_15m": {"verdict": s15.get("verdict"), "best": s15.get("best_overall")},
        "lockbox": {"verdict": lockbox.get("verdict"), "best": lockbox.get("best_overall")},
        "upgraded_1h": a1h,
        "upgraded_15m": a15,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ULTRA rule search")
    parser.add_argument("--horizon", choices=("1h", "15m"), default="1h")
    parser.add_argument("--all", action="store_true", help="Search 1h+15m and maybe upgrade")
    parser.add_argument("--lockbox", action="store_true", help="One-shot lockbox eval only")
    args = parser.parse_args()
    if args.lockbox:
        print(json.dumps(run_lockbox_eval(args.horizon), ensure_ascii=False, indent=2))
        return
    if args.all:
        out = run_refinement_cycle()
    else:
        s = run_search(args.horizon)
        out = {"search": s, "active": maybe_upgrade_active_rule(s)}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
