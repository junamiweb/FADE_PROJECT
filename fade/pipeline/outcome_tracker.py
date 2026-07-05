"""PRIMARY outcome tracker -- log forecasts and score against the next bar.



Tracks TWO ledgers:

  1. primary_outcomes.jsonl   -- sparse PRIMARY (Phase 1: conviction tier >= HIGH)

  2. eth_candidate_outcomes.jsonl -- ETH LOW_VR+min_hold=12 candidate (Phase 0 validation)



Every PRIMARY / candidate call is appended to JSONL. When the next 1h bar closes,

`score` fills hit/miss using actual forward return (no look-ahead).

ETH candidate v2: `score_eth_candidate_pnl()` scores full min_hold cycles via

`pnl_sim._equity` (5+5 bps round-trip) when planned_exit_bar_ts is in CSV.



Run:

    python -m fade.pipeline.outcome_tracker log btc_1h.csv

    python -m fade.pipeline.outcome_tracker log-candidate eth_1h.csv

    python -m fade.pipeline.outcome_tracker score

    python -m fade.pipeline.outcome_tracker report

    python -m fade.pipeline.outcome_tracker report-candidate

    python -m fade.pipeline.outcome_tracker run   # log + score + report (both)

"""



from __future__ import annotations



import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config, lean_config
from fade.core.data_loader import load_ohlcv
from fade.pipeline.eth_candidate_track import (
    SCORING_VERSION_V2,
    TRACK_ID,
    evaluate_eth_candidate,
)
from fade.pipeline.forecast_tiers import run_tiered_forecast
from fade.pipeline.pnl_sim import _equity
from fade.pipeline.pre_registration import get_candidate, init_manifest



DEFAULT_LEDGER = Path("fade/output/primary_outcomes.jsonl")

ETH_CANDIDATE_LEDGER = Path("fade/output/eth_candidate_outcomes.jsonl")





def _ledger_path(config: Config | None = None) -> Path:

    config = config or lean_config()

    return config.output_dir / "primary_outcomes.jsonl"





def _load_ledger(path: Path) -> list[dict]:

    if not path.exists():

        return []

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if line:

            rows.append(json.loads(line))

    return rows





def _save_ledger(path: Path, rows: list[dict]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(

        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),

        encoding="utf-8",

    )





def _bar_ts(primary_csv: str) -> str:

    df = load_ohlcv(primary_csv)

    return str(df.index[-1])





def _forward_return(csv_path: str, bar_ts: str) -> float | None:

    df = load_ohlcv(csv_path)

    ts = pd.Timestamp(bar_ts)

    if ts not in df.index:

        return None

    loc = df.index.get_loc(ts)

    if isinstance(loc, slice) or loc >= len(df) - 1:

        return None

    c0 = float(df["close"].iloc[loc])

    c1 = float(df["close"].iloc[loc + 1])

    return (c1 - c0) / c0


def _asset_csv(asset: str) -> str:
    return asset if str(asset).endswith(".csv") else f"{asset}.csv"


def _compute_hold_cycle_pnl(
    csv_path: str,
    entry_bar_ts: str,
    direction: str,
    min_hold: int,
    fee_bps: float,
) -> dict | None:
    """Net PnL for one min_hold cycle — same fee model as pnl_sim._equity / lockbox."""
    if not Path(csv_path).exists():
        return None
    df = load_ohlcv(csv_path)
    ts = pd.Timestamp(entry_bar_ts)
    if ts not in df.index:
        return None
    loc = df.index.get_loc(ts)
    if isinstance(loc, slice):
        return None
    exit_loc = loc + min_hold
    if exit_loc >= len(df):
        return None

    sign = 1.0 if direction == "UP" else -1.0
    fee_rate = fee_bps / 1e4
    entry_price = float(df["close"].iloc[loc])
    exit_price = float(df["close"].iloc[exit_loc])
    planned_exit_bar_ts = str(df.index[exit_loc])

    bar_rets = []
    for j in range(loc, exit_loc):
        c0 = float(df["close"].iloc[j])
        c1 = float(df["close"].iloc[j + 1])
        bar_rets.append((c1 - c0) / c0)

    pos = np.full(min_hold + 1, sign)
    pos[-1] = 0.0
    bar_ret_arr = np.array(bar_rets + [0.0])
    e = _equity(pos, bar_ret_arr, fee_rate, 0.0)
    net_pnl = float(np.prod(1.0 + e["strat_ret"]) - 1.0)

    return {
        "entry_bar_ts": entry_bar_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "planned_exit_bar_ts": planned_exit_bar_ts,
        "hold_cycle_pnl": round(net_pnl, 6),
        "hold_cycle_pnl_pct": round(net_pnl * 100, 4),
        "hold_cycle_outcome": "profit" if net_pnl > 0 else "loss",
        "fee_bps_round_trip": fee_bps * 2,
        "total_fees_pct": round(float(np.sum(
            np.abs(pos - np.concatenate([[0.0], pos[:-1]])) * fee_rate
        )) * 100, 4),
    }


def _is_pre_fix_candidate_row(r: dict) -> bool:
    """Legacy rows scored with next-bar only (before v2 hold-cycle PnL)."""
    return (
        bool(r.get("direction"))
        and r.get("scoring_version") != SCORING_VERSION_V2
        and r.get("outcome") in ("hit", "miss")
    )





def log_forecast(primary_csv: str, config: Config | None = None,

                 ledger: Path | None = None,

                 sparse_primary: bool = True) -> dict:

    """Append sparse PRIMARY for the latest bar if not already logged."""

    config = config or lean_config()

    ledger = ledger or _ledger_path(config)

    if not Path(primary_csv).exists():

        return {"status": "missing_file", "asset": primary_csv}



    fc = run_tiered_forecast(primary_csv, config=config, sparse_primary=sparse_primary)

    bar_ts = _bar_ts(primary_csv)

    asset = fc.get("asset", Path(primary_csv).stem)

    rows = _load_ledger(ledger)



    if any(r.get("asset") == asset and r.get("bar_ts") == bar_ts for r in rows):

        return {"status": "duplicate", "asset": asset, "bar_ts": bar_ts}



    primary = fc.get("primary", {})

    if primary.get("status") != "ok":

        entry = {

            "asset": asset,

            "bar_ts": bar_ts,

            "logged_at": datetime.now(timezone.utc).isoformat(),

            "primary_status": "no_signal",

            "reason": primary.get("reason", "no active tier"),

            "sparse_mode": sparse_primary,

            "outcome": None,

        }

        if primary.get("active_conviction_tier"):

            entry["active_conviction_tier"] = primary["active_conviction_tier"]

        rows.append(entry)

        _save_ledger(ledger, rows)

        return {"status": "logged_no_signal", "asset": asset, "bar_ts": bar_ts}



    entry = {

        "asset": asset,

        "bar_ts": bar_ts,

        "logged_at": datetime.now(timezone.utc).isoformat(),

        "direction": primary["direction"],

        "source": primary["source"],

        "label": primary.get("label", ""),

        "confidence_pct": primary.get("confidence_pct"),

        "sparse_mode": sparse_primary,

        "active_conviction_tier": primary.get("active_conviction_tier"),

        "regime": fc.get("regime"),

        "conviction_tier": fc.get("tiers", {}).get("conviction", {}).get("label"),

        "outcome": None,

        "actual_direction": None,

        "forward_return_pct": None,

    }

    rows.append(entry)

    _save_ledger(ledger, rows)

    return {"status": "logged", "asset": asset, "bar_ts": bar_ts,

            "direction": entry["direction"], "source": entry["source"]}





def log_eth_candidate(csv_path: str = "eth_1h.csv",

                    ledger: Path | None = None) -> dict:

    """Log ETH candidate track signal (LOW_VR + min_hold=12) for latest bar."""

    init_manifest()

    ledger = ledger or ETH_CANDIDATE_LEDGER

    cand = get_candidate(TRACK_ID) or {}

    ev = evaluate_eth_candidate(csv_path)



    if ev.get("status") == "duplicate":

        return ev



    bar_ts = ev.get("bar_ts")

    rows = _load_ledger(ledger)

    if any(r.get("track_id") == TRACK_ID and r.get("bar_ts") == bar_ts for r in rows):

        return {"status": "duplicate", "track_id": TRACK_ID, "bar_ts": bar_ts}



    entry = {

        "track_id": TRACK_ID,

        "candidate_status": cand.get("status", "candidate_not_validated"),

        "asset": ev.get("asset", Path(csv_path).stem),

        "bar_ts": bar_ts,

        "logged_at": datetime.now(timezone.utc).isoformat(),

        "config": cand.get("config", {"vr_regime": "LOW_VR", "min_hold_bars": 12}),

        "vr_regime": ev.get("vr_regime"),

        "vr_gate_pass": ev.get("vr_gate_pass"),

        "path_lean3_direction": ev.get("path_lean3_direction"),

        "traded": ev.get("traded"),

        "signal_status": ev.get("status"),

        "outcome": None,

    }



    if ev.get("status") == "signal" and ev.get("direction"):
        entry["direction"] = ev["direction"]
        entry["scoring_version"] = SCORING_VERSION_V2
        hc = ev.get("hold_cycle") or {}
        entry["entry_bar_ts"] = hc.get("entry_bar_ts", bar_ts)
        entry["entry_price"] = hc.get("entry_price")
        entry["planned_exit_bar_ts"] = hc.get("planned_exit_bar_ts")
        entry["fee_bps"] = hc.get("fee_bps", entry["config"].get("fee_bps_reference", 5))
        entry["hold_cycle_pnl"] = None
        entry["hold_cycle_outcome"] = None
    else:

        entry["primary_status"] = "no_signal"

        entry["reason"] = (

            "vr_gate_fail" if not ev.get("vr_gate_pass") else

            "min_hold_no_trade" if not ev.get("traded") else

            "no_path_lean3"

        )



    rows.append(entry)

    _save_ledger(ledger, rows)

    return {

        "status": "logged" if entry.get("direction") else "logged_no_signal",

        "track_id": TRACK_ID,

        "bar_ts": bar_ts,

        "direction": entry.get("direction"),

    }





def score_outcomes(config: Config | None = None, ledger: Path | None = None,

                   csv_key: str = "asset") -> dict:

    """Fill hit/miss for pending entries where the next bar exists in CSV."""

    config = config or lean_config()

    ledger = ledger or _ledger_path(config)

    rows = _load_ledger(ledger)

    if not rows:

        return {"status": "empty", "scored": 0}



    scored = 0

    for r in rows:

        if r.get("outcome") is not None:

            continue

        if r.get("primary_status") == "no_signal":

            continue

        if not r.get("direction"):

            continue

        asset = r.get(csv_key, r.get("asset"))

        csv = f"{asset}.csv" if not str(asset).endswith(".csv") else asset

        if not Path(csv).exists():

            continue

        fwd = _forward_return(csv, r["bar_ts"])

        if fwd is None:

            continue

        actual = "UP" if fwd > 0 else "DOWN"

        hit = actual == r["direction"]

        r["forward_return_pct"] = round(fwd * 100, 4)

        r["actual_direction"] = actual

        r["outcome"] = "hit" if hit else "miss"

        r["scored_at"] = datetime.now(timezone.utc).isoformat()

        scored += 1



    _save_ledger(ledger, rows)

    return {"status": "ok", "scored": scored, "pending": _count_pending(rows)}





def score_eth_candidate(ledger: Path | None = None) -> dict:
    """Score next-bar directional hit (legacy supplementary metric)."""
    ledger = ledger or ETH_CANDIDATE_LEDGER
    return score_outcomes(ledger=ledger, csv_key="asset")


def score_eth_candidate_pnl(ledger: Path | None = None) -> dict:
    """Score hold-cycle net PnL when planned_exit_bar_ts has closed in CSV."""
    ledger = ledger or ETH_CANDIDATE_LEDGER
    rows = _load_ledger(ledger)
    if not rows:
        return {"status": "empty", "scored": 0, "pending": 0}

    scored = 0
    for r in rows:
        if r.get("scoring_version") != SCORING_VERSION_V2:
            continue
        if r.get("hold_cycle_pnl") is not None:
            continue
        if not r.get("direction") or not r.get("entry_bar_ts"):
            continue

        asset = r.get("asset", "eth_1h")
        csv_path = _asset_csv(asset)
        cfg = r.get("config", {})
        min_hold = int(cfg.get("min_hold_bars", 12))
        fee_bps = float(r.get("fee_bps") or cfg.get("fee_bps_reference", 5))

        result = _compute_hold_cycle_pnl(
            csv_path, r["entry_bar_ts"], r["direction"], min_hold, fee_bps,
        )
        if result is None:
            continue

        r["planned_exit_bar_ts"] = result["planned_exit_bar_ts"]
        r["exit_price"] = result["exit_price"]
        r["hold_cycle_pnl"] = result["hold_cycle_pnl"]
        r["hold_cycle_pnl_pct"] = result["hold_cycle_pnl_pct"]
        r["hold_cycle_outcome"] = result["hold_cycle_outcome"]
        r["fee_bps_round_trip"] = result["fee_bps_round_trip"]
        r["total_fees_pct"] = result["total_fees_pct"]
        r["hold_cycle_scored_at"] = datetime.now(timezone.utc).isoformat()
        scored += 1

    _save_ledger(ledger, rows)
    pending = sum(
        1 for r in rows
        if r.get("scoring_version") == SCORING_VERSION_V2
        and r.get("direction")
        and r.get("hold_cycle_pnl") is None
    )
    return {"status": "ok", "scored": scored, "pending": pending}


def _count_pending_hold_cycles(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("scoring_version") == SCORING_VERSION_V2
        and r.get("direction")
        and r.get("hold_cycle_pnl") is None
    )





def _count_pending(rows: list[dict]) -> int:

    return sum(1 for r in rows if r.get("outcome") is None and r.get("direction"))





def build_report(config: Config | None = None, ledger: Path | None = None,

                 recent: int = 30, track_id: str | None = None) -> dict:

    config = config or lean_config()

    ledger = ledger or _ledger_path(config)

    rows = _load_ledger(ledger)

    if track_id:

        rows = [r for r in rows if r.get("track_id") == track_id]

    scored = [r for r in rows if r.get("outcome") in ("hit", "miss")]



    def _bucket(key_fn):

        buckets: dict[str, list[dict]] = {}

        for r in scored:

            k = key_fn(r)

            buckets.setdefault(k, []).append(r)

        out = {}

        for k, rs in sorted(buckets.items()):

            hits = sum(1 for x in rs if x["outcome"] == "hit")

            out[k] = {"n": len(rs), "hit_rate": round(hits / len(rs), 4),

                      "hits": hits, "misses": len(rs) - hits}

        return out



    recent_rows = scored[-recent:] if recent else scored

    rh = sum(1 for x in recent_rows if x["outcome"] == "hit")



    cand = get_candidate(track_id) if track_id else None

    val = (cand or {}).get("validation", {})

    min_n = val.get("min_signals", 100)

    min_hit = val.get("min_hit_rate", 0.55)

    n_scored = len(scored)

    overall = round(sum(1 for r in scored if r["outcome"] == "hit") / n_scored, 4) if scored else None

    validated = n_scored >= min_n and overall is not None and overall >= min_hit



    report = {

        "ledger": str(ledger),

        "track_id": track_id,

        "total_logged": len(rows),

        "total_scored": n_scored,

        "pending": _count_pending(rows),

        "overall_hit_rate": overall,

        "recent_n": len(recent_rows),

        "recent_hit_rate": round(rh / len(recent_rows), 4) if recent_rows else None,

        "by_asset": _bucket(lambda r: r["asset"]),

        "by_source": _bucket(lambda r: r.get("source", r.get("track_id", "?"))),

        "last_5": [{k: r.get(k) for k in ("asset", "bar_ts", "direction", "source",

                                           "track_id", "outcome", "forward_return_pct")}

                   for r in scored[-5:]],

    }

    if track_id:

        report["validation"] = {

            "min_signals": min_n,

            "min_hit_rate": min_hit,

            "candidate_status": cand.get("status") if cand else None,

            "validated": validated,

            "progress": f"{n_scored}/{min_n} signals",

        }

    return report


def build_candidate_report(recent: int = 30) -> dict:
    """ETH candidate report — next-bar (legacy) + hold-cycle PnL (validation metric)."""
    ledger = ETH_CANDIDATE_LEDGER
    rows = _load_ledger(ledger)
    rows = [r for r in rows if r.get("track_id") == TRACK_ID]

    pre_fix = [r for r in rows if _is_pre_fix_candidate_row(r)]
    next_bar_scored = [
        r for r in rows
        if r.get("outcome") in ("hit", "miss") and r.get("direction")
    ]
    hold_scored = [
        r for r in rows
        if r.get("scoring_version") == SCORING_VERSION_V2
        and r.get("hold_cycle_pnl") is not None
    ]

    def _next_bar_rate(scored_rows: list[dict]) -> float | None:
        if not scored_rows:
            return None
        hits = sum(1 for r in scored_rows if r["outcome"] == "hit")
        return round(hits / len(scored_rows), 4)

    def _hold_profitable_rate(scored_rows: list[dict]) -> float | None:
        if not scored_rows:
            return None
        wins = sum(1 for r in scored_rows if r.get("hold_cycle_outcome") == "profit")
        return round(wins / len(scored_rows), 4)

    recent_nb = next_bar_scored[-recent:] if recent else next_bar_scored
    recent_hc = hold_scored[-recent:] if recent else hold_scored

    cand = get_candidate(TRACK_ID) or {}
    val = cand.get("validation", {})
    min_n = val.get("min_signals", 100)
    min_profitable = val.get("min_profitable_rate", val.get("min_hit_rate", 0.55))
    n_hold = len(hold_scored)
    hold_rate = _hold_profitable_rate(hold_scored)
    validated = n_hold >= min_n and hold_rate is not None and hold_rate >= min_profitable

    avg_pnl = None
    if hold_scored:
        avg_pnl = round(
            float(np.mean([r["hold_cycle_pnl"] for r in hold_scored])) * 100, 4,
        )

    return {
        "ledger": str(ledger),
        "track_id": TRACK_ID,
        "total_logged": len(rows),
        "pre_fix_next_bar_only": {
            "n": len(pre_fix),
            "note": "pre-fix, next-bar metric only — excluded from n=100 hold-cycle criterion",
            "rows": [
                {
                    "bar_ts": r.get("bar_ts"),
                    "direction": r.get("direction"),
                    "outcome": r.get("outcome"),
                    "forward_return_pct": r.get("forward_return_pct"),
                }
                for r in pre_fix
            ],
        },
        "next_bar_directional": {
            "label": "next-bar directional hit (supplementary)",
            "total_scored": len(next_bar_scored),
            "overall_hit_rate": _next_bar_rate(next_bar_scored),
            "recent_n": len(recent_nb),
            "recent_hit_rate": _next_bar_rate(recent_nb),
            "pending": _count_pending(rows),
        },
        "hold_cycle_pnl": {
            "label": "hold-cycle PnL, net of fees (validation metric)",
            "scoring_version": SCORING_VERSION_V2,
            "total_scored": n_hold,
            "pending": _count_pending_hold_cycles(rows),
            "profitable_rate": hold_rate,
            "avg_pnl_pct": avg_pnl,
            "recent_n": len(recent_hc),
            "recent_profitable_rate": _hold_profitable_rate(recent_hc),
            "fee_bps_round_trip": val.get("fee_bps_round_trip", 10),
        },
        "validation": {
            "min_signals": min_n,
            "min_profitable_rate": min_profitable,
            "metric": val.get("primary_metric", "hold_cycle_profitable_rate"),
            "metric_changed_utc": val.get("metric_changed_utc"),
            "start_collecting_v2_utc": val.get("start_collecting_v2_utc"),
            "candidate_status": cand.get("status"),
            "validated": validated,
            "progress": f"{n_hold}/{min_n} hold-cycles",
        },
        "last_5_hold_cycles": [
            {k: r.get(k) for k in (
                "entry_bar_ts", "planned_exit_bar_ts", "direction",
                "hold_cycle_pnl_pct", "hold_cycle_outcome", "outcome",
            )}
            for r in hold_scored[-5:]
        ],
        "last_5_next_bar": [
            {k: r.get(k) for k in (
                "bar_ts", "direction", "outcome", "forward_return_pct", "scoring_era",
            )}
            for r in next_bar_scored[-5:]
        ],
    }


def _print_candidate_report(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print(f"ETH CANDIDATE TRACK ({r['track_id']})")
    print(line)
    print(f"  ledger       : {r['ledger']}")
    print(f"  total logged : {r['total_logged']}")

    pf = r.get("pre_fix_next_bar_only", {})
    if pf.get("n"):
        print(f"\n  PRE-FIX (excluded from n=100 hold-cycle): {pf['n']} signal(s)")
        for row in pf.get("rows", []):
            print(f"    {str(row.get('bar_ts', ''))[:16]}  "
                  f"pred={row.get('direction', '?'):<4}  "
                  f"next-bar={row.get('outcome', '?')}  "
                  f"ret={row.get('forward_return_pct', '?')}")

    nb = r["next_bar_directional"]
    print(f"\n  {nb['label'].upper()}")
    print(f"    scored   : {nb['total_scored']}   pending: {nb['pending']}")
    if nb["overall_hit_rate"] is not None:
        print(f"    hit rate : {nb['overall_hit_rate']:.1%}  "
              f"(recent {nb['recent_n']}: {nb['recent_hit_rate']:.1%})")

    hc = r["hold_cycle_pnl"]
    print(f"\n  {hc['label'].upper()}  [{hc['scoring_version']}]")
    print(f"    scored   : {hc['total_scored']}   pending: {hc['pending']}")
    if hc["profitable_rate"] is not None:
        print(f"    profitable: {hc['profitable_rate']:.1%}  "
              f"avg PnL: {hc['avg_pnl_pct']:+.3f}%  "
              f"(recent {hc['recent_n']}: {hc['recent_profitable_rate']:.1%})")
    print(f"    fees     : {hc['fee_bps_round_trip']} bps round-trip")

    if v := r.get("validation"):
        print(f"\n  VALIDATION (hold-cycle) : {v['progress']}  "
              f"need profitable>={v['min_profitable_rate']:.0%}")
        print(f"  status       : {v['candidate_status']}  validated={v['validated']}")
        if v.get("metric_changed_utc"):
            print(f"  metric v2 since: {v['metric_changed_utc']}")

    if r.get("last_5_hold_cycles"):
        print("\n  LAST HOLD-CYCLES (v2)")
        for x in r["last_5_hold_cycles"]:
            print(f"    entry={str(x.get('entry_bar_ts', ''))[:16]}  "
                  f"exit={str(x.get('planned_exit_bar_ts', ''))[:16]}  "
                  f"pred={x.get('direction', '?'):<4}  "
                  f"pnl={x.get('hold_cycle_pnl_pct', '?'):+.3f}%  "
                  f"{x.get('hold_cycle_outcome', '?')}")

    print(line + "\n")





def _print_report(r: dict, title: str = "PRIMARY OUTCOME TRACKER") -> None:

    line = "=" * 68

    print("\n" + line)

    print(title)

    print(line)

    print(f"  ledger       : {r['ledger']}")

    print(f"  logged       : {r['total_logged']}   scored: {r['total_scored']}   "

          f"pending: {r['pending']}")

    if r["overall_hit_rate"] is not None:

        print(f"  overall hit  : {r['overall_hit_rate']:.1%}")

        print(f"  recent ({r['recent_n']}) hit : {r['recent_hit_rate']:.1%}")



    if v := r.get("validation"):

        print(f"\n  VALIDATION   : {v['progress']}  need hit>={v['min_hit_rate']:.0%}")

        print(f"  status       : {v['candidate_status']}  "

              f"validated={v['validated']}")



    for title2, key in (("BY ASSET", "by_asset"), ("BY SOURCE", "by_source")):

        if not r.get(key):

            continue

        print(f"\n  {title2}")

        print(f"  {'key':<14}{'n':>6}{'hit%':>8}")

        for k, v in r[key].items():

            print(f"  {k:<14}{v['n']:>6}{v['hit_rate']:>8.1%}")



    if r.get("last_5"):

        print("\n  LAST SCORED")

        for x in r["last_5"]:

            ret = x.get("forward_return_pct")

            ret_s = f"{ret:+.2f}%" if ret is not None else "?"

            mark = x.get("outcome", "?")

            src = x.get("source") or x.get("track_id", "?")

            print(f"    {x.get('asset','?'):<10} {str(x.get('bar_ts',''))[:16]}  "

                  f"pred={x.get('direction','?'):<4} src={src:<16} {mark}  ret={ret_s}")

    print(line + "\n")





def main() -> None:

    parser = argparse.ArgumentParser(description="FADE outcome tracker")

    parser.add_argument(

        "command",

        choices=("log", "log-candidate", "score", "score-candidate",

                 "report", "report-candidate", "run", "run-all"),

        default="run-all",

        nargs="?",

    )

    parser.add_argument("csv", nargs="*", help="1h CSV(s) for log command")

    parser.add_argument("--recent", type=int, default=30)

    parser.add_argument("--legacy-primary", action="store_true")

    args = parser.parse_args()



    config = lean_config()

    init_manifest()



    if args.command == "log":

        files = args.csv or ["btc_1h.csv", "eth_1h.csv"]

        for f in files:

            r = log_forecast(f, config=config, sparse_primary=not args.legacy_primary)

            print(f"  {r.get('asset', f):<12} {r['status']}"

                  + (f"  {r.get('direction')} ({r.get('source')})" if r.get("direction") else ""))

        return



    if args.command == "log-candidate":

        f = (args.csv or ["eth_1h.csv"])[0]

        r = log_eth_candidate(f)

        print(f"  eth_candidate {r['status']}"

              + (f"  {r.get('direction')}" if r.get("direction") else ""))

        return



    if args.command == "score":

        r = score_outcomes(config=config)

        print(f"primary scored {r.get('scored', 0)}  pending {r.get('pending', 0)}")

        return



    if args.command == "score-candidate":
        r_nb = score_eth_candidate()
        r_pnl = score_eth_candidate_pnl()
        print(f"candidate next-bar scored {r_nb.get('scored', 0)}  "
              f"pending {r_nb.get('pending', 0)}")
        print(f"candidate hold-cycle scored {r_pnl.get('scored', 0)}  "
              f"pending {r_pnl.get('pending', 0)}")
        return



    if args.command == "report":

        _print_report(build_report(config=config, recent=args.recent))

        return



    if args.command == "report-candidate":
        _print_candidate_report(build_candidate_report(recent=args.recent))
        return



    if args.command == "run":

        for f in (args.csv or ["btc_1h.csv", "eth_1h.csv"]):

            log_forecast(f, config=config, sparse_primary=not args.legacy_primary)

        score_outcomes(config=config)

        _print_report(build_report(config=config, recent=args.recent))

        return



    # run-all: primary + eth candidate

    for f in (args.csv or ["btc_1h.csv", "eth_1h.csv"]):

        log_forecast(f, config=config, sparse_primary=not args.legacy_primary)

    log_eth_candidate("eth_1h.csv")
    score_outcomes(config=config)
    score_eth_candidate()
    score_eth_candidate_pnl()
    _print_report(build_report(config=config, recent=args.recent))
    _print_candidate_report(build_candidate_report(recent=args.recent))





if __name__ == "__main__":

    main()


