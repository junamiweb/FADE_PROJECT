"""Forward validation snapshot — PRIMARY sparse + ETH candidate progress.

Writes council-readable JSON for Track A / E monitoring.

Run:
    python -m fade.pipeline.forward_validation_snapshot
    python -m fade.pipeline.forward_validation_snapshot --run  # log+score first
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fade.config import lean_config
from fade.pipeline.outcome_tracker import (
    build_candidate_report,
    build_report,
    build_ultra_report,
    log_eth_candidate,
    log_forecast,
    log_ultra_forecast,
    score_eth_candidate,
    score_eth_candidate_pnl,
    score_outcomes,
    score_ultra_outcomes,
)
from fade.pipeline.ultra_rule_search import load_active_rule

OUTPUT_PATH = Path("fade/output/forward_validation_snapshot.json")
DEFAULT_CSVS = ("btc_1h.csv", "eth_1h.csv")


def _phase_a_status(primary: dict) -> dict:
    scored = primary.get("total_scored", 0)
    sparse_signals = sum(
        v.get("n", 0) for k, v in (primary.get("by_source") or {}).items()
    )
    return {
        "track_id": "A",
        "study_id": "phase2_a_sparse_product_v1",
        "sparse_primary_logged": primary.get("total_logged", 0),
        "sparse_scored": scored,
        "overall_hit": primary.get("overall_hit_rate"),
        "recent_hit": primary.get("recent_hit_rate"),
        "target_hit": 0.58,
        "target_min_signals": 500,
        "progress": f"{scored}/500 scored signals",
        "status": "collecting" if scored < 500 else "ready_for_review",
        "note": "forward live truth — holdout replay is exploratory only",
    }


def _phase_e_stub(primary: dict) -> dict:
    scored = primary.get("total_scored", 0)
    return {
        "track_id": "E",
        "study_id": "phase2_e_decay_meter_v1",
        "status": "waiting_data" if scored < 90 else "ready_to_compute",
        "forward_scored": scored,
        "min_for_90d_proxy": 90,
        "note": "decay meter activates when PRIMARY ledger has ~90d of scored sparse signals",
    }


def _eth_candidate_status(cand: dict) -> dict:
    val = cand.get("validation", {})
    hc = cand.get("hold_cycle_pnl", {})
    return {
        "track_id": "eth_candidate",
        "track_name": "eth_low_vr_minhold12",
        "hold_cycles_scored": hc.get("total_scored", 0),
        "hold_cycles_pending": hc.get("pending", 0),
        "profitable_rate": hc.get("profitable_rate"),
        "progress": val.get("progress", "0/100 hold-cycles"),
        "validated": val.get("validated", False),
        "candidate_status": val.get("candidate_status"),
        "next_bar_hit_supplementary": cand.get("next_bar_directional", {}).get("overall_hit_rate"),
    }


def _ultra_status(ultra: dict) -> dict:
    val = ultra.get("validation", {})
    active = load_active_rule()
    return {
        "track_id": "ULTRA",
        "study_id": "ultra_next_bar_90_target_v1",
        "rule": active.get("rule_id", "cross_elite_bigmove05"),
        "holdout_reference_hit": active.get("holdout_hit"),
        "holdout_reference_n": active.get("holdout_n"),
        "scored": ultra.get("total_scored", 0),
        "overall_hit": ultra.get("overall_hit_rate"),
        "target_hit": val.get("min_hit_rate", 0.90),
        "target_min_signals": val.get("min_signals", 50),
        "progress": val.get("progress", "0/50 scored signals"),
        "validated": val.get("validated", False),
        "status": "validated" if val.get("validated") else "collecting",
        "note": val.get(
            "target_note",
            f"90% forward target; holdout exploratory ~{active.get('holdout_hit')}",
        ),
    }


def build_snapshot(run_cycle: bool = False) -> dict:
    config = lean_config()
    if run_cycle:
        for f in DEFAULT_CSVS:
            if Path(f).exists():
                log_forecast(f, config=config, sparse_primary=True)
        if Path("eth_1h.csv").exists():
            log_eth_candidate("eth_1h.csv")
        log_ultra_forecast()
        score_outcomes(config=config)
        score_eth_candidate()
        score_eth_candidate_pnl()
        score_ultra_outcomes()

    primary = build_report(config=config, recent=30)
    candidate = build_candidate_report(recent=30)
    ultra = build_ultra_report(recent=30)

    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_track_a": _phase_a_status(primary),
        "eth_candidate": _eth_candidate_status(candidate),
        "ultra_next_bar": _ultra_status(ultra),
        "phase2_track_e": _phase_e_stub(primary),
        "primary_report": primary,
        "eth_candidate_report": candidate,
        "ultra_report": ultra,
        "actions": {
            "hourly": "GitHub Action .github/workflows/outcome-tracker.yml (cron :05)",
            "manual": "python -m fade.pipeline.forward_validation_snapshot --run",
        },
    }


def _print(snap: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("FORWARD VALIDATION SNAPSHOT")
    print(line)
    a = snap["phase2_track_a"]
    e = snap["eth_candidate"]
    u = snap.get("ultra_next_bar", {})
    print(f"  Track A sparse PRIMARY: {a['progress']}  hit={a['overall_hit']}")
    print(f"  ETH candidate:          {e['progress']}  validated={e['validated']}")
    print(f"  ULTRA next-bar (90%):   {u.get('progress', '?')}  hit={u.get('overall_hit')}")
    print(f"  Track E decay meter:    {snap['phase2_track_e']['status']}")
    print(f"  -> {OUTPUT_PATH}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward validation snapshot")
    parser.add_argument("--run", action="store_true", help="log+score before snapshot")
    args = parser.parse_args()
    snap = build_snapshot(run_cycle=args.run)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    _print(snap)


if __name__ == "__main__":
    main()
