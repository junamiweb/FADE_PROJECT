"""Pre-registration manifest — anti data-snooping protocol (Phase 0+).

Every candidate config must be registered BEFORE any final-test evaluation.
Lockbox v1 (18% sealed batch 30) is BURNED — no further tests on it.

Run:
    python -m fade.pipeline.pre_registration show
    python -m fade.pipeline.pre_registration burn-lockbox
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path("fade/output/pre_registration.json")
LOCKBOX_MANIFEST = Path("fade/output/lockbox_manifest.json")

# Frozen at Phase 0 start (2026-07-05).
DEFAULT_MANIFEST = {
    "version": 1,
    "protocol": (
        "Register config + success criteria BEFORE running on any final-test data. "
        "Lockbox v1 burned after batch 33. Live outcome_tracker for forward validation."
    ),
    "lockboxes": {
        "v1_18pct": {
            "status": "BURNED",
            "burned_at_utc": "2026-07-05T07:30:00+00:00",
            "burned_reason": (
                "Used for path_lean3 one-shot (batch 30) and regime_minhold one-shot "
                "(batch 33). No further config tests permitted on this slice."
            ),
            "manifest_file": "fade/output/lockbox_manifest.json",
        },
        "v2": {
            "status": "RESERVED",
            "note": "Next final-test lockbox must be NEWEST data sealed AFTER all "
                    "candidates pre-registered; hash recorded before any eval.",
        },
    },
    "candidates": [
        {
            "track_id": "eth_low_vr_minhold12",
            "status": "candidate_not_validated",
            "asset": "eth_1h",
            "config": {"vr_regime": "LOW_VR", "min_hold_bars": 12, "fee_bps_reference": 5},
            "selection_history": {
                "where_selected": "batch 32 holdout 70/30 grid (3 VR x 9 min_hold)",
                "selected_before_lockbox_test": True,
                "lockbox_search_on_slice": False,
                "lockbox_v1_one_shot_result": {
                    "batch": 33,
                    "return_5bps": 0.232,
                    "directional_hit": 0.5246,
                    "note": "Informational only — lockbox v1 now BURNED; NOT validation.",
                },
                "holdout_grid_winner_return_5bps": 0.942,
                "same_class_risk_as_btc": (
                    "Post-hoc grid on holdout (like BTC HIGH_VR+48). "
                    "Lockbox did NOT select this combo; one-shot confirmed positive "
                    "but cannot certify without forward live track."
                ),
            },
            "validation": {
                "method": "live_outcome_tracker_hold_cycle_pnl",
                "ledger": "fade/output/eth_candidate_outcomes.jsonl",
                "primary_metric": "hold_cycle_profitable_rate",
                "min_signals": 100,
                "min_profitable_rate": 0.55,
                "fee_bps": 5,
                "fee_bps_round_trip": 10,
                "metric_changed_utc": "2026-07-05T08:30:00+00:00",
                "metric_change_note": (
                    "v2: hold-cycle net PnL via pnl_sim._equity (aligns with lockbox batch 33). "
                    "Pre-fix signals (next-bar only) excluded from n=100."
                ),
                "start_collecting_utc": "2026-07-05T07:30:00+00:00",
                "start_collecting_v2_utc": "2026-07-05T08:30:00+00:00",
                "legacy_next_bar": {
                    "supplementary_only": True,
                    "min_hit_rate": 0.55,
                },
            },
            "pre_registered_utc": "2026-07-05T07:30:00+00:00",
        },
    ],
    "pending_preregistrations": [],
}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return json.loads(json.dumps(DEFAULT_MANIFEST))


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def init_manifest() -> dict:
    m = load_manifest()
    if not MANIFEST_PATH.exists():
        save_manifest(m)
    _mark_lockbox_burned(m)
    return m


def _mark_lockbox_burned(m: dict) -> None:
    if LOCKBOX_MANIFEST.exists():
        lb = json.loads(LOCKBOX_MANIFEST.read_text(encoding="utf-8"))
        for box in lb.get("lockboxes", []):
            box["status"] = "BURNED"
            box["burned_at_utc"] = datetime.now(timezone.utc).isoformat()
            box["burned_reason"] = m["lockboxes"]["v1_18pct"]["burned_reason"]
        lb["burn_policy"] = "v1 burned — use v2 for next one-shot after pre-register"
        LOCKBOX_MANIFEST.write_text(json.dumps(lb, indent=2), encoding="utf-8")


def get_candidate(track_id: str) -> dict | None:
    m = load_manifest()
    for c in m.get("candidates", []):
        if c["track_id"] == track_id:
            return c
    return None


def register_candidate(entry: dict) -> dict:
    """Append a pre-registered candidate (must be done BEFORE final test)."""
    m = load_manifest()
    m.setdefault("candidates", []).append({
        **entry,
        "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
        "status": entry.get("status", "candidate_not_validated"),
    })
    save_manifest(m)
    return entry


def _print_manifest(m: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    print("FADE PRE-REGISTRATION MANIFEST")
    print(line)
    for k, v in m.get("lockboxes", {}).items():
        print(f"  lockbox {k}: {v.get('status')}  {v.get('note', v.get('burned_reason', ''))[:60]}")
    print()
    for c in m.get("candidates", []):
        print(f"  [{c['track_id']}]  status={c['status']}")
        print(f"    config: {c.get('config')}")
        hist = c.get("selection_history", {})
        print(f"    selected: {hist.get('where_selected')}")
        print(f"    lockbox_search: {hist.get('lockbox_search_on_slice')}")
        val = c.get("validation", {})
        metric = val.get("primary_metric", "hit_rate")
        min_n = val.get("min_signals")
        if metric == "hold_cycle_profitable_rate":
            print(f"    validate via: {val.get('method')}  need n>={min_n} "
                  f"profitable>={val.get('min_profitable_rate')} (hold-cycle PnL)")
            if val.get("metric_changed_utc"):
                print(f"    metric v2 since: {val['metric_changed_utc']}")
        else:
            print(f"    validate via: {val.get('method')}  need n>={min_n} "
                  f"hit>={val.get('min_hit_rate')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE pre-registration manifest")
    parser.add_argument("cmd", choices=("show", "init", "burn-lockbox"), default="show", nargs="?")
    args = parser.parse_args()
    if args.cmd == "init" or args.cmd == "burn-lockbox":
        m = init_manifest()
        print("Initialized / lockbox v1 marked BURNED.")
    else:
        m = load_manifest()
    _print_manifest(m)


if __name__ == "__main__":
    main()
