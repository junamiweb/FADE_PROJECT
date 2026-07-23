"""Fully autonomous multi-asset blind lockbox — refresh → seal → reveal.

No human reveal approval. Each asset in ``blind_universe.json`` gets its own
blind window. When ``min_bars`` accumulate, the window is frozen and
immediately one-shot evaluated (burned).

Run (Action hourly):
    python -m fade.pipeline.blind_autonomy tick
    python -m fade.pipeline.blind_autonomy status
    python -m fade.pipeline.blind_autonomy bootstrap
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fade.pipeline import lockbox_v2 as lb2
from fade.pipeline.pre_registration import get_candidate, register_candidate
from fade.utils.logging import get_logger

log = get_logger("blind_autonomy")

UNIVERSE_PATH = Path("fade/output/blind_universe.json")
BOARD_PATH = Path("fade/output/blind_autonomy_board.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_universe() -> dict:
    return json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))


def ensure_preregistered(universe: dict) -> None:
    track = universe["study_track_id"]
    if get_candidate(track):
        return
    register_candidate({
        "track_id": track,
        "status": "candidate_collecting_blind",
        "asset": "multi",
        "config": {
            **universe["method"],
            "universe_file": str(UNIVERSE_PATH),
            "autonomous": True,
        },
        "selection_history": {
            "where_selected": "blind_autonomy bootstrap",
            "note": "Multi-asset blind forward lockbox; seals start at bootstrap time",
        },
        "validation": {
            "method": "lockbox_v2_autonomous_reveal",
            "primary_metric": "success_contract",
            "success_contract": universe["success_contract"],
            "fee_bps": universe["method"]["fee_bps"],
            "auto_reveal": True,
        },
    })
    log.info("pre-registered %s", track)


def _refresh_binance(asset_id: str, symbol: str, csv_path: Path) -> dict:
    from download_history import _fetch_and_save
    _fetch_and_save(str(csv_path), "1h", "2017-08-01", symbol=symbol)
    return {"id": asset_id, "source": "binance", "ok": True, "path": str(csv_path)}


def _refresh_yahoo(asset_id: str, symbol: str, csv_path: Path) -> dict:
    from fade.pipeline.stock_reversal_benchmark import fetch_yahoo_hourly
    fresh = fetch_yahoo_hourly(symbol, period="730d")
    if csv_path.exists():
        old = pd.read_csv(csv_path, parse_dates=["timestamp"])
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True)
        merged = (
            pd.concat([old, fresh], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )
    else:
        merged = fresh
    merged.to_csv(csv_path, index=False)
    return {
        "id": asset_id,
        "source": "yahoo",
        "ok": True,
        "path": str(csv_path),
        "rows": int(len(merged)),
        "last": str(merged["timestamp"].iloc[-1]),
    }


def refresh_universe(universe: dict) -> list[dict]:
    results = []
    for a in universe["assets"]:
        csv_path = Path(a["csv"])
        try:
            if a["source"] == "binance":
                results.append(_refresh_binance(a["id"], a["symbol"], csv_path))
            elif a["source"] == "yahoo":
                results.append(_refresh_yahoo(a["id"], a["symbol"], csv_path))
            else:
                results.append({"id": a["id"], "ok": False, "error": "unknown source"})
            time.sleep(0.35)
        except Exception as exc:  # noqa: BLE001
            log.exception("refresh failed %s", a["id"])
            results.append({"id": a["id"], "ok": False, "error": str(exc)})
    return results


def ensure_and_advance(universe: dict) -> list[dict]:
    """Create WAITING seals if needed; advance toward SEALED; auto-reveal."""
    track = universe["study_track_id"]
    events: list[dict] = []
    for a in universe["assets"]:
        csv_path = Path(a["csv"])
        if not csv_path.exists():
            events.append({"asset": a["id"], "status": "missing_csv"})
            continue
        asset_stem = csv_path.stem
        min_bars = int(a.get("min_bars") or 500)

        seal_res = lb2.cmd_seal(
            str(csv_path),
            track_id=track,
            after=None,
            min_bars=min_bars,
        )
        status = seal_res.get("status")
        events.append({
            "asset": a["id"],
            "seal_status": status,
            "n_available": seal_res.get("n_available"),
            "bars_still_needed": seal_res.get("bars_still_needed"),
            "n_lockbox": seal_res.get("n_lockbox"),
        })

        if status == "SEALED" or status == "already_sealed":
            # already_sealed means SEALED and waiting for reveal
            rev = lb2.cmd_reveal(
                track,
                confirm_burn=True,
                asset=asset_stem,
                autonomous=True,
            )
            events[-1]["reveal"] = {
                "status": rev.get("status"),
                "pass": (rev.get("result") or {}).get("success_contract_pass"),
                "ret": ((rev.get("result") or {}).get("strategy") or {}).get("total_return"),
                "bh": ((rev.get("result") or {}).get("buy_and_hold") or {}).get("total_return"),
            }
            log.info(
                "auto-reveal %s -> %s pass=%s",
                a["id"], rev.get("status"),
                (rev.get("result") or {}).get("success_contract_pass"),
            )
    return events


def build_board(universe: dict, refresh: list[dict], events: list[dict]) -> dict:
    m = lb2._load()
    track = universe["study_track_id"]
    seals = [s for s in m.get("seals", []) if s.get("track_id") == track]
    burned = [s for s in seals if s.get("status") == "BURNED"]
    waiting = [s for s in seals if s.get("status") == "WAITING"]
    sealed = [s for s in seals if s.get("status") == "SEALED"]
    passes = [
        s for s in burned
        if (s.get("burn_result_summary") or {}).get("success_contract_pass")
    ]
    return {
        "generated_utc": _utc_now(),
        "track_id": track,
        "autonomous": True,
        "universe_n": len(universe["assets"]),
        "refresh_ok": sum(1 for r in refresh if r.get("ok")),
        "refresh_fail": sum(1 for r in refresh if not r.get("ok")),
        "seals_waiting": len(waiting),
        "seals_sealed": len(sealed),
        "seals_burned": len(burned),
        "contract_passes": len(passes),
        "pass_assets": [s.get("asset") for s in passes],
        "waiting": [
            {
                "asset": s.get("asset"),
                "n_available": s.get("n_available"),
                "bars_still_needed": s.get("bars_still_needed"),
                "blind_after_utc": s.get("blind_after_utc"),
            }
            for s in waiting
        ],
        "burned_summary": [
            {
                "asset": s.get("asset"),
                **(s.get("burn_result_summary") or {}),
            }
            for s in burned
        ],
        "last_events": events,
        "north_star_he": (
            "אוטונומי מלא: ייבוא → חלון עיוור → חתימה → בדיקה חד-פעמית על "
            "קריפטו+מניות. הצלחה רק לפי חוזה PnL נטו."
        ),
    }


def tick(*, do_refresh: bool = True) -> dict:
    universe = load_universe()
    ensure_preregistered(universe)
    refresh = refresh_universe(universe) if do_refresh else []
    events = ensure_and_advance(universe)
    board = build_board(universe, refresh, events)
    BOARD_PATH.write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    return board


def bootstrap() -> dict:
    """Open blind windows for all universe assets (after=now). No metrics."""
    universe = load_universe()
    ensure_preregistered(universe)
    # Refresh once so CSVs exist, then open seals
    refresh = refresh_universe(universe)
    track = universe["study_track_id"]
    opened = []
    for a in universe["assets"]:
        csv_path = Path(a["csv"])
        if not csv_path.exists():
            opened.append({"asset": a["id"], "status": "missing_csv"})
            continue
        r = lb2.cmd_seal(
            str(csv_path),
            track_id=track,
            after=None,
            min_bars=int(a.get("min_bars") or 500),
        )
        opened.append({
            "asset": a["id"],
            "status": r.get("status"),
            "blind_after_utc": r.get("blind_after_utc"),
            "bars_still_needed": r.get("bars_still_needed"),
        })
    board = build_board(universe, refresh, opened)
    board["bootstrap"] = True
    BOARD_PATH.write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    return board


def status() -> dict:
    if BOARD_PATH.exists():
        board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    else:
        board = {"note": "no board yet — run bootstrap or tick"}
    m = lb2._load()
    board["live_seals"] = [
        {
            "track_id": s.get("track_id"),
            "asset": s.get("asset"),
            "status": s.get("status"),
            "n_available": s.get("n_available"),
            "bars_still_needed": s.get("bars_still_needed"),
            "pass": (s.get("burn_result_summary") or {}).get("success_contract_pass"),
        }
        for s in m.get("seals", [])
    ]
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous multi-asset blind lockbox")
    parser.add_argument(
        "cmd",
        choices=("tick", "bootstrap", "status", "refresh"),
        help="tick=refresh+seal+auto-reveal; bootstrap=open blinds; status; refresh",
    )
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    if args.cmd == "bootstrap":
        out = bootstrap()
    elif args.cmd == "tick":
        out = tick(do_refresh=not args.no_refresh)
    elif args.cmd == "refresh":
        u = load_universe()
        out = {"refresh": refresh_universe(u)}
    else:
        out = status()

    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
