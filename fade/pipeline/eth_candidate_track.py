"""ETH candidate track — LOW_VR + min_hold=12 (Phase 0 live validation).

Pre-registered candidate (batch 32 holdout grid). NOT production.
Forward validation ONLY via outcome_tracker ledger — no lockbox reuse.

Config frozen:
  vr_regime = LOW_VR  (tertiles fit on pre-lockbox 82% only)
  min_hold  = 12 bars
  direction = path_lean3 inference on eth_1h

Run:
    python -m fade.pipeline.eth_candidate_track evaluate eth_1h.csv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fade.config import Config, lean_config
from fade.core.data_loader import load_ohlcv
from fade.core.inference import infer_latest
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.final_lockbox import DEFAULT_LOCKBOX_FRAC
from fade.pipeline.pre_registration import get_candidate

TRACK_ID = "eth_low_vr_minhold12"
STATE_PATH = Path("fade/output/eth_candidate_state.json")
SCORING_VERSION_V2 = "v2_hold_cycle_pnl"


def _frozen_vr_regime(csv_path: str) -> tuple[str | None, dict]:
    """VR regime at latest bar; thresholds frozen on pre-lockbox slice."""
    config = Config()
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    vr = compute_vol_ratio(ret, config.vol_ratio_short_window, config.vol_ratio_long_window)
    cut = int(len(df) * (1.0 - DEFAULT_LOCKBOX_FRAC))
    dev_vr = vr.iloc[:cut].dropna()
    low = float(dev_vr.quantile(1 / 3))
    high = float(dev_vr.quantile(2 / 3))
    reg = assign_vr_regime(vr, low, high)
    latest = str(reg.iloc[-1]) if reg.notna().any() else None
    return latest, {"low_threshold": low, "high_threshold": high, "cut_bar": cut}


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"position": 0.0, "bars_in_position": 0, "last_bar_ts": None}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _target_direction(infer: dict) -> float:
    if infer.get("status") != "ok":
        return 0.0
    return 1.0 if infer.get("pred") == 1 else -1.0


def hold_cycle_meta(
    csv_path: str,
    entry_bar_ts: str,
    min_hold: int,
    fee_bps: float = 5.0,
) -> dict | None:
    """Entry/exit metadata for a min_hold cycle (no look-ahead beyond CSV end)."""
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
    return {
        "entry_bar_ts": entry_bar_ts,
        "entry_price": float(df["close"].iloc[loc]),
        "planned_exit_bar_ts": str(df.index[exit_loc]),
        "min_hold_bars": min_hold,
        "fee_bps": fee_bps,
        "fee_bps_round_trip": fee_bps * 2,
        "scoring_version": SCORING_VERSION_V2,
    }


def _min_hold_step(state: dict, want: float, min_hold: int) -> tuple[float, bool]:
    """Return (new_position, traded)."""
    cur = state["position"]
    bars = state["bars_in_position"]
    if want == cur:
        if cur != 0:
            state["bars_in_position"] = bars + 1
        return cur, False
    if bars >= min_hold or cur == 0:
        state["position"] = want
        state["bars_in_position"] = 1 if want != 0 else 0
        return want, True
    state["bars_in_position"] = bars + 1
    return cur, False


def evaluate_eth_candidate(
    csv_path: str = "eth_1h.csv",
    min_hold: int | None = None,
    required_regime: str | None = None,
) -> dict:
    cand = get_candidate(TRACK_ID) or {}
    cfg = cand.get("config", {})
    min_hold = min_hold if min_hold is not None else cfg.get("min_hold_bars", 12)
    required_regime = required_regime or cfg.get("vr_regime", "LOW_VR")

    if not Path(csv_path).exists():
        return {"track_id": TRACK_ID, "status": "missing_file"}

    bar_ts = str(load_ohlcv(csv_path).index[-1])
    vr_regime, vr_meta = _frozen_vr_regime(csv_path)
    infer = infer_latest(csv_path, config=lean_config())
    raw_target = _target_direction(infer)

    # Gate: only active in required VR regime
    gated_want = raw_target if vr_regime == required_regime else 0.0

    state = _load_state()
    if state.get("last_bar_ts") == bar_ts:
        return {"track_id": TRACK_ID, "status": "duplicate", "bar_ts": bar_ts}

    new_pos, traded = _min_hold_step(state, gated_want, min_hold)
    state["last_bar_ts"] = bar_ts
    _save_state(state)

    direction = None
    hold_cycle = None
    fee_bps = float(cfg.get("fee_bps_reference", 5))
    if traded and new_pos != 0:
        direction = "UP" if new_pos > 0 else "DOWN"
        hold_cycle = hold_cycle_meta(csv_path, bar_ts, min_hold, fee_bps=fee_bps)

    return {
        "track_id": TRACK_ID,
        "status": "signal" if direction else "no_signal",
        "bar_ts": bar_ts,
        "asset": Path(csv_path).stem,
        "vr_regime": vr_regime,
        "vr_gate_pass": vr_regime == required_regime,
        "path_lean3_status": infer.get("status"),
        "path_lean3_direction": infer.get("direction"),
        "min_hold": min_hold,
        "traded": traded,
        "direction": direction,
        "position_after": new_pos,
        "hold_cycle": hold_cycle,
        "candidate_status": cand.get("status", "candidate_not_validated"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vr_thresholds": {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in vr_meta.items()},
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ETH candidate track evaluator")
    parser.add_argument("command", default="evaluate", nargs="?")
    parser.add_argument("csv", default="eth_1h.csv")
    args = parser.parse_args()
    r = evaluate_eth_candidate(args.csv)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
