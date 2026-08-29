"""PEPE follow-up — pre-registered true-OOS study of the neglected ADVANCE lead.

Background: altcoin_screen_v1 (evaluated 2026-07-05) found PEPE's path_lean3
holdout hit = 0.6082 (n=27,764 bars) — the highest in the repo — and flagged
ADVANCE. Nobody followed up.

This study evaluates path_lean3 on data the screen never saw: all bars AFTER
2026-07-05 (the screen's evaluation date). Rules are mined and frozen on data
<= cutoff, then applied unchanged to the post-cutoff window.

Also reported: net PnL on the OOS window with min_hold in {1, 24, 48}
(pre-specified, all three reported) at taker 5 bps/side and maker 2 bps/side.

Success (pre-registered): OOS hit >= 0.56 (sampling margin below 0.6082) AND
at least one min_hold variant net-positive at 5 bps.

Run:
    python -m fade.pipeline.pepe_followup_holdout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import lean_config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.predictor import collect_calibration_samples, predict_calibrated
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.pipeline.pnl_reality_check_v2 import _min_hold_positions, _simulate_variant
from fade.pipeline.pre_registration import load_manifest, save_manifest

STUDY_ID = "pepe_followup_v1"
CUTOFF_UTC = "2026-07-05T13:10:00+00:00"   # altcoin_screen_v1 evaluated_utc
MIN_HOLDS = (1, 24, 48)
TAKER_BPS = 5.0
MAKER_BPS = 2.0
SLIP_BPS = 0.0
BARS_PER_YEAR = 24 * 365
CSV = "pepe_1h.csv"
OUTPUT_PATH = Path("fade/output/pepe_followup_holdout.json")


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if any(s.get("study_id") == STUDY_ID for s in studies):
        return
    studies.append({
        "study_id": STUDY_ID,
        "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "path_lean3 on PEPE (0.6082 holdout hit in altcoin_screen_v1, "
            "flagged ADVANCE, never followed up) survives on true OOS data "
            "accumulated after the screen's evaluation date."
        ),
        "oos_definition": f"all bars after {CUTOFF_UTC} (screen evaluation time)",
        "method": (
            "mine + freeze path_lean3 rules on data <= cutoff (walk_forward on "
            "dev, stable-rule selection, calibration on dev); apply frozen "
            "rules unchanged to post-cutoff bars"
        ),
        "pnl_variants": {"min_hold_bars": list(MIN_HOLDS),
                         "taker_bps_side": TAKER_BPS, "maker_bps_side": MAKER_BPS},
        "success_criteria": {
            "primary": "OOS directional hit >= 0.56",
            "secondary": "at least one min_hold variant net-positive at 5 bps taker",
        },
        "known_risks": "~2 years history only; memecoin liquidity/spread worse "
                       "than majors; OOS window is ~8 weeks (n~1300 bars)",
        "leakage_guard": "rules, thresholds, calibration all frozen on <= cutoff",
        "on_failure": "REJECT — do not re-mine or adjust min_hold on the OOS window",
    })
    save_manifest(m)


def run_study() -> dict:
    _ensure_preregistered()
    if not Path(CSV).exists():
        return {"status": "missing_csv", "csv": CSV}

    config = lean_config()
    df = load_ohlcv(CSV)
    cutoff = pd.Timestamp(CUTOFF_UTC)

    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    close = df["close"].reindex(atoms.index)

    dev_mask = atoms.index <= cutoff
    oos_mask = atoms.index > cutoff
    dev_atoms, oos_atoms = atoms[dev_mask], atoms[oos_mask]
    dev_fwd = fwd[dev_mask]

    n_oos = int(oos_mask.sum())
    if n_oos < 300:
        return {"status": "insufficient_oos", "n_oos": n_oos,
                "hint": "refresh pepe_1h.csv to extend past the cutoff"}

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return {"status": "no_stable_rules_on_dev"}

    cal = CalibrationStore(config.cache_dir / "_pepe_followup_cal.json")
    cal.data = {"bins": cal._empty_bins(), "runs": 0, "history": []}
    thresholds = ev.compute_thresholds(dev_atoms, config)
    dev_disc = ev.discretize(dev_atoms, thresholds)
    dev_events = ev.build_events(dev_disc, config, allowed=set(frozen.index))
    dev_preds = predict_calibrated(dev_events, frozen, cal, positive={})
    if not dev_preds.empty:
        samples = collect_calibration_samples(dev_preds, dev_fwd)
        if samples:
            cal.update(samples)

    oos_disc = ev.discretize(oos_atoms, thresholds)
    oos_events = ev.build_events(oos_disc, config, allowed=set(frozen.index))
    preds = predict_calibrated(oos_events, frozen, cal, positive={})
    if preds.empty:
        return {"status": "no_oos_predictions", "n_oos_bars": n_oos}

    oos_close = close[oos_mask]
    bar_ret = oos_close.pct_change().shift(-1)
    out = preds.join(bar_ret.rename("bar_ret")).dropna(subset=["bar_ret"])

    pred_up = out["pred"].to_numpy().astype(int)
    rets = out["bar_ret"].to_numpy()
    actual_up = (rets > 0).astype(int)
    hit = float(np.mean(pred_up == actual_up))
    n_covered = len(out)

    # Binomial-ish sanity: standard error of the hit estimate.
    se = float(np.sqrt(hit * (1 - hit) / max(n_covered, 1)))

    variants = {}
    raw_target = np.where(pred_up == 1, 1.0, -1.0)
    for fee_name, fee_bps in (("taker", TAKER_BPS), ("maker", MAKER_BPS)):
        fee_rate = fee_bps / 1e4
        for mh in MIN_HOLDS:
            pos = _min_hold_positions(raw_target, mh)
            v = _simulate_variant(f"min_hold_{mh}", pos, rets, fee_rate,
                                  SLIP_BPS / 1e4, BARS_PER_YEAR)
            variants[f"{fee_name}_min_hold_{mh}"] = v

    bh_equity = np.cumprod(1.0 + rets)
    bh_total = float(bh_equity[-1] - 1.0)

    any_positive_taker = any(
        variants[f"taker_min_hold_{mh}"]["total_return"] > 0 for mh in MIN_HOLDS
    )
    passes = hit >= 0.56 and any_positive_taker

    return {
        "status": "ok",
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": CUTOFF_UTC,
        "n_dev_bars": int(dev_mask.sum()),
        "n_oos_bars": n_oos,
        "n_oos_covered": n_covered,
        "oos_span": f"{out.index.min()} -> {out.index.max()}",
        "n_frozen_rules": int(len(frozen)),
        "oos_hit": round(hit, 4),
        "oos_hit_se": round(se, 4),
        "original_screen_hit": 0.6082,
        "buy_hold_oos": round(bh_total, 4),
        "pnl_variants": variants,
        "any_min_hold_positive_taker": any_positive_taker,
        "passes": passes,
        "overall_verdict": (
            f"PASS — OOS hit {hit:.4f} >= 0.56 with a net-positive min_hold variant"
            if passes else
            f"REJECT — OOS hit {hit:.4f} (need >= 0.56) "
            f"/ positive-variant={any_positive_taker}"
        ),
    }


def _print(r: dict) -> None:
    line = "=" * 74
    print("\n" + line)
    print("PEPE FOLLOW-UP (true OOS after screen date, pre-registered)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}  {r}")
        print(line + "\n")
        return
    print(f"  dev bars: {r['n_dev_bars']:,}   OOS bars: {r['n_oos_bars']:,} "
          f"(covered {r['n_oos_covered']:,})   rules: {r['n_frozen_rules']}")
    print(f"  OOS span: {r['oos_span']}")
    print(f"  OOS hit: {r['oos_hit']} (se {r['oos_hit_se']})  vs original screen "
          f"{r['original_screen_hit']}  |  buy&hold OOS {r['buy_hold_oos']*100:+.1f}%")
    print()
    for name, v in r["pnl_variants"].items():
        print(f"    {name:<22} ret={v['total_return']*100:+7.1f}%  "
              f"sharpe={v['sharpe']:+.2f}  trades={v['n_changes']}  "
              f"costDrag={v['cost_drag']*100:.1f}%")
    print(f"\n  {r['overall_verdict']}")
    print(f"  -> {OUTPUT_PATH}")
    print(line + "\n")


def main() -> None:
    argparse.ArgumentParser(description="PEPE follow-up true-OOS study").parse_args()
    r = run_study()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
