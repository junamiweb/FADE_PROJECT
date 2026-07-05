"""Tiered forecast — three validated confidence levels in one view.

Research-backed tiers (all passed strict holdout):
  FREQUENT  — 15m + 30m + 1h all agree (historically ~54.7% hit, ~65% of time)
  BALANCED  — 1h direction, any move (historically ~53.3% hit, ~85% coverage)
  QUALITY   — all three agree AND 1h calibrated prob >= 55% (stricter filter)

Run:
    python -m fade.pipeline.forecast_tiers btc_1h.csv
    python -m fade.pipeline.forecast_tiers btc_1h.csv --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from fade.config import Config, lean_config
from fade.core.conviction import read_conviction_state
from fade.core.inference import infer_latest
from fade.core.regimes import assign_regimes
from fade.core.significant_changes import detect_significant_changes
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv

RES_INTERVALS = ("15m", "30m", "1h")

# Console-only ASCII labels (JSON keeps Hebrew tier labels).
_CONSOLE_TIER_LABEL = {
    "conviction": "STREAK>=2 (conviction)",
    "frequent": "FULL AGREEMENT (15+30+60m)",
    "balanced": "HOURLY (general direction)",
    "quality": "QUALITY (agreement + high confidence)",
}

# Weak-conflict abstain thresholds (validated batch 27).
_ABSTAIN_CONVICTION_VS_BALANCED_PCT = 53.0
_ABSTAIN_FREQUENT_OVER_CONVICTION_PCT = 52.5

# Phase 1: abstain-by-default — PRIMARY only when conviction tier >= HIGH.
_SPARSE_CONVICT_TIER_IDS = frozenset({"elite", "strong", "high"})


def _res_files(primary_csv: str) -> dict[str, str]:
    stem = Path(primary_csv).stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return {res: f"{prefix}_{res}.csv" for res in RES_INTERVALS
            if Path(f"{prefix}_{res}.csv").exists()}


def run_tiered_forecast(primary_csv: str = "btc_1h.csv",
                        config: Config | None = None,
                        sparse_primary: bool = True) -> dict:
    config = config or lean_config()
    root = Path(primary_csv).parent
    res_files = _res_files(primary_csv)

    per_res: dict[str, dict] = {}
    for res, fname in res_files.items():
        path = root / fname
        if not path.exists():
            per_res[res] = {"status": "missing"}
            continue
        per_res[res] = infer_latest(str(path), config)

    base = per_res.get("1h", {"status": "missing"})
    result: dict = {
        "asset": Path(primary_csv).stem,
        "per_resolution": {k: {kk: vv for kk, vv in v.items()
                                 if kk != "rules_used"} for k, v in per_res.items()},
        "tiers": {},
    }

    # Regime from primary (1h) file
    if Path(primary_csv).exists():
        df = load_ohlcv(primary_csv)
        pool = atoms_mod.compute_atom_pool(df, config)
        atoms = pool[list(config.atom_columns)].dropna()
        changes = detect_significant_changes(pool.reindex(atoms.index), config=config)
        result["regime"] = str(assign_regimes(changes, config.post_shock_bars).iloc[-1])

    ok = [r for r in per_res.values() if r.get("status") == "ok"]
    preds = {k: v["pred"] for k, v in per_res.items() if v.get("status") == "ok"}

    # --- FREQUENT: all three agree ---
    if len(preds) == 3 and len(set(preds.values())) == 1:
        d = next(iter(ok))["direction"]
        probs = [r["calibrated_prob_pct"] for r in ok]
        result["tiers"]["frequent"] = {
            "active": True,
            "label": "הסכמה מלאה (15+30+60 דקות)",
            "direction": d,
            "avg_calibrated_pct": round(sum(probs) / len(probs), 1),
            "historical_hit": "~54.7%",
            "note": "פועל ~65% מהזמן",
        }
    else:
        dirs = {k: v.get("direction", "?") for k, v in per_res.items() if v.get("status") == "ok"}
        result["tiers"]["frequent"] = {
            "active": False,
            "label": "הסכמה מלאה",
            "reason": f"אין הסכמה: {dirs}",
            "note": "ממתין להסכמת כל שלוש הרזולוציות",
        }

    # --- BALANCED: 1h signal ---
    if base.get("status") == "ok":
        result["tiers"]["balanced"] = {
            "active": True,
            "label": "שעתי (כיוון כללי)",
            "direction": base["direction"],
            "calibrated_pct": base["calibrated_prob_pct"],
            "n_rules": base["n_rules"],
            "historical_hit": "~54.6%",
            "note": "כיסוי ~85%",
        }
    else:
        result["tiers"]["balanced"] = {
            "active": False,
            "label": "שעתי",
            "reason": base.get("status", "missing"),
        }

    # --- QUALITY: all agree + strong calibrated prob ---
    if result["tiers"].get("frequent", {}).get("active"):
        avg_p = result["tiers"]["frequent"]["avg_calibrated_pct"]
        if avg_p >= 55.0:
            result["tiers"]["quality"] = {
                "active": True,
                "label": "איכותי (הסכמה + ביטחון גבוה)",
                "direction": result["tiers"]["frequent"]["direction"],
                "calibrated_pct": avg_p,
                "historical_hit": "~57% (סף 1% על 15 דקות)",
                "note": "אות חזק, כיסוי נמוך (~1-5%)",
            }
        else:
            result["tiers"]["quality"] = {
                "active": False,
                "label": "איכותי",
                "reason": f"הסכמה קיימת אבל ביטחון נמוך ({avg_p}%)",
            }
    else:
        result["tiers"]["quality"] = {
            "active": False,
            "label": "איכותי",
            "reason": "דורש הסכמה מלאה קודם",
        }

    result["status"] = "ok" if any(t.get("active") for t in result["tiers"].values()) else "no_signal"

    # --- CONVICTION (validated reversal mechanism, batch 20-21) ---
    conv = read_conviction_state(primary_csv, config)
    result["conviction"] = conv
    if conv.get("vr_regime"):
        result["vr_regime"] = conv["vr_regime"]
    if conv.get("active_tier"):
        t = conv["active_tier"]
        tier_out = {
            "active": True,
            "label": t["label"],
            "direction": t["direction"],
            "calibrated_pct": t["calibrated_pct"],
            "historical_hit": f"~{t['calibrated_pct']}%",
            "note": t["note"],
            "streak_len": conv["streak_len"],
            "tf_agree": conv["tf_agree"],
        }
        if conv.get("vr_regime"):
            tier_out["vr_regime"] = conv["vr_regime"]
        result["tiers"]["conviction"] = tier_out
    else:
        inactive = {
            "active": False,
            "label": "conviction",
            "reason": f"streak={conv['streak_len']} tf_agree={conv['tf_agree']} (no tier)",
            "streak_len": conv["streak_len"],
            "tf_agree": conv["tf_agree"],
        }
        if conv.get("vr_regime"):
            inactive["vr_regime"] = conv["vr_regime"]
        if conv.get("vr_filter_reason"):
            inactive["reason"] = conv["vr_filter_reason"]
            inactive["vr_filter_note"] = (
                f"HIGH_VR: streak>={conv['streak_len']} suppressed (require streak>=3)"
            )
        result["tiers"]["conviction"] = inactive

    result["primary"] = _pick_primary(result, sparse=sparse_primary)
    if result["primary"].get("status") == "ok":
        result["status"] = "ok"

    return result


# Priority: quality > conviction > frequent > balanced (conviction wins over atomic rules).
_PRIMARY_ORDER = ("quality", "conviction", "frequent", "balanced")


def _attach_vr_context(result: dict, out: dict) -> dict:
    conv_state = result.get("conviction") or {}
    if vr := conv_state.get("vr_regime"):
        out["vr_regime"] = vr
    if conv_state.get("vr_filter_reason") and out.get("source") == "conviction":
        out["vr_filter_note"] = (
            f"HIGH_VR filter: conviction suppressed (streak={conv_state.get('streak_len')})"
        )
    elif conv_state.get("vr_filter_reason") and out.get("status") == "no_signal":
        out["vr_filter_note"] = conv_state.get("vr_filter_reason")
    return out


def _pick_primary(result: dict, sparse: bool = True) -> dict:
    tiers = result.get("tiers", {})
    conv = tiers.get("conviction", {})
    freq = tiers.get("frequent", {})
    bal = tiers.get("balanced", {})
    conv_state = result.get("conviction") or {}
    active_tid = (conv_state.get("active_tier") or {}).get("id")

    # Phase 1 sparse mode: abstain unless HIGH+ conviction or quality tier.
    if sparse:
        quality = tiers.get("quality", {})
        if quality.get("active"):
            chosen = {"source": "quality", **quality}
        elif conv.get("active") and active_tid in _SPARSE_CONVICT_TIER_IDS:
            chosen = {"source": "conviction", **conv}
        else:
            reason = "sparse_abstain"
            if conv.get("active") and active_tid:
                reason = f"sparse_abstain (tier={active_tid}, need high/strong/elite)"
            return _attach_vr_context(result, {
                "status": "no_signal",
                "reason": reason,
                "active_conviction_tier": active_tid,
            })
        conflict = None
        conflict_resolved = None
        pct = chosen.get("calibrated_pct") or chosen.get("avg_calibrated_pct")
        return _attach_vr_context(result, {
            "status": "ok",
            "source": chosen["source"],
            "label": chosen.get("label", chosen["source"]),
            "direction": chosen["direction"],
            "confidence_pct": pct,
            "historical_hit": chosen.get("historical_hit"),
            "sparse_mode": True,
            "active_conviction_tier": active_tid,
        })

    chosen = None
    conflict_resolved = None
    # Unanimous multi-res (frequent) beats conviction when they disagree.
    if conv.get("active") and freq.get("active") and conv["direction"] != freq["direction"]:
        chosen = {"source": "frequent", **freq}
        conflict_resolved = "frequent_over_conviction"
    else:
        for key in _PRIMARY_ORDER:
            t = tiers.get(key, {})
            if t.get("active"):
                chosen = {"source": key, **t}
                break

    if not chosen:
        return _attach_vr_context(result, {"status": "no_signal", "reason": "no active tier"})

    conflict = None
    if conflict_resolved:
        conflict = {
            "conviction": conv["direction"],
            "frequent": freq["direction"],
            "note": "frequent unanimous wins over conviction on conflict",
        }
    elif (chosen["source"] == "conviction" and bal.get("active")
            and bal.get("direction") != chosen.get("direction")):
        conflict = {
            "balanced": bal["direction"],
            "conviction": chosen["direction"],
            "note": "atomic 1h rules disagree; primary uses conviction",
        }
    elif (chosen["source"] not in ("conviction", "frequent") and conv.get("active")
          and conv.get("direction") != chosen.get("direction")):
        conflict = {
            "primary": chosen["direction"],
            "conviction": conv["direction"],
            "note": "conviction active but lower tier selected",
        }

    # Abstain on weak conflict: low confidence when tiers disagree.
    if conflict_resolved == "frequent_over_conviction":
        avg_conf = freq.get("avg_calibrated_pct")
        if avg_conf is not None and avg_conf < _ABSTAIN_FREQUENT_OVER_CONVICTION_PCT:
            abstain = {
                "status": "no_signal",
                "reason": "weak_conflict_abstain",
                "conflict": conflict,
                "conflict_resolved": conflict_resolved,
            }
            return _attach_vr_context(result, abstain)
    elif (chosen["source"] == "conviction" and bal.get("active")
          and bal.get("direction") != chosen.get("direction")):
        conv_pct = chosen.get("calibrated_pct") or chosen.get("avg_calibrated_pct")
        if conv_pct is not None and conv_pct < _ABSTAIN_CONVICTION_VS_BALANCED_PCT:
            return _attach_vr_context(result, {
                "status": "no_signal",
                "reason": "weak_conflict_abstain",
                "conflict": conflict,
            })

    pct = chosen.get("calibrated_pct") or chosen.get("avg_calibrated_pct")
    out = {
        "status": "ok",
        "source": chosen["source"],
        "label": chosen.get("label", chosen["source"]),
        "direction": chosen["direction"],
        "confidence_pct": pct,
        "historical_hit": chosen.get("historical_hit"),
        "conflict": conflict,
    }
    if conflict_resolved:
        out["conflict_resolved"] = conflict_resolved
    return _attach_vr_context(result, out)


def _console_tier_label(key: str, tier: dict) -> str:
    if key == "conviction" and tier.get("active"):
        return _CONSOLE_TIER_LABEL["conviction"]
    return _CONSOLE_TIER_LABEL.get(key, key)


def _console_inactive_reason(key: str, tier: dict) -> str:
    reason = tier.get("reason", "?")
    if not isinstance(reason, str):
        return str(reason)
    if key == "frequent":
        if "אין הסכמה" in reason:
            return reason.replace("אין הסכמה:", "no agreement:").replace("אין הסכמה", "no agreement")
    if key == "quality":
        if reason.startswith("דורש"):
            return "requires full agreement first"
        if "ביטחון נמוך" in reason:
            m = re.search(r"\(([\d.]+)%\)", reason)
            pct = m.group(1) if m else "?"
            return f"agreement exists but low confidence ({pct}%)"
    if key == "balanced":
        return reason  # status strings like no_match, missing
    return reason


def _print(r: dict) -> None:
    print("\n" + "=" * 62)
    print(f"FADE TIERED FORECAST - {r.get('asset', '?').upper()}")
    print("=" * 62)
    if regime := r.get("regime"):
        print(f"  Regime: {regime}")
    if vr := r.get("vr_regime"):
        print(f"  VR regime: {vr}")

    p = r.get("primary", {})
    print()
    if p.get("status") == "ok":
        src = p.get("source", "?")
        label = _CONSOLE_TIER_LABEL.get(src, src)
        print("  >>> PRIMARY <<<")
        print(f"    {p['direction']}  {p.get('confidence_pct', '?')}%  "
              f"({src}: {label})")
        if hr := p.get("historical_hit"):
            print(f"    Historical: {hr}")
        if vr_p := p.get("vr_regime"):
            print(f"    VR regime: {vr_p}")
        if note := p.get("vr_filter_note"):
            print(f"    VR filter: {note}")
        if c := p.get("conflict"):
            print(f"    CONFLICT: balanced={c.get('balanced')} "
                  f"conviction={c.get('conviction', c.get('primary'))}")
            print(f"    -> {c['note']}")
    else:
        print("  >>> PRIMARY: no signal <<<")
        if reason := p.get("reason"):
            print(f"    Reason: {reason}")
        if vr_p := p.get("vr_regime"):
            print(f"    VR regime: {vr_p}")
        if note := p.get("vr_filter_note"):
            print(f"    VR filter: {note}")

    print()
    for key in ("conviction", "frequent", "balanced", "quality"):
        t = r["tiers"].get(key, {})
        star = " *" if t.get("active") else ""
        print(f"  [{key.upper()}]{star}  {_console_tier_label(key, t)}")
        if t.get("active"):
            print(f"    Direction : {t['direction']}")
            if conf := t.get("calibrated_pct") or t.get("avg_calibrated_pct"):
                print(f"    Confidence: {conf}%")
            if sl := t.get("streak_len"):
                print(f"    Streak    : {sl} bars")
            if ta := t.get("tf_agree"):
                print(f"    TF agree  : {ta}/4")
            if hr := t.get("historical_hit"):
                print(f"    Historical: {hr}")
            if note := t.get("note"):
                print(f"    Note      : {note}")
        else:
            reason = _console_inactive_reason(key, t)
            print(f"    Inactive  : {reason}")
            if note := t.get("vr_filter_note"):
                print(f"    VR filter : {note}")
        print()

    print("  Per resolution:")
    for res, info in r.get("per_resolution", {}).items():
        st = info.get("status", "?")
        if st == "ok":
            print(f"    {res:<4} {info['direction']:>4}  {info['calibrated_prob_pct']}%  "
                  f"({info['n_rules']} rules)")
        else:
            print(f"    {res:<4} {st}")
    print("=" * 62 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE tiered forecast")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--legacy-primary", action="store_true",
                        help="Use pre-Phase-1 PRIMARY (all tiers, not sparse)")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    result = run_tiered_forecast(args.csv, sparse_primary=not args.legacy_primary)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print(result)


if __name__ == "__main__":
    main()
