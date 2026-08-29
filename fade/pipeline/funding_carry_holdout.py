"""Funding carry (cash-and-carry) — pre-registered holdout study.

Hypothesis: a delta-neutral cash-and-carry position (long spot + short perp)
on BTC/ETH earns positive NET yield from funding payments, with ZERO price
prediction. Income from market structure, not forecasting.

Mechanics: perpetual funding is exchanged every 8h. When funding > 0, shorts
receive it. A carry position collects funding while price exposure nets to ~0
(long spot cancels short perp). Costs: entering/exiting means trading BOTH
legs — 2 legs x fee per side.

Variants (all pre-specified):
  - passive:  always in carry (enter once) — gross ceiling of the trade.
  - gated:    in carry only while trailing 7d mean funding > 0 (threshold 0
              fixed a priori — no fitting anywhere).
Fees: taker 5 bps/side x 2 legs = 20 bps per round trip; maker sensitivity
2 bps/side x 2 legs = 8 bps per round trip.

Split: 70/30 chronological. Success (pre-registered): gated variant on the
holdout achieves annualized net carry > 3% on BOTH assets AND max drawdown
smaller in magnitude than the annualized return.

Run:
    python -m fade.pipeline.funding_carry_holdout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.pipeline.pre_registration import load_manifest, save_manifest

STUDY_ID = "funding_carry_v1"
HOLDOUT_FRAC = 0.30
GATE_WINDOW_PERIODS = 21          # 7 days x 3 funding periods/day
PERIODS_PER_YEAR = 3 * 365
TAKER_RT = 20 / 1e4               # 2 legs x 5 bps x entry+exit... see note below
MAKER_RT = 8 / 1e4
OUTPUT_PATH = Path("fade/output/funding_carry_holdout.json")

# --- realism_v2 sensitivity (same data, stricter costs — NOT a fresh test) ---
# Capital models: funding accrues on 1x notional, but the position ties up
# more capital than the notional. Reported per model:
#   efficient    1.0x — portfolio margin, spot serves as perp collateral
#   conservative 1.5x — spot + 50% margin buffer on the short perp
#   naive        2.0x — spot + full 1x collateral held aside
CAPITAL_MODELS = {"efficient_1x": 1.0, "conservative_1_5x": 1.5, "naive_2x": 2.0}
# Extra cost of crossing the spot-perp basis on entry/exit (assumption).
BASIS_SLIPPAGE_RT = 5 / 1e4
# Risk-free benchmark (assumption, documented — approx short T-bill yield).
RF_ANNUAL = 0.04
# Worst-window stress length: 90 days of 8h periods.
STRESS_WINDOW_PERIODS = 270

ASSETS = {"btc": "funding_btc.csv", "eth": "funding_eth.csv"}


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if any(s.get("study_id") == STUDY_ID for s in studies):
        return
    studies.append({
        "study_id": STUDY_ID,
        "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Delta-neutral cash-and-carry (long spot + short perp) on BTC/ETH "
            "earns positive net yield from funding, with zero price prediction."
        ),
        "variants": {
            "passive": "always in carry — gross ceiling",
            "gated": "in carry only while trailing 7d mean funding > 0 "
                     "(threshold 0 fixed a priori, no fitting)",
        },
        "costs": {
            "taker_round_trip_bps": 20,
            "maker_round_trip_bps": 8,
            "note": "2 legs (spot+perp), fee charged on each enter/exit",
        },
        "data_split": "holdout_70_30_chronological",
        "success_criteria": {
            "primary": "gated annualized net carry > 3% on holdout, both assets",
            "secondary": "abs(max_drawdown) < annualized return",
        },
        "leakage_guard": "gate uses trailing 7d funding mean only; threshold fixed at 0",
        "not_atoms": "income strategy — no price prediction, no path_lean3",
        "on_failure": "REJECT — do not tune gate window or threshold on holdout",
    })
    save_manifest(m)


def _load_funding(path: str) -> pd.Series:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    s = pd.Series(df["funding_rate"].astype(float).values, index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _carry_stats(period_ret: np.ndarray, ppy: float) -> dict:
    equity = np.cumprod(1.0 + period_ret)
    n = len(period_ret)
    total = float(equity[-1] - 1.0)
    ann = float(equity[-1] ** (ppy / max(n, 1)) - 1.0) if n else 0.0
    sd = float(np.std(period_ret))
    sharpe = float(np.mean(period_ret) / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = float(np.min(equity / peak - 1.0)) if n else 0.0
    return {"total_return": round(total, 4), "annual_return": round(ann, 4),
            "sharpe": round(sharpe, 3), "max_drawdown": round(dd, 4)}


def run_asset(sym: str, funding_csv: str) -> dict:
    if not Path(funding_csv).exists():
        return {"asset": sym, "status": "missing_funding_csv"}

    funding = _load_funding(funding_csv)
    if len(funding) < 1000:
        return {"asset": sym, "status": "insufficient_data", "n": len(funding)}

    # Carry P&L per 8h period: short perp RECEIVES funding when rate > 0.
    # Delta-neutral -> price return cancels; per-period return = funding rate.
    carry_ret = funding.copy()

    gate_signal = funding.rolling(GATE_WINDOW_PERIODS, min_periods=GATE_WINDOW_PERIODS).mean()
    # Position decided from data <= t applies to the NEXT period's funding.
    in_carry = (gate_signal > 0).astype(float).shift(1).fillna(0.0)

    frame = pd.DataFrame({"carry": carry_ret, "in": in_carry}).dropna()
    split = int(len(frame) * (1 - HOLDOUT_FRAC))
    hold = frame.iloc[split:]

    results = {}
    for fee_name, rt_cost in (("taker", TAKER_RT), ("maker", MAKER_RT)):
        # Passive: single entry at start, single exit at end.
        passive = hold["carry"].to_numpy().copy()
        passive[0] -= rt_cost / 2
        passive[-1] -= rt_cost / 2
        # Gated: cost of rt_cost/2 on each transition (enter or exit both legs).
        pos = hold["in"].to_numpy()
        prev = np.concatenate([[0.0], pos[:-1]])
        transitions = np.abs(pos - prev)
        gated = pos * hold["carry"].to_numpy() - transitions * (rt_cost / 2)
        n_switches = int(np.sum(transitions > 0))

        results[fee_name] = {
            "passive": _carry_stats(passive, PERIODS_PER_YEAR),
            "gated": {**_carry_stats(gated, PERIODS_PER_YEAR),
                      "n_switches": n_switches,
                      "active_periods": int(np.sum(pos > 0)),
                      "coverage_pct": round(float(np.mean(pos > 0) * 100), 1)},
        }

    g = results["taker"]["gated"]
    passes = (
        g["annual_return"] > 0.03
        and abs(g["max_drawdown"]) < g["annual_return"]
    )

    # --- realism_v2 sensitivity: stricter costs on the gated taker variant ---
    pos = hold["in"].to_numpy()
    prev = np.concatenate([[0.0], pos[:-1]])
    transitions = np.abs(pos - prev)
    stressed = (pos * hold["carry"].to_numpy()
                - transitions * ((TAKER_RT + BASIS_SLIPPAGE_RT) / 2))
    stressed_ann = _carry_stats(stressed, PERIODS_PER_YEAR)["annual_return"]

    capital_view = {}
    for name, mult in CAPITAL_MODELS.items():
        on_capital = stressed_ann / mult
        capital_view[name] = {
            "annual_return_on_capital": round(on_capital, 4),
            "excess_over_rf": round(on_capital - RF_ANNUAL, 4),
        }

    carry_arr = hold["carry"].to_numpy()
    worst_ann = None
    if len(carry_arr) >= STRESS_WINDOW_PERIODS:
        roll = pd.Series(carry_arr).rolling(STRESS_WINDOW_PERIODS).sum().dropna()
        worst_ann = round(float(roll.min()) * (PERIODS_PER_YEAR / STRESS_WINDOW_PERIODS), 4)

    realism = {
        "note": "sensitivity on same data — stricter costs only, not a fresh hypothesis test",
        "assumptions": {"basis_slippage_rt_bps": BASIS_SLIPPAGE_RT * 1e4,
                        "rf_annual": RF_ANNUAL},
        "gated_taker_plus_basis_annual_on_notional": round(stressed_ann, 4),
        "return_on_capital": capital_view,
        "worst_rolling_90d_funding_annualized": worst_ann,
        "beats_rf_at_conservative_capital": (
            capital_view["conservative_1_5x"]["excess_over_rf"] > 0
        ),
    }

    return {
        "realism_v2": realism,
        "asset": sym,
        "status": "evaluated",
        "n_periods_total": len(frame),
        "n_periods_holdout": len(hold),
        "holdout_span": f"{hold.index.min()} -> {hold.index.max()}",
        "funding_mean_annualized_holdout": round(float(hold["carry"].mean()) * PERIODS_PER_YEAR, 4),
        "fees": results,
        "passes": passes,
    }


def run_all() -> dict:
    _ensure_preregistered()
    results = {sym: run_asset(sym, csv) for sym, csv in ASSETS.items()}
    evaluated = [r for r in results.values() if r.get("status") == "evaluated"]
    n_pass = sum(1 for r in evaluated if r.get("passes"))
    overall = n_pass == len(ASSETS) and len(evaluated) == len(ASSETS)
    verdict = (
        f"PASS — gated carry clears 3% net annualized with contained DD on both assets"
        if overall else
        f"REJECT — only {n_pass}/{len(evaluated)} assets pass"
    )
    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_verdict": verdict,
        "results": results,
    }


def _print(r: dict) -> None:
    line = "=" * 74
    print("\n" + line)
    print("FUNDING CARRY (cash-and-carry, holdout, pre-registered)")
    print(line)
    for sym, item in r["results"].items():
        if item.get("status") != "evaluated":
            print(f"  {sym}: {item['status']}")
            continue
        print(f"\n  {sym.upper()}  holdout {item['holdout_span']}  "
              f"({item['n_periods_holdout']} periods of 8h)")
        print(f"    raw funding annualized: {item['funding_mean_annualized_holdout']*100:+.2f}%")
        for fee_name in ("taker", "maker"):
            f = item["fees"][fee_name]
            p, g = f["passive"], f["gated"]
            print(f"    [{fee_name}] passive: ann={p['annual_return']*100:+.2f}%  "
                  f"sharpe={p['sharpe']}  DD={p['max_drawdown']*100:.1f}%")
            print(f"    [{fee_name}] gated  : ann={g['annual_return']*100:+.2f}%  "
                  f"sharpe={g['sharpe']}  DD={g['max_drawdown']*100:.1f}%  "
                  f"cov={g['coverage_pct']}%  switches={g['n_switches']}")
        rv = item.get("realism_v2")
        if rv:
            print(f"    realism_v2 (gated taker + {rv['assumptions']['basis_slippage_rt_bps']:.0f}bps basis): "
                  f"ann on notional {rv['gated_taker_plus_basis_annual_on_notional']*100:+.2f}%")
            for name, c in rv["return_on_capital"].items():
                print(f"      {name:<18} on-capital {c['annual_return_on_capital']*100:+.2f}%  "
                      f"excess vs rf({rv['assumptions']['rf_annual']*100:.0f}%): "
                      f"{c['excess_over_rf']*100:+.2f}%")
            print(f"      worst rolling 90d funding annualized: "
                  f"{(rv['worst_rolling_90d_funding_annualized'] or 0)*100:+.2f}%")
        print(f"    -> {'PASS' if item['passes'] else 'FAIL'}")
    print(f"\n  {r['overall_verdict']}")
    print(f"  -> {OUTPUT_PATH}")
    print(line + "\n")


def main() -> None:
    argparse.ArgumentParser(description="Funding carry holdout").parse_args()
    r = run_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
