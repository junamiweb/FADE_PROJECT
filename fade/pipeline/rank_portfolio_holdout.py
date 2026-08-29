"""Weekly cross-sectional rank portfolio — pre-registered holdout study.

Hypothesis (H1, primary): a weekly long-losers / short-winners portfolio over
the 37-crypto panel earns positive net Sharpe. Rebalancing once a week makes
fees negligible; long-short construction is market-neutral.

H2 (secondary, reported only): momentum — the exact opposite legs.

Construction (all pre-specified):
  - Every Monday 00:00 UTC rank assets by trailing 7d return (computed from
    hourly closes, data <= rebalance time only).
  - Long bottom quintile, short top quintile, equal weight within legs,
    hold one week.
  - Fees: 5 bps/side on actual turnover (maker 2 bps sensitivity reported).
  - Assets need >= 8d of history at a rebalance to be ranked.

Split: 70/30 chronological on the weekly timeline. Success (pre-registered):
H1 on holdout — annualized Sharpe > 0.5 net (taker), total return > 0,
>= 30 holdout weeks.

Run:
    python -m fade.pipeline.rank_portfolio_holdout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.pre_registration import load_manifest, save_manifest

STUDY_ID = "rank_portfolio_weekly_v1"
HOLDOUT_FRAC = 0.30
WEEKS_PER_YEAR = 52
TAKER_SIDE = 5 / 1e4
MAKER_SIDE = 2 / 1e4
QUINTILE = 0.2
MIN_ASSETS_RANKED = 10
OUTPUT_PATH = Path("fade/output/rank_portfolio_holdout.json")

EXCLUDE = {"spy", "aapl"}


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if any(s.get("study_id") == STUDY_ID for s in studies):
        return
    studies.append({
        "study_id": STUDY_ID,
        "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {
            "H1_primary": "weekly reversal — long bottom-quintile 7d losers, "
                          "short top-quintile winners, positive net Sharpe",
            "H2_secondary": "weekly momentum (opposite legs) — reported only",
        },
        "construction": {
            "rebalance": "Monday 00:00 UTC",
            "ranking": "trailing 7d return from hourly closes (<= t only)",
            "legs": "long bottom quintile / short top quintile, equal weight",
            "hold": "one week",
            "min_assets_ranked": MIN_ASSETS_RANKED,
        },
        "panel": "37 crypto 1h series (root *_1h.csv excl SPY, AAPL)",
        "costs": {"taker_bps_side": 5, "maker_bps_side": 2,
                  "charged_on": "actual turnover per leg"},
        "data_split": "holdout_70_30_chronological_weekly",
        "success_criteria": {
            "primary": "H1 holdout: annualized net Sharpe > 0.5 (taker), "
                       "total_return > 0, >= 30 weeks",
        },
        "leakage_guard": "rankings use closes <= rebalance timestamp only",
        "not_atoms": "cross-sectional portfolio — no path_lean3 vocabulary",
        "on_failure": "REJECT — do not tune quintile, window, or frequency on holdout",
    })
    save_manifest(m)


def _load_panel_closes() -> pd.DataFrame:
    closes = {}
    for f in sorted(Path(".").glob("*_1h.csv")):
        sym = f.stem.replace("_1h", "").lower()
        if sym in EXCLUDE:
            continue
        try:
            closes[sym] = load_ohlcv(str(f))["close"]
        except Exception:
            continue
    return pd.DataFrame(closes)


def _portfolio_run(week_ret: pd.DataFrame, rank_ret: pd.DataFrame,
                   reversal: bool, fee_side: float) -> dict:
    """Simulate weekly quintile long-short portfolio.

    week_ret: realized NEXT-week return per asset (row = rebalance date).
    rank_ret: trailing 7d return per asset at rebalance (row-aligned).
    """
    weekly_pnl = []
    prev_weights: pd.Series | None = None
    n_weeks_used = 0

    for ts in rank_ret.index:
        ranks = rank_ret.loc[ts].dropna()
        realized = week_ret.loc[ts]
        ranks = ranks[realized.reindex(ranks.index).notna()]
        if len(ranks) < MIN_ASSETS_RANKED:
            weekly_pnl.append(0.0)
            prev_weights = None
            continue

        k = max(1, int(len(ranks) * QUINTILE))
        order = ranks.sort_values()
        losers = order.index[:k]
        winners = order.index[-k:]
        w = pd.Series(0.0, index=ranks.index)
        if reversal:
            w[losers] = 1.0 / k / 2
            w[winners] = -1.0 / k / 2
        else:
            w[winners] = 1.0 / k / 2
            w[losers] = -1.0 / k / 2

        gross = float((w * realized.reindex(w.index)).sum())
        if prev_weights is None:
            turnover = float(w.abs().sum())
        else:
            aligned = w.subtract(prev_weights, fill_value=0.0)
            turnover = float(aligned.abs().sum())
        cost = turnover * fee_side
        weekly_pnl.append(gross - cost)
        prev_weights = w
        n_weeks_used += 1

    pnl = np.array(weekly_pnl)
    equity = np.cumprod(1.0 + pnl)
    total = float(equity[-1] - 1.0)
    sd = float(np.std(pnl))
    sharpe = float(np.mean(pnl) / sd * np.sqrt(WEEKS_PER_YEAR)) if sd > 0 else 0.0
    ann = float(equity[-1] ** (WEEKS_PER_YEAR / max(len(pnl), 1)) - 1.0)
    peak = np.maximum.accumulate(equity)
    dd = float(np.min(equity / peak - 1.0))
    return {
        "total_return": round(total, 4),
        "annual_return": round(ann, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 4),
        "n_weeks": len(pnl),
        "n_weeks_traded": n_weeks_used,
        "avg_weekly_pnl_bps": round(float(np.mean(pnl)) * 1e4, 2),
    }


def run_all() -> dict:
    _ensure_preregistered()
    closes = _load_panel_closes()
    print(f"Panel: {closes.shape[1]} assets, {len(closes)} hourly rows")

    mondays = pd.date_range(
        closes.index.min().normalize(), closes.index.max().normalize(),
        freq="W-MON", tz="UTC",
    )
    weekly_close = closes.reindex(mondays, method="ffill")

    rank_ret = weekly_close.pct_change(1)           # trailing 7d return at t
    week_ret = weekly_close.pct_change(1).shift(-1)  # realized t -> t+1w

    valid = rank_ret.notna().sum(axis=1) >= MIN_ASSETS_RANKED
    rank_ret, week_ret = rank_ret[valid], week_ret[valid]
    week_ret = week_ret.iloc[:-1]
    rank_ret = rank_ret.iloc[:-1]

    split = int(len(rank_ret) * (1 - HOLDOUT_FRAC))
    hold_rank, hold_week = rank_ret.iloc[split:], week_ret.iloc[split:]
    dev_rank, dev_week = rank_ret.iloc[:split], week_ret.iloc[:split]

    out = {"dev": {}, "holdout": {}}
    for label, (rr, wr) in (("dev", (dev_rank, dev_week)),
                            ("holdout", (hold_rank, hold_week))):
        for h_name, reversal in (("H1_reversal", True), ("H2_momentum", False)):
            for fee_name, fee in (("taker", TAKER_SIDE), ("maker", MAKER_SIDE)):
                out[label][f"{h_name}_{fee_name}"] = _portfolio_run(wr, rr, reversal, fee)

    h1 = out["holdout"]["H1_reversal_taker"]
    passes = (h1["sharpe"] > 0.5 and h1["total_return"] > 0
              and h1["n_weeks_traded"] >= 30)
    verdict = (
        f"PASS — H1 reversal holdout Sharpe {h1['sharpe']}, return "
        f"{h1['total_return']*100:+.1f}% over {h1['n_weeks_traded']} weeks"
        if passes else
        f"REJECT — H1 reversal holdout Sharpe {h1['sharpe']} "
        f"(need >0.5), return {h1['total_return']*100:+.1f}%"
    )

    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_assets": int(closes.shape[1]),
        "n_weeks_total": int(len(rank_ret)),
        "n_weeks_holdout": int(len(hold_rank)),
        "holdout_span": f"{hold_rank.index.min().date()} -> {hold_rank.index.max().date()}",
        "results": out,
        "passes": passes,
        "overall_verdict": verdict,
    }


def _print(r: dict) -> None:
    line = "=" * 74
    print("\n" + line)
    print("WEEKLY RANK PORTFOLIO (holdout, pre-registered)")
    print(line)
    print(f"  panel={r['panel_assets']} assets  weeks={r['n_weeks_total']} "
          f"(holdout {r['n_weeks_holdout']}: {r['holdout_span']})")
    for label in ("dev", "holdout"):
        print(f"\n  [{label}]")
        for name, v in r["results"][label].items():
            print(f"    {name:<22} ret={v['total_return']*100:+7.1f}%  "
                  f"ann={v['annual_return']*100:+7.1f}%  sharpe={v['sharpe']:+.2f}  "
                  f"DD={v['max_drawdown']*100:.1f}%  avg/wk={v['avg_weekly_pnl_bps']:+.1f}bps")
    print(f"\n  {r['overall_verdict']}")
    print(f"  -> {OUTPUT_PATH}")
    print(line + "\n")


def main() -> None:
    argparse.ArgumentParser(description="Weekly rank portfolio holdout").parse_args()
    r = run_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
