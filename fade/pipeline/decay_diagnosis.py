"""Decay diagnosis — why did streak-reversal weaken 4-5pp in 2025-2026?

Three SEPARATE hypotheses (not mixed):

  H1  Regime-specific temporal shift — is decay concentrated in vol/funding
      sub-periods, or uniform across 2025-2026?
  H2  Monotonic time trend — quarterly rolling reversal-index 2017->today;
      is there a consistent negative slope even within 2017-2024?
  H3  Microstructure change — do we have spread/liquidity data? (Honest limit.)

After running, classifies into decision branch A / B / C for autonomous follow-up.

Run:
    python -m fade.pipeline.decay_diagnosis
    python -m fade.pipeline.decay_diagnosis btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.regimes import assign_vr_regime, compute_vol_ratio
from fade.pipeline.trend_structure import _signed_streak
from fade.utils.logging import get_logger

log = get_logger("decay_diagnosis")

RECENT_YEARS = (2025, 2026)
MIN_SUPPORT = 80
STREAK_MIN = 3
N_SHUFFLES = 500


def _reversal_hit(streak: np.ndarray, up: np.ndarray, mask: np.ndarray,
                  min_streak: int = STREAK_MIN) -> dict:
    """Contrarian reversal hit-rate on masked bars."""
    sel = mask & (np.abs(streak) >= min_streak)
    k = int(sel.sum())
    if k < MIN_SUPPORT:
        return {"n": k, "status": "low_support"}
    hits = 0
    for i in np.flatnonzero(sel):
        s = streak[i]
        hits += int(up[i] == (0 if s > 0 else 1))
    hit = hits / k
    return {"n": k, "hit": round(hit, 4), "edge": round(hit - 0.5, 4), "status": "ok"}


def _reversion_index(ret: np.ndarray, streak: np.ndarray, mask: np.ndarray,
                     min_streak: int = 2) -> dict:
    """Aggregate reversion index = 0.5 - continuation after |streak|>=min_streak."""
    up = (ret > 0).astype(int)
    cont_hits = cont_n = 0
    for m in range(min_streak, 8):
        for sgn in (m, -m):
            sel = mask & (streak == sgn)
            k = int(sel.sum())
            if k < 30:
                continue
            nxt = up[sel]
            cont_hits += int((nxt == (1 if sgn > 0 else 0)).sum())
            cont_n += k
    if cont_n < MIN_SUPPORT:
        return {"status": "low_support", "n": cont_n}
    cont = cont_hits / cont_n
    rev = 0.5 - cont
    return {
        "status": "ok", "n": cont_n,
        "continuation": round(cont, 4),
        "reversion_index": round(rev, 4),
    }


def _perm_p(hit: float, n: int, base: float, rng: np.random.Generator) -> float:
    """Two-sided permutation p vs base rate."""
    null = rng.binomial(n, base, N_SHUFFLES) / n
    dev = abs(hit - 0.5)
    return round((1 + int(np.sum(np.abs(null - base) >= dev))) / (1 + N_SHUFFLES), 4)


def _load_funding(path: str = "funding_btc.csv") -> pd.Series | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return pd.Series(df["funding_rate"].astype(float).values, index=ts).sort_index()


def hypothesis1_regime_shift(csv_path: str, funding_path: str = "funding_btc.csv") -> dict:
    """H1: vol-regime and funding-regime breakdown for recent vs earlier."""
    config = Config()
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    up = (ret > 0).astype(int).to_numpy()
    vr = compute_vol_ratio(ret, config.vol_ratio_short_window, config.vol_ratio_long_window)
    years = df.index.year.to_numpy()

    # VR thresholds from pre-2025 dev only (no look-ahead).
    pre_mask = years < RECENT_YEARS[0]
    vr_pre = vr[pre_mask].dropna()
    low_thr = float(vr_pre.quantile(1 / 3))
    high_thr = float(vr_pre.quantile(2 / 3))
    vr_reg = assign_vr_regime(vr, low_thr, high_thr)

    funding = _load_funding(funding_path)
    fund_reg = pd.Series("NEUTRAL", index=df.index, dtype=object)
    if funding is not None:
        aligned = funding.reindex(df.index, method="ffill")
        pre_fund = aligned[pre_mask].dropna()
        if len(pre_fund) > 100:
            p10, p90 = float(pre_fund.quantile(0.10)), float(pre_fund.quantile(0.90))
            fund_reg.loc[aligned <= p10] = "EXTREME_NEG"
            fund_reg.loc[aligned >= p90] = "EXTREME_POS"

    recent = years >= RECENT_YEARS[0]
    earlier = ~recent

    overall = {
        "recent": _reversal_hit(streak, up, recent),
        "earlier": _reversal_hit(streak, up, earlier),
    }
    if overall["recent"].get("hit") and overall["earlier"].get("hit"):
        overall["delta_pp"] = round(
            (overall["recent"]["hit"] - overall["earlier"]["hit"]) * 100, 2)

    by_vr = {}
    for reg in ("LOW_VR", "NORMAL", "HIGH_VR"):
        m = (vr_reg == reg).to_numpy()
        by_vr[reg] = {
            "recent": _reversal_hit(streak, up, recent & m),
            "earlier": _reversal_hit(streak, up, earlier & m),
        }
        r, e = by_vr[reg]["recent"], by_vr[reg]["earlier"]
        if r.get("hit") and e.get("hit"):
            by_vr[reg]["delta_pp"] = round((r["hit"] - e["hit"]) * 100, 2)

    by_fund = {}
    for reg in ("EXTREME_NEG", "NEUTRAL", "EXTREME_POS"):
        m = (fund_reg == reg).to_numpy()
        by_fund[reg] = {
            "recent": _reversal_hit(streak, up, recent & m),
            "earlier": _reversal_hit(streak, up, earlier & m),
        }
        r, e = by_fund[reg]["recent"], by_fund[reg]["earlier"]
        if r.get("hit") and e.get("hit"):
            by_fund[reg]["delta_pp"] = round((r["hit"] - e["hit"]) * 100, 2)

    # Uniformity: spread of regime-specific deltas in recent period
    vr_deltas = [by_vr[r]["delta_pp"] for r in by_vr if "delta_pp" in by_vr[r]]
    fund_deltas = [by_fund[r]["delta_pp"] for r in by_fund if "delta_pp" in by_fund[r]]
    spread_vr = max(vr_deltas) - min(vr_deltas) if len(vr_deltas) >= 2 else 0.0
    spread_fund = max(fund_deltas) - min(fund_deltas) if len(fund_deltas) >= 2 else 0.0

    # Concentrated if one regime decayed much more than others
    concentrated_vr = spread_vr >= 3.0
    concentrated_fund = spread_fund >= 3.0

    # Find strongest regime in recent (highest hit among VR with ok status)
    best_recent_reg = None
    best_recent_hit = 0.0
    for reg, data in by_vr.items():
        r = data.get("recent", {})
        if r.get("status") == "ok" and r.get("hit", 0) > best_recent_hit:
            best_recent_hit = r["hit"]
            best_recent_reg = reg

    uniform = not concentrated_vr and not concentrated_fund
    verdict = (
        f"Decay spread across VR regimes: {spread_vr:.1f}pp "
        f"({'CONCENTRATED' if concentrated_vr else 'uniform'}). "
        f"Funding spread: {spread_fund:.1f}pp. "
        f"Strongest recent VR: {best_recent_reg} ({best_recent_hit:.1%})."
    )

    return {
        "hypothesis": "H1_regime_shift",
        "asset": Path(csv_path).stem,
        "overall": overall,
        "by_vr_regime": by_vr,
        "by_funding_regime": by_fund,
        "vr_delta_spread_pp": round(spread_vr, 2),
        "fund_delta_spread_pp": round(spread_fund, 2),
        "decay_uniform": uniform,
        "decay_concentrated": concentrated_vr or concentrated_fund,
        "strongest_recent_vr": best_recent_reg,
        "strongest_recent_vr_hit": best_recent_hit,
        "verdict": verdict,
    }


def hypothesis2_time_trend(csv_path: str) -> dict:
    """H2: quarterly rolling reversion_index 2017->today."""
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change().to_numpy()
    streak = _signed_streak(ret)
    idx = df.index

    quarters = pd.PeriodIndex(idx, freq="Q")
    unique_q = sorted(set(quarters))
    q_arr = quarters.to_numpy()
    rows = []
    for q in unique_q:
        if q.year < 2017:
            continue
        mask = q_arr == q
        ri = _reversion_index(ret, streak, mask)
        if ri.get("status") != "ok":
            continue
        rows.append({
            "quarter": str(q),
            "year": q.year,
            "reversion_index": ri["reversion_index"],
            "n": ri["n"],
        })

    if len(rows) < 8:
        return {"hypothesis": "H2_time_trend", "status": "insufficient_data"}

    xs = np.arange(len(rows), dtype=float)
    ys = np.array([r["reversion_index"] for r in rows])
    slope_full, intercept_full = np.polyfit(xs, ys, 1)

    # 2017-2024 only
    pre24 = [r for r in rows if r["year"] <= 2024]
    xs_pre = np.arange(len(pre24), dtype=float)
    ys_pre = np.array([r["reversion_index"] for r in pre24])
    slope_pre, _ = np.polyfit(xs_pre, ys_pre, 1) if len(pre24) >= 4 else (float("nan"), 0)

    # Simple significance: correlation of quarter index with rev_index
    corr_pre = float(np.corrcoef(xs_pre, ys_pre)[0, 1]) if len(pre24) >= 4 else float("nan")
    corr_full = float(np.corrcoef(xs, ys)[0, 1])

    # Monotonic decline: negative slope AND corr < -0.2
    monotonic_pre = slope_pre < -0.001 and corr_pre < -0.15 if corr_pre == corr_pre else False
    monotonic_full = slope_full < -0.001 and corr_full < -0.15

    recent_q = [r for r in rows if r["year"] in RECENT_YEARS]
    recent_mean = float(np.mean([r["reversion_index"] for r in recent_q])) if recent_q else None
    pre24_mean = float(np.mean([r["reversion_index"] for r in pre24])) if pre24 else None

    return {
        "hypothesis": "H2_time_trend",
        "asset": Path(csv_path).stem,
        "n_quarters": len(rows),
        "quarterly": rows[-12:],  # last 12 for display
        "slope_per_quarter_pre2024": round(float(slope_pre), 5) if slope_pre == slope_pre else None,
        "slope_per_quarter_full": round(float(slope_full), 5),
        "corr_pre2024": round(corr_pre, 3) if corr_pre == corr_pre else None,
        "corr_full": round(corr_full, 3),
        "mean_rev_index_pre2024": round(pre24_mean, 4) if pre24_mean else None,
        "mean_rev_index_2025_26": round(recent_mean, 4) if recent_mean else None,
        "monotonic_decline_pre2024": monotonic_pre,
        "monotonic_decline_full": monotonic_full,
        "verdict": (
            f"Pre-2024 slope={slope_pre:.5f}/q corr={corr_pre:+.3f} "
            f"({'MONOTONIC DECLINE' if monotonic_pre else 'no clear monotonic trend pre-2024'}). "
            f"Full slope={slope_full:.5f}/q. "
            f"Mean rev_index: pre-2024={pre24_mean:.4f}, 2025-26={recent_mean:.4f}."
            if pre24_mean and recent_mean else "inconclusive"
        ),
    }


def hypothesis3_microstructure(csv_path: str) -> dict:
    """H3: honest assessment of microstructure data availability."""
    df = load_ohlcv(csv_path)
    ret = df["close"].pct_change()
    pool = atoms_mod.compute_atom_pool(df, Config()).dropna()
    pool_years = pool.index.year

    # Available weak proxies only (NOT true spread/liquidity)
    vol_z = pool["volume_zscore"] if "volume_zscore" in pool.columns else None
    rp = pool["range_pct"] if "range_pct" in pool.columns else None

    recent = pool_years >= RECENT_YEARS[0]
    earlier = pool_years < RECENT_YEARS[0]

    proxies = {}
    if vol_z is not None:
        proxies["volume_zscore_median"] = {
            "recent": round(float(vol_z[recent].median()), 4),
            "earlier": round(float(vol_z[earlier].median()), 4),
        }
    if rp is not None:
        proxies["range_pct_median"] = {
            "recent": round(float(rp[recent].median()), 6),
            "earlier": round(float(rp[earlier].median()), 6),
        }

    # Quarterly rev_index for correlation with volume (weak proxy test)
    ret_arr = ret.reindex(pool.index).to_numpy()
    streak = _signed_streak(ret_arr)
    quarters = pd.PeriodIndex(pool.index, freq="Q")
    q_arr = quarters.to_numpy()
    q_rows = []
    for q in sorted(set(quarters)):
        if q.year < 2017:
            continue
        mask = q_arr == q
        ri = _reversion_index(ret_arr, streak, mask)
        if ri.get("status") != "ok":
            continue
        vz_med = float(vol_z[mask].median()) if vol_z is not None else float("nan")
        q_rows.append({"quarter": str(q), "rev_index": ri["reversion_index"], "vol_z_med": vz_med})

    corr_vol_rev = None
    if len(q_rows) >= 8 and vol_z is not None:
        v = [r["vol_z_med"] for r in q_rows if r["vol_z_med"] == r["vol_z_med"]]
        r = [r["rev_index"] for r in q_rows if r["vol_z_med"] == r["vol_z_med"]]
        if len(v) >= 8:
            corr_vol_rev = round(float(np.corrcoef(v, r)[0, 1]), 3)

    return {
        "hypothesis": "H3_microstructure",
        "asset": Path(csv_path).stem,
        "data_available": {
            "bid_ask_spread": False,
            "order_book_depth": False,
            "exchange_count": False,
            "funding_rate": Path("funding_btc.csv").exists(),
            "volume_ohlcv": True,
            "range_pct_proxy": rp is not None,
        },
        "weak_proxies_only": proxies,
        "corr_quarterly_vol_z_vs_rev_index": corr_vol_rev,
        "verdict": (
            "NO direct microstructure data (spread, liquidity, exchange count). "
            "volume_zscore/range_pct are weak OHLCV proxies only — cannot confirm "
            "microstructure hypothesis. "
            + (f"Quarterly corr(vol_z, rev_index)={corr_vol_rev} (exploratory, not causal)."
               if corr_vol_rev is not None else "")
        ),
    }


def classify_branch(h1: dict, h2: dict, h3: dict, stock: dict | None) -> dict:
    """Map diagnosis results to decision branch A / B / C."""
    reasons = []

    h1_concentrated = h1.get("decay_concentrated", False)
    h1_uniform = h1.get("decay_uniform", True)
    h2_monotonic = h2.get("monotonic_decline_pre2024", False)
    strongest_vr = h1.get("strongest_recent_vr")

    stock_today = stock.get("comparison", {}) if stock else {}
    stocks_weak = all(
        stock_today.get(k, {}).get("reversion_index", 1) < 0.02
        for k in ("SPY", "AAPL") if k in stock_today
    ) if stock_today else False
    btc_today_weak = stock_today.get("BTC_today", {}).get("reversion_index", 1) < 0.03
    btc_2018_strong = stock_today.get("BTC_2018_2019", {}).get("reversion_index", 0) > 0.03

    # Branch B: monotonic pre-2024 decline + market maturity (stocks weak, BTC was strong)
    if h2_monotonic and stocks_weak and btc_today_weak and btc_2018_strong:
        branch = "B"
        reasons.append("H2: monotonic decline pre-2024")
        reasons.append("H5: stocks near-efficient today, BTC 2018-19 was stronger")
    elif h1_concentrated and not h2_monotonic:
        branch = "A"
        reasons.append(f"H1: decay concentrated in regimes (spread VR={h1.get('vr_delta_spread_pp')}pp)")
        reasons.append(f"Strongest recent regime: {strongest_vr}")
        reasons.append("H2: no monotonic pre-2024 trend")
    elif h1_uniform and h2_monotonic:
        branch = "B"
        reasons.append("H1: uniform decay + H2: monotonic trend -> market maturation")
    else:
        branch = "C"
        reasons.append("Inconclusive: mixed signals")
        if h1_uniform:
            reasons.append("H1: decay appears uniform across regimes")
        if not h2_monotonic:
            reasons.append("H2: no clear monotonic pre-2024 slope")
        if not stocks_weak:
            reasons.append("H5: stocks still show some reversal (not fully efficient)")
        reasons.append("DEFAULT -> conservative branch A (turnover/regime exploit)")

    return {
        "branch": branch,
        "reasons": reasons,
        "strongest_vr_for_pnl": strongest_vr or "LOW_VR",
    }


def run_diagnosis(csv_paths: list[str], stock_result: dict | None = None) -> dict:
    results = []
    for csv in csv_paths:
        if not Path(csv).exists():
            continue
        h1 = hypothesis1_regime_shift(csv)
        h2 = hypothesis2_time_trend(csv)
        h3 = hypothesis3_microstructure(csv)
        branch = classify_branch(h1, h2, h3, stock_result)
        results.append({"csv": csv, "H1": h1, "H2": h2, "H3": h3, "branch": branch})

    # Overall branch: use BTC primary
    primary = next((r for r in results if "btc" in r["csv"]), results[0] if results else None)
    overall_branch = primary["branch"]["branch"] if primary else "C"
    return {"assets": results, "overall_branch": overall_branch,
            "primary_branch_detail": primary["branch"] if primary else {}}


def _print_report(r: dict) -> None:
    line = "=" * 78
    print("\n" + line)
    print("FADE DECAY DIAGNOSIS (H1 regime / H2 trend / H3 microstructure)")
    print(line)
    for asset in r.get("assets", []):
        print(f"\n  === {asset['csv']} ===")
        h1 = asset["H1"]
        print(f"  H1: {h1['verdict']}")
        o = h1["overall"]
        if o.get("recent", {}).get("hit"):
            print(f"      overall recent={o['recent']['hit']} earlier={o['earlier'].get('hit')} "
                  f"delta={o.get('delta_pp')}pp")
        for reg, data in h1.get("by_vr_regime", {}).items():
            rr, ee = data.get("recent", {}), data.get("earlier", {})
            if rr.get("hit") and ee.get("hit"):
                print(f"      VR {reg}: recent={rr['hit']} earlier={ee['hit']} "
                      f"delta={data.get('delta_pp')}pp n={rr['n']}")

        h2 = asset["H2"]
        print(f"  H2: {h2.get('verdict', h2.get('status'))}")

        h3 = asset["H3"]
        print(f"  H3: {h3['verdict']}")

        b = asset["branch"]
        print(f"  -> BRANCH {b['branch']}: {'; '.join(b['reasons'])}")

    print(line)
    print(f"  OVERALL BRANCH: {r.get('overall_branch')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE decay diagnosis")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    parser.add_argument("--json-out", default="fade/output/decay_diagnosis.json")
    parser.add_argument("--with-stock", action="store_true",
                        help="Load stock benchmark JSON if present")
    args = parser.parse_args()

    stock = None
    stock_path = Path("fade/output/stock_reversal_benchmark.json")
    if args.with_stock and stock_path.exists():
        stock = json.loads(stock_path.read_text(encoding="utf-8"))

    result = run_diagnosis(args.csv, stock)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print_report(result)
    return result


if __name__ == "__main__":
    main()
