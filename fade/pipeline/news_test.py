"""Does news sentiment add a real, out-of-sample edge? — strict holdout.

News is treated as just another atom source. We align GDELT daily tone/volume
to daily BTC bars and run the SAME quarantined 70/30 holdout used for price
rules (fade.pipeline.holdout), with a permutation test for significance.

Three models on the IDENTICAL holdout dates and outcome, so the comparison is
fair:
    price   — the core-5 price atoms (our known daily baseline; historically no
              daily edge, so it is the honest control).
    news    — news_tone, news_tone_chg, news_volume_z only.
    combined— price + news atoms together.

Honest interpretation:
    * If NEWS beats PRICE and clears p<=0.05 on the holdout, sentiment carries
      orthogonal information worth keeping.
    * If it does not, we throw it out — exactly as regime-weighting was thrown
      out. A pretty in-sample number is not enough.

No look-ahead: tone/volume carry their own publication date; tone_chg and the
volume z-score use trailing windows only; rules are mined on the development
split and frozen before touching the holdout.

Run:
    python -m fade.pipeline.news_test
    python -m fade.pipeline.news_test --price btc_daily.csv --news news_btc.csv
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import ATOM_COLUMNS, Config
from fade.core import atoms as atoms_mod
from fade.core import events as ev
from fade.core.data_loader import load_ohlcv
from fade.core.evaluator import predict
from fade.core.targets import score_predictions
from fade.pipeline.backtest import walk_forward
from fade.pipeline.holdout import _select_stable_rules
from fade.utils.logging import get_logger

log = get_logger("news_test")

NEWS_ATOMS = ("news_tone", "news_tone_chg", "news_volume_z")
VOL_Z_WINDOW = 30  # trailing days for the news-volume z-score


def _load_news(news_csv: str) -> pd.DataFrame:
    """Daily news frame -> trailing-only news atoms, indexed by UTC midnight."""
    raw = pd.read_csv(news_csv, parse_dates=["date"])
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.normalize()
    raw = raw.drop_duplicates("date").set_index("date").sort_index()

    tone = raw["news_tone"].astype(float)
    volume = raw.get("news_volume", pd.Series(index=raw.index, dtype=float)).astype(float)

    vol_mean = volume.rolling(VOL_Z_WINDOW).mean()
    vol_std = volume.rolling(VOL_Z_WINDOW).std().replace(0.0, np.nan)

    news = pd.DataFrame({
        "news_tone": tone,
        "news_tone_chg": tone.diff(),
        "news_volume_z": (volume - vol_mean) / vol_std,
    })
    return news


def _frozen_holdout(
    atoms: pd.DataFrame,
    fwd: pd.Series,
    config: Config,
    holdout_frac: float,
    n_shuffles: int,
    seed: int,
) -> dict:
    """Mine+freeze rules on dev, score frozen rules on quarantined holdout."""
    atoms = atoms.dropna()
    fwd = fwd.reindex(atoms.index)
    n = len(atoms)
    if n < config.min_support * 4:
        return {"status": "insufficient_data", "n": int(n)}

    split = int(n * (1.0 - holdout_frac))
    dev_atoms, hold_atoms = atoms.iloc[:split], atoms.iloc[split:]
    dev_fwd, hold_fwd = fwd.iloc[:split], fwd.iloc[split:]

    dev_bt = walk_forward(dev_atoms, dev_fwd, config)
    frozen = _select_stable_rules(dev_bt.stability, config)
    if frozen.empty:
        return {"status": "no_rules", "n_dev": int(split), "n_holdout": int(n - split)}

    thresholds = ev.compute_thresholds(dev_atoms, config)
    hold_disc = ev.discretize(hold_atoms, thresholds)
    hold_events = ev.build_events(hold_disc, config, allowed=set(frozen.index))
    preds = predict(hold_events, frozen)
    if preds.empty:
        return {"status": "no_coverage", "n_stable_rules": int(len(frozen))}

    valid = hold_fwd.reindex(preds.index).notna()
    pred_v = preds["pred"][valid].to_numpy()
    fwd_v = hold_fwd.reindex(preds.index)[valid].to_numpy()
    pred_sc, act_sc = score_predictions(pred_v, fwd_v, config.move_threshold)
    if len(pred_sc) == 0:
        return {"status": "no_coverage", "n_stable_rules": int(len(frozen))}

    hit = float(np.mean(pred_sc == act_sc))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(pred_sc == rng.permutation(act_sc))
                     for _ in range(n_shuffles)])
    p_value = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)

    return {
        "status": "ok",
        "n_stable_rules": int(len(frozen)),
        "coverage": int(len(pred_sc)),
        "hit_rate": round(hit, 4),
        "lift_vs_random": round(hit - 0.5, 4),
        "null_mean": round(float(null.mean()), 4),
        "p_value": round(p_value, 4),
    }


def verify_alignment(price_csv: str = "btc_daily.csv",
                     news_csv: str = "news_btc.csv") -> dict:
    """Sanity-check that news dates line up with price-movement dates.

    Confirms (a) both series are daily UTC-midnight, (b) how many dates overlap
    and where the gaps are, and (c) that same-day news tone actually relates to
    same-day return more than to shifted days. If a shifted lag correlated far
    more strongly than lag 0, that would signal a date misalignment / leak.
    """
    price = load_ohlcv(price_csv)
    close = price["close"]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    ret_same = close.pct_change()                    # return realised ON day D
    ret_next = close.shift(-1) / close - 1.0         # return realised on day D+1

    news = pd.read_csv(news_csv, parse_dates=["date"])
    news["date"] = pd.to_datetime(news["date"], utc=True).dt.normalize()
    news = news.drop_duplicates("date").set_index("date").sort_index()
    tone = news["news_tone"].astype(float)

    # Gap detection in the news archive (missing calendar days).
    full_range = pd.date_range(news.index.min(), news.index.max(), freq="D", tz="UTC")
    missing = full_range.difference(news.index)
    gaps = []
    if len(missing):
        # collapse consecutive missing days into ranges
        m = missing.sort_values()
        start = prev = m[0]
        for d in m[1:]:
            if (d - prev).days == 1:
                prev = d
                continue
            gaps.append((str(start.date()), str(prev.date())))
            start = prev = d
        gaps.append((str(start.date()), str(prev.date())))

    common = tone.index.intersection(close.index)

    def _corr(a: pd.Series, b: pd.Series) -> float:
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        return round(float(j.iloc[:, 0].corr(j.iloc[:, 1])), 4) if len(j) > 10 else float("nan")

    # tone(D) vs return over a range of lags. lag 0 = same day.
    lag_corr = {}
    for lag in range(-2, 3):
        lag_corr[lag] = _corr(tone, ret_same.shift(-lag))

    # Directional: does tone sign match same-day / next-day return sign?
    def _dir_hit(r: pd.Series) -> float:
        j = pd.concat([np.sign(tone), np.sign(r)], axis=1, join="inner").dropna()
        j = j[(j.iloc[:, 0] != 0) & (j.iloc[:, 1] != 0)]
        return round(float((j.iloc[:, 0] == j.iloc[:, 1]).mean()), 4) if len(j) else float("nan")

    return {
        "price_days": int(len(close)),
        "news_days": int(len(news)),
        "news_span": [str(news.index.min().date()), str(news.index.max().date())],
        "overlap_days": int(len(common)),
        "news_gaps": gaps,
        "tone_vs_return_by_lag": lag_corr,
        "tone_sign_vs_same_day": _dir_hit(ret_same),
        "tone_sign_vs_next_day": _dir_hit(ret_next),
    }


def _print_alignment(a: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print("FADE NEWS<->PRICE DATE ALIGNMENT CHECK")
    print(line)
    print(f"  price days   : {a['price_days']:,}")
    print(f"  news days    : {a['news_days']:,}   span {a['news_span'][0]} -> {a['news_span'][1]}")
    print(f"  overlap days : {a['overlap_days']:,}")
    if a["news_gaps"]:
        print(f"  news gaps    : {len(a['news_gaps'])} missing range(s)")
        for lo, hi in a["news_gaps"]:
            print(f"      {lo} -> {hi}")
    else:
        print("  news gaps    : none")
    print()
    print("  tone(D) correlation with return, by lag (lag 0 = same day):")
    for lag, c in a["tone_vs_return_by_lag"].items():
        tag = "  <- same day" if lag == 0 else (" (future ret)" if lag > 0 else " (past ret)")
        print(f"      lag {lag:+d} : {c:+.4f}{tag}")
    print()
    print(f"  tone sign vs SAME-day return : {a['tone_sign_vs_same_day']}")
    print(f"  tone sign vs NEXT-day return : {a['tone_sign_vs_next_day']}  (0.50 = no predictive link)")
    print(line)
    # Honest read of the alignment.
    lags = a["tone_vs_return_by_lag"]
    same = lags.get(0, float("nan"))
    strongest = max(lags, key=lambda k: abs(lags[k]) if lags[k] == lags[k] else -1)
    if strongest == 0:
        print("  OK: same-day tone tracks same-day return most strongly — dates aligned.")
    else:
        print(f"  NOTE: lag {strongest:+d} is strongest, not lag 0 — inspect for shift/leak.")
    print(line + "\n")


def run_news_test(
    price_csv: str = "btc_daily.csv",
    news_csv: str = "news_btc.csv",
    holdout_frac: float = 0.30,
    n_shuffles: int = 300,
    seed: int = 0,
    base_config: Config | None = None,
) -> dict:
    base_config = base_config or Config()

    price = load_ohlcv(price_csv)
    price_pool = atoms_mod.compute_atom_pool(price, base_config)
    price_pool.index = pd.to_datetime(price_pool.index, utc=True).normalize()
    fwd = atoms_mod.forward_return(price, 1)
    fwd.index = pd.to_datetime(fwd.index, utc=True).normalize()

    news = _load_news(news_csv)

    # Common daily dates across price atoms, news, and the target.
    common = price_pool.index.intersection(news.index).intersection(fwd.index)
    common = common.sort_values()
    price_pool = price_pool.loc[common]
    news = news.loc[common]
    fwd = fwd.loc[common]

    result: dict = {
        "price_csv": Path(price_csv).stem,
        "news_csv": Path(news_csv).stem,
        "common_days": int(len(common)),
        "date_span": [str(common.min())[:10], str(common.max())[:10]] if len(common) else None,
        "models": {},
    }
    if len(common) < base_config.min_support * 4:
        result["status"] = "insufficient_overlap"
        return result

    price_atoms = price_pool[list(ATOM_COLUMNS)]
    news_atoms = news[list(NEWS_ATOMS)]
    combined = pd.concat([price_atoms, news_atoms], axis=1)

    specs = {
        "price": (price_atoms, ATOM_COLUMNS),
        "news": (news_atoms, NEWS_ATOMS),
        "combined": (combined, tuple(ATOM_COLUMNS) + NEWS_ATOMS),
    }
    for name, (frame, cols) in specs.items():
        cfg = dataclasses.replace(base_config, atom_columns=tuple(cols))
        result["models"][name] = _frozen_holdout(
            frame, fwd, cfg, holdout_frac, n_shuffles, seed)
        log.info("model %-9s -> %s", name, result["models"][name])

    result["status"] = "ok"
    result["verdict"] = _verdict(result["models"])
    return result


def _verdict(models: dict) -> str:
    news = models.get("news", {})
    price = models.get("price", {})
    if news.get("status") != "ok":
        return f"INCONCLUSIVE - news model: {news.get('status')}"
    news_lift, news_p = news["lift_vs_random"], news["p_value"]
    price_lift = price.get("lift_vs_random", 0.0) if price.get("status") == "ok" else 0.0
    if news_lift <= 0:
        return "FAIL - news has no positive edge out-of-sample."
    if news_p > 0.05:
        return "WEAK - news edge positive but within shuffle noise (not significant)."
    if news_lift > price_lift:
        return "PASS - news carries significant edge beyond the price baseline."
    return "PASS(marginal) - news significant but not above the price baseline."


def _print(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print(f"FADE NEWS-SENTIMENT HOLDOUT TEST  ({r.get('price_csv')} + {r.get('news_csv')})")
    print(line)
    if r.get("date_span"):
        print(f"  common days: {r['common_days']:,}   span {r['date_span'][0]} -> {r['date_span'][1]}")
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}")
        print(line + "\n")
        return
    print(f"  {'model':<10}{'rules':>6}{'cover':>7}{'hit':>8}{'lift':>9}{'null':>8}{'p':>8}")
    for name, m in r["models"].items():
        if m.get("status") != "ok":
            print(f"  {name:<10}{'':>6}{'':>7}  {m.get('status')}")
            continue
        print(f"  {name:<10}{m['n_stable_rules']:>6}{m['coverage']:>7}{m['hit_rate']:>8}"
              f"{m['lift_vs_random']:>+9}{m['null_mean']:>8}{m['p_value']:>8}")
    print(line)
    print(f"  VERDICT: {r.get('verdict')}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE news sentiment holdout test")
    parser.add_argument("--price", default="btc_daily.csv")
    parser.add_argument("--news", default="news_btc.csv")
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--verify", action="store_true",
                        help="only run the date-alignment check")
    args = parser.parse_args()
    for c in (args.price, args.news):
        if not Path(c).exists():
            log.error("File not found: %s", c)
            return
    if args.verify:
        _print_alignment(verify_alignment(args.price, args.news))
        return
    _print_alignment(verify_alignment(args.price, args.news))
    _print(run_news_test(args.price, args.news, holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
