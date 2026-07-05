"""Conditional news-behaviour patterns — learn on the past, test on unseen future.

The user's insight: even if news is reactive on average, a specific BEHAVIOURAL
pattern may still be predictive — e.g. after a day of extreme negative coverage
("panic"), does the next day tend to bounce (mean-reversion) or keep falling
(momentum)? That is a pattern the tool could learn and reuse.

We test a small, PRE-REGISTERED set of conditional states (not an open-ended
search), each with a clear behavioural hypothesis, using the honest protocol:
    1. Split daily history chronologically: dev 70% + holdout 30% (quarantined).
    2. On dev only, measure each state's next-day up-rate; freeze a direction
       (the side it leaned, if support is enough).
    3. Apply the frozen directions to the holdout; score hit-rate + permutation
       p-value on the unseen tail.

Honesty guards:
    * States are fixed in advance (see STATES below) — no fishing.
    * Thresholds (tone/volume percentiles) are computed on dev only.
    * This is the 4th angle on the same daily news; treat any single p<=0.05 with
      multiple-comparison skepticism (Bonferroni note printed).

No look-ahead: state at day D uses only day-D (and trailing) news; outcome is the
return realised from D to D+1.

Run:
    python -m fade.pipeline.news_patterns --price btc_daily.csv --news news_btc.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.utils.logging import get_logger

log = get_logger("news_patterns")

VOL_Z_WINDOW = 30
MIN_STATE_SUPPORT = 20   # min dev days in a state to freeze a direction


def _build_frame(price_csv: str, news_csv: str) -> pd.DataFrame:
    price = load_ohlcv(price_csv)
    close = price["close"]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    ret_next = close.shift(-1) / close - 1.0

    news = pd.read_csv(news_csv, parse_dates=["date"])
    news["date"] = pd.to_datetime(news["date"], utc=True).dt.normalize()
    news = news.drop_duplicates("date").set_index("date").sort_index()
    tone = news["news_tone"].astype(float)
    volume = news["news_volume"].astype(float)

    vmean = volume.rolling(VOL_Z_WINDOW).mean()
    vstd = volume.rolling(VOL_Z_WINDOW).std().replace(0.0, np.nan)

    df = pd.DataFrame({
        "tone": tone,
        "tone_chg": tone.diff(),
        "vol_z": (volume - vmean) / vstd,
        "ret_next": ret_next,
    }).dropna()
    return df


def _state_masks(df: pd.DataFrame, thr: dict) -> dict[str, pd.Series]:
    """Pre-registered conditional states (thresholds fitted on dev)."""
    return {
        # Panic: very negative tone today -> mean-reversion bounce? (contrarian)
        "panic_extreme_neg_tone": df["tone"] <= thr["tone_lo"],
        # Euphoria: very positive tone -> pullback? (contrarian)
        "euphoria_extreme_pos_tone": df["tone"] >= thr["tone_hi"],
        # Tone shock down: sharp day-over-day drop in tone.
        "tone_shock_down": df["tone_chg"] <= thr["chg_lo"],
        # Tone shock up.
        "tone_shock_up": df["tone_chg"] >= thr["chg_hi"],
        # Attention spike: high coverage volume (any tone).
        "attention_spike": df["vol_z"] >= 1.5,
    }


def run_patterns(price_csv: str, news_csv: str, holdout_frac: float = 0.30,
                 n_shuffles: int = 3000, seed: int = 0) -> dict:
    df = _build_frame(price_csv, news_csv)
    n = len(df)
    if n < 300:
        return {"status": "insufficient_data", "n": int(n)}

    split = int(n * (1.0 - holdout_frac))
    dev, hold = df.iloc[:split], df.iloc[split:]

    # Thresholds from DEV only (no look-ahead into holdout).
    thr = {
        "tone_lo": float(dev["tone"].quantile(0.10)),
        "tone_hi": float(dev["tone"].quantile(0.90)),
        "chg_lo": float(dev["tone_chg"].quantile(0.10)),
        "chg_hi": float(dev["tone_chg"].quantile(0.90)),
    }

    dev_masks = _state_masks(dev, thr)
    hold_masks = _state_masks(hold, thr)
    rng = np.random.default_rng(seed)

    results = []
    hold_up = (hold["ret_next"] > 0).astype(int).to_numpy()
    hold_base_up = float(hold_up.mean())

    for name, dmask in dev_masks.items():
        dev_days = dev[dmask]
        if len(dev_days) < MIN_STATE_SUPPORT:
            results.append({"state": name, "status": "low_dev_support",
                            "dev_n": int(len(dev_days))})
            continue

        # Frozen direction = the side this state leaned on DEV.
        dev_up_rate = float((dev_days["ret_next"] > 0).mean())
        frozen_dir = 1 if dev_up_rate >= 0.5 else 0

        hmask = hold_masks[name].to_numpy()
        hold_n = int(hmask.sum())
        if hold_n < 10:
            results.append({"state": name, "status": "low_holdout_support",
                            "dev_n": int(len(dev_days)), "hold_n": hold_n,
                            "dev_up_rate": round(dev_up_rate, 4)})
            continue

        # Hit-rate of the frozen direction on the holdout state days.
        actual = hold_up[hmask]
        hit = float(np.mean(actual == frozen_dir))

        # Permutation: random state days of same size from the holdout.
        null = np.empty(n_shuffles)
        for i in range(n_shuffles):
            idx = rng.choice(len(hold_up), size=hold_n, replace=False)
            null[i] = np.mean(hold_up[idx] == frozen_dir)
        p = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)

        results.append({
            "state": name,
            "status": "ok",
            "dev_n": int(len(dev_days)),
            "dev_up_rate": round(dev_up_rate, 4),
            "frozen_dir": "up" if frozen_dir == 1 else "down",
            "hold_n": hold_n,
            "hold_hit_rate": round(hit, 4),
            "hold_base_up": round(hold_base_up, 4),
            "edge_vs_base": round(hit - (hold_base_up if frozen_dir == 1
                                         else 1 - hold_base_up), 4),
            "p_value": round(p, 4),
        })

    tested = [r for r in results if r["status"] == "ok"]
    n_tested = len(tested)
    any_sig = [r for r in tested if r["p_value"] <= 0.05]
    bonferroni = 0.05 / n_tested if n_tested else float("nan")
    survive_bonf = [r for r in tested if r["p_value"] <= bonferroni]

    return {
        "status": "ok",
        "n_days": int(n),
        "n_dev": int(split),
        "n_holdout": int(n - split),
        "thresholds": {k: round(v, 4) for k, v in thr.items()},
        "states": results,
        "n_states_tested": n_tested,
        "n_p_below_0.05": len(any_sig),
        "bonferroni_alpha": round(bonferroni, 4) if bonferroni == bonferroni else None,
        "n_survive_bonferroni": len(survive_bonf),
        "verdict": _verdict(tested, survive_bonf),
    }


def _verdict(tested: list, survive_bonf: list) -> str:
    if not tested:
        return "INCONCLUSIVE - no state had enough support."
    if survive_bonf:
        names = ", ".join(r["state"] for r in survive_bonf)
        return (f"SIGNAL - {names} survives multiple-comparison correction "
                f"on the holdout. Worth a closer look.")
    raw_sig = [r for r in tested if r["p_value"] <= 0.05]
    if raw_sig:
        return ("NO ROBUST SIGNAL - some states hit raw p<=0.05 but none survive "
                "Bonferroni; consistent with chance across several angles.")
    return "NO SIGNAL - no conditional news pattern predicts the unseen holdout."


def _print(r: dict) -> None:
    line = "=" * 76
    print("\n" + line)
    print("FADE CONDITIONAL NEWS-PATTERN TEST  (learn on past, test on unseen future)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}  {r}")
        print(line + "\n")
        return
    print(f"  days: {r['n_days']:,}  (dev {r['n_dev']} / holdout {r['n_holdout']})")
    print()
    print(f"  {'state':<26}{'dev_n':>6}{'dir':>5}{'hold_n':>7}{'hit':>7}"
          f"{'edge':>8}{'p':>8}")
    for s in r["states"]:
        if s["status"] != "ok":
            print(f"  {s['state']:<26}{s.get('dev_n', 0):>6}   -   {s['status']}")
            continue
        print(f"  {s['state']:<26}{s['dev_n']:>6}{s['frozen_dir']:>5}{s['hold_n']:>7}"
              f"{s['hold_hit_rate']:>7}{s['edge_vs_base']:>+8}{s['p_value']:>8}")
    print(line)
    print(f"  states tested: {r['n_states_tested']}   raw p<=0.05: {r['n_p_below_0.05']}"
          f"   Bonferroni alpha: {r['bonferroni_alpha']}   "
          f"survive: {r['n_survive_bonferroni']}")
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE conditional news-pattern test")
    parser.add_argument("--price", default="btc_daily.csv")
    parser.add_argument("--news", default="news_btc.csv")
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()
    for c in (args.price, args.news):
        if not Path(c).exists():
            log.error("File not found: %s", c)
            return
    _print(run_patterns(args.price, args.news, holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
