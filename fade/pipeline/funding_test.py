"""Does perpetual funding rate predict the next 8h move? — strict holdout.

Funding rate is orthogonal to price and has a mechanical reason to LEAD it:
extreme positive funding = over-leveraged longs (squeeze risk DOWN); extreme
negative = crowded shorts (squeeze risk UP). The behavioural hypothesis is
CONTRARIAN mean-reversion, tested honestly.

Alignment (no look-ahead): funding is stamped every 8h at T (00/08/16 UTC). The
value known at T is used to predict the price move from T to the NEXT funding
stamp T+8h, built from hourly closes (btc_1h.csv). Funding at T is public at T,
the move happens after — no leakage.

Pre-registered states (thresholds fitted on the DEV split only):
    extreme_pos_funding  (top decile)   -> hypothesis: next move DOWN
    extreme_neg_funding  (bottom decile) -> hypothesis: next move UP
    funding_rising / funding_falling    -> momentum of funding
Plus a continuous check: correlation of funding with the next 8h return, and a
strict 70/30 chronological holdout with a permutation p-value on the frozen
contrarian rule.

Run:
    python -m fade.pipeline.funding_test
    python -m fade.pipeline.funding_test --funding funding_btc.csv --price btc_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.utils.logging import get_logger

log = get_logger("funding_test")

MIN_SUPPORT = 20


def _build_frame(funding_csv: str, price_csv: str) -> pd.DataFrame:
    """Align funding stamps to the price move over the NEXT 8h interval."""
    fund = pd.read_csv(funding_csv)
    fund["timestamp"] = pd.to_datetime(fund["timestamp"], utc=True, format="mixed")
    # Funding stamps occasionally carry a 1ms offset; snap to the exact hour so
    # they align cleanly with hourly closes.
    fund["timestamp"] = fund["timestamp"].dt.floor("h")
    fund = fund.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    price = load_ohlcv(price_csv)
    close = price["close"]
    close.index = pd.to_datetime(close.index, utc=True)

    # Price at each funding stamp (nearest hourly close at/just before the stamp).
    close_at = close.reindex(fund.index, method="ffill")
    # Next funding stamp's price -> realised 8h forward return.
    fwd_price = close_at.shift(-1)
    ret_next = fwd_price / close_at - 1.0

    df = pd.DataFrame({
        "funding": fund["funding_rate"],
        "funding_chg": fund["funding_rate"].diff(),
        "ret_next": ret_next,
    }).dropna()
    return df


def _holdout_contrarian(df: pd.DataFrame, holdout_frac: float,
                        n_shuffles: int, seed: int) -> dict:
    """Freeze a contrarian rule on dev deciles; score on quarantined holdout."""
    n = len(df)
    split = int(n * (1.0 - holdout_frac))
    dev, hold = df.iloc[:split], df.iloc[split:]

    lo = float(dev["funding"].quantile(0.10))
    hi = float(dev["funding"].quantile(0.90))

    # Contrarian prediction: extreme positive funding -> DOWN (0);
    # extreme negative funding -> UP (1). Neutral zone excluded.
    def _preds(frame: pd.DataFrame) -> pd.Series:
        p = pd.Series(np.nan, index=frame.index)
        p[frame["funding"] >= hi] = 0
        p[frame["funding"] <= lo] = 1
        return p

    hold_pred = _preds(hold)
    mask = hold_pred.notna() & hold["ret_next"].notna()
    pred = hold_pred[mask].to_numpy()
    actual = (hold["ret_next"][mask] > 0).astype(int).to_numpy()
    cov = int(len(pred))
    if cov < MIN_SUPPORT:
        return {"status": "low_coverage", "coverage": cov}

    hit = float(np.mean(pred == actual))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(pred == rng.permutation(actual))
                     for _ in range(n_shuffles)])
    p_value = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)
    return {
        "status": "ok",
        "dev_lo_decile": round(lo, 6),
        "dev_hi_decile": round(hi, 6),
        "coverage": cov,
        "hit_rate": round(hit, 4),
        "lift_vs_random": round(hit - 0.5, 4),
        "null_mean": round(float(null.mean()), 4),
        "p_value": round(p_value, 4),
    }


def run_funding_test(funding_csv: str = "funding_btc.csv",
                     price_csv: str = "btc_1h.csv",
                     holdout_frac: float = 0.30,
                     n_shuffles: int = 3000, seed: int = 0) -> dict:
    df = _build_frame(funding_csv, price_csv)
    if len(df) < 300:
        return {"status": "insufficient_data", "n": int(len(df))}

    # Continuous cross-checks (whole sample, descriptive).
    corr_level = float(df["funding"].corr(df["ret_next"]))
    corr_chg = float(df["funding_chg"].corr(df["ret_next"]))
    # Sign persistence: does positive funding go with negative next return?
    pos_next_up = float((df.loc[df["funding"] > 0, "ret_next"] > 0).mean())
    neg_next_up = float((df.loc[df["funding"] < 0, "ret_next"] > 0).mean())

    holdout = _holdout_contrarian(df, holdout_frac, n_shuffles, seed)

    result = {
        "status": "ok",
        "n": int(len(df)),
        "span": [str(df.index.min())[:10], str(df.index.max())[:10]],
        "funding_vs_next_ret_corr": round(corr_level, 4),
        "funding_chg_vs_next_ret_corr": round(corr_chg, 4),
        "pos_funding_next_up_rate": round(pos_next_up, 4),
        "neg_funding_next_up_rate": round(neg_next_up, 4),
        "contrarian_holdout": holdout,
    }
    result["verdict"] = _verdict(result)
    return result


def _verdict(r: dict) -> str:
    h = r.get("contrarian_holdout", {})
    if h.get("status") != "ok":
        return f"INCONCLUSIVE - {h.get('status')}"
    lift, p = h["lift_vs_random"], h["p_value"]
    if lift <= 0:
        return "FAIL - contrarian funding rule has no edge on unseen holdout."
    if p <= 0.05:
        return "PASS - funding contrarian edge survives strict holdout (p<=0.05)."
    return "WEAK - positive but within shuffle noise (not significant)."


def _print(r: dict) -> None:
    line = "=" * 70
    print("\n" + line)
    print("FADE FUNDING-RATE TEST  (does 8h funding predict the next 8h move?)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}  {r}")
        print(line + "\n")
        return
    print(f"  points: {r['n']:,}   span {r['span'][0]} -> {r['span'][1]}")
    print()
    print("  -- descriptive (whole sample) --")
    print(f"  corr(funding, next 8h ret)      : {r['funding_vs_next_ret_corr']:+.4f}")
    print(f"  corr(funding_chg, next 8h ret)  : {r['funding_chg_vs_next_ret_corr']:+.4f}")
    print(f"  next-up rate | funding > 0      : {r['pos_funding_next_up_rate']:.4f}")
    print(f"  next-up rate | funding < 0      : {r['neg_funding_next_up_rate']:.4f}")
    print()
    print("  -- strict holdout: frozen contrarian rule (dev deciles) --")
    h = r["contrarian_holdout"]
    if h.get("status") == "ok":
        print(f"  coverage        : {h['coverage']}")
        print(f"  hit-rate        : {h['hit_rate']}")
        print(f"  lift vs random  : {h['lift_vs_random']:+.4f}")
        print(f"  shuffle null    : {h['null_mean']:.4f}")
        print(f"  p-value         : {h['p_value']:.4f}")
    else:
        print(f"  {h.get('status')}")
    print(line)
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE funding-rate holdout test")
    parser.add_argument("--funding", default="funding_btc.csv")
    parser.add_argument("--price", default="btc_1h.csv")
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    args = parser.parse_args()
    for c in (args.funding, args.price):
        if not Path(c).exists():
            log.error("File not found: %s", c)
            return
    _print(run_funding_test(args.funding, args.price, holdout_frac=args.holdout_frac))


if __name__ == "__main__":
    main()
