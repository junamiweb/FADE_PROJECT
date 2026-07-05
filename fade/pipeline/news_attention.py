"""Attention hypothesis — does a coverage-VOLUME spike precede a big MOVE?

The tone test failed: sentiment is a lagging indicator of direction. But that
tested the wrong thing. The honest hypothesis is different:

    A surge in news ATTENTION (coverage volume), regardless of tone, may precede
    elevated VOLATILITY / larger absolute moves — not a directional edge.

This is the "panic/hype" idea: when everyone suddenly writes about bitcoin, the
next bar tends to be bigger (either way). We test magnitude, not direction.

No look-ahead: the volume z-score at day D uses a trailing window ending at D;
we relate it to the return realised AFTER D.

Metric: split days into "spike" (volume z >= threshold) vs "normal", compare the
mean absolute next-day return. A permutation test shuffles the spike labels to
see if the magnitude gap is real or noise.

Run:
    python -m fade.pipeline.news_attention --price btc_daily.csv --news news_btc.csv
    python -m fade.pipeline.news_attention --z 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.utils.logging import get_logger

log = get_logger("news_attention")

VOL_Z_WINDOW = 30


def _controlled_gap(ctrl: pd.DataFrame, spike: pd.Series) -> float | None:
    """Weighted spike-vs-normal |ret| gap within price-volatility quintiles."""
    gaps, weights = [], []
    for _, grp in ctrl.groupby("vbucket"):
        s = grp[spike.loc[grp.index]]
        nrm = grp[~spike.loc[grp.index]]
        if len(s) >= 5 and len(nrm) >= 5:
            gaps.append(float(s["abs_ret"].mean() - nrm["abs_ret"].mean()))
            weights.append(len(s))
    return float(np.average(gaps, weights=weights)) if gaps else None


def _permute_controlled(
    ctrl: pd.DataFrame,
    spike: pd.Series,
    n_shuffles: int,
    seed: int,
) -> tuple[float | None, float]:
    """Stratified permutation: shuffle spike labels WITHIN each vol bucket."""
    observed = _controlled_gap(ctrl, spike)
    if observed is None:
        return None, float("nan")
    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = spike.copy()
        for _, grp in ctrl.groupby("vbucket"):
            idx = grp.index
            perm.loc[idx] = rng.permutation(spike.loc[idx].to_numpy())
        g = _controlled_gap(ctrl, perm)
        null[i] = g if g is not None else 0.0
    p = (1 + int(np.sum(null >= observed))) / (1 + n_shuffles)
    return observed, p


def run_attention(price_csv: str, news_csv: str, z_threshold: float = 1.5,
                  n_shuffles: int = 2000, seed: int = 0) -> dict:
    price = load_ohlcv(price_csv)
    close = price["close"]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    ret_next = close.shift(-1) / close - 1.0          # move realised AFTER day D

    news = pd.read_csv(news_csv, parse_dates=["date"])
    news["date"] = pd.to_datetime(news["date"], utc=True).dt.normalize()
    news = news.drop_duplicates("date").set_index("date").sort_index()
    volume = news["news_volume"].astype(float)

    vmean = volume.rolling(VOL_Z_WINDOW).mean()
    vstd = volume.rolling(VOL_Z_WINDOW).std().replace(0.0, np.nan)
    vol_z = (volume - vmean) / vstd

    df = pd.concat([vol_z.rename("vol_z"), ret_next.rename("ret_next")],
                   axis=1, join="inner").dropna()
    if len(df) < 100:
        return {"status": "insufficient_data", "n": int(len(df))}

    df["abs_ret"] = df["ret_next"].abs()
    spike = df["vol_z"] >= z_threshold
    n_spike = int(spike.sum())
    if n_spike < 20:
        return {"status": "too_few_spikes", "n_spike": n_spike,
                "hint": "lower --z"}

    spike_absmove = float(df.loc[spike, "abs_ret"].mean())
    normal_absmove = float(df.loc[~spike, "abs_ret"].mean())
    observed_gap = spike_absmove - normal_absmove

    # Permutation: shuffle the spike labels, recompute the gap.
    rng = np.random.default_rng(seed)
    absr = df["abs_ret"].to_numpy()
    labels = spike.to_numpy()
    k = labels.sum()
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(labels)
        null[i] = absr[perm].mean() - absr[~perm].mean()
    p_value = (1 + int(np.sum(null >= observed_gap))) / (1 + n_shuffles)

    # Correlation as a continuous cross-check.
    corr = float(df["vol_z"].corr(df["abs_ret"]))

    # Directional check too — confirm it is NOT a direction signal.
    dir_hit = float(np.mean((df.loc[spike, "ret_next"] > 0)))

    # --- CONFOUND CONTROL: volatility clustering ------------------------
    # News volume spikes near volatile periods; next-day moves may be big just
    # from volatility autocorrelation, not news. Control by matching on the
    # price's OWN recent realised volatility (trailing 5-day mean |return|,
    # ending at D — no look-ahead) and re-testing the gap WITHIN vol buckets.
    daily_ret = close.pct_change()
    recent_vol = daily_ret.abs().rolling(5).mean().shift(1)  # ends at D-1 -> known at D
    ctrl = df.join(recent_vol.rename("recent_vol")).dropna()
    controlled = None
    controlled_p = float("nan")
    if len(ctrl) > 200:
        ctrl["vbucket"] = pd.qcut(ctrl["recent_vol"], 5, labels=False, duplicates="drop")
        sp = ctrl["vol_z"] >= z_threshold
        controlled = _controlled_gap(ctrl, sp)
        if controlled is not None:
            controlled, controlled_p = _permute_controlled(ctrl, sp, n_shuffles, seed)

    # Also: correlation of news vol_z with next |ret| after removing the linear
    # effect of recent price volatility (partial-correlation style residuals).
    partial_corr = None
    if len(ctrl) > 200:
        x = ctrl["vol_z"].to_numpy()
        y = ctrl["abs_ret"].to_numpy()
        z = ctrl["recent_vol"].to_numpy()
        def _resid(a, ctrl_var):
            A = np.vstack([ctrl_var, np.ones_like(ctrl_var)]).T
            coef, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ coef
        rx, ry = _resid(x, z), _resid(y, z)
        if rx.std() > 0 and ry.std() > 0:
            partial_corr = float(np.corrcoef(rx, ry)[0, 1])

    # Verdict: raw gap must be significant; controlled residual needs its OWN
    # stratified permutation p-value before we claim news adds info beyond price.
    raw_ok = observed_gap > 0 and p_value <= 0.05
    ctrl_ok = controlled is not None and controlled > 0
    ctrl_sig = controlled_p <= 0.05 if controlled_p == controlled_p else False
    ctrl_frac = (controlled / observed_gap) if (controlled is not None and observed_gap) else None

    if raw_ok and ctrl_ok and ctrl_sig:
        verdict = ("PASS - controlled gap is significant on its own (p<=0.05); "
                   "news volume adds volatility info beyond price clustering.")
    elif raw_ok and ctrl_ok:
        verdict = ("CLOSED - raw effect is real but controlled residual is NOT "
                   "significant (p>0.05). Mostly price volatility, not news.")
    elif raw_ok and not ctrl_ok:
        verdict = ("CONFOUNDED - raw effect vanishes once price volatility is "
                   "controlled. It was volatility clustering, not news.")
    elif observed_gap > 0:
        verdict = "WEAK - positive but within shuffle noise."
    else:
        verdict = "FAIL - coverage spikes do not precede larger moves."

    return {
        "status": "ok",
        "n_days": int(len(df)),
        "z_threshold": z_threshold,
        "n_spike_days": n_spike,
        "spike_abs_move": round(spike_absmove, 5),
        "normal_abs_move": round(normal_absmove, 5),
        "abs_move_gap": round(observed_gap, 5),
        "gap_ratio": round(spike_absmove / normal_absmove, 3) if normal_absmove else None,
        "vol_z_vs_abs_ret_corr": round(corr, 4),
        "spike_up_fraction": round(dir_hit, 4),
        "null_gap_mean": round(float(null.mean()), 5),
        "p_value": round(p_value, 4),
        "controlled_gap": round(controlled, 5) if controlled is not None else None,
        "controlled_gap_fraction": round(ctrl_frac, 3) if ctrl_frac is not None else None,
        "controlled_p_value": round(controlled_p, 4) if controlled_p == controlled_p else None,
        "partial_corr_vs_recent_vol": round(partial_corr, 4) if partial_corr is not None else None,
        "verdict": verdict,
    }


def _print(r: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print("FADE NEWS-ATTENTION TEST  (coverage spike -> bigger move?)")
    print(line)
    if r.get("status") != "ok":
        print(f"  status: {r.get('status')}  {r}")
        print(line + "\n")
        return
    print(f"  days analysed        : {r['n_days']:,}")
    print(f"  spike threshold      : vol_z >= {r['z_threshold']}  "
          f"({r['n_spike_days']} spike days)")
    print()
    print(f"  mean |next move| spike : {r['spike_abs_move']*100:.3f}%")
    print(f"  mean |next move| normal: {r['normal_abs_move']*100:.3f}%")
    print(f"  gap                    : {r['abs_move_gap']*100:+.3f}%  "
          f"(ratio {r['gap_ratio']}x)")
    print(f"  vol_z vs |ret| corr    : {r['vol_z_vs_abs_ret_corr']:+.4f}")
    print(f"  spike-day up fraction  : {r['spike_up_fraction']}  (0.50 = no direction)")
    print(f"  permutation p-value    : {r['p_value']:.4f}")
    print()
    print("  -- confound control: price's own recent volatility --")
    cg = r.get("controlled_gap")
    if cg is not None:
        print(f"  gap within vol-buckets : {cg*100:+.3f}%")
    else:
        print("  gap within vol-buckets : n/a")
    if r.get("controlled_gap_fraction") is not None:
        print(f"  fraction of raw gap    : {r['controlled_gap_fraction']} "
              f"(1.0 = news fully independent of price vol)")
    cp = r.get("controlled_p_value")
    if cp is not None:
        print(f"  controlled p-value       : {cp:.4f}  "
              f"(stratified permutation within vol buckets)")
    if r.get("partial_corr_vs_recent_vol") is not None:
        print(f"  partial corr (ctrl vol): {r['partial_corr_vs_recent_vol']:+.4f}  "
              f"(vs raw {r['vol_z_vs_abs_ret_corr']:+.4f})")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE news-attention (volume) test")
    parser.add_argument("--price", default="btc_daily.csv")
    parser.add_argument("--news", default="news_btc.csv")
    parser.add_argument("--z", type=float, default=1.5)
    args = parser.parse_args()
    for c in (args.price, args.news):
        if not Path(c).exists():
            log.error("File not found: %s", c)
            return
    _print(run_attention(args.price, args.news, z_threshold=args.z))


if __name__ == "__main__":
    main()
