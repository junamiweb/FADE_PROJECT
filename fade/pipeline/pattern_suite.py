"""Integrated pattern suite — four directions, individually and together.

1. PHASE TRANSITION: resample 1s BTC to intermediate bars (2s..60s) and locate
   where momentum (rev_index<0) flips to mean-reversion (rev_index>0).

2. ETH vs BTC: same reversion metrics on ETH 1h; cross-asset signal correlation
   and joint-prediction hit rate when both agree.

3. VOLATILITY-CONDITIONED REVERSAL: is intraday streak-reversal stronger in
   high-vol vs low-vol regimes? (dev-fit vol median, tested on holdout)

4. ORTHOGONAL ENSEMBLE: combine contrarian streak-signals from 5m/15m/30m/1h on
   a shared 1h grid; measure holdout hit rate for single vs majority vs unanimous.

5. INTEGRATED: summary tying all four together.

Honest protocol throughout: streak uses bars strictly before t; vol uses shifted
rolling window; macro/daily not needed here; 70/30 chronological holdout;
permutation p-values where stated.

Run:
    python -m fade.pipeline.pattern_suite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.trend_structure import _signed_streak

MIN_SUPPORT = 50
HOLDOUT_FRAC = 0.30
ENSEMBLE_FILES = ["btc_5m.csv", "btc_15m.csv", "btc_30m.csv", "btc_1h.csv"]


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def _streak_metrics(ret: np.ndarray, holdout_frac: float = HOLDOUT_FRAC,
                    n_shuffles: int = 1500, seed: int = 0) -> dict:
    """Continuation after streak>=2 on holdout + reversion_index."""
    streak = _signed_streak(ret)
    up = (ret > 0).astype(int)
    finite = np.isfinite(ret)
    n = len(ret)
    split = int(n * (1 - holdout_frac))
    hold = np.zeros(n, dtype=bool)
    hold[split:] = True
    hold &= finite

    cont_hits = cont_n = 0
    for m in range(2, 8):
        for sgn in (m, -m):
            sel = hold & (streak == sgn)
            k = int(sel.sum())
            if k < MIN_SUPPORT:
                continue
            nxt = up[sel]
            cont_hits += int((nxt == (1 if sgn > 0 else 0)).sum())
            cont_n += k
    if cont_n < MIN_SUPPORT:
        return {"status": "low_support", "n": int(n)}
    cont = cont_hits / cont_n
    rev = 0.5 - cont
    rng = np.random.default_rng(seed)
    base = float(up[hold].mean())
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null[i] = rng.choice(up[hold], size=cont_n, replace=False).mean()
    dev = abs(cont - 0.5)
    p = (1 + int(np.sum(np.abs(null - base) >= dev))) / (1 + n_shuffles)
    return {
        "status": "ok", "n": int(n), "streak_n": cont_n,
        "continuation": round(cont, 4),
        "reversion_index": round(rev, 4),
        "behaviour": "reversion" if rev > 0 else "momentum",
        "p_value": round(p, 4),
    }


# ------------------------------------------------------------------ 1 phase --
def phase_transition(csv_1s: str = "btc_1s.csv") -> dict:
    if not Path(csv_1s).exists():
        return {"status": "missing_file"}
    df = load_ohlcv(csv_1s)
    rules = [("1s", None), ("2s", "2s"), ("5s", "5s"), ("10s", "10s"),
             ("15s", "15s"), ("30s", "30s"), ("60s", "60s")]
    rows = []
    for label, rule in rules:
        sub = df if rule is None else _resample_ohlcv(df, rule)
        ret = sub["close"].pct_change().to_numpy()
        m = _streak_metrics(ret)
        if m.get("status") != "ok":
            rows.append({"scale": label, "status": m["status"]})
            continue
        rows.append({"scale": label, **m})

    ok = [r for r in rows if r.get("status") == "ok"]
    crossover = None
    for i in range(len(ok) - 1):
        if ok[i]["reversion_index"] < 0 and ok[i + 1]["reversion_index"] > 0:
            crossover = f"{ok[i]['scale']} -> {ok[i+1]['scale']}"
            break
    return {
        "status": "ok" if ok else "insufficient",
        "source": Path(csv_1s).stem,
        "span": f"{df.index[0].date()} .. {df.index[-1].date()}",
        "rows": rows,
        "crossover": crossover,
        "caveat": "1s window ~7 days; tick-sensitive — directional hint only",
    }


# ------------------------------------------------------------------ 2 ETH ----
def asset_comparison(btc_1h: str = "btc_1h.csv", eth_1h: str = "eth_1h.csv",
                   holdout_frac: float = HOLDOUT_FRAC) -> dict:
    assets = {}
    for path in (btc_1h, eth_1h):
        if not Path(path).exists():
            continue
        df = load_ohlcv(path)
        ret = df["close"].pct_change().to_numpy()
        assets[Path(path).stem] = _streak_metrics(ret, holdout_frac=holdout_frac)

    if len(assets) < 2:
        return {"status": "missing_files"}

    # cross-asset signal correlation + joint hit on aligned 1h grid
    btc = load_ohlcv(btc_1h)
    eth = load_ohlcv(eth_1h)

    def _contrarian_series(df: pd.DataFrame) -> pd.Series:
        r = df["close"].pct_change()
        streak = _signed_streak(r.to_numpy())
        sig = pd.Series(0.0, index=df.index)
        mask = np.abs(streak) >= 2
        sig.iloc[mask] = -np.sign(streak[mask])
        return sig

    bsig = _contrarian_series(btc).resample("1h").last()
    esig = _contrarian_series(eth).resample("1h").last()
    bret = btc["close"].pct_change().resample("1h").sum()
    eret = eth["close"].pct_change().resample("1h").sum()

    aligned = pd.DataFrame({"b_sig": bsig, "e_sig": esig, "b_ret": bret, "e_ret": eret}).dropna()
    n = len(aligned)
    split = int(n * (1 - holdout_frac))
    hold = aligned.iloc[split:]

    sig_corr = float(hold[["b_sig", "e_sig"]].corr().iloc[0, 1])
    ret_corr = float(hold[["b_ret", "e_ret"]].corr().iloc[0, 1])

    def _hit(sig_col: str, ret_col: str, mask: pd.Series) -> tuple[float, int]:
        sub = hold[mask]
        if len(sub) < MIN_SUPPORT:
            return float("nan"), 0
        hit = ((sub[sig_col] > 0) & (sub[ret_col] > 0)) | ((sub[sig_col] < 0) & (sub[ret_col] < 0))
        return float(hit.mean()), len(sub)

    b_only = hold["b_sig"] != 0
    e_only = hold["e_sig"] != 0
    both = (hold["b_sig"] != 0) & (hold["e_sig"] != 0)
    agree = both & (np.sign(hold["b_sig"]) == np.sign(hold["e_sig"]))
    disagree = both & (np.sign(hold["b_sig"]) != np.sign(hold["e_sig"]))

    b_hit, b_n = _hit("b_sig", "b_ret", b_only)
    e_hit, e_n = _hit("e_sig", "e_ret", e_only)
    joint_hit, j_n = _hit("b_sig", "b_ret", agree)  # predict BTC when both agree
    disagree_b, d_n = _hit("b_sig", "b_ret", disagree)

    return {
        "status": "ok",
        "btc": assets.get("btc_1h", {}),
        "eth": assets.get("eth_1h", {}),
        "cross": {
            "n_holdout": int(len(hold)),
            "signal_corr": round(sig_corr, 4),
            "return_corr": round(ret_corr, 4),
            "btc_solo_hit": round(b_hit, 4), "btc_solo_n": b_n,
            "eth_solo_hit": round(e_hit, 4), "eth_solo_n": e_n,
            "both_agree_hit": round(joint_hit, 4), "both_agree_n": j_n,
            "both_disagree_hit": round(disagree_b, 4), "both_disagree_n": d_n,
        },
    }


# ----------------------------------------------------------- 3 vol-conditioned
def vol_conditioned(csv: str = "btc_1h.csv", vol_window: int = 24,
                    holdout_frac: float = HOLDOUT_FRAC,
                    n_shuffles: int = 2000, seed: int = 0) -> dict:
    df = load_ohlcv(csv)
    ret = df["close"].pct_change()
    streak = _signed_streak(ret.to_numpy())
    # trailing vol, shifted — no look-ahead
    vol = ret.rolling(vol_window).std().shift(1)
    up = (ret > 0).astype(int)

    frame = pd.DataFrame({"ret": ret, "streak": streak, "vol": vol, "up": up}).dropna()
    n = len(frame)
    split = int(n * (1 - holdout_frac))
    dev, hold = frame.iloc[:split], frame.iloc[split:]
    med = float(dev["vol"].median())

    rng = np.random.default_rng(seed)
    rows = []
    base_up = float(hold["up"].mean())
    for label, mask_fn in (("low-vol", lambda v: v <= med),
                           ("high-vol", lambda v: v > med)):
        sub = hold[(np.abs(hold["streak"]) >= 2) & mask_fn(hold["vol"])]
        k = len(sub)
        if k < MIN_SUPPORT:
            rows.append({"regime": label, "n": k, "status": "low"})
            continue
        # reversal strength: P(next opposes streak)
        rev = []
        for _, row in sub.iterrows():
            if row["streak"] > 0:
                rev.append(1 - row["up"])
            else:
                rev.append(row["up"])
        rev_rate = float(np.mean(rev))
        null = np.empty(n_shuffles)
        allup = hold["up"].to_numpy()
        for i in range(n_shuffles):
            samp = rng.choice(allup, size=k, replace=False)
            null[i] = samp.mean()
        # test if reversal deviates from base
        dev_abs = abs(rev_rate - 0.5)
        p = (1 + int(np.sum(np.abs(null - base_up) >= dev_abs))) / (1 + n_shuffles)
        rows.append({
            "regime": label, "n": k, "status": "ok",
            "reversal_strength": round(rev_rate, 4),
            "p_value": round(p, 4),
        })
    gap = None
    ok = [r for r in rows if r.get("status") == "ok"]
    if len(ok) == 2:
        gap = round(ok[1]["reversal_strength"] - ok[0]["reversal_strength"], 4)
    return {
        "status": "ok", "asset": Path(csv).stem,
        "vol_window": vol_window, "dev_median": round(med, 6),
        "rows": rows, "high_minus_low": gap,
    }


# ---------------------------------------------------------- 4 ensemble -------
def _contrarian_on_grid(csv: str, grid: str = "1h") -> pd.Series:
    df = load_ohlcv(csv)
    r = df["close"].pct_change()
    streak = _signed_streak(r.to_numpy())
    sig = pd.Series(0.0, index=df.index)
    mask = np.abs(streak) >= 2
    sig.iloc[mask] = -np.sign(streak[mask])
    return sig.resample(grid).last()


def orthogonal_ensemble(files: list[str] | None = None, grid: str = "1h",
                        holdout_frac: float = HOLDOUT_FRAC,
                        n_shuffles: int = 2000, seed: int = 0) -> dict:
    files = files or ENSEMBLE_FILES
    sigs = {}
    for c in files:
        if Path(c).exists():
            sigs[Path(c).stem] = _contrarian_on_grid(c, grid)
    if len(sigs) < 2:
        return {"status": "insufficient_files"}

    # target: NEXT 1h return (signal at bar close t predicts t -> t+1)
    btc1h = load_ohlcv("btc_1h.csv")
    target = btc1h["close"].pct_change().shift(-1)
    df = pd.DataFrame(sigs)
    df["ret"] = target
    df = df.dropna()
    n = len(df)
    split = int(n * (1 - holdout_frac))
    hold = df.iloc[split:]

    cols = list(sigs.keys())

    def _dir_1h(row: pd.Series) -> float:
        v = float(row.get("btc_1h", row.iloc[-1]))
        return v if v != 0 else 0.0

    def _dir_majority(row: pd.Series) -> float:
        active = row[cols][row[cols] != 0]
        if len(active) < 2:
            return 0.0
        pos = int((active > 0).sum())
        neg = int((active < 0).sum())
        if pos >= 2:
            return 1.0
        if neg >= 2:
            return -1.0
        return 0.0

    def _dir_unanimous(row: pd.Series) -> float:
        active = row[cols][row[cols] != 0]
        if len(active) != len(cols):
            return 0.0
        if active.nunique() != 1:
            return 0.0
        return float(active.iloc[0])

    def _hit_on_dir(sub: pd.DataFrame, directions: pd.Series) -> tuple[float, int]:
        valid = directions != 0
        sub = sub.loc[valid]
        d = directions.loc[valid]
        if len(sub) < MIN_SUPPORT:
            return float("nan"), 0
        hit = ((d > 0) & (sub["ret"] > 0)) | ((d < 0) & (sub["ret"] < 0))
        return float(hit.mean()), len(sub)

    mode_dirs = {
        "1h_solo": hold.apply(_dir_1h, axis=1),
        "majority_2+": hold.apply(_dir_majority, axis=1),
        "unanimous_4": hold.apply(_dir_unanimous, axis=1),
    }
    # same-bar 1h (diagnostic — matches streak literature)
    same_bar = hold["btc_1h"].replace(0, np.nan) if "btc_1h" in hold.columns else hold[cols[-1]].replace(0, np.nan)
    same_ret = load_ohlcv("btc_1h.csv")["close"].pct_change().reindex(hold.index)
    same_mask = same_bar.notna() & same_ret.notna()
    if same_mask.sum() >= MIN_SUPPORT:
        d = same_bar[same_mask]
        r = same_ret[same_mask]
        sb_hit = float((((d > 0) & (r > 0)) | ((d < 0) & (r < 0))).mean())
    else:
        sb_hit = float("nan")
    rng = np.random.default_rng(seed)
    results = []
    for name, directions in mode_dirs.items():
        hit, k = _hit_on_dir(hold, directions)
        if k < MIN_SUPPORT:
            results.append({"mode": name, "n": k, "status": "low"})
            continue
        null = np.empty(n_shuffles)
        for i in range(n_shuffles):
            samp = rng.choice(hold["ret"].to_numpy(), size=k, replace=False)
            dirs = rng.choice([-1.0, 1.0], size=k)
            null[i] = float(np.mean(
                ((dirs > 0) & (samp > 0)) | ((dirs < 0) & (samp < 0))
            ))
        edge = hit - 0.5
        p = (1 + int(np.sum(null >= hit))) / (1 + n_shuffles)
        results.append({
            "mode": name, "n": k, "hit": round(hit, 4),
            "edge": round(edge, 4), "p_value": round(p, 4), "status": "ok",
        })
    best = max((r for r in results if r.get("status") == "ok"),
               key=lambda x: x["hit"], default=None)
    return {
        "status": "ok", "grid": grid, "n_holdout": int(len(hold)),
        "base_up": round(float((hold["ret"] > 0).mean()), 4),
        "same_bar_1h_hit": round(sb_hit, 4) if sb_hit == sb_hit else None,
        "modes": results,
        "best_mode": best["mode"] if best else None,
        "best_hit": best["hit"] if best else None,
    }


# ------------------------------------------------------- 5 integrated --------
def run_suite() -> dict:
    return {
        "phase": phase_transition(),
        "eth": asset_comparison(),
        "vol": vol_conditioned(),
        "ensemble": orthogonal_ensemble(),
    }


def _integrated_verdict(suite: dict) -> str:
    parts = []
    ph = suite.get("phase", {})
    if ph.get("crossover"):
        parts.append(f"phase flip ~{ph['crossover']}")
    eth = suite.get("eth", {})
    if eth.get("status") == "ok":
        b = eth["btc"].get("reversion_index", 0)
        e = eth["eth"].get("reversion_index", 0)
        parts.append(f"ETH rev={e:+.3f} vs BTC rev={b:+.3f}")
        c = eth.get("cross", {})
        if c.get("both_agree_hit"):
            parts.append(f"BTC+ETH agree hit={c['both_agree_hit']:.1%}(n={c['both_agree_n']})")
    vol = suite.get("vol", {})
    if vol.get("high_minus_low") is not None:
        parts.append(f"high-vol boost={vol['high_minus_low']:+.3f}")
    ens = suite.get("ensemble", {})
    if ens.get("best_mode"):
        parts.append(f"ensemble best={ens['best_mode']} hit={ens['best_hit']:.1%}")
    return " | ".join(parts) if parts else "inconclusive"


# ------------------------------------------------------------- print ----------
def _print_all(suite: dict) -> None:
    line = "=" * 72

    print("\n" + line)
    print("A) PHASE TRANSITION  (1s resampled -> where momentum flips)")
    print(line)
    ph = suite["phase"]
    if ph.get("status") != "ok":
        print(f"  status: {ph.get('status')}")
    else:
        print(f"  source: {ph['source']}  span: {ph['span']}")
        print(f"  {'scale':<8}{'continue':>10}{'rev_index':>12}{'behaviour':>12}{'p':>8}")
        for r in ph["rows"]:
            if r.get("status") != "ok":
                print(f"  {r['scale']:<8}  {r.get('status')}")
                continue
            star = " *" if r["p_value"] <= 0.05 else ""
            print(f"  {r['scale']:<8}{r['continuation']:>10}{r['reversion_index']:>+12}"
                  f"{r['behaviour']:>12}{r['p_value']:>8}{star}")
        print(f"  CROSSOVER: {ph.get('crossover', 'not found in range')}")
        print(f"  caveat: {ph.get('caveat')}")

    print("\n" + line)
    print("B) ETH vs BTC  (reversion ladder + cross-asset signals)")
    print(line)
    eth = suite["eth"]
    if eth.get("status") != "ok":
        print(f"  status: {eth.get('status')}")
    else:
        for name in ("btc", "eth"):
            m = eth[name]
            print(f"  {name}: rev_index={m.get('reversion_index'):+.4f}  "
                  f"continue={m.get('continuation')}  p={m.get('p_value')}")
        c = eth["cross"]
        print(f"  signal_corr={c['signal_corr']}  return_corr={c['return_corr']}")
        print(f"  BTC solo hit={c['btc_solo_hit']} (n={c['btc_solo_n']})  "
              f"ETH solo={c['eth_solo_hit']} (n={c['eth_solo_n']})")
        print(f"  BOTH agree -> BTC hit={c['both_agree_hit']} (n={c['both_agree_n']})  "
              f"disagree={c['both_disagree_hit']} (n={c['both_disagree_n']})")

    print("\n" + line)
    print("C) VOLATILITY-CONDITIONED REVERSAL  (btc 1h, holdout)")
    print(line)
    vol = suite["vol"]
    if vol.get("status") != "ok":
        print(f"  status: {vol.get('status')}")
    else:
        print(f"  vol window={vol['vol_window']}h  dev median={vol['dev_median']}")
        for r in vol["rows"]:
            if r.get("status") != "ok":
                print(f"  {r['regime']}: n={r['n']} low")
                continue
            star = " *" if r["p_value"] <= 0.05 else ""
            print(f"  {r['regime']:<10} n={r['n']}  reversal={r['reversal_strength']}"
                  f"  p={r['p_value']}{star}")
        print(f"  high-minus-low gap: {vol.get('high_minus_low')}")

    print("\n" + line)
    print("D) ORTHOGONAL ENSEMBLE  (5m+15m+30m+1h contrarian, 1h grid)")
    print(line)
    ens = suite["ensemble"]
    if ens.get("status") != "ok":
        print(f"  status: {ens.get('status')}")
    else:
        print(f"  holdout n={ens['n_holdout']}  base up={ens['base_up']}  "
              f"(same-bar 1h ref={ens.get('same_bar_1h_hit')})")
        print(f"  {'mode':<16}{'n':>8}{'hit':>8}{'edge':>8}{'p':>8}")
        for r in ens["modes"]:
            if r.get("status") != "ok":
                print(f"  {r['mode']:<16}{r['n']:>8}  low")
                continue
            star = " *" if r["p_value"] <= 0.05 else ""
            print(f"  {r['mode']:<16}{r['n']:>8}{r['hit']:>8}{r['edge']:>+8}{r['p_value']:>8}{star}")

    print("\n" + line)
    print("E) INTEGRATED")
    print(line)
    print(f"  {_integrated_verdict(suite)}")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE integrated pattern suite")
    parser.parse_args()
    suite = run_suite()
    _print_all(suite)


if __name__ == "__main__":
    main()
