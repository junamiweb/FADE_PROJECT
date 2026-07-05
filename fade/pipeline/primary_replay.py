"""Score PRIMARY forecast logic on holdout -- honest OOS validation.

Measures conviction tiers and conflict buckets (conviction vs multi-TF
contrarian unanimous) on the quarantined last 30%. Fixed definitions only.

Run:
    python -m fade.pipeline.primary_replay
    python -m fade.pipeline.primary_replay btc_1h.csv eth_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.conviction import TIER_DEFS
from fade.core.data_loader import load_ohlcv
from fade.pipeline.conviction_gate import _contrarian_grid
from fade.pipeline.trend_structure import _signed_streak

HOLDOUT_FRAC = 0.30
MULTI_IV = ("5m", "15m", "30m", "1h")


def _prefix(csv: str) -> str:
    stem = Path(csv).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _multi_grids(prefix: str) -> dict[str, pd.Series]:
    out = {}
    for iv in MULTI_IV:
        p = f"{prefix}_{iv}.csv"
        if Path(p).exists():
            out[iv] = _contrarian_grid(p)
    return out


def _active_tier(slen: int, streak_dir: str, tf_agree: int, multi_dir: str,
                   aligned: bool) -> tuple[str, str] | None:
    """Return (tier_id, direction) or None."""
    for tid, _label, min_s, min_k, _hit, _note in TIER_DEFS:
        if min_k == 0:
            if slen < min_s or streak_dir == "FLAT":
                continue
            return tid, streak_dir
        if min_s == 0:
            if tf_agree < min_k or multi_dir == "FLAT":
                continue
            return tid, multi_dir
        if slen < min_s or tf_agree < min_k or not aligned:
            continue
        return tid, streak_dir
    return None


def score_asset(csv_1h: str, holdout_frac: float = HOLDOUT_FRAC) -> dict:
    prefix = _prefix(csv_1h)
    df = load_ohlcv(csv_1h)
    ret = df["close"].pct_change()
    fwd = ret.shift(-1)
    streak = _signed_streak(ret.to_numpy())
    grids = _multi_grids(prefix)

    frame = pd.DataFrame({"streak": streak, "slen": np.abs(streak), "fwd": fwd},
                         index=df.index)
    for iv, s in grids.items():
        frame[iv] = s
    frame = frame.dropna(subset=["fwd"])
    cols = [c for c in MULTI_IV if c in frame.columns]

    n = len(frame)
    split = int(n * (1 - holdout_frac))
    hold = frame.iloc[split:]

    rows = []
    for ts, row in hold.iterrows():
        s = int(row["streak"]) if np.isfinite(row["streak"]) else 0
        slen = int(row["slen"])
        streak_dir = "FLAT" if slen < 2 else ("UP" if s < 0 else "DOWN")
        dirs = []
        for c in cols:
            v = row[c]
            if v == 0 or not np.isfinite(v):
                dirs.append(None)
            else:
                dirs.append("UP" if v > 0 else "DOWN")
        ok = [d for d in dirs if d is not None]
        up = sum(1 for d in ok if d == "UP")
        dn = sum(1 for d in ok if d == "DOWN")
        if up >= dn and up > 0:
            multi_dir, tf_agree = "UP", up
        elif dn > 0:
            multi_dir, tf_agree = "DOWN", dn
        else:
            multi_dir, tf_agree = "FLAT", 0
        aligned = streak_dir != "FLAT" and multi_dir != "FLAT" and streak_dir == multi_dir

        tier = _active_tier(slen, streak_dir, tf_agree, multi_dir, aligned)
        if not tier:
            continue
        _tid, conv_dir = tier
        actual_up = int(row["fwd"] > 0)
        hit = int((conv_dir == "UP") == actual_up)

        # proxy "frequent": 15m+30m+1h contrarian unanimous (3 TF)
        core = [d for c, d in zip(cols, dirs) if c in ("15m", "30m", "1h") and d]
        freq_dir = None
        if len(core) == 3 and len(set(core)) == 1:
            freq_dir = core[0]
        conflict = freq_dir is not None and freq_dir != conv_dir
        policy_hit = hit
        if conflict:
            policy_hit = int((freq_dir == "UP") == actual_up)

        rows.append({"hit_conv": hit, "hit_primary": policy_hit, "conflict": conflict,
                     "tier": _tid})

    if not rows:
        return {"asset": Path(csv_1h).stem, "status": "no_rows"}

    r = pd.DataFrame(rows)
    conv_hit = float(r["hit_conv"].mean())
    prim_hit = float(r["hit_primary"].mean())
    conf = r[r["conflict"]]
    no_conf = r[~r["conflict"]]

    return {
        "asset": Path(csv_1h).stem,
        "status": "ok",
        "holdout_bars": len(hold),
        "primary_signals": len(r),
        "conviction_hit": round(conv_hit, 4),
        "primary_hit_new_rule": round(prim_hit, 4),
        "conflict_n": int(r["conflict"].sum()),
        "conflict_conv_hit": round(float(conf["hit_conv"].mean()), 4) if len(conf) else None,
        "conflict_primary_hit": round(float(conf["hit_primary"].mean()), 4) if len(conf) else None,
        "aligned_n": len(no_conf),
        "aligned_hit": round(float(no_conf["hit_conv"].mean()), 4) if len(no_conf) else None,
    }


def _print(results: list[dict]) -> None:
    line = "=" * 72
    print("\n" + line)
    print("PRIMARY REPLAY (holdout 30%, fixed rules)")
    print(line)
    print(f"  {'asset':<12}{'signals':>8}{'conv_hit':>10}{'primary':>10}"
          f"{'conflict':>10}{'conf_old':>10}{'conf_new':>10}")
    for r in results:
        if r.get("status") != "ok":
            print(f"  {r.get('asset', '?'):<12}  {r.get('status')}")
            continue
        print(f"  {r['asset']:<12}{r['primary_signals']:>8}"
              f"{r['conviction_hit']:>10.4f}{r['primary_hit_new_rule']:>10.4f}"
              f"{r['conflict_n']:>10}"
              f"{r['conflict_conv_hit'] or 0:>10.4f}"
              f"{r['conflict_primary_hit'] or 0:>10.4f}")
    print(line)
    print("  primary = frequent wins on conflict (conviction vs 15+30+1h unanimous)")
    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE primary replay")
    parser.add_argument("csv", nargs="*", default=["btc_1h.csv", "eth_1h.csv"])
    args = parser.parse_args()
    results = []
    for csv in args.csv:
        if not Path(csv).exists():
            results.append({"asset": csv, "status": "missing"})
            continue
        results.append(score_asset(csv))
    _print(results)


if __name__ == "__main__":
    main()
