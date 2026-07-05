"""BTC->ETH lead-lag probe (v0.3 pre-registered exploratory).

Pre-registered study — holdout/exploratory only, no lockbox v1.
Tests whether BTC 1h return / reversal signal leads ETH next bar(s).

Run:
    python -m fade.pipeline.lead_lag_probe
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fade.core.data_loader import load_ohlcv
from fade.pipeline.pre_registration import load_manifest, save_manifest
from fade.pipeline.trend_structure import _signed_streak

STUDY_ID = "lead_lag_btc_eth_1h"
OUTPUT_PATH = Path("fade/output/lead_lag_probe.json")
LAGS = (0, 1, 2, 4)


def _ensure_preregistered() -> None:
    m = load_manifest()
    studies = m.setdefault("studies", [])
    if not any(s.get("study_id") == STUDY_ID for s in studies):
        studies.append({
            "study_id": STUDY_ID,
            "pre_registered_utc": datetime.now(timezone.utc).isoformat(),
            "pair": "BTC 1h -> ETH 1h",
            "lags_bars": list(LAGS),
            "metrics": ["return_corr", "reversal_hit_eth_given_btc_streak"],
            "data_split": "full_history_exploratory",
            "lockbox_policy": "v1 BURNED; no final claim without v2",
        })
        save_manifest(m)


def _align_btc_eth() -> pd.DataFrame:
    btc = load_ohlcv("btc_1h.csv")
    eth = load_ohlcv("eth_1h.csv")
    bret = btc["close"].pct_change().rename("btc_ret")
    eret = eth["close"].pct_change().rename("eth_ret")
    frame = pd.concat([bret, eret], axis=1).dropna()
    streak = _signed_streak(frame["btc_ret"].to_numpy())
    frame["btc_streak"] = streak
    return frame


def run_probe() -> dict:
    _ensure_preregistered()
    frame = _align_btc_eth()
    n = len(frame)

    corr_rows = []
    for lag in LAGS:
        if lag == 0:
            c = float(frame["btc_ret"].corr(frame["eth_ret"]))
        else:
            c = float(frame["btc_ret"].iloc[:-lag].corr(frame["eth_ret"].iloc[lag:]))
        corr_rows.append({"lag_bars": lag, "return_corr": round(c, 4)})

    # BTC streak>=2 contrarian -> ETH next bar direction
    rev_rows = []
    streak = frame["btc_streak"].to_numpy()
    eth_up = (frame["eth_ret"] > 0).astype(int).to_numpy()
    for lag in (1, 2, 4):
        hits = total = 0
        for i in range(len(streak) - lag):
            if abs(streak[i]) < 2:
                continue
            pred_eth_up = 1 if streak[i] < 0 else 0  # contrarian
            actual = eth_up[i + lag]
            hits += int(pred_eth_up == actual)
            total += 1
        if total >= 100:
            rev_rows.append({
                "lag_bars": lag,
                "n": total,
                "hit_rate": round(hits / total, 4),
                "edge": round(hits / total - 0.5, 4),
            })

    return {
        "study_id": STUDY_ID,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_aligned_bars": n,
        "return_correlations": corr_rows,
        "btc_streak_reversal_on_eth": rev_rows,
        "verdict": (
            f"Same-bar corr={corr_rows[0]['return_corr']:.3f}; "
            f"best ETH lag hit={max(rev_rows, key=lambda x: x['hit_rate'])['hit_rate'] if rev_rows else 'n/a'} "
            "(exploratory, not OOS lockbox)."
        ),
    }


def _print(r: dict) -> None:
    print("\n" + "=" * 60)
    print("LEAD-LAG PROBE  BTC 1h -> ETH 1h")
    print("=" * 60)
    print(f"  aligned bars: {r['n_aligned_bars']:,}")
    print("\n  Return correlations:")
    for row in r["return_correlations"]:
        print(f"    lag {row['lag_bars']}: {row['return_corr']:+.4f}")
    print("\n  BTC streak>=2 contrarian -> ETH direction:")
    for row in r.get("btc_streak_reversal_on_eth", []):
        print(f"    lag {row['lag_bars']}: hit={row['hit_rate']}  n={row['n']}")
    print(f"\n  {r['verdict']}")
    print("=" * 60 + "\n")


def main() -> None:
    r = run_probe()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(r, indent=2), encoding="utf-8")
    _print(r)


if __name__ == "__main__":
    main()
