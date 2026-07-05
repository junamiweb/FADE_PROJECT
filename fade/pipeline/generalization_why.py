"""Why lean path sets fail to generalize -- diagnostic holdout study.

path_lean3 wins on btc_1h (54.64%) but path_min only beats core5 on 2/5 assets.
This module asks WHY by pairing strict holdout results with per-asset mechanism
diagnostics on the SAME quarantined holdout slice (no look-ahead):

    - streak reversion strength (does the asset even mean-revert after runs?)
    - close_pos / range_pct signal on holdout (path_lean3's non-path atoms)
    - rule count & coverage (does mining find enough stable rules?)
    - data depth (total bars)

Then correlates diagnostics with (path_lean3 hit - core5 hit) to see which factor
explains where the lean set wins vs loses.

Run:
    python -m fade.pipeline.generalization_why
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import ATOM_SETS, Config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.pipeline.holdout import holdout_test
from fade.pipeline.trend_structure import _signed_streak

DEFAULT_ASSETS = ["btc_1h.csv", "eth_1h.csv", "btc_15m.csv", "btc_30m.csv", "btc_5m.csv"]
HOLDOUT_FRAC = 0.30
MIN_SUPPORT = 100


def _holdout_diagnostics(csv_path: str, holdout_frac: float = HOLDOUT_FRAC) -> dict:
  """Mechanism stats on the quarantined holdout slice only."""
  df = load_ohlcv(csv_path)
  ret = df["close"].pct_change()
  pool = atoms_mod.compute_atom_pool(df, Config()).dropna()
  n = len(pool)
  split = int(n * (1 - holdout_frac))
  hold = pool.iloc[split:]
  hold_ret = ret.reindex(hold.index)
  fwd_up = (hold_ret.shift(-1) > 0).astype(float)  # next-bar direction

  streak = _signed_streak(hold_ret.to_numpy())
  # reversion index: 0.5 - continuation after |streak|>=2
  cont_hits = cont_n = 0
  for m in range(2, 8):
    for sgn in (m, -m):
      sel = streak == sgn
      k = int(sel.sum())
      if k < MIN_SUPPORT:
        continue
      nxt = (hold_ret.to_numpy()[sel] > 0).astype(int)
      cont_hits += int((nxt == (1 if sgn > 0 else 0)).sum())
      cont_n += k
  cont = cont_hits / cont_n if cont_n else float("nan")
  rev_index = 0.5 - cont

  # close_pos contrarian: close near HIGH of bar -> predict down next bar
  cp = hold["close_pos"]
  hi = cp >= cp.quantile(2 / 3)
  lo = cp <= cp.quantile(1 / 3)
  hi_n = int(hi.sum())
  lo_n = int(lo.sum())
  hi_hit = float((fwd_up[hi] < 0.5).mean()) if hi_n >= MIN_SUPPORT else float("nan")
  lo_hit = float((fwd_up[lo] > 0.5).mean()) if lo_n >= MIN_SUPPORT else float("nan")
  cp_edge = float(np.nanmean([hi_hit, lo_hit]) - 0.5) if hi_hit == hi_hit and lo_hit == lo_hit else float("nan")

  # range_pct: vol-of-bar proxy, median on holdout
  rp_med = float(hold["range_pct"].median())

  return {
    "n_total": int(n),
    "n_holdout": int(len(hold)),
    "rev_index": round(rev_index, 4) if rev_index == rev_index else None,
    "streak_n": int(cont_n),
    "close_pos_edge": round(cp_edge, 4) if cp_edge == cp_edge else None,
    "range_pct_med": round(rp_med, 6),
  }


def run_why(assets: list[str] | None = None, holdout_frac: float = HOLDOUT_FRAC) -> dict:
  assets = assets or DEFAULT_ASSETS
  rows = []
  for csv in assets:
    if not Path(csv).exists():
      continue
    asset = Path(csv).stem
    diag = _holdout_diagnostics(csv, holdout_frac)
    lifts = {}
    hits = {}
    rules = {}
    for name in ("core5", "path_lean3"):
      cfg = dataclasses.replace(Config(), atom_columns=ATOM_SETS[name])
      r = holdout_test(csv, holdout_frac=holdout_frac, config=cfg)
      if r.get("status") == "ok":
        hits[name] = r["holdout_hit_rate"]
        lifts[name] = r["holdout_lift_vs_random"]
        rules[name] = r["n_stable_rules"]
      else:
        hits[name] = lifts[name] = rules[name] = None
    delta = None
    if hits.get("path_lean3") is not None and hits.get("core5") is not None:
      delta = round(hits["path_lean3"] - hits["core5"], 4)
    rows.append({
      "asset": asset,
      **diag,
      "core5_hit": hits.get("core5"),
      "lean3_hit": hits.get("path_lean3"),
      "delta": delta,
      "lean3_rules": rules.get("path_lean3"),
      "core5_rules": rules.get("core5"),
      "lean_wins": delta is not None and delta > 0,
    })
  return {"status": "ok", "rows": rows}


def _verdict(rows: list[dict]) -> str:
  ok = [r for r in rows if r.get("delta") is not None]
  if not ok:
    return "inconclusive"
  wins = sum(1 for r in ok if r["lean_wins"])
  # simple: does rev_index correlate with delta?
  revs = [r["rev_index"] for r in ok if r["rev_index"] is not None]
  deltas = [r["delta"] for r in ok if r["rev_index"] is not None]
  corr_rev = float(np.corrcoef(revs, deltas)[0, 1]) if len(revs) >= 3 else float("nan")
  cps = [r["close_pos_edge"] for r in ok if r["close_pos_edge"] is not None]
  deltas_cp = [r["delta"] for r in ok if r["close_pos_edge"] is not None]
  corr_cp = float(np.corrcoef(cps, deltas_cp)[0, 1]) if len(cps) >= 3 else float("nan")
  parts = [f"path_lean3 beats core5 on {wins}/{len(ok)} assets"]
  if corr_rev == corr_rev:
    parts.append(f"corr(rev_index, delta)={corr_rev:+.3f}")
  if corr_cp == corr_cp:
    parts.append(f"corr(close_pos_edge, delta)={corr_cp:+.3f}")
  return " | ".join(parts)


def _print(r: dict) -> None:
  line = "=" * 78
  print("\n" + line)
  print("WHY LEAN SETS FAIL TO GENERALIZE (holdout diagnostics + path_lean3 vs core5)")
  print(line)
  if r.get("status") != "ok":
    print(f"  status: {r.get('status')}")
    print(line + "\n")
    return
  print(f"  {'asset':<12}{'bars':>8}{'rev_idx':>9}{'cp_edge':>9}{'core5':>8}"
        f"{'lean3':>8}{'delta':>8}{'win':>5}")
  for x in r["rows"]:
    w = "Y" if x.get("lean_wins") else "N"
    print(f"  {x['asset']:<12}{x['n_total']:>8}{str(x.get('rev_index','?')):>9}"
          f"{str(x.get('close_pos_edge','?')):>9}{str(x.get('core5_hit','?')):>8}"
          f"{str(x.get('lean3_hit','?')):>8}{str(x.get('delta','?')):>8}{w:>5}")
  print(line)
  print(f"  VERDICT: {_verdict(r['rows'])}")
  print("  rev_idx>0 = mean-reversion on holdout | cp_edge = close_pos contrarian edge")
  print(line + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description="FADE generalization diagnostics")
  parser.add_argument("assets", nargs="*", default=None)
  args = parser.parse_args()
  files = args.assets if args.assets else DEFAULT_ASSETS
  _print(run_why(files))


if __name__ == "__main__":
  main()
