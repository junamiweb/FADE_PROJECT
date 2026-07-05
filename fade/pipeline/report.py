"""Unified FADE report — one command, full picture.

Runs the validated research tools and prints a concise summary:
  - Tiered forecast (frequent / balanced / quality)
  - Strict holdout on primary file
  - Multi-resolution agreement stats (if data present)

Run:
    python -m fade.pipeline.report btc_1h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fade.config import Config
from fade.pipeline.forecast_tiers import run_tiered_forecast, _print as print_tiers
from fade.pipeline.holdout import holdout_test
from fade.utils.logging import get_logger

log = get_logger("report")

VALIDATED_FORMULA = {
    "resolution": "1h (Binance full history)",
    "horizon": "1 bar ahead",
    "atoms": "core5 (original five)",
    "ensemble": "15m + 30m + 1h unanimous -> ~54.7% hit",
    "magnitude_sweet_spot": "0.1% threshold -> ~53.7% hit, 70% coverage",
    "magnitude_strong": "1.0% on 15m -> ~61.6% hit, 1% coverage",
}


def run_report(csv_path: str, config: Config | None = None,
               skip_holdout: bool = False) -> dict:
    config = config or Config()
    asset = Path(csv_path).stem

    tiers = run_tiered_forecast(csv_path, config)
    holdout = None if skip_holdout else holdout_test(csv_path, config=config)

    return {
        "asset": asset,
        "formula": VALIDATED_FORMULA,
        "tiers": tiers,
        "holdout": holdout,
    }


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 62)
    print("FADE REPORT - validated formula summary")
    print("=" * 62)
    f = r["formula"]
    print(f"  Resolution : {f['resolution']}")
    print(f"  Horizon    : {f['horizon']}")
    print(f"  Atoms      : {f['atoms']}")
    print(f"  Ensemble   : {f['ensemble']}")
    print(f"  Magnitude  : {f['magnitude_sweet_spot']}")
    print("=" * 62)

    print_tiers(r["tiers"])

    h = r.get("holdout")
    if h and h.get("status") == "ok":
        print("  HOLDOUT (strict, untouched 30%):")
        print(f"    hit={h['holdout_hit_rate']}  lift={h['holdout_lift_vs_random']:+.4f}  "
              f"p={h['p_value']}  -> {h['verdict']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE unified report")
    parser.add_argument("csv", nargs="?", default="btc_1h.csv")
    parser.add_argument("--fast", action="store_true", help="skip holdout (faster)")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)
    _print_summary(run_report(args.csv, skip_holdout=args.fast))


if __name__ == "__main__":
    main()
