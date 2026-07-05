"""Generate research charts for an asset.

Run:
    python -m fade.pipeline.plot btc.csv
    python -m fade.pipeline.plot btc.csv --bars 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fade.config import Config
from fade.utils.logging import get_logger
from fade.viz.charts import generate_charts

log = get_logger("plot")


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE research charts")
    parser.add_argument("csv", help="path to OHLCV CSV")
    parser.add_argument("--bars", type=int, default=336,
                        help="bars shown on price chart (default 336 ~ 2 weeks)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        log.error("File not found: %s", args.csv)
        sys.exit(1)

    result = generate_charts(args.csv, Config(), last_bars=args.bars)
    print("\n" + "=" * 60)
    print(f"FADE CHARTS — {result['asset'].upper()}")
    print("=" * 60)
    print(f"  Output dir: {result['output_dir']}")
    for name, path in result["charts"].items():
        status = path if path else "(skipped — insufficient data)"
        print(f"  {name:<16}: {status}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
