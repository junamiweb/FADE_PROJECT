"""Minimal logging.

Only essential metrics are logged (per the cleanup policy). We deliberately
avoid dumping raw datasets or verbose per-row traces.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "fade") -> logging.Logger:
    """Return a lazily-configured module logger."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root = logging.getLogger("fade")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("fade") else f"fade.{name}")


def log_metric(logger: logging.Logger, name: str, value) -> None:
    """Log a single aggregated metric (keeps logs lean and grep-able)."""
    if isinstance(value, float):
        logger.info("metric %-24s = %.4f", name, value)
    else:
        logger.info("metric %-24s = %s", name, value)
