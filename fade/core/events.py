"""Event construction.

Atoms are discretised into LOW/MID/HIGH via quantiles, then combined into
2-4 atom "events". Events are hashable string keys, de-duplicated, and rare
events are filtered out (noise reduction / anti-overfitting).

Key format:  "return_1h=HIGH|volatility=LOW"   (atoms sorted for canonicality)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from fade.config import ATOM_COLUMNS, Config


def compute_thresholds(atoms: pd.DataFrame, config: Config) -> dict[str, tuple[float, float]]:
    """Compute (low, high) quantile cut points per atom from a *training* frame.

    Thresholds are derived only from the data passed in; in the walk-forward
    backtest this is always the training slice, preventing look-ahead leakage.
    """
    fixed = getattr(config, "atom_fixed_thresholds", {}) or {}
    thresholds: dict[str, tuple[float, float]] = {}
    for atom in config.atom_columns:
        if atom in fixed:
            thresholds[atom] = fixed[atom]
            continue
        col = atoms[atom]
        lo = float(col.quantile(config.quantile_low))
        hi = float(col.quantile(config.quantile_high))
        thresholds[atom] = (lo, hi)
    return thresholds


def discretize(atoms: pd.DataFrame, thresholds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Map each atom value to LOW/MID/HIGH using precomputed thresholds."""
    out = {}
    for atom in thresholds:
        lo, hi = thresholds[atom]
        col = atoms[atom].to_numpy()
        state = np.where(col <= lo, "LOW", np.where(col >= hi, "HIGH", "MID"))
        out[atom] = state
    return pd.DataFrame(out, index=atoms.index)


def build_events(
    disc: pd.DataFrame,
    config: Config,
    allowed: set[str] | None = None,
    blocked: set[str] | None = None,
) -> pd.DataFrame:
    """Build a long-format (timestamp, event) table from discretised atoms.

    Each row spawns one event per atom-combination of the configured sizes.
    ``blocked`` events (negative memory) are dropped *before* anything else.
    If ``allowed`` is provided, only those event keys are kept (used at
    evaluation time to restrict to mined rules).
    """
    frames = []
    for size in config.event_sizes:
        for combo in itertools.combinations(config.atom_columns, size):
            # Canonical, vectorised string key: "atom=STATE|atom=STATE".
            key = combo[0] + "=" + disc[combo[0]]
            for atom in combo[1:]:
                key = key + "|" + atom + "=" + disc[atom]
            frames.append(pd.Series(key.to_numpy(), index=disc.index, name="event"))

    events = pd.concat(frames)
    events = events.to_frame()
    if blocked:
        events = events[~events["event"].isin(blocked)]
    if allowed is not None:
        events = events[events["event"].isin(allowed)]
    return events


def frequent_events(events: pd.DataFrame, min_frequency: int) -> set[str]:
    """Return the set of event keys occurring at least ``min_frequency`` times."""
    counts = events["event"].value_counts()
    return set(counts[counts >= min_frequency].index)
