"""Tiny on-disk cache for intermediate artifacts.

Design goals (per FADE core principles):
- Avoid recomputation of deterministic steps (e.g. atom features).
- Store only essential intermediate artifacts, keyed by a content hash.
- Never persist full raw datasets repeatedly; callers pass compact keys.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def hash_key(*parts: Any) -> str:
    """Build a stable short hash from arbitrary hashable/serialisable parts."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, pd.DataFrame):
            # Hash shape + a cheap fingerprint of the values, not the whole frame.
            h.update(str(p.shape).encode())
            h.update(pd.util.hash_pandas_object(p, index=True).values.tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:16]


class DiskCache:
    """Pickle-backed cache. Minimal by design."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        return None

    def set(self, key: str, value: Any) -> None:
        with self._path(key).open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def memoize(self, key: str, fn: Callable[[], Any]) -> Any:
        """Return cached value for ``key`` or compute, store, and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fn()
        self.set(key, value)
        return value

    def clear(self) -> None:
        """Remove all cached artifacts (cleanup policy helper)."""
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink()
