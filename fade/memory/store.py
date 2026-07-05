"""Two-layer persistent memory (per-asset in v0.2).

positive_rules_{asset}.json   -> validated stable rules
negative_patterns_{asset}.json -> anti-patterns (checked BEFORE mining)

Legacy shared files (positive_rules.json) are migrated once on first load.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class MemoryStore:
    def __init__(self, memory_dir: str | Path, asset: str | None = None) -> None:
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.asset = asset.lower() if asset else None

        if self.asset:
            self.positive_path = self.dir / f"positive_rules_{self.asset}.json"
            self.negative_path = self.dir / f"negative_patterns_{self.asset}.json"
            self._migrate_legacy_if_needed()
        else:
            self.positive_path = self.dir / "positive_rules.json"
            self.negative_path = self.dir / "negative_patterns.json"

        self.positive: dict[str, dict] = self._load(self.positive_path)
        self.negative: dict[str, dict] = self._load(self.negative_path)

    def _migrate_legacy_if_needed(self) -> None:
        """Copy legacy shared memory into per-asset files if new files absent."""
        legacy_pos = self.dir / "positive_rules.json"
        legacy_neg = self.dir / "negative_patterns.json"
        if not self.positive_path.exists() and legacy_pos.exists():
            shutil.copy2(legacy_pos, self.positive_path)
        if not self.negative_path.exists() and legacy_neg.exists():
            shutil.copy2(legacy_neg, self.negative_path)

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def blocked_events(self) -> set[str]:
        return {e for e, v in self.negative.items() if v.get("blacklisted")}

    def record_failure(self, event: str, reason: str, min_failures: int) -> None:
        entry = self.negative.get(event, {"failures": 0, "reason": reason})
        entry["failures"] += 1
        entry["reason"] = reason
        entry["blacklisted"] = entry["failures"] >= min_failures
        self.negative[event] = entry

    def upsert_positive(self, event: str, stats: dict) -> None:
        self.positive[event] = stats

    def prune_positive(self, valid_events: set[str]) -> None:
        self.positive = {k: v for k, v in self.positive.items() if k in valid_events}

    def save(self) -> None:
        self.positive_path.write_text(
            json.dumps(self.positive, indent=2, sort_keys=True)
        )
        self.negative_path.write_text(
            json.dumps(self.negative, indent=2, sort_keys=True)
        )
