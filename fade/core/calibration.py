"""Frequency-based probability calibration (no ML).

Raw rule-vote probabilities are binned; each bin tracks how often the
predicted direction was correct across runs. As more outcomes accumulate,
bucket hit-rates converge to calibrated probabilities.

This is the mechanism that "improves with more attempts" while staying
within FADE's rule-accumulation philosophy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Fixed bins over [0.50, 1.00] — pre-registered, not tuned post-hoc.
BIN_WIDTH = 0.05
BIN_EDGES = [round(0.50 + i * BIN_WIDTH, 2) for i in range(11)]  # 0.50 .. 1.00


def _bin_index(prob: float) -> int:
    """Map a probability in [0.5, 1.0] to a bin index."""
    p = float(np.clip(prob, 0.5, 0.9999))
    return min(int((p - 0.5) / BIN_WIDTH), len(BIN_EDGES) - 2)


class CalibrationStore:
    """Persistent reliability-table calibration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if not data.get("bins"):
                    data["bins"] = self._empty_bins()
                return data
            except json.JSONDecodeError:
                pass
        return {"bins": self._empty_bins(), "runs": 0, "history": []}

    def _ensure_bins(self) -> None:
        if not self.data.get("bins"):
            self.data["bins"] = self._empty_bins()

    @staticmethod
    def _empty_bins() -> list[dict]:
        bins = []
        for i in range(len(BIN_EDGES) - 1):
            bins.append({
                "lo": BIN_EDGES[i],
                "hi": BIN_EDGES[i + 1],
                "hits": 0,
                "total": 0,
            })
        return bins

    def calibrate(self, raw_prob: float) -> float:
        """Map raw probability to calibrated estimate (Laplace-smoothed)."""
        self._ensure_bins()
        idx = _bin_index(raw_prob)
        b = self.data["bins"][idx]
        # Laplace smoothing: never report 0% or 100% from sparse bins.
        return (b["hits"] + 1) / (b["total"] + 2)

    def update(self, samples: list[tuple[float, int]]) -> dict:
        """Ingest (raw_prob, correct) pairs from a completed run.

        Returns calibration metrics for this run (ECE, Brier).
        """
        self._ensure_bins()
        ece_num, ece_den = 0.0, 0
        brier_sum, n = 0.0, 0

        for raw_prob, correct in samples:
            idx = _bin_index(raw_prob)
            b = self.data["bins"][idx]
            b["total"] += 1
            b["hits"] += int(correct)

            cal = self.calibrate(raw_prob)
            ece_num += abs(cal - raw_prob)
            ece_den += 1
            brier_sum += (cal - correct) ** 2
            n += 1

        ece = ece_num / ece_den if ece_den else float("nan")
        brier = brier_sum / n if n else float("nan")

        self.data["runs"] += 1
        self.data["history"].append({
            "run": self.data["runs"],
            "samples": n,
            "ece": round(ece, 4) if ece == ece else None,
            "brier": round(brier, 4) if brier == brier else None,
        })
        # Keep only last 20 run summaries (cleanup policy).
        self.data["history"] = self.data["history"][-20:]
        return {"ece": ece, "brier": brier, "samples": n}

    def summary(self) -> dict:
        """Aggregate calibration state for reporting."""
        bins = self.data["bins"]
        filled = [b for b in bins if b["total"] > 0]
        return {
            "runs": self.data["runs"],
            "total_samples": sum(b["total"] for b in bins),
            "filled_bins": len(filled),
            "history": self.data["history"],
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))
