"""Minimal research plots for FADE v0.2 (matplotlib only).

Three charts:
  1. Price + significant changes (recent window)
  2. Calibration reliability diagram
  3. Replay lift over expanding windows
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fade.config import Config
from fade.core import atoms as atoms_mod
from fade.core.calibration import CalibrationStore
from fade.core.data_loader import load_ohlcv
from fade.core.significant_changes import detect_significant_changes
from fade.pipeline.replay import run_replay

CHANGE_COLORS = {
    "PRICE_UP": "#2ecc71",
    "PRICE_DOWN": "#e74c3c",
    "VOLUME_SPIKE": "#3498db",
    "COMBINED_UP": "#27ae60",
    "COMBINED_DOWN": "#c0392b",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_price_shocks(
    df: pd.DataFrame,
    changes: pd.DataFrame,
    asset: str,
    out_path: Path,
    last_bars: int = 336,
) -> Path:
    """Price line with shock markers on the most recent window."""
    window = df.iloc[-last_bars:]
    ch = changes.reindex(window.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(window.index, window["close"], color="#2c3e50", linewidth=0.9, label="close")

    for ctype, color in CHANGE_COLORS.items():
        mask = ch["change_type"] == ctype
        if mask.any():
            ax.scatter(
                window.index[mask],
                window.loc[mask, "close"],
                s=18,
                c=color,
                alpha=0.75,
                label=ctype,
                zorder=3,
            )

    ax.set_title(f"{asset.upper()} — price + significant changes (last {last_bars} bars)")
    ax.set_ylabel("close")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_calibration(
    calibration: CalibrationStore,
    asset: str,
    out_path: Path,
) -> Path | None:
    """Reliability diagram: bin midpoint vs observed hit-rate."""
    bins = calibration.data.get("bins") or []
    mids, obs, sizes = [], [], []
    for b in bins:
        if b["total"] < 5:
            continue
        mid = (b["lo"] + b["hi"]) / 2
        rate = b["hits"] / b["total"]
        mids.append(mid)
        obs.append(rate)
        sizes.append(b["total"])

    if not mids:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, label="perfect")
    ax.scatter(mids, obs, s=[min(200, 20 + n) for n in sizes], alpha=0.8, c="#8e44ad")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("predicted probability (bin midpoint)")
    ax.set_ylabel("observed hit-rate")
    ax.set_title(f"{asset.upper()} — calibration reliability (n={sum(sizes)})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_replay_lift(
    replay_result: dict,
    asset: str,
    out_path: Path,
) -> Path | None:
    """Bar chart of lift vs random across replay windows."""
    windows = replay_result.get("windows") or []
    if not windows:
        return None

    labels = [str(w["window"]) for w in windows]
    lifts = [w["lift_vs_random"] if w["lift_vs_random"] is not None else 0 for w in windows]
    colors = ["#27ae60" if l >= 0 else "#e74c3c" for l in lifts]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, lifts, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("replay window")
    ax.set_ylabel("lift vs random")
    trend = replay_result.get("lift_trend")
    title = f"{asset.upper()} — replay lift"
    if trend is not None:
        title += f"  (trend {trend:+.4f})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_charts(
    csv_path: str,
    config: Config | None = None,
    last_bars: int = 336,
) -> dict:
    """Build all three charts for an asset; return output paths."""
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    out_dir = _ensure_dir(config.output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    changes = detect_significant_changes(atoms, fwd, config)
    calibration = CalibrationStore(config.memory_dir / f"calibration_{asset}.json")

    paths: dict[str, str | None] = {}
    paths["price_shocks"] = str(
        plot_price_shocks(
            df, changes, asset,
            out_dir / f"{asset}_{stamp}_price_shocks.png",
            last_bars=last_bars,
        )
    )
    cal_path = plot_calibration(
        calibration, asset, out_dir / f"{asset}_{stamp}_calibration.png"
    )
    paths["calibration"] = str(cal_path) if cal_path else None

    replay = run_replay(csv_path, config)
    rep_path = plot_replay_lift(
        replay, asset, out_dir / f"{asset}_{stamp}_replay_lift.png"
    )
    paths["replay_lift"] = str(rep_path) if rep_path else None

    return {"asset": asset, "output_dir": str(out_dir), "charts": paths}
