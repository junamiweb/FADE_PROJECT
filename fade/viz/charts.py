"""Minimal research plots for FADE v0.2 (matplotlib only).

Three charts:
  1. Price + significant changes (recent window)
  2. Calibration reliability diagram
  3. Replay lift over expanding windows

Kid mode (--kid): Hebrew labels, star rankings, simpler vocabulary.
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
from fade.core.conviction import read_conviction_state
from fade.core.data_loader import load_ohlcv
from fade.core.significant_changes import detect_significant_changes
from fade.pipeline.replay import run_replay
from fade.viz import kid_labels as kid

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


def _change_legend(ctype: str, kid_friendly: bool) -> str:
    if kid_friendly:
        return kid.he(kid.CHANGE_TYPE_HE.get(ctype, ctype))
    return ctype


def plot_price_shocks(
    df: pd.DataFrame,
    changes: pd.DataFrame,
    asset: str,
    out_path: Path,
    last_bars: int = 336,
    kid_friendly: bool = False,
) -> Path:
    """Price line with shock markers on the most recent window."""
    if kid_friendly:
        kid.apply_kid_font()

    window = df.iloc[-last_bars:]
    ch = changes.reindex(window.index)
    fig, ax = plt.subplots(figsize=(12, 5))
    close_label = kid.he("מחיר") if kid_friendly else "close"
    ax.plot(window.index, window["close"], color="#2c3e50", linewidth=1.2, label=close_label)

    for ctype, color in CHANGE_COLORS.items():
        mask = ch["change_type"] == ctype
        if mask.any():
            ax.scatter(
                window.index[mask],
                window.loc[mask, "close"],
                s=28 if kid_friendly else 18,
                c=color,
                alpha=0.85,
                label=_change_legend(ctype, kid_friendly),
                zorder=3,
            )

    if kid_friendly:
        name = kid.asset_name_he(asset)
        ax.set_title(kid.he(f"מה קרה לאחרונה ב{name}?"))
        ax.set_ylabel(kid.he("מחיר"))
        ax.set_xlabel(kid.he("זמן"))
    else:
        ax.set_title(f"{asset.upper()} — price + significant changes (last {last_bars} bars)")
        ax.set_ylabel("close")
    ax.legend(loc="upper left", fontsize=9 if kid_friendly else 7, ncol=2 if kid_friendly else 3)
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
    kid_friendly: bool = False,
) -> Path | None:
    """Reliability diagram: bin midpoint vs observed hit-rate."""
    if kid_friendly:
        kid.apply_kid_font()

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

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, label=kid.he("מושלם") if kid_friendly else "perfect")
    ax.scatter(mids, obs, s=[min(220, 30 + n) for n in sizes], alpha=0.85, c="#8e44ad")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.02)
    if kid_friendly:
        ax.set_xlabel(kid.he("מה חשבנו שיקרה (%)"))
        ax.set_ylabel(kid.he("מה באמת קרה (%)"))
        ax.set_title(kid.he(f"{kid.asset_name_he(asset)} — האם החיזוי שלנו מדויק?"))
    else:
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
    kid_friendly: bool = False,
) -> Path | None:
    """Bar chart of lift vs random across replay windows."""
    if kid_friendly:
        kid.apply_kid_font()

    windows = replay_result.get("windows") or []
    if not windows:
        return None

    if kid_friendly:
        ranked = sorted(
            [w for w in windows if w.get("lift_vs_random") is not None],
            key=lambda w: w["lift_vs_random"],
            reverse=True,
        )
        labels = [kid.he(kid.rank_label(i)) for i in range(1, len(ranked) + 1)]
        lifts = [w["lift_vs_random"] for w in ranked]
        bar_labels = [kid.he(kid.lift_label(l)) for l in lifts]
    else:
        ranked = windows
        labels = [str(w["window"]) for w in windows]
        lifts = [w["lift_vs_random"] if w["lift_vs_random"] is not None else 0 for w in windows]
        bar_labels = None

    colors = ["#27ae60" if l >= 0 else "#e74c3c" for l in lifts]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, lifts, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    if kid_friendly:
        ax.set_xlabel(kid.he("דירוג תקופות (מהטוב לפחות טוב)"))
        ax.set_ylabel(kid.he("כמה זה טוב יותר מניחוש אקראי"))
        ax.set_title(kid.he(f"{kid.asset_name_he(asset)} — איזו תקופה הייתה הכי מוצלחת?"))
        for bar, note in zip(bars, bar_labels or []):
            height = bar.get_height()
            y = height + 0.003 if height >= 0 else height - 0.008
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                note,
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=11,
            )
    else:
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


def plot_tier_ranking(
    asset: str,
    out_path: Path,
    primary_csv: str | None = None,
    kid_friendly: bool = True,
) -> Path:
    """Horizontal bar chart ranking conviction tiers for kids."""
    if kid_friendly:
        kid.apply_kid_font()

    y_labels = []
    hits = []
    colors = []
    active_id = None
    if primary_csv:
        state = read_conviction_state(primary_csv)
        active_id = (state.get("active_tier") or {}).get("id")

    for tid, label, star_count, _ in kid.TIER_RANK_HE:
        y_labels.append(kid.he(kid.tier_bar_label(tid, label, star_count)))
        hits.append(kid.TIER_HIT_PCT[tid])
        if tid == active_id:
            colors.append("#f39c12")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(y_labels))
    bars = ax.barh(y_pos, hits, color=colors, alpha=0.9, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlim(50, 65)
    if kid_friendly:
        ax.set_xlabel(kid.he("כמה פעמים צדקנו (מתוך 100)"))
        ax.set_title(kid.he(f"{kid.asset_name_he(asset)} — דירוג רמות החיזוי (מהחזק לחלש)"))
        if active_id:
            ax.text(
                0.98, 0.02,
                kid.he("הרמה המודגשת = הרמה הפעילה עכשיו"),
                transform=ax.transAxes,
                ha="right",
                fontsize=11,
                color="#d35400",
            )
    else:
        ax.set_xlabel("calibrated hit-rate (%)")
        ax.set_title(f"{asset.upper()} — conviction tier ranking")
    for bar, pct in zip(bars, hits):
        ax.text(
            pct + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.0f}%",
            va="center",
            fontsize=11,
        )
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_charts(
    csv_path: str,
    config: Config | None = None,
    last_bars: int = 336,
    kid_friendly: bool = False,
) -> dict:
    """Build all charts for an asset; return output paths."""
    config = config or Config()
    asset = Path(csv_path).stem.lower()
    out_dir = _ensure_dir(config.output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "_kid" if kid_friendly else ""

    df = load_ohlcv(csv_path)
    atoms = atoms_mod.compute_atoms(df, config)
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(atoms.index)
    changes = detect_significant_changes(atoms, fwd, config)
    calibration = CalibrationStore(config.memory_dir / f"calibration_{asset}.json")

    paths: dict[str, str | None] = {}
    paths["price_shocks"] = str(
        plot_price_shocks(
            df, changes, asset,
            out_dir / f"{asset}_{stamp}{suffix}_price_shocks.png",
            last_bars=last_bars,
            kid_friendly=kid_friendly,
        )
    )
    cal_path = plot_calibration(
        calibration, asset,
        out_dir / f"{asset}_{stamp}{suffix}_calibration.png",
        kid_friendly=kid_friendly,
    )
    paths["calibration"] = str(cal_path) if cal_path else None

    replay = run_replay(csv_path, config)
    rep_path = plot_replay_lift(
        replay, asset,
        out_dir / f"{asset}_{stamp}{suffix}_replay_lift.png",
        kid_friendly=kid_friendly,
    )
    paths["replay_lift"] = str(rep_path) if rep_path else None

    if kid_friendly:
        paths["tier_ranking"] = str(
            plot_tier_ranking(
                asset,
                out_dir / f"{asset}_{stamp}{suffix}_tier_ranking.png",
                primary_csv=csv_path,
                kid_friendly=True,
            )
        )

    return {"asset": asset, "output_dir": str(out_dir), "charts": paths, "kid_friendly": kid_friendly}
