"""Central configuration for FADE v0.1.

All tunable knobs live here so runs are reproducible and the anti-overfitting
thresholds are visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Core atom set (the original strict five).
ATOM_COLUMNS = (
    "return_1h",
    "return_6h",
    "volatility",
    "volume_zscore",
    "trend_slope",
)

# Named atom sets for the grid search (all OHLCV-only, see core/atoms.py pool).
ATOM_SETS = {
    "core5": ATOM_COLUMNS,
    "plus7": ATOM_COLUMNS + ("range_pct", "close_pos"),
    "full9": ATOM_COLUMNS + ("range_pct", "close_pos", "return_accel", "volume_trend"),
    # Path-aware sets: inject the streak (trajectory) mechanism into the engine's
    # vocabulary so it can express the reversal structure the state atoms cannot.
    "core5_path": ATOM_COLUMNS + ("streak_signed",),
    "path_min": ("return_1h", "volatility", "volume_zscore", "streak_signed"),
    # lean_search winner on btc_1h holdout (54.64%, 3 atoms); not yet validated cross-asset
    "path_lean3": ("close_pos", "range_pct", "streak_signed"),
    # magnitude-conditioned path atom variants
    "path_big": ("close_pos", "range_pct", "streak_big"),
    "path_both": ("close_pos", "range_pct", "streak_signed", "streak_big"),
    # path_lean3 + named candle flags (holdout research only, batch 29)
    "path_candles": (
        "close_pos",
        "range_pct",
        "streak_signed",
        "doji",
        "bullish_engulfing",
        "bearish_engulfing",
    ),
}

# Default lean set (validated holdout winner, batch 17).
DEFAULT_ATOM_SET = "path_lean3"


def lean_config(atomset: str | None = None) -> Config:
    """Config with a named atom set from ATOM_SETS (default: path_lean3)."""
    import dataclasses
    name = atomset or DEFAULT_ATOM_SET
    return dataclasses.replace(Config(), atom_columns=ATOM_SETS[name])


@dataclass(frozen=True)
class Config:
    # --- active feature set (swappable for the grid search) ---
    atom_columns: tuple[str, ...] = ATOM_COLUMNS

    # --- horizons / windows (all expressed in 1H bars) ---
    return_6h_window: int = 6
    volatility_window: int = 24
    volume_window: int = 24
    trend_window: int = 12          # within the requested 6-24h band
    forward_horizon: int = 1        # bars ahead. Horizon sweep (holdout, all of
                                    # 15m/30m/1h) showed 1 bar has the highest
                                    # drift-adjusted skill; edge decays with horizon.

    # Minimum |forward return| to count as a material move (0 = any direction).
    # Mining excludes neutral bars; evaluation scores only material outcomes.
    move_threshold: float = 0.0

    # --- event construction ---
    event_sizes: tuple[int, ...] = (2, 3)   # combine 2-4 atom states
    quantile_low: float = 1.0 / 3.0
    quantile_high: float = 2.0 / 3.0

    # Structural (non-quantile) LOW/HIGH cut points for atoms whose meaning is
    # discrete, not distributional. streak_signed uses fixed +/-2 so that
    # LOW = down-run>=2, HIGH = up-run>=2, MID = short/flat (-1,0,+1). This keeps
    # the reversal-carrying long-run buckets pure instead of letting a quantile
    # split merge single bars with deep streaks. Fixed cuts are a definition, not
    # data-fit, so they introduce no look-ahead.
    atom_fixed_thresholds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "streak_signed": (-2.0, 2.0),
            "streak_big": (-2.0, 2.0),
            # Binary candle flags: LOW=absent (0), HIGH=present (1).
            "doji": (0.5, 0.5),
            "bullish_engulfing": (0.5, 0.5),
            "bearish_engulfing": (0.5, 0.5),
        }
    )

    # streak_big: a bar counts as a "big" move when |return| > k * trailing vol.
    streak_big_k: float = 1.0

    # --- anti-overfitting thresholds ---
    min_event_frequency: int = 25   # rare events are discarded (noise filter)
    min_support: int = 25           # min occurrences for a rule to be kept
    min_confidence: float = 0.55    # directional edge required to keep a rule

    # --- rule decay (older evidence weighs less) ---
    decay_half_life_h: float = 720.0  # 30 days; None-like disabled if <= 0

    # --- walk-forward backtest ---
    n_folds: int = 5
    initial_train_frac: float = 0.4  # expanding window start size

    # --- stability / memory promotion ---
    stability_min_folds: int = 3        # must appear in >= this many folds
    stability_min_consistency: float = 0.75  # directional agreement across folds
    negative_min_failures: int = 2      # failures before blacklisting a pattern

    # --- significant change detection (from OHLCV only) ---
    price_shock_sigma: float = 2.5   # |return_1h| > sigma * volatility
    volume_shock_z: float = 2.0      # |volume_zscore| > threshold
    post_shock_bars: int = 4         # POST_SHOCK regime length after a shock

    # --- volatility ratio (VR) regime gate ---
    # short_vol / long_vol on 1h returns. LOW_VR (compressing vol) is mean-reversion
    # friendly; HIGH_VR (expanding vol) favours momentum/trend. Thresholds are
    # optional fixed cuts; leave None to fit on dev only (see vr_gate_test).
    vol_ratio_short_window: int = 24    # hours
    vol_ratio_long_window: int = 168    # hours (1 week)
    vr_low_threshold: float | None = None   # VR <= this -> LOW_VR
    vr_high_threshold: float | None = None  # VR >= this -> HIGH_VR

    # --- historical replay ---
    replay_windows: int = 8

    # --- regime-weighted forecasting ---
    # Rescales forecast confidence by a regime's OOS reliability. DISABLED by
    # default: a chronological split test (fade.pipeline.regime_eval) showed
    # regime reliability is NOT stable across time (POST_SHOCK's edge did not
    # persist), so weighting worsened Brier/ECE on BTC and ETH. Kept as an
    # opt-in knob for future research, not shipped as a live feature.
    regime_weighting_enabled: bool = False

    # --- paths ---
    root: Path = field(default=Path(__file__).resolve().parent)

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"
