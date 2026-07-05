"""LSTM / sequential ML challenger — sandbox only, NOT part of FADE core.

Optional rolling-window sequence model (K=12 bars) on path_lean3 atoms plus
return_1h. Uses the same chronological 70/30 holdout protocol as ml_challenger;
holdout is never seen during training. Backend preference: PyTorch, then
Keras/TensorFlow; skipped with a message if neither is available.

This module is explicitly outside the scope-guard "no ML" invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fade.config import Config, lean_config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.targets import score_predictions

from fade.pipeline.ml_challenger import (
    FADE_BASELINE,
    FADE_BASELINE_RANGE,
    HOLDOUT_FRAC,
    N_SHUFFLES,
    P_VALUE_MAX,
    _verdict,
)

WINDOW_K = 12
FEATURE_COLS = ("close_pos", "range_pct", "streak_signed", "return_1h")

# --- backend detection (PyTorch > Keras/TF > unavailable) -------------------
_LSTM_BACKEND: str | None = None
_torch = None
_keras = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _torch = torch
    _LSTM_BACKEND = "pytorch"
except ImportError:
    try:
        import keras
        from keras import layers, models

        _keras = keras
        _LSTM_BACKEND = "keras"
    except ImportError:
        try:
            import tensorflow as tf
            from tensorflow import keras as tf_keras
            from tensorflow.keras import layers, models

            _keras = tf_keras
            _LSTM_BACKEND = "keras"
        except ImportError:
            _LSTM_BACKEND = None


def lstm_backend_available() -> bool:
    return _LSTM_BACKEND is not None


def lstm_backend_name() -> str | None:
    return _LSTM_BACKEND


def _build_sequential_dataset(
    csv_path: str | Path,
    config: Config,
    window: int = WINDOW_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build (n, K, F) windows aligned with next-bar direction labels."""
    df = load_ohlcv(csv_path)
    pool = atoms_mod.compute_atom_pool(df, config)
    feats = pool[list(FEATURE_COLS)].dropna()
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(feats.index)
    valid = fwd.notna()
    feats = feats.loc[valid]
    fwd = fwd.loc[valid]

    arr = feats.to_numpy(dtype=np.float32)
    y = (fwd > 0).astype(np.int64).to_numpy()
    fwd_arr = fwd.to_numpy(dtype=np.float64)

    if len(arr) < window + 20:
        return (
            np.empty((0, window, len(FEATURE_COLS)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            {
                "asset": Path(csv_path).stem.lower(),
                "n_total": len(arr),
                "feature_cols": list(FEATURE_COLS),
                "window": window,
            },
        )

    # Causal windows: each sample uses bars [t-K+1 .. t]; label = direction at t.
    windows = sliding_window_view(arr, (window, arr.shape[1]))[:, 0, :, :]
    y_seq = y[window - 1 :]
    fwd_seq = fwd_arr[window - 1 :]

    meta = {
        "asset": Path(csv_path).stem.lower(),
        "n_total": int(len(y_seq)),
        "feature_cols": list(FEATURE_COLS),
        "window": window,
    }
    return windows.copy(), y_seq, fwd_seq, meta


def _score_holdout(
    pred_hold: np.ndarray,
    y_hold: np.ndarray,
    fwd_hold: np.ndarray,
    config: Config,
    seed: int,
    n_shuffles: int,
) -> dict[str, Any]:
    pred_v, act_v = score_predictions(pred_hold, fwd_hold, config.move_threshold)
    coverage = int(len(pred_v))
    if coverage == 0:
        return {"status": "no_coverage", "n_predictions": 0}

    hit = float(np.mean(pred_v == act_v))
    lift = hit - 0.5

    rng = np.random.default_rng(seed)
    null_hits = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null_hits[i] = np.mean(pred_v == rng.permutation(act_v))
    n_ge = int(np.sum(null_hits >= hit))
    p_value = (1 + n_ge) / (1 + n_shuffles)

    return {
        "status": "ok",
        "n_predictions": coverage,
        "holdout_hit_rate": round(hit, 4),
        "holdout_lift_vs_random": round(lift, 4),
        "null_mean": round(float(np.mean(null_hits)), 4),
        "null_std": round(float(np.std(null_hits)), 4),
        "p_value": round(p_value, 4),
        "verdict": _verdict(lift, p_value),
    }


def _fade_comparison(asset: str, hit: float) -> tuple[str | None, float | None]:
    fade_ref = FADE_BASELINE.get(asset)
    if fade_ref is not None:
        return f"{fade_ref:.2%} (FADE path_lean3 holdout ref)", round(hit - fade_ref, 4)
    return (
        f"{FADE_BASELINE_RANGE[0]:.0%}–{FADE_BASELINE_RANGE[1]:.0%} (FADE typical)",
        None,
    )


# --- PyTorch model -----------------------------------------------------------
def _train_predict_pytorch(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_hold: np.ndarray,
    seed: int,
    epochs: int = 25,
    batch_size: int = 64,
) -> np.ndarray:
    torch = _torch
    nn = torch.nn

    class _LSTMClassifier(nn.Module):
        def __init__(self, n_features: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    torch.manual_seed(seed)
    n_features = X_dev.shape[2]
    model = _LSTMClassifier(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_dev)
    y_t = torch.from_numpy(y_dev.astype(np.float32))
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_hold))
        pred = (torch.sigmoid(logits) >= 0.5).numpy().astype(int)
    return pred


# --- Keras model -------------------------------------------------------------
def _train_predict_keras(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_hold: np.ndarray,
    seed: int,
    epochs: int = 25,
    batch_size: int = 64,
) -> np.ndarray:
    keras = _keras
    layers = keras.layers
    models = keras.models

    n_features = X_dev.shape[2]
    model = models.Sequential(
        [
            layers.Input(shape=(X_dev.shape[1], n_features)),
            layers.LSTM(32),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_dev,
        y_dev,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.0,
    )
    prob = model.predict(X_hold, verbose=0).ravel()
    return (prob >= 0.5).astype(int)


def run_lstm_challenger(
    csv_path: str | Path,
    holdout_frac: float = HOLDOUT_FRAC,
    n_shuffles: int = N_SHUFFLES,
    seed: int = 0,
    config: Config | None = None,
    window: int = WINDOW_K,
    epochs: int = 25,
) -> dict:
    """Train a simple LSTM on dev; score holdout direction predictions."""
    if _LSTM_BACKEND is None:
        return {
            "asset": Path(csv_path).stem.lower(),
            "status": "no_backend",
            "verdict": (
                "SKIP - no PyTorch or Keras/TensorFlow installed. "
                "Install torch (sandbox): pip install torch"
            ),
        }

    config = config or lean_config()
    X, y, fwd, meta = _build_sequential_dataset(csv_path, config, window=window)
    n = len(y)
    split = int(n * (1.0 - holdout_frac))
    if split < 50 or (n - split) < 20:
        return {
            **meta,
            "status": "too_short",
            "verdict": "INCONCLUSIVE - series too short for 70/30 split.",
        }

    X_dev, X_hold = X[:split], X[split:]
    y_dev, y_hold = y[:split], y[split:]
    fwd_hold = fwd[split:]

    if _LSTM_BACKEND == "pytorch":
        pred_hold = _train_predict_pytorch(X_dev, y_dev, X_hold, seed, epochs=epochs)
    else:
        pred_hold = _train_predict_keras(X_dev, y_dev, X_hold, seed, epochs=epochs)

    scored = _score_holdout(pred_hold, y_hold, fwd_hold, config, seed, n_shuffles)
    if scored.get("status") != "ok":
        return {
            **meta,
            **scored,
            "n_dev": split,
            "n_holdout": n - split,
            "verdict": "INCONCLUSIVE - no scorable holdout predictions.",
        }

    asset = meta["asset"]
    fade_note, vs_fade = _fade_comparison(asset, scored["holdout_hit_rate"])

    return {
        **meta,
        **scored,
        "backend": _LSTM_BACKEND,
        "model": "lstm",
        "atomset": "path_lean3+return_1h",
        "n_dev": split,
        "n_holdout": n - split,
        "fade_baseline_ref": fade_note,
        "vs_fade_baseline": vs_fade,
    }


def print_lstm_report(r: dict) -> None:
    line = "=" * 66
    asset = r.get("asset", "?").upper()
    print("\n" + line)
    print(f"FADE LSTM CHALLENGER (SANDBOX - NOT CORE) - {asset}")
    print(line)
    print("  Scope: optional sequential challenger; core inference untouched.")
    print(
        f"  Backend: {r.get('backend', '?')}  |  model: LSTM  |  "
        f"window K={r.get('window', WINDOW_K)}"
    )
    if r.get("feature_cols"):
        print(f"  Features per bar: {', '.join(r['feature_cols'])}")
    if r.get("status") != "ok":
        print(f"\n  {r.get('verdict', r.get('status'))}")
        print(line + "\n")
        return
    print(f"  Split: dev={r['n_dev']} windows  |  holdout={r['n_holdout']} windows (quarantined)")
    print(f"  Holdout predictions (all bars) : {r['n_predictions']}")
    print()
    print(f"  Holdout hit-rate                 : {r['holdout_hit_rate']:.2%}")
    print(f"  Lift vs random                   : {r['holdout_lift_vs_random']:+.4f}")
    print(f"  Shuffle null (mean +/- std)      : {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
    print(f"  Permutation p-value              : {r['p_value']:.4f}")
    print()
    print(f"  FADE rule-based holdout ref      : {r['fade_baseline_ref']}")
    if r.get("vs_fade_baseline") is not None:
        sign = "+" if r["vs_fade_baseline"] >= 0 else ""
        print(f"  LSTM vs FADE baseline            : {sign}{r['vs_fade_baseline']:.4f}")
    print()
    print(f"  VERDICT: {r['verdict']}")
    print(line + "\n")


def print_comparison(gb: dict, lstm: dict) -> None:
    """Side-by-side GB vs LSTM vs FADE baseline."""
    if gb.get("status") != "ok" and lstm.get("status") != "ok":
        return
    line = "-" * 66
    asset = gb.get("asset") or lstm.get("asset", "?")
    print("\n" + line)
    print(f"COMPARISON (SANDBOX - NOT CORE) - {asset.upper()}")
    print(line)
    fade_ref = FADE_BASELINE.get(asset)
    fade_str = f"{fade_ref:.2%}" if fade_ref is not None else "53–54% (typical)"
    print(f"  {'Model':<22} {'Holdout hit':>12}  {'vs FADE ref':>12}  {'p-value':>10}")
    print(f"  {'-' * 22} {'-' * 12}  {'-' * 12}  {'-' * 10}")
    if gb.get("status") == "ok":
        vs = gb.get("vs_fade_baseline")
        vs_s = f"{vs:+.4f}" if vs is not None else "n/a"
        print(
            f"  {'GradientBoosting':<22} {gb['holdout_hit_rate']:>11.2%}  "
            f"{vs_s:>12}  {gb['p_value']:>10.4f}"
        )
    if lstm.get("status") == "ok":
        vs = lstm.get("vs_fade_baseline")
        vs_s = f"{vs:+.4f}" if vs is not None else "n/a"
        print(
            f"  {'LSTM (K=12 seq)':<22} {lstm['holdout_hit_rate']:>11.2%}  "
            f"{vs_s:>12}  {lstm['p_value']:>10.4f}"
        )
    print(f"  {'FADE rules (ref)':<22} {fade_str:>12}  {'n/a':>12}  {'n/a':>10}")
    print(line + "\n")
