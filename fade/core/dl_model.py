"""Deep-learning forecast model — core inference (PyTorch LSTM).

Causal constraints
------------------
* Features at bar T use only data available at T (atoms + forward-filled news).
* Labels are the next bar's direction (forward_horizon).
* Chronological split: train on the first (1 - holdout_frac), score the rest.
* No look-ahead: news is daily and forward-filled onto later hours only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fade.config import Config, lean_config
from fade.core import atoms as atoms_mod
from fade.core.data_loader import load_ohlcv
from fade.core.news_features import attach_news_to_pool
from fade.core.targets import score_predictions
from fade.utils.logging import get_logger

log = get_logger("dl_model")

_DL_BACKEND: str | None = None
_torch = None
_nn = None
_DataLoader = None
_TensorDataset = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _torch = torch
    _nn = nn
    _DataLoader = DataLoader
    _TensorDataset = TensorDataset
    _DL_BACKEND = "pytorch"
except ImportError:
    _DL_BACKEND = None


def dl_backend_available() -> bool:
    return _DL_BACKEND is not None


def dl_backend_name() -> str | None:
    return _DL_BACKEND


DEFAULT_WINDOW_K = 12
DEFAULT_FEATURE_COLS = ("close_pos", "range_pct", "streak_signed", "return_1h")


def _model_path(memory_dir: Path, asset: str) -> Path:
    return Path(memory_dir) / f"dl_model_{asset}.json"



if _nn is not None:

    class LSTMClassifier(_nn.Module):
        """Single-layer LSTM + linear head → logit for P(up)."""

        def __init__(self, n_features: int, hidden: int = 32):
            super().__init__()
            self.lstm = _nn.LSTM(n_features, hidden, batch_first=True)
            self.head = _nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (batch, window, n_features)
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)

else:  # pragma: no cover — backend missing

    class LSTMClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed")


def _score_holdout(
    pred_score: np.ndarray | None,
    fwd_score: np.ndarray,
    config: Config,
    seed: int,
    n_shuffles: int,
) -> dict[str, Any]:
    if pred_score is None:
        return {"status": "no_predictions", "n_predictions": 0, "verdict": "INCONCLUSIVE"}
    pred_v, act_v = score_predictions(pred_score, fwd_score, config.move_threshold)
    coverage = int(len(pred_v))
    if coverage == 0:
        return {"status": "no_coverage", "n_predictions": 0, "verdict": "INCONCLUSIVE"}

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
    }

def build_sequential_dataset(
    csv_path: str | Path,
    config: Config,
    feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLS,
    window: int = DEFAULT_WINDOW_K,
    news_csv: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build (X, y, fwd, meta) with causal windows.

    X shape: (n_windows, window, n_features)
    y: direction of the *last* bar in each window's next-horizon return.
    """
    df = load_ohlcv(csv_path)
    pool = atoms_mod.compute_atom_pool(df, config)

    # Carry over news columns already present in the source (e.g. a CSV that
    # was pre-merged via attach_news_to_pool before being saved) — the atom
    # pool above is rebuilt from OHLCV only and would otherwise drop them.
    news_cols = ("news_tone", "news_tone_chg", "news_vol_z")
    for col in news_cols:
        if col in df.columns and col not in pool.columns:
            pool[col] = df[col].to_numpy()

    if news_csv:
        pool = attach_news_to_pool(pool, str(news_csv))

    available = [c for c in feature_cols if c in pool.columns]
    if not available:
        empty_meta = {"asset": Path(csv_path).stem.lower(), "n_total": 0}
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            empty_meta,
        )

    feats = pool[available].dropna()
    fwd = atoms_mod.forward_return(df, config.forward_horizon).reindex(feats.index)
    valid = fwd.notna()
    feats = feats.loc[valid]
    fwd = fwd.loc[valid]

    arr = feats.to_numpy(dtype=np.float32)
    y = (fwd > 0).astype(np.int64).to_numpy()
    fwd_arr = fwd.to_numpy(dtype=np.float64)
    asset = Path(csv_path).stem.lower()
    meta = {"asset": asset, "n_total": int(len(arr)), "feature_cols_used": available}

    if len(arr) < window + 20:
        return (
            np.empty((0, window, len(available)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            meta,
        )

    windows = sliding_window_view(arr, (window, arr.shape[1]))[:, 0, :, :]
    # Label of window ending at i is the forward return *after* bar i.
    y_win = y[window - 1:]
    fwd_win = fwd_arr[window - 1:]
    # Drop the last window — its forward return would require a future bar.
    if len(windows) > 1:
        windows, y_win, fwd_win = windows[:-1], y_win[:-1], fwd_win[:-1]
    return windows.copy(), y_win, fwd_win, meta
def _train_predict_pytorch(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_score: np.ndarray,
    seed: int,
    epochs: int = 25,
) -> np.ndarray:
    """Train LSTM on (X_dev, y_dev) and return binary predictions for X_score."""
    if _torch is None:
        raise RuntimeError("PyTorch is not installed")
    if len(X_dev) == 0 or len(X_score) == 0:
        return np.zeros(len(X_score), dtype=int)

    torch = _torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMClassifier(X_dev.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    x_t = torch.from_numpy(np.ascontiguousarray(X_dev))
    y_t = torch.from_numpy(y_dev.astype(np.float32))
    loader = _DataLoader(_TensorDataset(x_t, y_t), batch_size=64, shuffle=True)

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        xs = torch.from_numpy(np.ascontiguousarray(X_score))
        probs = torch.sigmoid(model(xs)).cpu().numpy()
    return (probs >= 0.5).astype(int)


def train_and_evaluate(
    csv_path: str | Path,
    config: Config | None = None,
    holdout_frac: float = 0.30,
    n_shuffles: int = 300,
    seed: int = 0,
    feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLS,
    window: int = DEFAULT_WINDOW_K,
    news_csv: str | Path | None = None,
    epochs: int = 25,
) -> dict[str, Any]:
    if not dl_backend_available():
        return {"status": "no_backend", "verdict": "SKIP - PyTorch not installed."}
    config = config or lean_config()
    X, y, fwd, meta = build_sequential_dataset(
        csv_path, config, feature_cols, window, news_csv
    )
    if X.shape[0] == 0:
        return {**meta, "status": "too_short", "verdict": "INCONCLUSIVE"}

    n = len(X)
    if holdout_frac <= 0:
        pred_last = _train_predict_pytorch(X, y, X[-1:], seed, epochs)
        direction = "UP" if int(pred_last[0]) == 1 else "DOWN"
        return {
            **meta,
            "status": "ok",
            "backend": _DL_BACKEND,
            "model": "lstm",
            "n_dev": n,
            "n_holdout": 0,
            "feature_cols": list(feature_cols),
            "window": window,
            "news_csv": str(news_csv) if news_csv else None,
            "seed": seed,
            "epochs": epochs,
            "last_direction": direction,
            "direction": direction,
            "raw_prob_pct": 50.0,
        }

    split = int(n * (1.0 - holdout_frac))
    if split < 50 or (n - split) < 20:
        return {**meta, "status": "too_short", "verdict": "INCONCLUSIVE"}

    pred_hold = _train_predict_pytorch(X[:split], y[:split], X[split:], seed, epochs)
    scored = _score_holdout(pred_hold, fwd[split:], config, seed, n_shuffles)
    pred_all = _train_predict_pytorch(X, y, X[-1:], seed, epochs)
    last_dir = "UP" if int(pred_all[0]) == 1 else "DOWN"
    return {
        **meta,
        **scored,
        "backend": _DL_BACKEND,
        "model": "lstm",
        "n_dev": split,
        "n_holdout": n - split,
        "feature_cols": list(feature_cols),
        "window": window,
        "news_csv": str(news_csv) if news_csv else None,
        "seed": seed,
        "epochs": epochs,
        "last_direction": last_dir,
        "direction": last_dir,
    }


def persist_record(record: dict[str, Any], memory_dir: Path) -> Path:
    path = _model_path(memory_dir, record.get("asset", "synthetic"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path


def load_record(memory_dir: Path, asset: str) -> dict[str, Any] | None:
    path = _model_path(memory_dir, asset)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def forecast_latest(
    csv_path: str | Path,
    config: Config | None = None,
    feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLS,
    window: int = DEFAULT_WINDOW_K,
    news_csv: str | Path | None = None,
    seed: int = 0,
    epochs: int = 25,
) -> dict[str, Any]:
    if not dl_backend_available():
        return {"status": "no_backend", "verdict": "SKIP - PyTorch not installed."}
    return train_and_evaluate(
        csv_path,
        config,
        holdout_frac=0.0,
        seed=seed,
        feature_cols=feature_cols,
        window=window,
        news_csv=news_csv,
        epochs=epochs,
    )