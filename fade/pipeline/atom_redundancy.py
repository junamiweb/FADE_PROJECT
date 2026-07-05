"""Atom redundancy and effective dimensionality analysis.

Empirically quantifies how collinear the 9 candidate atoms are and identifies
an orthogonal subset. Core evidence for the hypothesis that the full pool spans
only ~2 independent dimensions (momentum/returns vs volatility/range).

Run from repo root:
    python -m fade.pipeline.atom_redundancy
    python -m fade.pipeline.atom_redundancy --csv btc_1h.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fade.config import Config
from fade.core.atoms import compute_atom_pool, forward_return
from fade.core.data_loader import load_ohlcv

try:
    from sklearn.feature_selection import mutual_info_classif

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# The nine candidate atoms under test (excludes later pool additions).
ATOM_POOL_9 = (
    "return_1h",
    "return_6h",
    "volatility",
    "volume_zscore",
    "trend_slope",
    "range_pct",
    "close_pos",
    "return_accel",
    "volume_trend",
)

CORR_THRESHOLD = 0.6
PCA_VAR_THRESHOLD = 0.95
MI_BINS = 20


def _quantile_discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Map continuous values to integer bins via quantile edges."""
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(len(x), dtype=int)
    # digitize against interior edges; cap at n_bins - 1
    bins = np.digitize(x, edges[1:-1], right=False)
    return np.minimum(bins, len(edges) - 2)


def _mi_from_contingency(counts: np.ndarray) -> float:
    """Mutual information (nats) from a contingency table."""
    joint = counts.astype(float)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(joint / (px * py))
        log_ratio = np.where(joint > 0, log_ratio, 0.0)
    return float(np.sum(joint * log_ratio))


def _histogram_mi_univariate(x: np.ndarray, y: np.ndarray, n_bins: int = MI_BINS) -> float:
    """Histogram-based MI between one continuous feature and discrete target."""
    x_disc = _quantile_discretize(x, n_bins)
    y_disc = y.astype(int)
    n_x = int(x_disc.max()) + 1
    n_y = int(y_disc.max()) + 1
    counts = np.zeros((n_x, n_y))
    for xi, yi in zip(x_disc, y_disc, strict=True):
        counts[xi, yi] += 1
    return _mi_from_contingency(counts)


def _histogram_mi_multivariate(
    features: np.ndarray, y: np.ndarray, n_bins: int = MI_BINS
) -> float:
    """Joint MI between multiple continuous features and a discrete target."""
    y_disc = y.astype(int)
    disc_cols = [_quantile_discretize(features[:, j], n_bins) for j in range(features.shape[1])]
    disc = np.column_stack(disc_cols)
    # Flatten multi-dimensional bins into a single index.
    multipliers = np.cumprod([1] + [n_bins] * (disc.shape[1] - 1))[: disc.shape[1]]
    flat_x = (disc * multipliers).sum(axis=1)
    n_x = int(flat_x.max()) + 1
    n_y = int(y_disc.max()) + 1
    counts = np.zeros((n_x, n_y))
    for xi, yi in zip(flat_x, y_disc, strict=True):
        counts[xi, yi] += 1
    return _mi_from_contingency(counts)


def _mi_per_atom_sklearn(X: np.ndarray, y: np.ndarray, columns: list[str]) -> pd.Series:
    scores = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return pd.Series(scores, index=columns).sort_values(ascending=False)


def _mi_per_atom_histogram(X: np.ndarray, y: np.ndarray, columns: list[str]) -> pd.Series:
    scores = {col: _histogram_mi_univariate(X[:, i], y) for i, col in enumerate(columns)}
    return pd.Series(scores).sort_values(ascending=False)


def _joint_mi(
    atoms: pd.DataFrame, selected: list[str], y: np.ndarray, n_bins: int = MI_BINS
) -> float:
    if not selected:
        return 0.0
    X = atoms[selected].to_numpy()
    if _HAS_SKLEARN and len(selected) == 1:
        return float(_mi_per_atom_sklearn(X, y, selected).iloc[0])
    return _histogram_mi_multivariate(X, y, n_bins)


def _pca_explained_variance(X_std: np.ndarray) -> tuple[np.ndarray, int]:
    """PCA via covariance eigendecomposition; returns ratios and n for >=95%."""
    cov = np.cov(X_std, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total = eigvals.sum()
    if total <= 0:
        ratios = np.zeros_like(eigvals)
    else:
        ratios = eigvals / total
    cum = np.cumsum(ratios)
    n_95 = int(np.searchsorted(cum, PCA_VAR_THRESHOLD) + 1)
    return ratios, n_95


def _print_matrix(label: str, mat: pd.DataFrame, decimals: int = 3) -> None:
    print(f"\n{label}")
    print(mat.round(decimals).to_string())


def _greedy_orthogonal_subset(
    atoms: pd.DataFrame,
    y: np.ndarray,
    mi_ranking: pd.Series,
    corr: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Greedy selection: max MI first, then max marginal MI gain, |r|<0.6."""
    columns = list(atoms.columns)
    selected: list[str] = [mi_ranking.index[0]]
    steps: list[tuple[str, float]] = [(selected[0], float(mi_ranking.iloc[0]))]

    while len(selected) < len(columns):
        base_mi = _joint_mi(atoms, selected, y)
        best_atom: str | None = None
        best_gain = 0.0

        for atom in columns:
            if atom in selected:
                continue
            max_corr = max(abs(corr.loc[atom, s]) for s in selected)
            if max_corr >= CORR_THRESHOLD:
                continue
            gain = _joint_mi(atoms, selected + [atom], y) - base_mi
            if gain > best_gain:
                best_gain = gain
                best_atom = atom

        if best_atom is None:
            break
        selected.append(best_atom)
        steps.append((best_atom, best_gain))

    return selected, steps


def _verdict(
    pca_n_95: int,
    mi_ranking: pd.Series,
    pearson: pd.DataFrame,
    subset: list[str],
    steps: list[tuple[str, float]],
) -> None:
    """Print concise summary of redundancy findings."""
    high_pairs: list[str] = []
    cols = list(pearson.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = abs(pearson.loc[a, b])
            if r >= CORR_THRESHOLD:
                high_pairs.append(f"{a}~{b} ({r:.2f})")

    redundant = [c for c in cols if c not in subset]
    saturated_at = len(subset)
    if len(steps) > 1 and steps[-1][1] <= 1e-4:
        saturated_at = len(subset) - 1

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    print(f"  Effective independent dimensions (PCA >=95%): {pca_n_95}")
    print(f"  Greedy orthogonal basis ({len(subset)} atoms): {subset}")
    print(f"  Selection saturates after {saturated_at} atom(s)")
    if redundant:
        print(f"  Redundant / absorbed atoms: {redundant}")
    if high_pairs:
        print(f"  High-|r| pairs (>= {CORR_THRESHOLD}): {', '.join(high_pairs)}")
    print(
        f"  Top predictive atom: {mi_ranking.index[0]} "
        f"(MI={mi_ranking.iloc[0]:.4f})"
    )
    print("=" * 66 + "\n")


def run_atom_redundancy(csv_path: str, horizon: int = 1) -> dict:
    """Run full redundancy analysis; returns key metrics dict."""
    df = load_ohlcv(csv_path)
    config = Config(forward_horizon=horizon)
    pool = compute_atom_pool(df, config)[list(ATOM_POOL_9)]
    target = forward_return(df, horizon)
    y_binary = (target > 0).astype(int)

    aligned = pool.join(y_binary.rename("up"), how="inner").dropna()
    atoms = aligned.drop(columns=["up"])
    y = aligned["up"].to_numpy()

    pearson = atoms.corr(method="pearson")
    spearman = atoms.corr(method="spearman")

    X = atoms.to_numpy()
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    X_std = np.nan_to_num(X_std, nan=0.0)

    var_ratios, n_95 = _pca_explained_variance(X_std)
    columns = list(atoms.columns)

    if _HAS_SKLEARN:
        mi_ranking = _mi_per_atom_sklearn(X, y, columns)
    else:
        mi_ranking = _mi_per_atom_histogram(X, y, columns)

    subset, steps = _greedy_orthogonal_subset(atoms, y, mi_ranking, pearson)

    return {
        "n_rows": len(atoms),
        "pearson": pearson,
        "spearman": spearman,
        "var_ratios": var_ratios,
        "n_95": n_95,
        "mi_ranking": mi_ranking,
        "subset": subset,
        "steps": steps,
        "mi_backend": "sklearn" if _HAS_SKLEARN else "histogram",
    }


def _print_results(r: dict) -> None:
    line = "=" * 66
    print("\n" + line)
    print("FADE ATOM REDUNDANCY ANALYSIS")
    print(line)
    print(f"  rows (after NaN drop) : {r['n_rows']:,}")
    print(f"  MI estimator          : {r['mi_backend']}")

    _print_matrix("Pearson correlation", r["pearson"])
    _print_matrix("Spearman correlation", r["spearman"])

    print("\nPCA explained-variance ratio per component:")
    for i, vr in enumerate(r["var_ratios"], start=1):
        cum = float(np.cumsum(r["var_ratios"])[i - 1])
        print(f"  PC{i}: {vr:.4f}  (cumulative {cum:.4f})")
    print(f"  Components for >=95% variance: {r['n_95']}")

    print("\nMutual information with forward-direction target (ranked):")
    for atom, score in r["mi_ranking"].items():
        print(f"  {atom:16s}  {score:.6f}")

    print("\nGreedy orthogonal subset (|r| < 0.6, marginal MI gain):")
    for i, (atom, gain) in enumerate(r["steps"]):
        if i == 0:
            print(f"  1. {atom}  (seed MI={gain:.6f})")
        else:
            print(f"  {i + 1}. {atom}  (marginal gain={gain:.6f})")
    if len(r["steps"]) < len(r["mi_ranking"]):
        print(f"  -> saturated at {len(r['subset'])} atoms")

    _verdict(r["n_95"], r["mi_ranking"], r["pearson"], r["subset"], r["steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE atom redundancy analysis")
    parser.add_argument("--csv", default="btc_1h.csv", help="OHLCV CSV path")
    parser.add_argument("--horizon", type=int, default=1, help="Forward-return horizon")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"File not found: {args.csv}")
        return

    results = run_atom_redundancy(args.csv, horizon=args.horizon)
    _print_results(results)


if __name__ == "__main__":
    main()
