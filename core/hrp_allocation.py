"""Hierarchical Risk Parity (HRP) — Lopez de Prado AFML Chapter 16.

Replaces mean-variance optimization (Markowitz) for cross-sleeve allocation.
HRP advantages:
    1. No matrix inversion (numerically stable even with noisy covariance)
    2. Cluster-based diversification (intuitive AND robust)
    3. Allocations stable to small changes in returns
    4. Empirically outperforms Markowitz out-of-sample

Algorithm:
    1. Compute return correlation matrix across sleeves
    2. Cluster sleeves via hierarchical agglomerative clustering
    3. Quasi-diagonalize the covariance matrix (reorder by cluster proximity)
    4. Recursive bisection: allocate within clusters by inverse-variance

Result: per-sleeve allocation weights that automatically diversify across
uncorrelated bets and concentrate in clusters with lower variance.

Reference:
    Lopez de Prado (2016) "Building Diversified Portfolios that Outperform
    Out-of-Sample"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation to distance: d_ij = sqrt(0.5 * (1 - rho_ij))."""
    return np.sqrt(0.5 * (1 - corr))


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Reorder asset indices so similar items are adjacent (per linkage tree)."""
    link = link.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_idx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df0]).sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()


def recursive_bisection(cov: np.ndarray, sort_idx: list[int]) -> np.ndarray:
    """Recursively assign inverse-variance weights through the cluster tree."""
    w = pd.Series(1.0, index=sort_idx)
    clusters = [sort_idx]
    while len(clusters) > 0:
        clusters = [cl[s:e] for cl in clusters for s, e in (
            (0, len(cl) // 2), (len(cl) // 2, len(cl))) if e - s > 1 or len(cl) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c1 = clusters[i]
            c2 = clusters[i + 1]
            cov1 = cov[np.ix_(c1, c1)]
            cov2 = cov[np.ix_(c2, c2)]
            # Inverse-variance weights per cluster
            w1_iv = 1.0 / np.diag(cov1)
            w1 = w1_iv / w1_iv.sum() if w1_iv.sum() > 0 else np.ones(len(c1)) / len(c1)
            w2_iv = 1.0 / np.diag(cov2)
            w2 = w2_iv / w2_iv.sum() if w2_iv.sum() > 0 else np.ones(len(c2)) / len(c2)
            # Cluster-level variance
            v1 = float(w1.T @ cov1 @ w1)
            v2 = float(w2.T @ cov2 @ w2)
            # Allocate between clusters inversely to their variance
            alpha = 1 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            for idx in c1:
                w[idx] *= alpha
            for idx in c2:
                w[idx] *= (1 - alpha)
        # Filter to clusters with >1 member (terminals fully resolved)
        clusters = [cl for cl in clusters if len(cl) > 1]
    return w.values


def hrp_weights(returns_df: pd.DataFrame) -> dict[str, float]:
    """Compute HRP weights for a return-series DataFrame.

    Args:
        returns_df: columns = sleeve names, rows = daily returns

    Returns: {sleeve: weight} summing to 1.0
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        n = max(returns_df.shape[1], 1)
        return {col: 1.0 / n for col in returns_df.columns}

    cov = returns_df.cov().values
    corr = returns_df.corr().values
    # Replace NaN
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    # Distance + linkage
    dist = correlation_distance(corr)
    # Convert to condensed form
    condensed = squareform(dist, checks=False)
    if (condensed == 0).all():
        # All correlated 1; equal weights
        n = returns_df.shape[1]
        return {col: 1.0 / n for col in returns_df.columns}
    link = linkage(condensed, method="single")
    # Quasi-diagonalize
    sort_idx = quasi_diagonalize(link)
    # Bisection
    weights = recursive_bisection(cov, sort_idx)
    # Map back to sleeve names (in original column order)
    result = {col: 0.0 for col in returns_df.columns}
    for idx, w in zip(sort_idx, weights):
        result[returns_df.columns[idx]] = float(w)
    return result


def compute_sleeve_hrp(days: int = 60) -> dict:
    """Compute HRP weights across all sleeves using live daily returns from pnl_db."""
    from core.pnl_db import get_sleeve_returns
    sleeves = ["bah_btc", "xsmom", "pro_trend", "oversold_bounce", "overbought_fade",
                "basis_arb", "grid_trader", "intraday_momentum", "consolidation_breakout"]
    series = {}
    for s in sleeves:
        r = get_sleeve_returns(s, days=days)
        if len(r) >= 14:
            series[s] = list(reversed(r))  # chronological
    if not series:
        return {"error": "no_data"}
    # Align lengths
    min_len = min(len(s) for s in series.values())
    aligned = {k: v[-min_len:] for k, v in series.items()}
    df = pd.DataFrame(aligned)
    weights = hrp_weights(df)
    return {
        "weights": weights,
        "n_sleeves": len(weights),
        "n_observations": min_len,
        "verdict": "HRP-allocated; use as multiplier on sleeve baselines",
    }


def main():
    print("=" * 70)
    print("HIERARCHICAL RISK PARITY — Lopez de Prado AFML Ch 16")
    print("=" * 70)
    r = compute_sleeve_hrp(days=60)
    if r.get("error"):
        print(f"\n{r['error']} — sleeves need >=14 days of recorded returns each")
        print("This will populate as the rig accumulates daily P&L data.")
        return
    print(f"\nN sleeves: {r['n_sleeves']}  N observations: {r['n_observations']}")
    print("\nHRP weights (sum to 1.0):")
    sorted_w = sorted(r["weights"].items(), key=lambda x: -x[1])
    for sleeve, w in sorted_w:
        bar = "█" * int(w * 50)
        print(f"  {sleeve:<24s}  {w*100:>5.2f}%  {bar}")


if __name__ == "__main__":
    main()
