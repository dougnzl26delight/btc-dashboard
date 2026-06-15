"""Hierarchical Risk Parity portfolio combiner.

López de Prado, JoPM (2016) "Building Diversified Portfolios that Outperform
Out-of-Sample".

Avoids the instability of Markowitz mean-variance by:
  1. Clustering correlated assets via single-linkage on correlation distance
  2. Quasi-diagonalizing: reorder assets so correlated ones are adjacent
  3. Recursive bisection: split into halves, allocate inversely to risk

Key advantage: doesn't require inverting the covariance matrix, so handles
ill-conditioned cov (correlated strategies) gracefully. Standard practitioner
choice when combining N >= 3 strategies with non-trivial correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_distance(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1.0 - corr) / 2.0) ** 0.5


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Reorder leaf indices via the linkage matrix."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    n_items = link[-1, 3]
    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        i = df0.index
        j = df0.values - n_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Variance of an inverse-variance-weighted portfolio of `items`."""
    cov_slice = cov.loc[items, items]
    inv = 1.0 / np.diag(cov_slice.values)
    inv = inv / inv.sum()
    w = inv.reshape(-1, 1)
    return float(np.dot(np.dot(w.T, cov_slice.values), w).item())


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP allocation weights. Sums to 1."""
    if returns.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns)

    cov = returns.cov()
    corr = returns.corr()
    dist = _correl_distance(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    sort_ix_int = _quasi_diag(link)
    sort_ix = [returns.columns[i] for i in sort_ix_int]

    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        new_clusters: list[list] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
            for asset in left:
                w[asset] *= alpha
            for asset in right:
                w[asset] *= (1.0 - alpha)
            new_clusters.extend([left, right])
        clusters = new_clusters

    return w.reindex(returns.columns)
