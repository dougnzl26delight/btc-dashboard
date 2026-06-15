"""Purged K-Fold Cross-Validation — López de Prado AFML Chapter 7.

Standard k-fold CV LEAKS information in financial data because labels overlap
in time (a label formed today references future bars; a test sample using a
nearby label has seen some of the same data).

PURGING: drop training observations whose label-formation window overlaps
         the test set's window.
EMBARGOING: skip a buffer period after each test split before resuming
            training (handles serial autocorrelation in residuals).

Result: each fold gives a TRULY out-of-sample Sharpe. Multiple folds give
you a confidence interval. Decision rule for live deployment:
    95% CI lower bound > 0.5 = robust edge worth real capital.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats


def purged_kfold_indices(
    n_samples: int,
    n_splits: int = 10,
    embargo_pct: float = 0.01,
    label_window: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index pairs for purged k-fold CV.

    Args:
        n_samples: total number of observations
        n_splits: number of CV folds
        embargo_pct: fraction of observations to skip AFTER each test set
        label_window: how many bars forward each label sees (purge bandwidth)

    Returns:
        list of (train_indices, test_indices) for each fold
    """
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    embargo_size = int(n_samples * embargo_pct)
    folds = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        test_idx = indices[test_start:test_end]

        # Train: everything OUTSIDE test ± purge/embargo
        purge_start = max(0, test_start - label_window)
        purge_end = min(n_samples, test_end + embargo_size)
        train_mask = (indices < purge_start) | (indices >= purge_end)
        train_idx = indices[train_mask]
        folds.append((train_idx, test_idx))
    return folds


def compute_oos_sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio from return array."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def purged_cv_sharpe(
    returns: np.ndarray | list[float],
    n_splits: int = 10,
    embargo_pct: float = 0.01,
    label_window: int = 1,
    periods_per_year: int = 365,
) -> dict:
    """Run purged k-fold CV on a return series. Reports per-fold Sharpes + summary.

    Use when you have a continuous strategy return series. For per-trade label
    based CV (e.g., triple-barrier outcomes), use the indices to slice your
    feature/label matrix instead.
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < n_splits * 10:
        return {"error": f"insufficient_samples: need >= {n_splits * 10}, have {n}"}

    folds = purged_kfold_indices(n, n_splits, embargo_pct, label_window)
    fold_sharpes = []
    for train_idx, test_idx in folds:
        test_returns = arr[test_idx]
        fold_sharpe = compute_oos_sharpe(test_returns, periods_per_year)
        fold_sharpes.append(fold_sharpe)

    arr_sh = np.array(fold_sharpes)
    mean = float(arr_sh.mean())
    std = float(arr_sh.std(ddof=1)) if len(arr_sh) > 1 else 0.0
    # 95% confidence interval via t-distribution
    if std > 0 and len(arr_sh) > 1:
        t_crit = float(stats.t.ppf(0.975, len(arr_sh) - 1))
        margin = t_crit * std / np.sqrt(len(arr_sh))
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = mean
        ci_high = mean

    verdict = _purged_verdict(ci_low)

    return {
        "n_folds": n_splits,
        "fold_sharpes": fold_sharpes,
        "mean_sharpe": mean,
        "std_sharpe": std,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "verdict": verdict,
    }


def _purged_verdict(ci_low: float) -> str:
    if ci_low > 0.5:
        return "DEPLOY — robust edge confirmed"
    if ci_low > 0.0:
        return "WEAK — small live test only"
    return "REJECT — likely overfit or no edge"


def main():
    """Demonstrate purged CV on synthetic data."""
    print("=" * 70)
    print("PURGED K-FOLD CV — Lopez de Prado AFML Ch 7 demonstration")
    print("=" * 70)
    np.random.seed(42)
    # Synthetic strategy with small positive edge + noise
    returns = np.random.normal(0.0008, 0.02, 500)  # 0.08%/day mean, 2% vol
    print(f"\nSynthetic data: {len(returns)} daily returns, mean={returns.mean()*100:.3f}%, "
          f"std={returns.std()*100:.2f}%")
    print(f"Naive annualized Sharpe: {compute_oos_sharpe(returns):.2f}")
    print()
    r = purged_cv_sharpe(returns, n_splits=10, embargo_pct=0.01)
    print(f"Purged CV results (n_splits=10):")
    print(f"  Mean Sharpe across folds:    {r['mean_sharpe']:+.2f}")
    print(f"  Std Sharpe across folds:     {r['std_sharpe']:.2f}")
    print(f"  95% CI:                      [{r['ci_95_low']:+.2f}, {r['ci_95_high']:+.2f}]")
    print(f"  Verdict:                     {r['verdict']}")
    print()
    print("Per-fold Sharpes:")
    for i, s in enumerate(r["fold_sharpes"]):
        print(f"  Fold {i+1:>2d}: {s:+.2f}")


if __name__ == "__main__":
    main()
