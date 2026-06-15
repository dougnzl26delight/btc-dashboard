"""Combinatorial Purged Cross-Validation. López de Prado AFML (2018) Ch. 7.4.

Standard k-fold CV is invalid for finance because (a) returns are
autocorrelated and (b) labels can leak across train/test boundaries.
CPCV addresses both:

  - Splits data into N groups
  - Generates all C(N, k) combinations of which groups are TEST
  - Each combination yields one OOS evaluation
  - Embargo period after each test group prevents label-leakage from train

Result: substantially more OOS observations than walk-forward, with the
same purging guarantees. Tighter standard errors on Sharpe estimates.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def cpcv_split(
    n_obs: int, n_groups: int = 6, k_test: int = 2, embargo: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) pairs for each CPCV combination.

    Train set excludes:
      - All test indices
      - `embargo` indices immediately after each test group (prevents the
        next-period label leak from training)
    """
    if k_test >= n_groups:
        raise ValueError(f"k_test ({k_test}) must be < n_groups ({n_groups})")

    group_size = n_obs // n_groups
    boundaries = []
    for i in range(n_groups):
        lo = i * group_size
        hi = (i + 1) * group_size if i < n_groups - 1 else n_obs
        boundaries.append((lo, hi))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_combo in combinations(range(n_groups), k_test):
        test_mask = np.zeros(n_obs, dtype=bool)
        embargo_mask = np.zeros(n_obs, dtype=bool)
        for g in test_combo:
            lo, hi = boundaries[g]
            test_mask[lo:hi] = True
            embargo_mask[hi : min(hi + embargo, n_obs)] = True

        train_mask = ~(test_mask | embargo_mask)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    return splits


def cpcv_evaluate_signal(
    prices: pd.Series,
    signal_fn,
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 5,
) -> dict:
    """Evaluate a stationary signal_fn under CPCV.

    For ML-trained signals, the caller would re-fit signal_fn within each
    train fold; this helper assumes a stationary signal that uses only
    past data within itself (e.g. our TSMOM family).
    """
    from core import backtest

    n = len(prices)
    splits = cpcv_split(n, n_groups, k_test, embargo)
    sig = signal_fn(prices)

    fold_sharpes: list[float] = []
    fold_returns: list[pd.Series] = []
    for _train_idx, test_idx in splits:
        sig_test = sig.copy()
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        sig_test.iloc[mask] = 0.0

        bt = backtest.run(prices, sig_test)
        bt_test = bt.iloc[test_idx]
        if bt_test.empty or bt_test["ret"].std() == 0:
            continue
        s = float(bt_test["ret"].mean() / bt_test["ret"].std() * np.sqrt(365))
        fold_sharpes.append(s)
        fold_returns.append(bt_test["ret"])

    if not fold_sharpes:
        return {"n_combinations": len(splits), "n_evaluated": 0, "passes": False}

    arr = np.array(fold_sharpes)
    return {
        "n_combinations": len(splits),
        "n_evaluated": len(fold_sharpes),
        "mean_sharpe_oos": float(arr.mean()),
        "std_sharpe_oos": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min_sharpe_oos": float(arr.min()),
        "max_sharpe_oos": float(arr.max()),
        "n_oos_obs": int(sum(len(r) for r in fold_returns)),
        "concatenated_returns": pd.concat(fold_returns).sort_index(),
        "passes": float(arr.mean()) > 0.5 and float(arr.min()) > 0.0,
    }
