"""Cross-validation for time-series strategies.

Standard k-fold CV is invalid for finance because:
  1. Returns are autocorrelated (samples not i.i.d.)
  2. Lookback windows leak future info into "training" labels

López de Prado, AFML (2018) Ch. 7: walk-forward + purged k-fold with embargo.
This module implements walk-forward (the simpler, more practical variant for
single-asset signal evaluation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest


def walk_forward(
    prices: pd.Series,
    signal_fn: Callable[[pd.Series], pd.Series],
    n_folds: int = 5,
    min_train: int = 365,
    embargo_days: int = 5,
) -> dict:
    """Expanding-window walk-forward CV.

    For each fold k:
      - Train window: [0, t_k]
      - Embargo: skip embargo_days after t_k
      - Test window: [t_k + embargo, t_k + embargo + fold_size]
      - signal_fn computed on prices through test end (uses only past data
        within the function — strategies must guarantee this)

    Returns dict with per-fold Sharpe, mean OOS Sharpe, and concatenated returns.
    """
    n = len(prices)
    if n < min_train + n_folds * 30:
        raise ValueError(f"need >= {min_train + n_folds * 30} obs, got {n}")

    fold_size = (n - min_train) // n_folds
    fold_summaries: list[dict] = []
    fold_sharpes: list[float] = []
    fold_returns: list[pd.Series] = []

    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_start = train_end + embargo_days
        test_end = min(test_start + fold_size, n)
        if test_end - test_start < 30:
            continue

        prices_to_test_end = prices.iloc[:test_end]
        sig = signal_fn(prices_to_test_end)

        # Mask training portion to zero so backtest only "trades" the test window
        sig_oos = sig.copy()
        sig_oos.iloc[:test_start] = 0.0

        bt = backtest.run(prices_to_test_end, sig_oos)
        bt_test = bt.iloc[test_start:test_end]
        if bt_test.empty:
            continue

        summary = backtest.summarize(bt_test)
        fold_summaries.append(summary)
        fold_sharpes.append(summary["sharpe"])
        fold_returns.append(bt_test["ret"])

    if not fold_summaries:
        return {"n_folds": 0, "passes": False, "reason": "no valid folds"}

    sharpes_arr = np.array(fold_sharpes)
    concat = pd.concat(fold_returns) if fold_returns else pd.Series(dtype=float)

    return {
        "n_folds": len(fold_summaries),
        "fold_sharpes": [float(s) for s in fold_sharpes],
        "mean_sharpe_oos": float(sharpes_arr.mean()),
        "std_sharpe_oos": float(sharpes_arr.std(ddof=1)) if len(sharpes_arr) > 1 else 0.0,
        "min_sharpe_oos": float(sharpes_arr.min()),
        "n_oos_obs": int(sum(len(r) for r in fold_returns)),
        "concatenated_returns": concat,
        # Passes if mean OOS Sharpe > 0.5 AND min fold > 0 (no catastrophic fold)
        "passes": float(sharpes_arr.mean()) > 0.5 and float(sharpes_arr.min()) > 0.0,
    }
