"""Factor decomposition — separate alpha from beta exposure.

A strategy that's just leveraged exposure to BTC adds no diversification,
even if its standalone Sharpe looks decent. We test this by regressing
strategy returns on the benchmark and inspecting the intercept's t-stat.

A real edge has alpha t-stat > 2.0 (conventional) and meaningful R^2 reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def decompose(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: int = 365,
) -> dict:
    """OLS regression of strategy on benchmark. Returns alpha, beta, t-stats."""
    df = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    n = len(df)
    if n < 30:
        return {
            "alpha_per_period": 0.0,
            "alpha_annualized": 0.0,
            "alpha_t": 0.0,
            "beta": 0.0,
            "r_squared": 0.0,
            "n_obs": n,
            "passes_alpha_t": False,
            "reason": "insufficient observations",
        }

    slope, intercept, r, _, _ = stats.linregress(df["b"], df["s"])

    fitted = intercept + slope * df["b"]
    residuals = df["s"] - fitted
    se_resid = float(residuals.std(ddof=2))
    x_mean = float(df["b"].mean())
    x_var = float(df["b"].var(ddof=1))
    se_intercept = (
        se_resid * np.sqrt(1.0 / n + x_mean * x_mean / (n * x_var))
        if x_var > 0
        else 0.0
    )
    alpha_t = float(intercept / se_intercept) if se_intercept > 0 else 0.0

    return {
        "alpha_per_period": float(intercept),
        "alpha_annualized": float(intercept * annualization),
        "alpha_t": alpha_t,
        "beta": float(slope),
        "r_squared": float(r * r),
        "n_obs": n,
        "passes_alpha_t": abs(alpha_t) > 2.0,
    }
