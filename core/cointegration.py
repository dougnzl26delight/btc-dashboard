"""Engle-Granger cointegration test + Ornstein-Uhlenbeck process fitting.

Standard stat-arb foundation. Different from naive z-score reversion because:
  - Tests STATIONARITY of the spread (not just current standard deviation)
  - Fits a proper OU mean-reversion model (gives half-life, not arbitrary thresholds)
  - Hedge ratio comes from regression, not naive 1:1

Reference:
  Engle & Granger (1987) "Co-integration and Error Correction"
  Lipton & Lopez de Prado (2020) "A Closed-Form Solution for Optimal
    Mean-Reverting Trading"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def engle_granger_test(
    series1: pd.Series, series2: pd.Series, p_value_threshold: float = 0.05
) -> dict:
    """Test cointegration between two price series.

    Step 1: Engle-Granger cointegration test
    Step 2: OLS hedge ratio
    Step 3: ADF on residuals (additional stationarity check)
    """
    aligned = pd.concat([series1.rename("s1"), series2.rename("s2")], axis=1).dropna()
    if len(aligned) < 100:
        return {"is_cointegrated": False, "reason": "insufficient data", "n_obs": len(aligned)}

    s1 = aligned["s1"].values
    s2 = aligned["s2"].values

    # Engle-Granger
    try:
        coint_score, coint_p, _ = coint(s1, s2)
    except Exception as e:
        return {"is_cointegrated": False, "reason": f"coint failed: {e}"}

    # OLS hedge ratio: s1 = α + β*s2
    cov_matrix = np.cov(s1, s2)
    hedge_ratio = float(cov_matrix[0, 1] / np.var(s2)) if np.var(s2) > 0 else 1.0
    intercept = float(s1.mean() - hedge_ratio * s2.mean())

    spread = s1 - hedge_ratio * s2 - intercept
    spread_series = pd.Series(spread, index=aligned.index)

    # ADF on spread
    try:
        adf_stat, adf_p, *_ = adfuller(spread)
    except Exception:
        adf_p = 1.0

    is_cointegrated = (coint_p < p_value_threshold) and (adf_p < p_value_threshold)

    return {
        "is_cointegrated": bool(is_cointegrated),
        "coint_p_value": float(coint_p),
        "adf_p_value": float(adf_p),
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread_series": spread_series,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
        "n_obs": int(len(aligned)),
    }


def fit_ou_process(spread: pd.Series, dt: float = 1.0) -> dict:
    """Fit Ornstein-Uhlenbeck SDE to a spread series.

    OU: dX = θ(μ - X)dt + σ dW
    Discretized AR(1): X_t = α + β*X_{t-1} + ε
      where β = e^(-θ*dt), μ = α/(1-β), θ = -ln(β)/dt
    """
    spread = spread.dropna()
    if len(spread) < 30:
        return {"error": "insufficient data"}

    x_lag = spread.iloc[:-1].values
    x_now = spread.iloc[1:].values
    x_lag_mean = float(x_lag.mean())
    y_mean = float(x_now.mean())

    var_x = float(np.var(x_lag))
    if var_x <= 0:
        return {"error": "no variance in spread"}

    beta = float(np.sum((x_lag - x_lag_mean) * (x_now - y_mean)) / (len(x_lag) * var_x))
    alpha = y_mean - beta * x_lag_mean

    if not (0 < beta < 1):
        return {"error": f"non-stationary AR(1) (β={beta:.3f}); not OU-mean-reverting"}

    theta = -np.log(beta) / dt
    mu = alpha / (1 - beta)

    residuals = x_now - (alpha + beta * x_lag)
    sigma_e = float(np.std(residuals))
    sigma = sigma_e * np.sqrt(2 * theta / (1 - beta * beta))

    half_life = float(np.log(2) / theta) if theta > 0 else float("inf")

    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life_periods": half_life,
        "ar1_beta": float(beta),
        "n_obs": int(len(spread)),
    }


def find_cointegrated_pairs(prices_df: pd.DataFrame, p_threshold: float = 0.05) -> pd.DataFrame:
    """Pairwise cointegration test across columns. Returns table of results."""
    cols = list(prices_df.columns)
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = engle_granger_test(prices_df[c1], prices_df[c2], p_threshold)
            rows.append({
                "pair_a": c1,
                "pair_b": c2,
                "is_cointegrated": r.get("is_cointegrated", False),
                "coint_p": r.get("coint_p_value", 1.0),
                "adf_p": r.get("adf_p_value", 1.0),
                "hedge_ratio": r.get("hedge_ratio", 0.0),
            })
    return pd.DataFrame(rows).sort_values("coint_p")


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from core import data

    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
    prices = pd.DataFrame({p: data.ohlcv_extended(p, days_back=730)["close"] for p in pairs}).dropna()
    print(f"Testing {len(pairs)} pairs over {len(prices)} days...\n")
    out = find_cointegrated_pairs(prices)
    print(out.to_string(index=False))
