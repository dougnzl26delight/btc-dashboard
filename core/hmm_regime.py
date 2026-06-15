"""Hidden Markov regime detection.

Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary
Time Series and the Business Cycle" — Markov-switching models for regime
identification. Implemented via statsmodels.MarkovRegression.

Two-state model on log returns:
  State 0: low-volatility regime (typically bull / steady)
  State 1: high-volatility regime (typically crisis / drawdown)

Returns the smoothed probabilities of each state plus the current most-likely
regime label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from core import data


def fit_hmm_2state(pair: str = "BTC/USDT", days_back: int = 730) -> dict:
    """Fit a 2-state Markov-switching model on daily log returns.

    Returns dict with:
      current_regime: 0 or 1 (lowest-vol = 0)
      regime_label: 'low_vol' or 'high_vol'
      regime_prob: probability of being in current_regime today
      vol_per_regime: estimated volatility for each state (annualized)
      mean_per_regime: mean return per state
      converged: bool
    """
    df = data.ohlcv_extended(pair, days_back=days_back)
    if len(df) < 100:
        return {"converged": False, "reason": "insufficient data"}

    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna() * 100  # in percent

    try:
        model = MarkovRegression(
            log_ret, k_regimes=2, trend="c", switching_variance=True
        )
        res = model.fit(disp=False)
    except Exception as e:
        return {"converged": False, "reason": f"fit failed: {e}"}

    # Order regimes by variance: state with lower variance = "low_vol"
    variances = res.params.filter(like="sigma2").values
    means = res.params.filter(like="const").values

    if len(variances) < 2 or len(means) < 2:
        return {"converged": False, "reason": "regime params missing"}

    low_idx = int(np.argmin(variances))
    high_idx = 1 - low_idx

    smoothed = res.smoothed_marginal_probabilities
    current_probs = smoothed.iloc[-1].values
    current_state_raw = int(np.argmax(current_probs))
    current_state = 0 if current_state_raw == low_idx else 1
    label = "low_vol" if current_state == 0 else "high_vol"

    return {
        "converged": True,
        "current_regime": current_state,
        "regime_label": label,
        "regime_prob": float(current_probs[current_state_raw]),
        "vol_per_regime_ann": {
            "low_vol": float(np.sqrt(variances[low_idx]) / 100 * np.sqrt(365)),
            "high_vol": float(np.sqrt(variances[high_idx]) / 100 * np.sqrt(365)),
        },
        "mean_per_regime_daily_pct": {
            "low_vol": float(means[low_idx]),
            "high_vol": float(means[high_idx]),
        },
        "smoothed_probs_low_vol_recent": [
            float(smoothed.iloc[-i, low_idx]) for i in range(min(10, len(smoothed)), 0, -1)
        ],
    }


def regime_probability_series(pair: str = "BTC/USDT", days_back: int = 730) -> pd.DataFrame:
    """Return time series of smoothed regime probabilities."""
    df = data.ohlcv_extended(pair, days_back=days_back)
    if len(df) < 100:
        return pd.DataFrame()
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna() * 100
    try:
        res = MarkovRegression(
            log_ret, k_regimes=2, trend="c", switching_variance=True
        ).fit(disp=False)
        smoothed = res.smoothed_marginal_probabilities
        smoothed.columns = ["state_0", "state_1"]
        return smoothed
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    import json
    print(json.dumps(fit_hmm_2state(), indent=2, default=str))
