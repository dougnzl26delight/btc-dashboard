"""Bailey/Lopez de Prado Deflated Sharpe + Harvey/Liu/Zhu t-stat hurdle.

Bailey & Lopez de Prado (2014), JoPM — corrects PSR for selection bias and
backtest overfitting given the number of trials searched.
Harvey, Liu, Zhu (2016), RFS — argues t > 3.0 as the multiple-testing-adjusted
hurdle for new factor claims.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
from scipy.stats import kurtosis, norm, skew


T_HURDLE = 3.0
ANNUALIZATION = 365  # crypto trades 24/7


def sharpe_ratio(
    returns: Iterable[float], periods_per_year: int = ANNUALIZATION
) -> float:
    r = np.asarray(list(returns), dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * sqrt(periods_per_year))


def t_stat(returns: Iterable[float]) -> float:
    r = np.asarray(list(returns), dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / sqrt(len(r))))


def deflated_sharpe(
    returns: Iterable[float],
    num_trials: int,
    periods_per_year: int = ANNUALIZATION,
) -> dict:
    """Probability the true Sharpe exceeds the trial-inflated null threshold."""
    r = np.asarray(list(returns), dtype=float)
    n = len(r)
    if n < 30 or num_trials < 1:
        return {
            "dsr": 0.0,
            "sr_annualized": 0.0,
            "sr_threshold_annualized": 0.0,
            "passes": False,
            "n_obs": n,
            "n_trials": num_trials,
            "reason": "insufficient observations" if n < 30 else "invalid trials",
        }

    sr = sharpe_ratio(r, periods_per_year=1)  # per-period
    sk = float(skew(r))
    kt = float(kurtosis(r, fisher=True))

    em = 0.5772156649  # Euler-Mascheroni
    z1 = norm.ppf(1 - 1.0 / num_trials)
    z2 = norm.ppf(1 - 1.0 / (num_trials * np.e))
    sr_zero = (1 - em) * z1 + em * z2

    sigma_sr = sqrt(max(0.0, (1 - sk * sr + (kt - 1) / 4 * sr * sr) / (n - 1)))
    # sr_zero above is in z-score units (expected max of N standard normals).
    # The trial-adjusted threshold in per-period SR units is sr_zero * sigma_sr.
    # PSR(threshold) = Phi((SR_obs - SR_threshold) / sigma_sr).
    sr_threshold_per_period = sr_zero * sigma_sr
    psr = float(norm.cdf((sr - sr_threshold_per_period) / sigma_sr)) if sigma_sr > 0 else 0.0
    return {
        "dsr": psr,
        "sr_annualized": sharpe_ratio(r, periods_per_year=periods_per_year),
        "sr_threshold_annualized": float(sr_threshold_per_period * sqrt(periods_per_year)),
        "passes": psr > 0.95,
        "n_obs": n,
        "n_trials": num_trials,
    }


def passes_quant_hurdle(
    returns: Iterable[float],
    num_trials: int,
    t_hurdle: float = T_HURDLE,
    periods_per_year: int = ANNUALIZATION,
) -> dict:
    """Combined gate: DSR > 0.95 AND |t-stat| > t_hurdle.

    A strategy must pass this BEFORE earning a `VALIDATED = True` flag.
    """
    dsr = deflated_sharpe(returns, num_trials, periods_per_year)
    t = t_stat(returns)
    return {
        **dsr,
        "t_stat": t,
        "passes_t": abs(t) > t_hurdle,
        "passes_combined": dsr["passes"] and abs(t) > t_hurdle,
    }
