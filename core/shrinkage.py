"""Ledoit-Wolf shrinkage covariance estimation.

Ledoit & Wolf (2003), 'Honey, I Shrunk the Sample Covariance Matrix'.

Sample covariance is unbiased but high variance, especially when the number
of assets approaches the number of observations. Shrinkage to a structured
target (typically scaled identity) trades a small bias for substantially
lower variance, dramatically improving out-of-sample portfolio behavior.

Standard practitioner upgrade — every serious portfolio optimization in
production uses some form of shrinkage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def shrinkage_cov(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    """Shrinkage-adjusted covariance matrix.

    method: 'ledoit_wolf' (default) or 'oas' (Oracle Approximating Shrinkage,
    typically slightly less aggressive shrinkage).
    """
    cleaned = returns.dropna()
    if len(cleaned) < returns.shape[1] + 5:
        raise ValueError("too few observations for stable shrinkage estimate")

    estimator = OAS() if method == "oas" else LedoitWolf()
    estimator.fit(cleaned.values)
    return pd.DataFrame(
        estimator.covariance_, index=returns.columns, columns=returns.columns
    )


def shrinkage_intensity(returns: pd.DataFrame) -> float:
    """The shrinkage coefficient α from Ledoit-Wolf in [0, 1].

    α = 0 → pure sample covariance. α = 1 → pure shrinkage target.
    Practical values are typically 0.1–0.3 for liquid asset returns.
    """
    cleaned = returns.dropna()
    est = LedoitWolf().fit(cleaned.values)
    return float(est.shrinkage_)
