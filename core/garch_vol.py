"""GARCH(1,1) conditional volatility forecasting.

Engle (1982) — autoregressive conditional heteroskedasticity (Nobel 2003).
Bollerslev (1986) — generalized GARCH(1,1) extension.

GARCH gives a smoother, autoregressive vol estimate that adapts faster to
new shocks than rolling realized vol. Standard practitioner tool for
conditional vol forecasting in risk and sizing models.

Uses the `arch` package (Sheppard et al.) — well-tested implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arch import arch_model


def garch_conditional_vol(returns: pd.Series, annualization: int = 365) -> pd.Series:
    """In-sample conditional volatility from GARCH(1,1). Annualized."""
    pct = returns.dropna() * 100
    if len(pct) < 100:
        raise ValueError(f"need >= 100 obs for stable GARCH, got {len(pct)}")

    model = arch_model(pct, vol="Garch", p=1, q=1, rescale=False, mean="Constant")
    res = model.fit(disp="off", show_warning=False)
    return (res.conditional_volatility / 100) * np.sqrt(annualization)


def garch_forecast_vol(
    returns: pd.Series, horizon: int = 1, annualization: int = 365
) -> float:
    """Out-of-sample conditional vol forecast `horizon` steps ahead. Annualized."""
    pct = returns.dropna() * 100
    if len(pct) < 100:
        raise ValueError(f"need >= 100 obs for stable GARCH, got {len(pct)}")

    model = arch_model(pct, vol="Garch", p=1, q=1, rescale=False, mean="Constant")
    res = model.fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=horizon, reindex=False)
    var_h = float(fc.variance.iloc[-1, horizon - 1])
    return float(np.sqrt(var_h) / 100 * np.sqrt(annualization))


def garch_params(returns: pd.Series) -> dict:
    """Return fitted GARCH(1,1) parameters and persistence (alpha + beta)."""
    pct = returns.dropna() * 100
    model = arch_model(pct, vol="Garch", p=1, q=1, rescale=False, mean="Constant")
    res = model.fit(disp="off", show_warning=False)
    p = res.params
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    omega = float(p.get("omega", 0.0))
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "stationary": (alpha + beta) < 1.0,
        "log_likelihood": float(res.loglikelihood),
    }
