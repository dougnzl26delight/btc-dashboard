"""Fractional Kelly position sizing.

Kelly (1956), Thorp (1962) — the criterion that maximizes long-run log-wealth.
Carver, Systematic Trading (2015) — practitioner standard is 0.25 x Kelly to
account for parameter estimation uncertainty (full Kelly is dangerously aggressive
when mean and variance are estimated from finite samples).

Full Kelly:        f* = mu / sigma^2     (continuous-time approx)
Fractional Kelly:  f  = 0.25 * f*        (Carver default; Thorp 1980s suggests 0.20-0.50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Full-Kelly optimal fraction from historical returns. Multiply by 0.25 for retail use."""
    if len(returns) < 30 or returns.var() == 0:
        return 0.0
    mu = float(returns.mean()) * periods_per_year
    sigma2 = float(returns.var()) * periods_per_year
    return mu / sigma2 if sigma2 > 0 else 0.0


def fractional_kelly_size(
    expected_return_ann: float,
    expected_variance_ann: float,
    fraction: float = 0.25,
    max_size: float = 0.20,
) -> float:
    """Fractional Kelly position as fraction of capital. Annualized inputs."""
    if expected_variance_ann <= 0:
        return 0.0
    full_kelly = expected_return_ann / expected_variance_ann
    return float(np.clip(fraction * full_kelly, -max_size, max_size))


def kelly_from_signal_history(
    signal: pd.Series,
    forward_returns: pd.Series,
    fraction: float = 0.25,
    max_size: float = 0.20,
    periods_per_year: int = 365,
) -> float:
    """Compute fractional Kelly size from joint signal × forward-return history."""
    df = pd.DataFrame({"sig": signal, "ret": forward_returns}).dropna()
    if len(df) < 30:
        return 0.0
    pnl = df["sig"] * df["ret"]
    if pnl.var() == 0:
        return 0.0
    mu_ann = float(pnl.mean()) * periods_per_year
    sigma2_ann = float(pnl.var()) * periods_per_year
    return fractional_kelly_size(mu_ann, sigma2_ann, fraction=fraction, max_size=max_size)
