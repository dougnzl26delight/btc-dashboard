"""Momentum-crash tail overlay.

Daniel & Moskowitz (2016), 'Momentum Crashes', JFE.

Momentum strategies have asymmetric crash risk — they get crushed during
sharp bear-market reversals (March 2009, BTC June 2022, etc.). Solution:
scale position down using the EWMA vol of the STRATEGY's own returns,
not just the underlying asset.

Use case: wrap any momentum strategy's position weight with crash_adjusted_size().
"""

from __future__ import annotations

from math import sqrt

import pandas as pd


def crash_adjusted_size(
    strategy_returns: pd.Series,
    target_vol: float = 0.15,
    span: int = 30,
    max_leverage: float = 1.0,
    annualization: int = 365,
) -> float:
    """Inverse-vol scaling on the strategy's own EWMA vol.

    When the strategy itself becomes volatile (typical pre-crash regime),
    the scale shrinks; when calm, returns to max_leverage.
    """
    if len(strategy_returns) < span:
        return 1.0
    realized = float(strategy_returns.ewm(span=span).std().iloc[-1] * sqrt(annualization))
    if realized <= 0:
        return 1.0
    return float(min(max_leverage, target_vol / realized))


def crash_adjusted_series(
    strategy_returns: pd.Series,
    target_vol: float = 0.15,
    span: int = 30,
    max_leverage: float = 1.0,
    annualization: int = 365,
) -> pd.Series:
    """Time series of crash-adjusted scalars for backtest application."""
    realized = strategy_returns.ewm(span=span).std() * sqrt(annualization)
    scale = (target_vol / realized).clip(upper=max_leverage)
    return scale.fillna(1.0)
