"""Perp funding-rate sentiment signal.

When perp funding is materially positive (longs paying shorts), the spot
position takes a contrarian negative tilt — crowded longs unwind. We z-score
recent funding rates and flip the sign.

Note: this is a sentiment overlay for spot, not the textbook cash-and-carry
basis trade (which needs both perp and spot legs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import data


def funding_signal(perp_pair: str = "BTC/USDT:USDT", lookback: int = 30) -> float:
    """Latest funding-based contrarian signal in [-1, 1]. lookback = days."""
    hist = data.funding_history(perp_pair, limit=lookback * 8)
    if hist.empty or len(hist) < lookback:
        return 0.0
    rates = hist["funding_rate"].astype(float)
    if rates.std() == 0:
        return 0.0
    z = (rates.iloc[-1] - rates.mean()) / rates.std()
    return float(np.clip(-z, -1, 1))


def funding_signal_series(
    perp_pair: str = "BTC/USDT:USDT",
    z_window_periods: int = 90,
    days_back: int = 730,
) -> pd.Series:
    """Daily-frequency funding contrarian signal as a time series.

    z_window_periods is in 8-hourly funding events (90 ≈ 30 days).
    Returns a UTC-indexed series of daily-resampled signal in [-1, 1].
    """
    hist = data.funding_history_extended(perp_pair, days_back=days_back)
    if hist.empty or len(hist) < z_window_periods:
        return pd.Series(dtype=float)

    rates = hist["funding_rate"].astype(float)
    rolling_mean = rates.rolling(z_window_periods).mean()
    rolling_std = rates.rolling(z_window_periods).std()
    z = (rates - rolling_mean) / rolling_std
    sig_8h = (-z).clip(-1, 1).fillna(0)

    # Average the (up to) 3 funding events per day onto a daily index
    sig_daily = sig_8h.resample("1D").mean().fillna(0)
    return sig_daily
