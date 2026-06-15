"""Time-series momentum signals.

Moskowitz/Ooi/Pedersen (2012) — single-horizon TSMOM
Asness/Moskowitz/Pedersen (2013) — multi-horizon ensemble across (1m, 3m, 12m)

Multi-horizon is preferred: averaging across horizons reduces overfit to
any single recent regime and is the canonical academic formulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tsmom_signal(
    prices: pd.Series,
    lookback_days: int = 60,
    vol_window_days: int = 60,
    target_vol: float = 0.20,
    annualization: int = 365,
) -> pd.Series:
    """Single-horizon vol-targeted TSMOM signal in [-1, 1]."""
    log_ret = np.log(prices / prices.shift(1))
    trailing = log_ret.rolling(lookback_days).sum()
    realized_vol = log_ret.rolling(vol_window_days).std() * np.sqrt(annualization)

    sign = np.sign(trailing)
    magnitude = (target_vol / realized_vol).clip(0, 1).fillna(0)
    return (sign * magnitude).fillna(0).clip(-1, 1)


def tsmom_signal_multi(
    prices: pd.Series,
    horizons: tuple[int, ...] = (30, 90, 180),
    target_vol: float = 0.20,
    annualization: int = 365,
) -> pd.Series:
    """Multi-horizon TSMOM ensemble (Asness/Moskowitz/Pedersen 2013).

    Equal-weighted average of vol-targeted single-horizon signals across
    the supplied horizons. Default (30, 90, 180) approximates the canonical
    1m / 3m / 6m ensemble adapted for crypto's higher vol.
    """
    log_ret = np.log(prices / prices.shift(1))
    sigs: list[pd.Series] = []
    for h in horizons:
        trailing = log_ret.rolling(h).sum()
        rv = log_ret.rolling(h).std() * np.sqrt(annualization)
        sign = np.sign(trailing)
        mag = (target_vol / rv).clip(0, 1).fillna(0)
        sigs.append((sign * mag).fillna(0).clip(-1, 1))
    return sum(sigs) / len(sigs)
