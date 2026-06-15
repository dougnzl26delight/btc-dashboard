"""Library of parameterized indicators for systematic sweeping.

Each function takes a price series and returns a signal in [-1, 1] aligned
to the input index. No lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tsmom_single(
    prices: pd.Series, lookback: int = 60, target_vol: float = 0.20, ann: int = 365
) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    trail = log_ret.rolling(lookback).sum()
    rv = log_ret.rolling(lookback).std() * np.sqrt(ann)
    sign = np.sign(trail)
    mag = (target_vol / rv).clip(0, 1).fillna(0)
    return (sign * mag).fillna(0).clip(-1, 1)


def tsmom_multi(
    prices: pd.Series, horizons=(30, 90, 180), target_vol: float = 0.20, ann: int = 365
) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    sigs = []
    for h in horizons:
        trail = log_ret.rolling(h).sum()
        rv = log_ret.rolling(h).std() * np.sqrt(ann)
        sign = np.sign(trail)
        mag = (target_vol / rv).clip(0, 1).fillna(0)
        sigs.append((sign * mag).fillna(0).clip(-1, 1))
    return sum(sigs) / len(sigs)


def ma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    raw = (fast_ma - slow_ma) / slow_ma
    return (raw / 0.05).clip(-1, 1).fillna(0)


def bollinger_revert(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std
    return (-z / n_std).clip(-1, 1).fillna(0)


def rsi_revert(prices: pd.Series, window: int = 14, low: float = 30, high: float = 70) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    sig = pd.Series(0.0, index=prices.index)
    sig[rsi < low] = 1.0
    sig[rsi > high] = -1.0
    return sig.fillna(0)


def donchian_breakout(prices: pd.Series, window: int = 20) -> pd.Series:
    upper = prices.rolling(window).max().shift(1)
    lower = prices.rolling(window).min().shift(1)
    sig = pd.Series(0.0, index=prices.index)
    sig[prices > upper] = 1.0
    sig[prices < lower] = -1.0
    return sig.fillna(0)


def zscore_revert(prices: pd.Series, window: int = 20) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std
    return (-z / 2.0).clip(-1, 1).fillna(0)


def vol_breakout(prices: pd.Series, window: int = 30, ann: int = 365) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(ann)
    rv_ema = rv.ewm(span=window).mean()
    vol_high = (rv > rv_ema).astype(float)
    direction = np.sign(prices.pct_change(window))
    return (vol_high * direction).fillna(0).clip(-1, 1)
