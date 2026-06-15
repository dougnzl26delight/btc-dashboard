"""Stablecoin supply expansion — macro liquidity indicator.

Stablecoins (USDT, USDC) are the primary on-ramp into crypto. Their total
market cap acts as a "fuel gauge" for crypto buying power. When stablecoin
supply expands, money is flowing in (bullish). When it contracts, money is
flowing out (bearish).

Reference: standard crypto-macro practitioner signal. CryptoQuant, Glassnode,
and Coin Metrics all publish variations of this. We compute it from CoinGecko
free market_chart endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import binance_extras


def fetch_total_stablecoin_supply(days: int = 180) -> pd.Series:
    """Sum of USDT + USDC market caps over time."""
    usdt = binance_extras.fetch_coingecko_market_chart("tether", days=days)
    usdc = binance_extras.fetch_coingecko_market_chart("usd-coin", days=days)
    if usdt.empty:
        return pd.Series(dtype=float)
    total = usdt["market_cap"].copy()
    if not usdc.empty:
        usdc_aligned = usdc["market_cap"].reindex(total.index, method="nearest").fillna(0)
        total = total + usdc_aligned
    return total


def stablecoin_signal(lookback: int = 14, z_window: int = 60) -> float:
    """Signal in [-1, 1]: positive when stablecoin supply is expanding fast."""
    supply = fetch_total_stablecoin_supply(days=180)
    if supply.empty or len(supply) < z_window:
        return 0.0

    pct_change = supply.pct_change(lookback)
    if pct_change.std() == 0:
        return 0.0

    rolling_mean = pct_change.rolling(z_window).mean()
    rolling_std = pct_change.rolling(z_window).std()
    z = (pct_change.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    if pd.isna(z):
        return 0.0
    return float(np.clip(z, -2, 2) / 2)
