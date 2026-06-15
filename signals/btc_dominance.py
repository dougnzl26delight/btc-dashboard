"""BTC dominance signal — alt-vs-BTC tilt indicator.

When BTC dominance rises, capital is rotating into BTC (favors BTC over alts).
When BTC dominance falls, capital is rotating into alts.

Implementation: trailing change in BTC dominance.
- For BTC pair: signal positive when dominance is rising
- For alt pairs: signal positive when dominance is falling
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import binance_extras


def btc_dominance_history(days: int = 90) -> pd.Series:
    """BTC market cap / total crypto market cap proxy from CoinGecko mcap series."""
    btc = binance_extras.fetch_coingecko_market_chart("bitcoin", days=days)
    if btc.empty:
        return pd.Series(dtype=float)
    return btc["market_cap"]


def btc_dominance_signal(pair: str, lookback: int = 14, z_window: int = 60) -> float:
    """Returns dominance-momentum signal scaled by whether the target pair
    benefits from BTC dominance going up (BTC) or down (alts)."""
    dom_proxy = btc_dominance_history(days=180)
    if dom_proxy.empty or len(dom_proxy) < z_window:
        return 0.0

    pct_change = dom_proxy.pct_change(lookback)
    if pct_change.std() == 0:
        return 0.0

    rolling_mean = pct_change.rolling(z_window).mean()
    rolling_std = pct_change.rolling(z_window).std()
    z = (pct_change.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    if pd.isna(z):
        return 0.0

    z_clipped = float(np.clip(z, -2, 2) / 2)
    # Invert sign for non-BTC pairs (rising BTC dominance is bearish for alts)
    return z_clipped if pair == "BTC/USDT" else -z_clipped
