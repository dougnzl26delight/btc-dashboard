"""Open interest signal — derivatives flow indicator.

Established crypto-practitioner signal:
  - Rising OI + rising price = trend (real money entering long)
  - Rising OI + falling price = capitulation/fear (forced selling)
  - Falling OI + rising price = short squeeze
  - Falling OI + falling price = exits/de-risking

Implementation: standardized OI delta, multiplied by price direction sign.
Positive when OI growth confirms price direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import binance_extras


def open_interest_signal(
    pair: str,
    lookback: int = 5,
    z_window: int = 14,
    days_history: int = 30,
) -> float:
    # Binance openInterestHist API only serves ~30 days. Tune windows down to fit.
    """Latest OI-confirmed trend signal in [-1, 1]."""
    oi_df = binance_extras.fetch_open_interest_history(pair, period="1d", limit=days_history)
    if oi_df.empty or len(oi_df) < z_window:
        return 0.0

    oi = oi_df["oi_value"]
    oi_delta = oi.pct_change(lookback)
    if oi_delta.std() == 0:
        return 0.0

    z = (oi_delta - oi_delta.rolling(z_window).mean()) / oi_delta.rolling(z_window).std()
    z = z.iloc[-1]
    if pd.isna(z):
        return 0.0

    # Confirm with price direction
    from core import data
    try:
        price_df = data.ohlcv_extended(pair, days_back=lookback * 3)
        if price_df.empty:
            return 0.0
        price_ret = price_df["close"].pct_change(lookback).iloc[-1]
        direction = 1 if price_ret > 0 else (-1 if price_ret < 0 else 0)
    except Exception:
        direction = 1

    # Signal: OI confirms direction → strong; OI contradicts → weak
    confirmation = float(np.clip(z, -2, 2) / 2)
    return float(np.clip(confirmation * direction, -1, 1))
