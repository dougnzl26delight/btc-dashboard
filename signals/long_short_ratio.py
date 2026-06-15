"""Long/short account ratio — contrarian sentiment indicator.

Binance publishes the global long/short account ratio for each perp.
Documented practitioner signal: when retail accounts are heavily positioned
one direction, the contrarian fade tends to win short-term.

Signal: -z(LS_ratio). When LS ratio is unusually high (crowd is long),
signal is negative (contrarian short bias).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import binance_extras


def long_short_signal(pair: str, z_window: int = 30, days_history: int = 90) -> float:
    df = binance_extras.fetch_long_short_ratio(pair, period="1d", limit=days_history)
    if df.empty or len(df) < z_window:
        return 0.0

    ratio = df["ratio"]
    if ratio.std() == 0:
        return 0.0

    rolling_mean = ratio.rolling(z_window).mean()
    rolling_std = ratio.rolling(z_window).std()
    z = (ratio.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    if pd.isna(z):
        return 0.0
    return float(np.clip(-z, -2, 2) / 2)
