"""Market regime detection — turn off strategies when state is hostile.

Two regime axes:
1. Volatility regime (realized 30-day vol of BTC) — scales sizing down in
   extreme vol environments.
2. Trend regime (price vs 200-day SMA) — long bias in bull, short bias in bear.
"""

from __future__ import annotations

import numpy as np

from core import data


def vol_regime(pair: str = "BTC/USDT", window: int = 30) -> dict:
    df = data.ohlcv(pair, limit=window * 3)
    if len(df) < window:
        return {"regime": "unknown", "scale": 1.0, "realized_vol": 0.0}
    ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    rv = float(ret.iloc[-window:].std() * np.sqrt(365))
    if rv > 1.0:
        return {"regime": "extreme", "scale": 0.25, "realized_vol": rv}
    if rv > 0.6:
        return {"regime": "high", "scale": 0.50, "realized_vol": rv}
    return {"regime": "normal", "scale": 1.0, "realized_vol": rv}


def trend_regime(pair: str = "BTC/USDT", sma_window: int = 200) -> dict:
    df = data.ohlcv(pair, limit=sma_window * 2)
    if len(df) < sma_window:
        return {"regime": "unknown", "long_ok": True, "short_ok": True, "price_vs_sma": 1.0}
    sma = float(df["close"].rolling(sma_window).mean().iloc[-1])
    last = float(df["close"].iloc[-1])
    if last >= sma:
        return {"regime": "bull", "long_ok": True, "short_ok": False, "price_vs_sma": last / sma}
    return {"regime": "bear", "long_ok": False, "short_ok": True, "price_vs_sma": last / sma}


def overall(pair: str = "BTC/USDT") -> dict:
    v = vol_regime(pair)
    t = trend_regime(pair)
    return {
        "vol": v,
        "trend": t,
        "scale": v["scale"],
        "long_ok": t["long_ok"],
        "short_ok": t["short_ok"],
    }
