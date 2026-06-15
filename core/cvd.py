"""Cumulative Volume Delta (CVD) approximation from kline OHLCV.

True CVD needs tick-by-tick trades with side flag, which Binance provides
via /api/v3/aggTrades but is expensive at scale. This approximation uses the
kline shape — a known proxy that captures ~70% of the signal:

    For each bar:
        if close > open: treat (close - open) / range * volume as BUY pressure
        if close < open: same logic but SELL
    CVD = cumulative net signed volume

Useful for detecting:
    - Distribution: price ranging/up but CVD declining (smart money exiting)
    - Accumulation: price ranging/down but CVD rising (smart money entering)
    - Climax: extreme CVD divergence from price = reversal signal

Cross-references: CVD divergence on the daily bar at cycle tops/bottoms is
one of the cleanest single-indicator setups in crypto historically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import data


def cvd_series(pair: str, days_back: int = 60) -> pd.DataFrame:
    """Compute approximate CVD series. Returns df with columns: close, cvd, signed_vol."""
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = df["close"] - df["open"]
    # Sign: positive close-open = buy pressure
    # Magnitude: body's share of range, scaled by volume
    signed_vol = (body / rng) * df["volume"]
    signed_vol = signed_vol.fillna(0)

    df["signed_vol"] = signed_vol
    df["cvd"] = signed_vol.cumsum()
    return df


def divergence_signal(pair: str, lookback: int = 30) -> dict:
    """Detect CVD divergence in recent N bars.

    Bullish divergence: price makes lower low, CVD makes higher low (accumulation)
    Bearish divergence: price makes higher high, CVD makes lower high (distribution)
    """
    df = cvd_series(pair, days_back=lookback + 5)
    if df.empty or len(df) < lookback:
        return {"signal": "no_data", "pair": pair}

    recent = df.iloc[-lookback:]
    price = recent["close"]
    cvd = recent["cvd"]

    price_low_idx = price.idxmin()
    price_high_idx = price.idxmax()
    cvd_at_price_low = cvd.loc[price_low_idx]
    cvd_at_price_high = cvd.loc[price_high_idx]

    # Compare to earlier extremes in the window (first half)
    early = recent.iloc[:lookback // 2]
    late = recent.iloc[lookback // 2:]
    early_price_low = early["close"].min()
    late_price_low = late["close"].min()
    early_price_high = early["close"].max()
    late_price_high = late["close"].max()
    early_cvd_at_low = early.loc[early["close"].idxmin(), "cvd"]
    late_cvd_at_low = late.loc[late["close"].idxmin(), "cvd"]
    early_cvd_at_high = early.loc[early["close"].idxmax(), "cvd"]
    late_cvd_at_high = late.loc[late["close"].idxmax(), "cvd"]

    bullish_div = (late_price_low < early_price_low) and (late_cvd_at_low > early_cvd_at_low)
    bearish_div = (late_price_high > early_price_high) and (late_cvd_at_high < early_cvd_at_high)

    signal = "neutral"
    if bullish_div:
        signal = "bullish_divergence"
    elif bearish_div:
        signal = "bearish_divergence"

    return {
        "pair": pair,
        "signal": signal,
        "current_cvd": float(cvd.iloc[-1]),
        "cvd_30d_change": float(cvd.iloc[-1] - cvd.iloc[0]) if len(cvd) >= lookback else 0,
        "price_30d_change_pct": float((price.iloc[-1] / price.iloc[0] - 1) * 100) if len(price) >= lookback else 0,
        "early_price_low": float(early_price_low),
        "late_price_low": float(late_price_low),
        "early_price_high": float(early_price_high),
        "late_price_high": float(late_price_high),
    }


def main():
    print(f"{'Pair':<10s} {'P chg 30d':>10s} {'CVD chg':>14s} {'Signal':<22s}")
    print("-" * 65)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]:
        try:
            d = divergence_signal(pair, lookback=30)
            print(f"{pair:<10s} {d.get('price_30d_change_pct', 0):>+9.1f}% "
                  f"{d.get('cvd_30d_change', 0):>+13,.0f}  {d.get('signal', '?'):<22s}")
        except Exception as e:
            print(f"{pair:<10s} ERROR: {e}")


if __name__ == "__main__":
    main()
