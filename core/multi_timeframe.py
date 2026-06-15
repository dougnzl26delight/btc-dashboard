"""Multi-timeframe analysis — 1h / 4h / 1d / 1w confluence.

Single-timeframe signals fire too often and chase noise. Pros require
ALIGNMENT across timeframes:
    - 1h direction matches 4h direction = high-frequency edge
    - 4h matches 1d = quality intraday setup
    - 1d matches 1w = position-trade-worthy

This module fetches multi-TF data for a pair, computes a directional read
per timeframe, returns a confluence score 0-1.

Use in sleeves:
    confluence < 0.5 = filter out (single-TF only)
    confluence 0.5-0.8 = normal sizing
    confluence > 0.8 = upsize via meta-confidence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import ccxt


_EX = ccxt.binance({"enableRateLimit": True, "timeout": 8000})


def fetch_ohlcv(pair: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV at a specific timeframe."""
    try:
        ohlcv = _EX.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")
        return df
    except Exception:
        return pd.DataFrame()


def directional_read(df: pd.DataFrame, ema_short: int = 20, ema_long: int = 50) -> dict:
    """Direction + strength for one timeframe.

    Direction: +1 (up), -1 (down), 0 (flat)
    Strength: 0-1 based on EMA gap and momentum
    """
    if df.empty or len(df) < ema_long + 5:
        return {"direction": 0, "strength": 0.0, "available": False}

    close = df["close"]
    ema_s = close.ewm(span=ema_short, adjust=False).mean()
    ema_l = close.ewm(span=ema_long, adjust=False).mean()

    current = float(close.iloc[-1])
    s = float(ema_s.iloc[-1])
    l = float(ema_l.iloc[-1])

    # Direction: above both EMAs (up), below both (down), between (flat)
    if current > s and s > l:
        direction = 1
    elif current < s and s < l:
        direction = -1
    else:
        direction = 0

    # Strength: relative EMA gap + recent momentum
    if l > 0:
        ema_gap = abs(s - l) / l
    else:
        ema_gap = 0
    # Last 5-bar return
    if len(close) >= 6:
        recent_ret = float(close.iloc[-1] / close.iloc[-6] - 1)
    else:
        recent_ret = 0

    strength = min(1.0, ema_gap * 10 + abs(recent_ret) * 5)
    return {
        "direction": direction,
        "strength": strength,
        "current_price": current,
        "ema_short": s,
        "ema_long": l,
        "available": True,
    }


def confluence(pair: str = "BTC/USDT") -> dict:
    """Compute multi-TF confluence score for a pair.

    Returns:
        {
            confluence_score: 0-1,
            net_direction: -1/+1,
            timeframes: {tf: {direction, strength, ...}},
            verdict: 'aligned_up' | 'aligned_down' | 'mixed' | 'choppy'
        }
    """
    tfs = ["1h", "4h", "1d", "1w"]
    reads = {}
    for tf in tfs:
        df = fetch_ohlcv(pair, timeframe=tf, limit=200)
        reads[tf] = directional_read(df)

    # Weighted directional consensus (higher TFs weighted more)
    tf_weights = {"1h": 1, "4h": 2, "1d": 3, "1w": 4}
    weighted_directions = []
    for tf, r in reads.items():
        if r.get("available"):
            weighted_directions.append(r["direction"] * tf_weights[tf] * r["strength"])
    total_weight = sum(tf_weights[tf] for tf, r in reads.items() if r.get("available"))

    if not weighted_directions or total_weight == 0:
        return {
            "pair": pair, "confluence_score": 0.0, "net_direction": 0,
            "verdict": "no_data", "timeframes": reads,
        }

    weighted_sum = sum(weighted_directions) / total_weight
    confluence_score = abs(weighted_sum)
    net_direction = 1 if weighted_sum > 0.2 else (-1 if weighted_sum < -0.2 else 0)

    if confluence_score > 0.7:
        verdict = "aligned_up" if net_direction > 0 else "aligned_down"
    elif confluence_score > 0.4:
        verdict = "trending_" + ("up" if net_direction > 0 else "down")
    else:
        verdict = "choppy"

    return {
        "pair": pair,
        "confluence_score": confluence_score,
        "net_direction": net_direction,
        "verdict": verdict,
        "weighted_signal": weighted_sum,
        "timeframes": reads,
    }


def main():
    print("=" * 80)
    print("MULTI-TIMEFRAME CONFLUENCE — 1h / 4h / 1d / 1w")
    print("=" * 80)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        c = confluence(pair)
        print()
        print(f"{pair}: {c['verdict']:<16s}  confluence: {c['confluence_score']:.2f}  net_dir: {c['net_direction']:+d}")
        for tf, r in c["timeframes"].items():
            if not r.get("available"):
                continue
            dir_arrow = "UP" if r["direction"] > 0 else ("DN" if r["direction"] < 0 else "->")
            print(f"  {tf:<3s}  {dir_arrow}  strength {r['strength']:.2f}  px ${r['current_price']:,.4f}")


if __name__ == "__main__":
    main()
