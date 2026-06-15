"""Liquidation cascade detector — free OI + funding-based proxy.

Coinglass and similar paid services aggregate exchange-wide leveraged position
data. Without paid feeds we proxy using:
    - Open Interest (Binance public)
    - Funding rate (already monitored)
    - 24h price change

High OI + high funding (in one direction) + 24h move opposite that direction
= liquidation cascade probable. Trade the wick reversal.

This is one of the most consistent intraday edges in crypto. Per [Kingfisher
liquidation maps], 60-70% of large directional moves end at liquidation
cluster levels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt


_EX_CACHE: dict[str, ccxt.Exchange] = {}


def _ex(name: str = "binance") -> ccxt.Exchange:
    if name not in _EX_CACHE:
        cls = getattr(ccxt, name)
        _EX_CACHE[name] = cls({
            "enableRateLimit": True,
            "timeout": 8000,
            "options": {"defaultType": "swap"},
        })
    return _EX_CACHE[name]


def open_interest(pair: str = "BTC/USDT") -> Optional[dict]:
    """Fetch open interest for a perp pair via Binance public API."""
    try:
        ex = _ex("binance")
        perp = pair if ":" in pair else f"{pair}:{pair.split('/')[1]}"
        oi = ex.fetch_open_interest(perp)
        return {
            "open_interest_usd": float(oi.get("openInterestAmount") or 0),
            "open_interest_contracts": float(oi.get("openInterestValue") or 0),
            "timestamp": oi.get("timestamp"),
        }
    except Exception:
        return None


def liquidation_pressure(pair: str = "BTC/USDT") -> dict:
    """Compose OI + funding + price-action into liquidation-cascade probability.

    Returns:
        {
            cascade_long_probability: 0-1   (longs likely to cascade)
            cascade_short_probability: 0-1  (shorts likely to cascade)
            edge_direction: 'fade_long' | 'fade_short' | 'no_edge'
            reasoning: list[str]
        }
    """
    try:
        from core import data
        from strategies.funding_basis_arb import latest_funding_bps_8h
    except Exception:
        return {"error": "imports_failed"}

    reasoning = []
    cascade_long_prob = 0.0
    cascade_short_prob = 0.0

    # 1. Funding-rate read
    funding_bps = latest_funding_bps_8h(pair)
    funding_annualized = funding_bps * 3 * 365 / 100
    if funding_bps > 5:
        cascade_long_prob += 0.4
        reasoning.append(f"Funding {funding_bps:+.2f}bp/8h = {funding_annualized:+.1f}% APR — longs paying heavily; crowded long")
    elif funding_bps < -5:
        cascade_short_prob += 0.4
        reasoning.append(f"Funding {funding_bps:+.2f}bp/8h — shorts paying; crowded short")

    # 2. OI trend
    oi = open_interest(pair)
    if oi:
        reasoning.append(f"Open Interest: ${oi.get('open_interest_usd', 0)/1e6:,.1f}M")

    # 3. 24h price move
    try:
        df = data.ohlcv_extended(pair, days_back=3)
        if not df.empty and len(df) >= 2:
            ret_24h = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
            ret_5d = float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) >= 6 else 0
            reasoning.append(f"24h move: {ret_24h*100:+.2f}%   5d move: {ret_5d*100:+.2f}%")

            # Cascade probability: funding direction + opposite price move
            if funding_bps > 5 and ret_24h < -0.04:
                # Crowded longs + sudden drop = long liquidation cascade
                cascade_long_prob += 0.4
                reasoning.append("CASCADE: high funding + recent drop = long liquidation pressure")
            elif funding_bps < -5 and ret_24h > 0.04:
                cascade_short_prob += 0.4
                reasoning.append("CASCADE: negative funding + recent rally = short squeeze risk")
    except Exception:
        pass

    # Determine edge
    if cascade_long_prob > 0.5:
        edge_direction = "fade_long"
        reasoning.append("EDGE: Long cascade likely. Fade rally OR buy the dip after capitulation.")
    elif cascade_short_prob > 0.5:
        edge_direction = "fade_short"
        reasoning.append("EDGE: Short squeeze risk. Buy the breakout OR fade panic-shorts.")
    else:
        edge_direction = "no_edge"

    return {
        "pair": pair,
        "cascade_long_probability": cascade_long_prob,
        "cascade_short_probability": cascade_short_prob,
        "edge_direction": edge_direction,
        "funding_bps_8h": funding_bps,
        "funding_annualized_pct": funding_annualized,
        "reasoning": reasoning,
    }


def main():
    print("=" * 80)
    print("LIQUIDATION PRESSURE READ — proxy via OI + funding + 24h move")
    print("=" * 80)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        r = liquidation_pressure(pair)
        print()
        print(f"{pair}:")
        print(f"  Edge: {r.get('edge_direction', '?')}")
        print(f"  P(long cascade):  {r.get('cascade_long_probability', 0)*100:.0f}%")
        print(f"  P(short cascade): {r.get('cascade_short_probability', 0)*100:.0f}%")
        for line in r.get("reasoning", []):
            print(f"    - {line}")


if __name__ == "__main__":
    main()
