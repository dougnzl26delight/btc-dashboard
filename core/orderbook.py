"""L2 order book analysis — depth, imbalance, spread, walls.

Pulls top-N levels of the order book via ccxt REST and computes:
    - Spread in basis points (mid-spread / mid-price)
    - Bid/ask depth imbalance (ratio of cumulative volume in N levels)
    - Wall detection (single orders > X bps of cum depth)
    - Price impact estimate (slippage for a given notional)

Use cases:
    - Pre-trade: skip if imbalance suggests adverse selection
    - Regime detection: persistent imbalance shifts = institutional flow
    - Slippage modeling: realistic execution cost per pair
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
        _EX_CACHE[name] = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 8000})
    return _EX_CACHE[name]


def fetch_book(pair: str, depth: int = 25, exchange: str = "binance") -> Optional[dict]:
    """Top-N order book. Returns {bids: [[px, qty]], asks: [[px, qty]]} or None."""
    try:
        ex = _ex(exchange)
        ob = ex.fetch_order_book(pair, depth)
        return {
            "bids": ob.get("bids", [])[:depth],
            "asks": ob.get("asks", [])[:depth],
            "timestamp": ob.get("timestamp"),
        }
    except Exception:
        return None


def analyze(pair: str, depth: int = 25, exchange: str = "binance",
            wall_threshold_bps: float = 50.0) -> dict:
    """Full order book analysis for one pair."""
    ob = fetch_book(pair, depth, exchange)
    if ob is None or not ob["bids"] or not ob["asks"]:
        return {"error": "no_book", "pair": pair}

    best_bid = ob["bids"][0][0]
    best_ask = ob["asks"][0][0]
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 10000 if mid > 0 else 0

    bid_qty = sum(b[1] for b in ob["bids"])
    ask_qty = sum(a[1] for a in ob["asks"])
    bid_notional = sum(b[0] * b[1] for b in ob["bids"])
    ask_notional = sum(a[0] * a[1] for a in ob["asks"])

    if (bid_notional + ask_notional) > 0:
        imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)
    else:
        imbalance = 0

    # Wall detection — orders larger than X bps of cumulative notional
    walls = []
    cum_bid_notional = 0
    for b in ob["bids"]:
        cum_bid_notional += b[0] * b[1]
        single_notional = b[0] * b[1]
        if cum_bid_notional > 0 and single_notional / cum_bid_notional > wall_threshold_bps / 100:
            walls.append({"side": "bid", "price": b[0], "qty": b[1], "notional": single_notional})
    cum_ask_notional = 0
    for a in ob["asks"]:
        cum_ask_notional += a[0] * a[1]
        single_notional = a[0] * a[1]
        if cum_ask_notional > 0 and single_notional / cum_ask_notional > wall_threshold_bps / 100:
            walls.append({"side": "ask", "price": a[0], "qty": a[1], "notional": single_notional})

    return {
        "pair": pair,
        "exchange": exchange,
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread_bps,
        "bid_qty_top25": bid_qty,
        "ask_qty_top25": ask_qty,
        "bid_notional_top25": bid_notional,
        "ask_notional_top25": ask_notional,
        "imbalance": imbalance,  # +1 = all bids, -1 = all asks
        "imbalance_interpretation": (
            "bullish_buyers_deep" if imbalance > 0.2 else
            "bearish_sellers_deep" if imbalance < -0.2 else
            "balanced"
        ),
        "walls_count": len(walls),
        "walls": walls[:3],  # top 3 to keep payload small
    }


def estimate_slippage_bps(pair: str, side: str, notional_usd: float,
                          exchange: str = "binance") -> dict:
    """Estimate the price impact of a market order.

    Walks the book, sums quantity-weighted price, computes effective vs mid.
    Useful for cost modeling on live execution.
    """
    ob = fetch_book(pair, 50, exchange)
    if ob is None:
        return {"error": "no_book"}
    if side == "buy":
        levels = ob["asks"]
    elif side == "sell":
        levels = ob["bids"]
    else:
        return {"error": "side must be buy|sell"}
    if not levels:
        return {"error": "empty_side"}

    mid = (ob["bids"][0][0] + ob["asks"][0][0]) / 2 if (ob["bids"] and ob["asks"]) else levels[0][0]
    remaining = notional_usd
    filled_notional = 0
    filled_qty = 0
    last_px = None
    for price, qty in levels:
        level_notional = price * qty
        if remaining <= 0:
            break
        take = min(remaining, level_notional)
        take_qty = take / price
        filled_qty += take_qty
        filled_notional += take_qty * price
        last_px = price
        remaining -= take

    if remaining > 0:
        return {"error": "insufficient_depth", "shortfall_usd": remaining}

    avg_price = filled_notional / filled_qty if filled_qty > 0 else 0
    slippage_bps = abs((avg_price - mid) / mid) * 10000 if mid > 0 else 0
    return {
        "pair": pair,
        "side": side,
        "notional_usd": notional_usd,
        "avg_fill_price": avg_price,
        "mid_price": mid,
        "slippage_bps": slippage_bps,
        "last_level_price": last_px,
        "levels_walked": int(filled_qty > 0),
    }


def main():
    """CLI: snapshot a few pairs."""
    print(f"{'Pair':<10s} {'Mid':>12s} {'SprBps':>7s} {'Imbal':>7s} {'Read':<24s} {'$5k buy bps':>12s}")
    print("-" * 90)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "TAO/USDT"]:
        a = analyze(pair)
        if "error" in a:
            print(f"{pair:<10s} ERROR: {a['error']}")
            continue
        slip = estimate_slippage_bps(pair, "buy", 5000)
        slip_str = f"{slip.get('slippage_bps', 0):.1f}" if "slippage_bps" in slip else "n/a"
        mid_str = f"${a['mid']:,.2f}" if a['mid'] > 100 else f"${a['mid']:,.4f}"
        print(f"{pair:<10s} {mid_str:>12s} {a['spread_bps']:>5.1f}bp "
              f"{a['imbalance']:>+6.0%} {a['imbalance_interpretation']:<24s} {slip_str:>10s}bp")


if __name__ == "__main__":
    main()
