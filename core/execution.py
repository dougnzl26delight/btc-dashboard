"""Liquidity-aware execution helpers.

Before placing a market order, check the visible order book depth. If the
trade is a meaningful fraction of available liquidity, abort or slice. For
retail-sized paper trades on liquid majors this rarely fires — but it's the
guarantee against accidentally walking the book.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


# Practitioner defaults — tune for your style and market.
MAX_PCT_OF_BOOK = 0.05      # trade <= 5% of top-20-level book depth
MAX_SPREAD_BPS = 50         # abort if spread > 50 bps (0.5%)
MAX_CHUNK_USDT = 5_000.0    # slice trades larger than this (TWAP)


def check_liquidity(
    pair: str,
    side: str,
    quote_amount: float,
    max_pct_of_book: float = MAX_PCT_OF_BOOK,
    max_spread_bps: float = MAX_SPREAD_BPS,
) -> dict:
    """Inspect the live book before trading. Returns dict with ok flag, reason,
    available_depth, pct_of_depth, spread_bps."""
    try:
        book = data._EX.fetch_order_book(pair, limit=20)
    except Exception as e:
        return {"ok": False, "reason": f"book fetch failed: {e}"}

    asks = book.get("asks") or []
    bids = book.get("bids") or []
    if not asks or not bids:
        return {"ok": False, "reason": "empty book"}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000 if mid else 999

    levels = asks[:20] if side == "buy" else bids[:20]
    depth_quote = sum(p * sz for p, sz in levels)
    pct = quote_amount / depth_quote if depth_quote > 0 else 1.0

    ok = (
        spread_bps <= max_spread_bps
        and pct <= max_pct_of_book
        and depth_quote > 0
    )

    if spread_bps > max_spread_bps:
        reason = f"spread {spread_bps:.1f} bps > {max_spread_bps:.0f}"
    elif pct > max_pct_of_book:
        reason = f"trade is {pct:.1%} of top-20 depth (max {max_pct_of_book:.0%})"
    else:
        reason = "ok"

    return {
        "ok": ok,
        "reason": reason,
        "available_depth_quote": float(depth_quote),
        "pct_of_depth": float(pct),
        "spread_bps": float(spread_bps),
        "mid_price": float(mid),
    }


def slice_order(quote_amount: float, max_chunk: float = MAX_CHUNK_USDT) -> list[float]:
    """Split a large trade into TWAP-style equal chunks."""
    if quote_amount <= max_chunk:
        return [quote_amount]
    n_chunks = int(quote_amount / max_chunk) + (1 if quote_amount % max_chunk > 0 else 0)
    chunk_size = quote_amount / n_chunks
    return [chunk_size] * n_chunks
