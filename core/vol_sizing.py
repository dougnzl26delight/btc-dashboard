"""Inverse-volatility position sizing within baskets.

Equal-weight is naive: a basket of [BTC, SOL, ONDO] at 33% each over-bets the
more volatile names. SOL is ~1.8x BTC vol; ONDO is ~3x. Equal-weight means
the basket P&L is dominated by ONDO swings, not your edge.

Inverse-vol weighting fixes this:
    weight_i = (1 / vol_i) / sum(1 / vol_j for all j)

Result: each position contributes ROUGHLY equal risk (in vol terms), so the
basket's outcome depends on your signal quality, not on which asset happens
to be vol-leader that week.

Used by oversold_bounce, overbought_fade, XSMOM. Falls back to equal-weight
if vol data is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import data


LOOKBACK_DAYS = 30
DEFAULT_VOL_FALLBACK = 0.05  # 5% daily — conservative for crypto


def _realized_vol(pair: str) -> float:
    """30-day realized daily vol (decimal). Returns DEFAULT_VOL_FALLBACK on error."""
    try:
        df = data.ohlcv_extended(pair, days_back=LOOKBACK_DAYS + 10)
        if df.empty or len(df) < LOOKBACK_DAYS:
            return DEFAULT_VOL_FALLBACK
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(rets.iloc[-LOOKBACK_DAYS:].std())
    except Exception:
        return DEFAULT_VOL_FALLBACK


def vol_weighted_allocation(pairs: Sequence[str], total_capital: float,
                             clip_to: tuple[float, float] | None = (0.5, 2.0)) -> dict[str, float]:
    """Allocate total_capital across pairs using inverse-vol weights.

    Args:
        pairs: list of pair symbols (e.g., ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        total_capital: total $ to spread across the basket
        clip_to: (min_mult, max_mult) — clip individual weights so no single
                 pair gets less than min_mult * equal-weight or more than
                 max_mult * equal-weight. Default (0.5, 2.0) prevents any single
                 position from being <0.5x or >2x the equal-weight allocation.

    Returns: {pair: capital_to_allocate}
    """
    if not pairs:
        return {}

    vols = {p: _realized_vol(p) for p in pairs}

    # Inverse-vol raw weights
    inv_vols = {p: (1.0 / v) if v > 0 else 0 for p, v in vols.items()}
    total_inv = sum(inv_vols.values())
    if total_inv <= 0:
        # Fallback: equal weight
        return {p: total_capital / len(pairs) for p in pairs}

    raw_weights = {p: iv / total_inv for p, iv in inv_vols.items()}

    # Optional clipping to prevent extreme concentration
    if clip_to:
        eq_weight = 1.0 / len(pairs)
        min_w, max_w = eq_weight * clip_to[0], eq_weight * clip_to[1]
        clipped = {p: max(min_w, min(max_w, w)) for p, w in raw_weights.items()}
        # Re-normalize after clipping
        total_clipped = sum(clipped.values())
        if total_clipped > 0:
            raw_weights = {p: w / total_clipped for p, w in clipped.items()}

    return {p: raw_weights[p] * total_capital for p in pairs}


def vol_weighted_per_unit_size(pair: str, basket_total: float,
                                 n_positions: int, target_vol_share: float = 1.0) -> float:
    """Single-pair version: given basket total and N total positions, return
    THIS pair's allocation under inverse-vol weighting.

    Useful when sizing iteratively (one pair at a time) rather than computing
    the full basket up front.
    """
    eq_weight = basket_total / max(n_positions, 1)
    vol = _realized_vol(pair)
    if vol <= 0:
        return eq_weight
    # Scale: inverse-vol relative to a 5% reference (rough crypto median)
    REFERENCE_VOL = 0.05
    scale = REFERENCE_VOL / vol
    return eq_weight * scale * target_vol_share


def main():
    """CLI: show vol-weighted allocation for the standard universe."""
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]
    basket = 15_000
    print(f"Vol-weighted allocation across {pairs}, basket ${basket:,.0f}:")
    print()
    allocations = vol_weighted_allocation(pairs, basket)
    for p, alloc in allocations.items():
        vol = _realized_vol(p)
        print(f"  {p:<12s}  vol={vol*100:>4.1f}%/day  alloc=${alloc:>7,.0f}  ({alloc/basket*100:>4.1f}% of basket)")
    print()
    total = sum(allocations.values())
    print(f"  Total allocated: ${total:,.0f}")
    eq = basket / len(pairs)
    print(f"  Vs equal-weight ${eq:,.0f} each")


if __name__ == "__main__":
    main()
