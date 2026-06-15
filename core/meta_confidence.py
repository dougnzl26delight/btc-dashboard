"""Meta-labeling for sleeve position sizing — López de Prado AFML Ch 3.7.

Primary model: signal direction (long / flat / short) — your existing sleeves.
Secondary model: signal CONFIDENCE → position size multiplier [0.5, 1.5].

Currently every signal gets the same allocation. A "barely qualified" entry
gets the same size as a "screaming setup." This wastes information.

Implementation: per-sleeve confidence function that returns 0.5-1.5x based on
how far ABOVE threshold the signal is. Multiplied into the existing gates
pipeline (sleeve CB × Sharpe × loss-streak × correlation × event × META).

Caps the multiplier between 0.5 and 1.5 so it cannot dominate the other gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import data


# Confidence range — never let meta-label dominate other risk gates
META_MIN = 0.5
META_MAX = 1.5


def _clip(x: float, lo: float = META_MIN, hi: float = META_MAX) -> float:
    return max(lo, min(hi, x))


def oversold_bounce_confidence() -> float:
    """Confidence for oversold_bounce. Higher when MORE pairs at DEEPER RSI."""
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT",
                "DOGE/USDT", "AVAX/USDT", "BNB/USDT", "DOT/USDT", "ATOM/USDT"]
    THRESHOLD = 25
    rsi_values = []
    for pair in universe:
        try:
            df = data.ohlcv_extended(pair, days_back=30)
            if df.empty:
                continue
            close = df["close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)
            rsi_values.append(float(rsi.iloc[-1]))
        except Exception:
            continue
    if not rsi_values:
        return 1.0
    n_oversold = sum(1 for r in rsi_values if r < THRESHOLD)
    # Strength factors:
    #   "barely 3" oversold pairs   -> 0.6x (weak)
    #   "5+ pairs deeply oversold"  -> 1.4x (strong)
    if n_oversold == 0:
        return 0.5
    avg_rsi_when_oversold = np.mean([r for r in rsi_values if r < THRESHOLD])
    breadth_score = min(n_oversold / 6, 1.0)  # 0-1 based on breadth
    depth_score = max(0, (THRESHOLD - avg_rsi_when_oversold) / 15)  # 0-1 based on depth
    raw = 0.7 + 0.4 * breadth_score + 0.4 * depth_score
    return _clip(raw)


def overbought_fade_confidence() -> float:
    """Confidence for overbought_fade — mirror of oversold but RSI > 70."""
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT",
                "DOGE/USDT", "AVAX/USDT", "BNB/USDT", "DOT/USDT", "ATOM/USDT"]
    THRESHOLD = 70
    rsi_values = []
    for pair in universe:
        try:
            df = data.ohlcv_extended(pair, days_back=30)
            if df.empty:
                continue
            close = df["close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)
            rsi_values.append(float(rsi.iloc[-1]))
        except Exception:
            continue
    if not rsi_values:
        return 1.0
    n_overbought = sum(1 for r in rsi_values if r > THRESHOLD)
    if n_overbought == 0:
        return 0.5
    avg_rsi_when_ob = np.mean([r for r in rsi_values if r > THRESHOLD])
    breadth_score = min(n_overbought / 6, 1.0)
    depth_score = max(0, (avg_rsi_when_ob - THRESHOLD) / 15)
    raw = 0.7 + 0.4 * breadth_score + 0.4 * depth_score
    return _clip(raw)


def bah_btc_confidence() -> float:
    """Confidence for BAH BTC — higher when CYCLE_SCORE is deeper bear (more accumulation upside)."""
    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        score = cp.get("score")
        if score is None:
            return 1.0
        # Deep bear (0-20) -> highest confidence to accumulate
        # Euphoria (80+) -> lowest confidence (de-risk)
        if score < 20:
            return META_MAX  # full conviction
        if score > 80:
            return META_MIN  # severe caution
        # Linear interpolation
        return _clip(META_MAX - (META_MAX - META_MIN) * (score / 100))
    except Exception:
        return 1.0


def xsmom_confidence() -> float:
    """Confidence for XSMOM — higher when momentum dispersion is wide (clearer winners/losers)."""
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ATOM/USDT"]
    returns = []
    for pair in universe:
        try:
            df = data.ohlcv_extended(pair, days_back=20)
            if df.empty or len(df) < 14:
                continue
            r = float(df["close"].iloc[-1] / df["close"].iloc[-14] - 1)
            returns.append(r)
        except Exception:
            continue
    if len(returns) < 4:
        return 1.0
    spread = max(returns) - min(returns)
    # 0% spread -> 0.5x (no signal). 30%+ spread -> 1.4x (clear winners)
    raw = 0.5 + 3.0 * spread
    return _clip(raw)


def basis_arb_confidence() -> float:
    """Confidence for basis_arb — higher when more pairs have stable strong funding."""
    try:
        from strategies.funding_basis_arb import (
            rank_universe_by_funding, ENTRY_FUNDING_BPS_8H
        )
        UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT"]
        ranked = rank_universe_by_funding(UNIVERSE)
        qualified = [r for r in ranked if r.get("qualifies_for_entry")]
        n = len(qualified)
        if n == 0:
            return 0.5
        if n >= 4:
            return META_MAX
        return _clip(0.7 + 0.2 * n)
    except Exception:
        return 1.0


def pro_trend_confidence() -> float:
    """Confidence for pro_trend — higher when Donchian breakout is wider + TSMOM stronger."""
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=40)
        if df.empty:
            return 1.0
        close = df["close"]
        donch_high = float(df["high"].rolling(20).max().iloc[-1])
        donch_low = float(df["low"].rolling(20).min().iloc[-1])
        current = float(close.iloc[-1])
        donch_range = donch_high - donch_low
        if donch_range <= 0:
            return 1.0
        # Position within Donchian range — extremes get more confidence
        pos = (current - donch_low) / donch_range  # 0-1
        edge_distance = max(pos, 1 - pos) * 2 - 1  # 0 at center, 1 at extremes
        tsmom_30 = float(close.iloc[-1] / close.iloc[-31] - 1) if len(close) >= 31 else 0
        # Combine edge proximity + momentum strength
        raw = 0.7 + 0.5 * edge_distance + min(abs(tsmom_30), 0.3)
        return _clip(raw)
    except Exception:
        return 1.0


# Confidence dispatcher per sleeve
CONFIDENCE_FUNCS = {
    "oversold_bounce": oversold_bounce_confidence,
    "overbought_fade": overbought_fade_confidence,
    "bah_btc": bah_btc_confidence,
    "xsmom": xsmom_confidence,
    "basis_arb": basis_arb_confidence,
    "pro_trend": pro_trend_confidence,
}


def get_meta_confidence(sleeve: str) -> float:
    """Public API: return meta-confidence scale multiplier for a sleeve."""
    fn = CONFIDENCE_FUNCS.get(sleeve)
    if fn is None:
        return 1.0
    try:
        return fn()
    except Exception:
        return 1.0


def main():
    """CLI: show current meta-confidence per sleeve."""
    print("=" * 70)
    print("META-CONFIDENCE per sleeve (Lopez de Prado AFML 3.7)")
    print("=" * 70)
    print(f"  Range: [{META_MIN}, {META_MAX}]  — multiplier on position size")
    print()
    print(f"{'Sleeve':<22s} {'Confidence':>11s}  Interpretation")
    print("-" * 70)
    for s in CONFIDENCE_FUNCS:
        c = get_meta_confidence(s)
        if c >= 1.3:
            verdict = "STRONG signal — upsize"
        elif c >= 1.0:
            verdict = "normal"
        elif c >= 0.7:
            verdict = "weak — reduce size"
        else:
            verdict = "VERY WEAK — minimal exposure"
        print(f"  {s:<20s}  {c:>10.2f}x  {verdict}")


if __name__ == "__main__":
    main()
