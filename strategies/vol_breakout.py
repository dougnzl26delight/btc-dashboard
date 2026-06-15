"""Volatility breakout — directional bet when realized vol expands.

Pattern: when 30-day realized vol breaks above its EWMA, the prevailing
direction (sign of trailing return) tends to extend. Documented in
Faber (2013) "A Quantitative Approach to Tactical Asset Allocation"
and standard CTA practitioner literature (Wisdom Tree, AHL).

VALIDATED = False. Counts as trial #49.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, evidence
from research import signals as res_sig


VALIDATED = False
NAME = "vol_breakout"
WINDOW = 30


def latest_signal(pair: str = "BTC/USDT") -> float:
    try:
        df = data.ohlcv_extended(pair, days_back=180)
    except Exception:
        return 0.0
    if len(df) < WINDOW * 3:
        return 0.0
    sig = res_sig.vol_breakout(df["close"], window=WINDOW)
    return float(sig.iloc[-1]) if not sig.empty else 0.0


def evaluate_strict(pair: str = "BTC/USDT") -> dict:
    """In-sample sanity check — full strict eval was done in research/sweep.py."""
    return {
        "see": "research/sweep.py results — vol_breakout_30 OOS Sharpe ~0.05",
        "validated": False,
    }


if __name__ == "__main__":
    print(f"vol_breakout latest signal: {latest_signal():+.3f}")
