"""Short-term momentum — 10-day TSMOM for fast-twitch directional plays.

Standard practitioner timeframe; complements the slower 30/90/180-day
TSMOM in `tsmom`. Adds a short-horizon axis to the strategy mix.
With realistic costs in mind, applied via continuous weights (not discrete).

VALIDATED = False. Counts as trial #50.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, evidence
from research import signals as res_sig


VALIDATED = False
NAME = "short_term_momentum"
LOOKBACK = 10


def latest_signal(pair: str = "BTC/USDT") -> float:
    try:
        df = data.ohlcv_extended(pair, days_back=120)
    except Exception:
        return 0.0
    if len(df) < LOOKBACK * 2:
        return 0.0
    sig = res_sig.tsmom_single(df["close"], lookback=LOOKBACK)
    return float(sig.iloc[-1]) if not sig.empty else 0.0


def evaluate_strict(pair: str = "BTC/USDT") -> dict:
    return {
        "see": "research/sweep.py — tsmom_single sweep (lookbacks 30, 60, 90, 120, 180, 365)",
        "validated": False,
    }


if __name__ == "__main__":
    print(f"short_term_momentum signal: {latest_signal():+.3f}")
