"""Stablecoin-supply expansion strategy.

Same signal across all crypto pairs (it's a macro liquidity indicator).
VALIDATED = False. Counts as trial #53.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.stablecoin_supply import stablecoin_signal


VALIDATED = False
NAME = "stablecoin_supply"

_cached_signal = None


def latest_signal(pair: str = "BTC/USDT") -> float:
    """Same macro signal applied across all pairs."""
    global _cached_signal
    if _cached_signal is None:
        try:
            _cached_signal = stablecoin_signal()
        except Exception:
            _cached_signal = 0.0
    return _cached_signal


if __name__ == "__main__":
    print(f"Stablecoin supply signal: {latest_signal():+.4f}")
