"""Contrarian long/short ratio strategy.

VALIDATED = False. Counts as trial #52.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.long_short_ratio import long_short_signal


VALIDATED = False
NAME = "long_short_ratio"


def latest_signal(pair: str = "BTC/USDT") -> float:
    try:
        return long_short_signal(pair)
    except Exception:
        return 0.0


if __name__ == "__main__":
    for p in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        print(f"{p}: L/S contrarian signal = {latest_signal(p):+.4f}")
