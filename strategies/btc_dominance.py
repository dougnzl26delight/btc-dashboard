"""BTC-dominance momentum strategy.

VALIDATED = False. Counts as trial #54.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.btc_dominance import btc_dominance_signal


VALIDATED = False
NAME = "btc_dominance"


def latest_signal(pair: str = "BTC/USDT") -> float:
    try:
        return btc_dominance_signal(pair)
    except Exception:
        return 0.0


if __name__ == "__main__":
    for p in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        print(f"{p}: dominance signal = {latest_signal(p):+.4f}")
