"""Open interest strategy — perp-derivatives flow signal.

Practitioner-standard signal in crypto. Free Binance public data.
VALIDATED = False. Counts as trial #51.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.open_interest import open_interest_signal


VALIDATED = False
NAME = "open_interest"


def latest_signal(pair: str = "BTC/USDT") -> float:
    try:
        return open_interest_signal(pair)
    except Exception:
        return 0.0


if __name__ == "__main__":
    for p in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        print(f"{p}: OI signal = {latest_signal(p):+.4f}")
