"""Cross-sectional momentum strategy across crypto majors.

Asness/Moskowitz/Pedersen (2013) "Value and Momentum Everywhere".
Rank pairs by trailing 30-day log return; long top tercile, short bottom.

For single-asset orchestrator integration, latest_signal(pair) returns
that pair's slot in the cross-section: +1 if top-ranked, -1 if bottom,
0 if middle.

VALIDATED = False. Counts as trial #48.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, evidence


VALIDATED = False
NAME = "xs_momentum"
UNIVERSE = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT")
LOOKBACK = 30


def latest_signal(pair: str = "BTC/USDT") -> float:
    if pair not in UNIVERSE:
        return 0.0
    try:
        prices = {p: data.ohlcv_extended(p, days_back=180)["close"] for p in UNIVERSE}
    except Exception:
        return 0.0
    aligned = pd.DataFrame(prices).dropna()
    if len(aligned) < LOOKBACK + 1:
        return 0.0
    log_ret = np.log(aligned / aligned.shift(1))
    trailing = log_ret.rolling(LOOKBACK).sum().iloc[-1]
    ranks = trailing.rank(pct=True)
    rank_for_pair = float(ranks.get(pair, 0.5))
    if rank_for_pair > 2.0 / 3.0:
        return 1.0
    if rank_for_pair < 1.0 / 3.0:
        return -1.0
    return 0.0


def evaluate_strict(pair: str = "BTC/USDT") -> dict:
    """Reuse the cross-sectional portfolio evaluator from research/."""
    from research.cross_sectional import evaluate_xs
    result = evaluate_xs(lookback=LOOKBACK, pairs=UNIVERSE, num_trials=50)
    evidence.record(NAME, "strict eval", result)
    return result


if __name__ == "__main__":
    print(f"latest signal for BTC: {latest_signal('BTC/USDT'):+.3f}")
    print(f"latest signal for ETH: {latest_signal('ETH/USDT'):+.3f}")
    print(f"latest signal for SOL: {latest_signal('SOL/USDT'):+.3f}")
