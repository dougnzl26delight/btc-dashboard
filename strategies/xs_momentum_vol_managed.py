"""Cross-sectional momentum with volatility management.

Han/Kang/Ryu (2024): cross-sectional momentum Sharpe 1.51 in crypto under
realistic assumptions, vs market portfolio 0.84.

Daniel/Moskowitz (2016): momentum suffers severe crashes. Volatility-managed
momentum (scale by 1/realized_vol) reduces tail crashes substantially.

Combined: rank-based long top tercile + scale by inverse strategy vol.

VALIDATED = False. Counts as trial #56.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


VALIDATED = False
NAME = "xs_momentum_vol_managed"
UNIVERSE = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT")
LOOKBACK = 30
TARGET_VOL_ANN = 0.20  # 20% annual target


def latest_signal(pair: str = "BTC/USDT") -> float:
    if pair not in UNIVERSE:
        return 0.0
    try:
        prices = {p: data.ohlcv_extended(p, days_back=180)["close"] for p in UNIVERSE}
    except Exception:
        return 0.0
    aligned = pd.DataFrame(prices).dropna()
    if len(aligned) < LOOKBACK + 30:
        return 0.0

    log_ret = np.log(aligned / aligned.shift(1))
    trailing = log_ret.rolling(LOOKBACK).sum().iloc[-1]
    ranks = trailing.rank(pct=True)

    # Raw rank-based signal
    rank_for_pair = float(ranks.get(pair, 0.5))
    if rank_for_pair > 2.0 / 3.0:
        raw = 1.0
    elif rank_for_pair < 1.0 / 3.0:
        raw = -1.0
    else:
        raw = 0.0

    if raw == 0:
        return 0.0

    # Vol management: scale by target_vol / pair_realized_vol
    pair_vol_ann = float(log_ret[pair].iloc[-LOOKBACK:].std() * np.sqrt(365))
    if pair_vol_ann <= 0:
        return raw
    scale = min(1.0, TARGET_VOL_ANN / pair_vol_ann)
    return raw * scale


if __name__ == "__main__":
    for p in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]:
        print(f"{p}: vol-managed XS momentum = {latest_signal(p):+.4f}")
