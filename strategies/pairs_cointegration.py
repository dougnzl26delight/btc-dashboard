"""Cointegrated pairs trading — proper stat-arb.

Different from our existing naive ETH/BTC z-score reversion:
  - Tests cointegration first (Engle-Granger + ADF)
  - Fits OU process for proper entry/exit thresholds
  - Hedge ratio from regression, not 1:1

Universe: scan top crypto pairs for cointegrated relationships.
Documented edge: stat arb has ~0.5-1.0 Sharpe in liquid pairs.

VALIDATED = False. Counts as trial #55.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core import cointegration, data


VALIDATED = False
NAME = "pairs_cointegration"

# Universe to scan for cointegrated relationships
SCAN_UNIVERSE = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT")
LOOKBACK_DAYS = 365
ENTRY_Z = 2.0
EXIT_Z = 0.5


def find_best_cointegrated_pair() -> tuple[str, str, dict] | None:
    """Scan universe and return the best cointegrated pair (lowest p-value)."""
    prices = pd.DataFrame({
        p: data.ohlcv_extended(p, days_back=LOOKBACK_DAYS)["close"]
        for p in SCAN_UNIVERSE
    }).dropna()

    results = cointegration.find_cointegrated_pairs(prices)
    cointegrated = results[results["is_cointegrated"]]
    if cointegrated.empty:
        return None
    best = cointegrated.iloc[0]
    return (
        best["pair_a"],
        best["pair_b"],
        {
            "coint_p": float(best["coint_p"]),
            "adf_p": float(best["adf_p"]),
            "hedge_ratio": float(best["hedge_ratio"]),
        },
    )


def latest_signal(pair: str = "BTC/USDT") -> float:
    """Return signal for `pair` based on its position in the best cointegrated pair.

    If `pair` is the LONG leg (s1), positive z-score → spread above mean → SHORT pair (signal -1).
    Inverse for SHORT leg.
    """
    best = find_best_cointegrated_pair()
    if best is None:
        return 0.0
    pair_a, pair_b, meta = best
    if pair not in (pair_a, pair_b):
        return 0.0

    prices_a = data.ohlcv_extended(pair_a, days_back=LOOKBACK_DAYS)["close"]
    prices_b = data.ohlcv_extended(pair_b, days_back=LOOKBACK_DAYS)["close"]
    test = cointegration.engle_granger_test(prices_a, prices_b)
    if not test["is_cointegrated"]:
        return 0.0

    spread = test["spread_series"]
    z = (spread.iloc[-1] - test["spread_mean"]) / test["spread_std"]

    if abs(z) < ENTRY_Z:
        return 0.0

    # Spread = s1 - β*s2. If z > 0, spread above mean → s1 overvalued vs s2 → short s1, long s2.
    if pair == pair_a:
        return float(-z / ENTRY_Z)  # contrarian on s1
    return float(z / ENTRY_Z)        # opposite for s2


if __name__ == "__main__":
    best = find_best_cointegrated_pair()
    if best:
        a, b, meta = best
        print(f"Best cointegrated pair: {a} / {b}")
        print(f"  coint p: {meta['coint_p']:.4f}, adf p: {meta['adf_p']:.4f}, hedge ratio: {meta['hedge_ratio']:.4f}")
        print(f"  Signal for {a}: {latest_signal(a):+.4f}")
        print(f"  Signal for {b}: {latest_signal(b):+.4f}")
    else:
        print("No cointegrated pairs found in scan universe.")
