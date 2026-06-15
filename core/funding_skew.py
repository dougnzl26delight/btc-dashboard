"""Multi-exchange funding rate comparison.

Pulls perp funding from Binance, Bybit, OKX (and any other supported by ccxt).
Computes:
    - Mean cross-venue funding
    - Dispersion (max - min)
    - Per-exchange skew vs mean

Use cases:
    - Cross-venue arb: when one exchange's funding diverges > 5bp from others,
      that's a tradeable opportunity (long on the cheap, short on the rich)
    - Signal confirmation: high cross-venue funding (all exchanges +ve) = strong
      long positioning = potential setup for short-fade
    - Regime detection: persistent skew often precedes price moves
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ccxt


# ccxt-supported derivatives exchanges with public funding data
SUPPORTED_EXCHANGES = ["binance", "bybit", "okx"]

# Per-exchange pair format (futures symbols differ from spot)
PERP_SUFFIX = {
    "binance": ":USDT",     # BTC/USDT:USDT
    "bybit": ":USDT",
    "okx": ":USDT",
}

_EX_CACHE: dict[str, ccxt.Exchange] = {}


def _ex(name: str) -> ccxt.Exchange:
    if name not in _EX_CACHE:
        cls = getattr(ccxt, name)
        _EX_CACHE[name] = cls({"enableRateLimit": True, "timeout": 8000,
                                "options": {"defaultType": "swap"}})
    return _EX_CACHE[name]


def fetch_funding(pair: str, exchange: str) -> Optional[float]:
    """Latest funding rate as decimal (e.g., 0.0001 = 0.01% per 8h).

    Returns None if unavailable. Uses ccxt's fetch_funding_rate which is
    pretty universal.
    """
    try:
        ex = _ex(exchange)
        perp_symbol = pair + PERP_SUFFIX.get(exchange, "")
        rate = ex.fetch_funding_rate(perp_symbol)
        return float(rate.get("fundingRate") or 0)
    except Exception:
        return None


def skew_analysis(pair: str, exchanges: Optional[list[str]] = None) -> dict:
    """Compare funding across exchanges. Returns mean, dispersion, per-venue skew."""
    if exchanges is None:
        exchanges = SUPPORTED_EXCHANGES

    rates = {}
    for ex_name in exchanges:
        r = fetch_funding(pair, ex_name)
        if r is not None:
            rates[ex_name] = r

    if not rates:
        return {"pair": pair, "rates": {}, "mean": None, "error": "no_data"}

    values = list(rates.values())
    mean = sum(values) / len(values)
    dispersion = max(values) - min(values) if len(values) > 1 else 0
    skews = {name: rate - mean for name, rate in rates.items()}

    # Annualized for human reading (3 fundings/day * 365)
    ANN_FACTOR = 3 * 365

    # Detect arb-ish opportunities (>5bp/8h dispersion = meaningful)
    arb_opportunity = abs(dispersion) > 0.0005
    arb_side = None
    if arb_opportunity and len(rates) > 1:
        long_ex = min(rates, key=rates.get)   # lowest funding = cheapest to be long
        short_ex = max(rates, key=rates.get)  # highest funding = most paid to be short
        arb_side = {"long": long_ex, "short": short_ex,
                    "edge_bps_per_8h": dispersion * 10000}

    # Regime read on absolute level
    if mean > 0.0002:  # > 2bp / 8h sustained
        regime = "longs_paying_heavily"  # bearish bias — longs are crowded
    elif mean < -0.0002:
        regime = "shorts_paying_heavily"  # bullish bias — shorts are crowded
    elif abs(mean) < 0.0001:
        regime = "neutral_low"
    else:
        regime = "normal"

    return {
        "pair": pair,
        "rates": rates,
        "mean_8h": mean,
        "mean_annualized_pct": mean * ANN_FACTOR * 100,
        "dispersion_8h": dispersion,
        "dispersion_bps": dispersion * 10000,
        "per_exchange_skew": skews,
        "regime": regime,
        "arb_opportunity": arb_opportunity,
        "arb_details": arb_side,
    }


def main():
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]
    print(f"{'Pair':<10s} {'Mean 8h':>9s} {'Mean ann%':>11s} {'Dispersion':>11s} {'Regime':<28s} {'Arb?':<10s}")
    print("-" * 90)
    for pair in pairs:
        s = skew_analysis(pair)
        if s.get("error"):
            print(f"{pair:<10s} {s['error']}")
            continue
        arb_str = "YES" if s.get("arb_opportunity") else "no"
        print(f"{pair:<10s} {s['mean_8h']*10000:>+7.2f}bp  {s['mean_annualized_pct']:>+9.2f}%  "
              f"{s['dispersion_bps']:>+9.2f}bp  {s['regime']:<28s} {arb_str:<10s}")
        if s.get("arb_details"):
            d = s["arb_details"]
            print(f"           ARB: long {d['long']} / short {d['short']}, edge {d['edge_bps_per_8h']:.2f}bp/8h")


if __name__ == "__main__":
    main()
