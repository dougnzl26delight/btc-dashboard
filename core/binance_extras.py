"""Extra Binance public-API fetches not in the standard ccxt path.

Open interest history and global long/short account ratio. Both come from
Binance USDT-margined futures public endpoints — no API key required.
Cached per-call via core.cache.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cache import get_or_fetch


_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_LS_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"


def _spot_to_perp_symbol(pair: str) -> str:
    """BTC/USDT -> BTCUSDT (Binance futures symbol convention)."""
    return pair.replace("/", "")


def _http_get_json(url: str, params: dict, timeout: int = 15):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": "CryptoRig/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json
        return json.loads(r.read().decode("utf-8", errors="ignore"))


def fetch_open_interest_history(pair: str, period: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Open interest history for a perp.

    period: '5m','15m','30m','1h','2h','4h','6h','12h','1d'
    Returns DataFrame indexed by ts with columns: oi, oi_value (USD).
    """
    symbol = _spot_to_perp_symbol(pair)
    key = f"oi_{symbol}_{period}_{limit}"

    def fetch():
        return _http_get_json(_OI_URL, {"symbol": symbol, "period": period, "limit": limit})

    raw = get_or_fetch(key, fetch, ttl_s=3600)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    df["oi_value"] = df["sumOpenInterestValue"].astype(float)
    return df.set_index("ts")[["oi", "oi_value"]]


def fetch_long_short_ratio(pair: str, period: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Global long/short account ratio for a perp.

    Returns DataFrame indexed by ts with columns: long_account, short_account, ratio.
    Ratio > 1 means more accounts are long than short.
    """
    symbol = _spot_to_perp_symbol(pair)
    key = f"lsratio_{symbol}_{period}_{limit}"

    def fetch():
        return _http_get_json(_LS_URL, {"symbol": symbol, "period": period, "limit": limit})

    raw = get_or_fetch(key, fetch, ttl_s=3600)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_account"] = df["longAccount"].astype(float)
    df["short_account"] = df["shortAccount"].astype(float)
    df["ratio"] = df["longShortRatio"].astype(float)
    return df.set_index("ts")[["long_account", "short_account", "ratio"]]


def fetch_coingecko_market_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    """CoinGecko market cap history. coin_id e.g. 'tether', 'usd-coin', 'bitcoin'."""
    url = "https://api.coingecko.com/api/v3/coins/{cid}/market_chart".format(cid=coin_id)
    key = f"cg_chart_{coin_id}_{days}"

    def fetch():
        return _http_get_json(url, {"vs_currency": "usd", "days": days, "interval": "daily"})

    raw = get_or_fetch(key, fetch, ttl_s=3600)
    if not raw or "market_caps" not in raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw["market_caps"], columns=["ts", "market_cap"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def fetch_btc_dominance() -> float | None:
    """Current BTC dominance (% of total crypto market cap)."""
    try:
        raw = get_or_fetch("cg_global", lambda: _http_get_json("https://api.coingecko.com/api/v3/global", {}), ttl_s=3600)
        return float(raw["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None
