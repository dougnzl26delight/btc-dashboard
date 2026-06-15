"""Market data router. Single point of access for prices and OHLCV.

Read-only — does not depend on Broker so strategies can import without
instantiating any trading client.
"""

from __future__ import annotations

import ccxt
import pandas as pd

from .cache import get_or_fetch


_EX = ccxt.binance({"enableRateLimit": True})

# US cloud hosts (Streamlit Community Cloud, GitHub Actions runners) geo-block
# Binance. When a Binance OHLCV call fails there, fall back to yfinance (Yahoo),
# which is reachable from the US and carries deep daily history. Daily/weekly
# only — which is all the dashboard's cycle signals need. On Dave's NZ laptop
# Binance works, so this fallback never fires there.
_YF_TF = {"1d": "1d", "1w": "1wk", "1wk": "1wk", "1h": "1h", "1H": "1h"}


def _yf_bars(pair: str, timeframe: str, days_back: int) -> list:
    """Geo-agnostic OHLCV via yfinance -> ccxt [ts_ms, o, h, l, c, v] list."""
    try:
        import yfinance as yf
        base = pair.split("/")[0].upper()
        sym = {"BTC": "BTC-USD", "ETH": "ETH-USD"}.get(base, f"{base}-USD")
        iv = _YF_TF.get(timeframe, "1d")
        period = "60d" if iv == "1h" else f"{max(int(days_back) + 5, 400)}d"
        df = yf.download(sym, period=period, interval=iv, progress=False,
                         auto_adjust=False, threads=False)
        if df is None or len(df) == 0:
            return []

        def _col(name):
            s = df[name]
            return s.iloc[:, 0] if hasattr(s, "columns") else s

        o, h, l, c = _col("Open"), _col("High"), _col("Low"), _col("Close")
        v = _col("Volume")
        out = []
        for i in range(len(df)):
            ms = int(pd.Timestamp(df.index[i]).timestamp() * 1000)
            out.append([ms, float(o.iloc[i]), float(h.iloc[i]), float(l.iloc[i]),
                        float(c.iloc[i]), float(v.iloc[i]) if v is not None else 0.0])
        return out
    except Exception:
        return []


def ohlcv(
    pair: str,
    timeframe: str = "1d",
    limit: int = 365,
    ttl_s: int = 3600,
) -> pd.DataFrame:
    """Fetch OHLCV bars for a pair, cached. Returns DataFrame indexed by UTC ts."""
    key = f"ohlcv_{pair}_{timeframe}_{limit}"

    def fetch():
        try:
            bars = [list(b) for b in _EX.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)]
            if bars:
                return bars
        except Exception:
            pass
        return _yf_bars(pair, timeframe, limit if timeframe.startswith("1d") else limit * 7)

    raw = get_or_fetch(key, fetch, ttl_s)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def ohlcv_extended(
    pair: str, days_back: int = 1500, timeframe: str = "1d", ttl_s: int = 86_400
) -> pd.DataFrame:
    """Paginated OHLCV walking backward from now. Cached daily."""
    import time as _time

    key = f"ohlcv_ext_{pair}_{timeframe}_{days_back}"

    def fetch():
        end_ms = int(_time.time() * 1000)
        start_ms = end_ms - days_back * 86_400 * 1000
        cursor = start_ms
        records: list[list] = []
        for _ in range(20):
            try:
                chunk = _EX.fetch_ohlcv(pair, timeframe=timeframe, since=cursor, limit=1000)
            except Exception:
                break
            if not chunk:
                break
            records.extend([list(b) for b in chunk])
            last_ts = chunk[-1][0]
            if last_ts >= end_ms or len(chunk) < 50:
                break
            cursor = last_ts + 1
        if not records:
            # Binance geo-blocked (US cloud) — deep daily history via yfinance
            return _yf_bars(pair, timeframe, days_back)
        return records

    raw = get_or_fetch(key, fetch, ttl_s)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def ticker(pair: str) -> dict:
    return _EX.fetch_ticker(pair)


def funding_rate(pair: str) -> dict:
    """Latest funding rate for a perp contract.

    ccxt unified pair format for Binance USDT-margined perps: 'BTC/USDT:USDT'.
    """
    return _EX.fetch_funding_rate(pair)


def funding_history(pair: str, limit: int = 100) -> pd.DataFrame:
    """Recent funding rate history for a perp contract (single API call)."""
    rates = _EX.fetch_funding_rate_history(pair, limit=limit)
    df = pd.DataFrame(
        [
            {
                "ts": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": r["fundingRate"],
            }
            for r in rates
        ]
    )
    return df.set_index("ts") if not df.empty else df


def funding_history_extended(pair: str, days_back: int = 730) -> pd.DataFrame:
    """Paginated funding history for `days_back` days. Walks backward via `since`."""
    import time

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 86_400 * 1000
    cursor = start_ms
    all_records: list[dict] = []

    for _ in range(60):  # safety cap on pagination
        try:
            chunk = _EX.fetch_funding_rate_history(pair, since=cursor, limit=1000)
        except Exception:
            break
        if not chunk:
            break
        all_records.extend(chunk)
        last_ts = chunk[-1].get("timestamp", 0)
        if last_ts >= end_ms or len(chunk) < 50:
            break
        cursor = last_ts + 1

    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "ts": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": r["fundingRate"],
            }
            for r in all_records
        ]
    )
    return df.drop_duplicates("ts").set_index("ts").sort_index()
