"""W5: FRED alternates + health check.

When FRED CSV endpoint is unreachable from the user's network, this
module provides yfinance-based proxies for the most important series.

Health check: `python -m core.fred_alternates health`
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Map: FRED series ID -> yfinance proxy ticker + description
PROXIES = {
    # Yield curve — TLT (20y) - SHY (1-3y) proxy spread
    "T10Y2Y":      {"proxy": "tlt_shy_spread",
                    "desc":   "TLT - SHY price ratio proxy for 10y-2y curve"},
    # 10y Treasury yield — ^TNX (yfinance has this directly)
    "DGS10":       {"proxy": "^TNX",
                    "desc":   "10y Treasury yield (in tenths of %)"},
    # HY spread — HYG/TLT ratio fallback
    "BAMLH0A0HYM2":{"proxy": "hyg_tlt_ratio",
                    "desc":   "HYG/TLT ratio proxy for HY spread"},
    # DXY — DX-Y.NYB on yfinance (or UUP ETF)
    "DTWEXBGS":    {"proxy": "UUP",
                    "desc":   "UUP ETF as DXY proxy"},
    # M2 Money Supply — no direct proxy, but ^SPX vs gold ratio captures monetary expansion
    "M2SL":        {"proxy": None, "desc": "no good proxy — keep as 'unavailable'"},
    # MOVE Index — yfinance ^MOVE
    "MOVE":        {"proxy": "^MOVE", "desc": "MOVE Index direct"},
    # VIX — yfinance ^VIX
    "VIXCLS":      {"proxy": "^VIX", "desc": "VIX direct"},
    # SPY for SPX
    "SP500":       {"proxy": "SPY", "desc": "SPY for S&P 500"},
}


def _yf(ticker: str, period: str = "5y") -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty: return None
        s = df["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def fred_alternate(series_id: str, period: str = "5y") -> Optional[pd.Series]:
    """Try to fetch a yfinance proxy for the requested FRED series.

    Returns None if no proxy is available or fetch fails.
    """
    spec = PROXIES.get(series_id)
    if spec is None or spec.get("proxy") is None: return None
    proxy = spec["proxy"]

    if proxy == "tlt_shy_spread":
        tlt = _yf("TLT", period=period)
        shy = _yf("SHY", period=period)
        if tlt is None or shy is None: return None
        df = pd.concat([tlt, shy], axis=1).dropna()
        df.columns = ["tlt", "shy"]
        # TLT (long duration) going UP relative to SHY = curve steepening
        # Map to T10Y2Y-like reading: positive = steeper
        return ((df["tlt"] / df["shy"]) - (df["tlt"] / df["shy"]).rolling(252).median()) * 100

    if proxy == "hyg_tlt_ratio":
        hyg = _yf("HYG", period=period)
        tlt = _yf("TLT", period=period)
        if hyg is None or tlt is None: return None
        df = pd.concat([hyg, tlt], axis=1).dropna()
        df.columns = ["hyg", "tlt"]
        # HYG/TLT falling = HY stress, equivalent to spreads widening
        # Map to bps-like: invert and scale
        ratio = df["hyg"] / df["tlt"]
        # Convert to bps-like: lower ratio = wider spread
        return (1 - ratio / ratio.rolling(252).median()) * 1000

    if proxy == "^TNX":
        s = _yf("^TNX", period=period)
        # ^TNX is already in basis points / 10 — divide
        return None if s is None else s / 10

    # Direct ticker proxy
    return _yf(proxy, period=period)


# ============================================================
# Health check
# ============================================================

def fred_health() -> dict:
    """Test if FRED CSV endpoint is reachable."""
    import time
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    t0 = time.time()
    try:
        r = requests.get(url, timeout=8,
                          headers={"User-Agent": "Mozilla/5.0"})
        elapsed = time.time() - t0
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "elapsed_s": round(elapsed, 2),
            "body_size": len(r.text) if r.text else 0,
            "url": url,
        }
    except Exception as e:
        return {
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(e)[:120],
            "url": url,
        }


def alternates_health() -> dict:
    """Test that yfinance alternates are reachable."""
    import time
    results = {}
    for series_id, spec in PROXIES.items():
        if spec.get("proxy") is None:
            results[series_id] = {"available": False, "reason": "no_proxy_defined"}
            continue
        t0 = time.time()
        s = fred_alternate(series_id)
        results[series_id] = {
            "available": s is not None and len(s.dropna()) > 30,
            "n_points": int(len(s.dropna())) if s is not None else 0,
            "elapsed_s": round(time.time() - t0, 2),
            "proxy": spec["proxy"],
            "latest_value": float(s.iloc[-1]) if s is not None and len(s) > 0 else None,
        }
    return results


def overall_health() -> dict:
    f = fred_health()
    a = alternates_health()
    n_alt_ok = sum(1 for v in a.values() if v.get("available"))
    return {
        "fred_reachable": f.get("ok", False),
        "fred_response_time_s": f.get("elapsed_s"),
        "fred_error": f.get("error"),
        "n_alternates_available": n_alt_ok,
        "n_alternates_total": len(a),
        "alternates_status": a,
        "recommendation": ("FRED healthy" if f.get("ok")
                            else f"FRED DOWN — use {n_alt_ok} yfinance proxies"),
    }


def main():
    import json
    h = overall_health()
    print("=" * 70)
    print("FRED + ALTERNATES HEALTH")
    print("=" * 70)
    print(f"  FRED reachable:     {h['fred_reachable']}")
    print(f"  FRED response time: {h.get('fred_response_time_s', '?')}s")
    if h.get("fred_error"):
        print(f"  FRED error:         {h['fred_error']}")
    print(f"  yfinance alternates: {h['n_alternates_available']}/{h['n_alternates_total']} working")
    print(f"  Recommendation:     {h['recommendation']}")
    print(f"\n  Per-alternate status:")
    for series, info in h["alternates_status"].items():
        if info.get("available"):
            print(f"    [OK ] {series:15s} via {info.get('proxy', '?'):15s}  "
                  f"latest={info.get('latest_value', '?')}")
        else:
            print(f"    [---] {series:15s} unavailable ({info.get('reason', 'fetch_failed')})")


if __name__ == "__main__":
    main()
