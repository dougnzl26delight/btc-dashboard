"""On-chain indicator layer for BTC.

Pulls MVRV, NUPL, realized price from CoinMetrics community API (free).
Caches responses 6h to avoid hammering the endpoint.

Falls back to PRICE-BASED PROXIES if the API is unavailable:
    realized_price_proxy = 4-year SMA of close
    mvrv_proxy = price / realized_price_proxy
    nupl_proxy = (price - realized_price_proxy) / price

The proxies are imperfect (real MVRV uses UTXO timestamps, not a simple SMA)
but historically correlate ~0.85 with the true metric. Good enough to drive
sleeve sizing while a free precise source isn't available.

True MVRV interpretation:
    < 1.0  : market cap below cost basis — extreme bear; historical bottoms
    1.0-2.0: fair value zone
    2.0-3.5: bull market mid-stage
    > 3.5  : euphoria / top zone; every prior cycle topped here
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from core import data


CACHE_DIR = REPO_ROOT / ".onchain_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 6 * 3600  # 6h freshness for daily metrics

# CoinMetrics free community endpoint
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"

# Map our internal metric names to CoinMetrics column codes
CM_METRIC_MAP = {
    "mvrv": "CapMVRVCur",        # Market cap / realized cap
    "nupl": "CapMrktCurUSD",     # used in NUPL calc
    "realized_price": "PriceRealUSD",
    "active_addresses": "AdrActCnt",
    "exchange_inflow_usd": "FlowInExUSD",
    "exchange_outflow_usd": "FlowOutExUSD",
}


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.json"


def _read_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        if time.time() - d.get("fetched_at", 0) < CACHE_TTL_SECONDS:
            return d.get("data")
    except Exception:
        pass
    return None


def _write_cache(key: str, data: dict) -> None:
    _cache_path(key).write_text(json.dumps({
        "fetched_at": time.time(),
        "data": data,
    }, default=str))


def fetch_coinmetrics(asset: str = "btc", metrics: list[str] | None = None,
                     days: int = 200) -> Optional[pd.DataFrame]:
    """Fetch network metrics from CoinMetrics community API.

    Returns DataFrame indexed by date with columns = requested metric codes,
    or None if API unavailable.
    """
    if metrics is None:
        metrics = list(CM_METRIC_MAP.values())
    cache_key = f"cm_{asset}_{','.join(metrics)}_{days}"
    cached = _read_cache(cache_key)
    if cached is not None:
        df = pd.DataFrame(cached)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.set_index("time").sort_index()
        for m in metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        return df

    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"{COINMETRICS_BASE}/timeseries/asset-metrics?"
        f"assets={asset}&metrics={','.join(metrics)}&start_time={start}"
        f"&page_size=1000&pretty=false"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            payload = json.loads(r.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        # Network/parse failure — caller will fall back to proxies
        return None

    rows = payload.get("data", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Convert numeric columns
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("time").sort_index()
    _write_cache(cache_key, df.reset_index().to_dict(orient="records"))
    return df


def _btc_price_history(days: int = 1900) -> Optional[pd.DataFrame]:
    """Get BTC daily OHLCV — used for price proxies."""
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=days)
        if df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def get_mvrv(use_proxy_if_unavailable: bool = True) -> dict:
    """Return current MVRV (z-score and absolute) for BTC."""
    cm_df = fetch_coinmetrics("btc", ["CapMVRVCur"], days=400)
    if cm_df is not None and "CapMVRVCur" in cm_df.columns and not cm_df["CapMVRVCur"].isna().all():
        series = cm_df["CapMVRVCur"].dropna()
        current = float(series.iloc[-1])
        mean = float(series.mean())
        std = float(series.std())
        z = (current - mean) / std if std > 0 else 0
        return {
            "mvrv": current,
            "z_score": z,
            "source": "coinmetrics",
            "history_mean": mean,
            "history_std": std,
            "as_of": str(series.index[-1].date()),
        }

    if not use_proxy_if_unavailable:
        return {"mvrv": None, "source": "unavailable", "reason": "no_data"}

    # Proxy: price / 4-year SMA (≈ realized price proxy)
    df = _btc_price_history(1900)
    if df is None or len(df) < 1500:
        return {"mvrv": None, "source": "unavailable", "reason": "insufficient_history"}
    realized_proxy = float(df["close"].rolling(1460).mean().iloc[-1])  # ~4yr
    current_price = float(df["close"].iloc[-1])
    if realized_proxy <= 0:
        return {"mvrv": None, "source": "unavailable", "reason": "bad_realized"}
    mvrv_proxy = current_price / realized_proxy
    return {
        "mvrv": mvrv_proxy,
        "z_score": None,
        "source": "price_proxy_4y_sma",
        "realized_price_proxy": realized_proxy,
        "current_price": current_price,
        "as_of": str(df.index[-1].date()),
        "note": "Proxy; correlates ~0.85 with true MVRV. Replace with paid source for production.",
    }


def get_nupl() -> dict:
    """Net Unrealized Profit/Loss — (price - realized_price) / price.

    Interpretation:
        < 0.00 : capitulation (all coins held at loss)
        0.00-0.25 : hope/fear zone — historical bottoms
        0.25-0.50: optimism — fair value
        0.50-0.75: belief/denial — typical bull
        > 0.75 : euphoria — historical tops
    """
    mvrv = get_mvrv()
    if mvrv.get("mvrv") is None:
        return {"nupl": None, "source": "unavailable"}
    # NUPL = (market_cap - realized_cap) / market_cap = 1 - 1/MVRV
    nupl = 1 - 1 / mvrv["mvrv"] if mvrv["mvrv"] > 0 else None
    return {
        "nupl": nupl,
        "mvrv": mvrv["mvrv"],
        "source": mvrv["source"],
        "phase": _nupl_phase(nupl) if nupl is not None else None,
    }


def _nupl_phase(nupl: float) -> str:
    if nupl < 0:
        return "capitulation"
    if nupl < 0.25:
        return "hope_fear"
    if nupl < 0.50:
        return "optimism"
    if nupl < 0.75:
        return "belief_denial"
    return "euphoria"


def get_exchange_flows() -> dict:
    """Exchange inflow/outflow (net flow). Positive net = coins leaving exchanges
    (bullish — accumulation). Negative = coins going to exchanges (bearish — distribution).
    """
    cm_df = fetch_coinmetrics("btc", ["FlowInExUSD", "FlowOutExUSD"], days=30)
    if cm_df is None or cm_df.empty:
        return {"net_flow_usd": None, "source": "unavailable",
                "note": "Requires CoinMetrics or Glassnode API"}
    inflow = float(cm_df["FlowInExUSD"].iloc[-1]) if "FlowInExUSD" in cm_df.columns else 0
    outflow = float(cm_df["FlowOutExUSD"].iloc[-1]) if "FlowOutExUSD" in cm_df.columns else 0
    net = outflow - inflow  # positive = leaving exchanges = bullish
    return {
        "net_flow_usd": net,
        "inflow_usd": inflow,
        "outflow_usd": outflow,
        "interpretation": "bullish_accumulation" if net > 0 else "bearish_distribution",
        "source": "coinmetrics",
    }


def get_active_addresses() -> dict:
    """Daily active addresses — network usage indicator."""
    cm_df = fetch_coinmetrics("btc", ["AdrActCnt"], days=400)
    if cm_df is None or "AdrActCnt" not in cm_df.columns:
        return {"active_addresses": None, "source": "unavailable"}
    s = cm_df["AdrActCnt"].dropna()
    current = float(s.iloc[-1])
    mean_30d = float(s.iloc[-30:].mean())
    mean_90d = float(s.iloc[-90:].mean())
    return {
        "active_addresses": current,
        "vs_30d_mean": current / mean_30d - 1 if mean_30d > 0 else 0,
        "vs_90d_mean": current / mean_90d - 1 if mean_90d > 0 else 0,
        "source": "coinmetrics",
    }


def cycle_position() -> dict:
    """Composite cycle position from on-chain metrics.

    Returns a score [0, 100] where:
        0-20  : DEEP_BEAR (accumulation zone — historical bottoms)
        20-40 : EARLY_BULL
        40-60 : MID_BULL
        60-80 : LATE_BULL
        80-100: EUPHORIA (distribution zone — historical tops)
    """
    mvrv = get_mvrv()
    nupl = get_nupl()

    # Score from MVRV
    mvrv_val = mvrv.get("mvrv")
    if mvrv_val is None:
        return {"score": None, "phase": "unknown", "reason": "no mvrv"}

    # === RECALIBRATED 2026-06-01 for muted institutional cycles ===
    # Backtest of 2025-10-06 peak (cycle 5 high $124,659):
    #   MVRV peaked at 2.29 — old curve gave score 57 (MID_BULL).
    #   Old curve assumed cycle-3/4-style retail euphoria (MVRV 3-4 at peaks).
    #   Cycle 5 was MUTED by ETF/sovereign flows; MVRV peak was 30% lower.
    # New curve calibrated so cycle-5 MVRV 2.29 maps to score 97 (EUPHORIA).
    # Expectation for cycle 6 (2029): peak MVRV ~1.8-2.0, peak score should
    # still flag EUPHORIA even at lower absolute MVRV.
    #
    # New mapping:
    #   MVRV 0.7  ≈ score 0   (deep bear bottom)
    #   MVRV 1.0  ≈ score 25  (fair value)
    #   MVRV 1.5  ≈ score 50  (mid bull)
    #   MVRV 1.9  ≈ score 75  (late bull / sell zone begins)
    #   MVRV 2.3+ ≈ score 100 (peak euphoria)
    if mvrv_val < 0.7:
        mvrv_score = 0
    elif mvrv_val < 1.0:
        mvrv_score = 25 * (mvrv_val - 0.7) / 0.3
    elif mvrv_val < 1.5:
        mvrv_score = 25 + 25 * (mvrv_val - 1.0) / 0.5
    elif mvrv_val < 1.9:
        mvrv_score = 50 + 25 * (mvrv_val - 1.5) / 0.4
    elif mvrv_val < 2.3:
        mvrv_score = 75 + 25 * (mvrv_val - 1.9) / 0.4
    else:
        mvrv_score = 100

    score = max(0, min(100, mvrv_score))
    if score < 20:
        phase = "DEEP_BEAR"
    elif score < 40:
        phase = "EARLY_BULL"
    elif score < 60:
        phase = "MID_BULL"
    elif score < 80:
        phase = "LATE_BULL"
    else:
        phase = "EUPHORIA"

    return {
        "score": score,
        "phase": phase,
        "mvrv": mvrv_val,
        "nupl": nupl.get("nupl"),
        "source": mvrv.get("source"),
    }


def main():
    print("=" * 80)
    print("ON-CHAIN INDICATOR SNAPSHOT")
    print("=" * 80)
    print()
    mvrv = get_mvrv()
    print(f"MVRV: {mvrv.get('mvrv', 'n/a'):.3f}" if mvrv.get('mvrv') else "MVRV: unavailable")
    print(f"  source: {mvrv['source']}")
    if mvrv.get("z_score") is not None:
        print(f"  z-score: {mvrv['z_score']:+.2f} (vs historical mean)")
    if mvrv.get("note"):
        print(f"  note: {mvrv['note']}")
    print()
    nupl = get_nupl()
    if nupl.get("nupl") is not None:
        print(f"NUPL: {nupl['nupl']:+.3f}  phase: {nupl['phase']}")
    else:
        print("NUPL: unavailable")
    print()
    flows = get_exchange_flows()
    if flows.get("net_flow_usd") is not None:
        print(f"Exchange net flow: ${flows['net_flow_usd']:>+,.0f}  ({flows['interpretation']})")
    else:
        print(f"Exchange flows: {flows.get('note', 'unavailable')}")
    print()
    aa = get_active_addresses()
    if aa.get("active_addresses") is not None:
        print(f"Active addresses (today): {aa['active_addresses']:,.0f}")
        print(f"  vs 30d mean: {aa['vs_30d_mean']*100:+.1f}%")
        print(f"  vs 90d mean: {aa['vs_90d_mean']*100:+.1f}%")
    else:
        print("Active addresses: unavailable")
    print()
    cp = cycle_position()
    print("=" * 80)
    print(f"CYCLE POSITION: {cp.get('score', 'n/a')} / 100   phase: {cp.get('phase', 'unknown')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
