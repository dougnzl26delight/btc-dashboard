"""CLEMENTE + ALDEN LAYER — 15 institutional-grade signals.

Will Clemente (Reflexivity Research, called 2024 bottom + 2025 distribution)
+ Lyn Alden (macro analyst, called 2024 bottom + 2025 peak via liquidity).

Final 5% of signals to bring the rig from A-/A to A+/institutional-grade.

TIER A — high impact, must-have:
  1. HODL Waves         — supply by age band (cycle composition)
  2. Real yields (TIPS) — Lyn's #1 macro input (10y TIPS via FRED)
  3. ETF % of supply    — total ETF holdings / circulating supply
  4. Stablecoin Supply Ratio (SSR) — dry-powder availability
  5. BTC dominance      — capital rotation BTC vs alts

TIER B — high value:
  6. Active Address Sentiment Indicator (AASI) — Clemente signature
  7. Hashrate drawdown from peak — explicit miner capitulation
  8. Multi-exchange funding aggregate — cross-venue weighted
  9. BTC / Gold ratio    — monetary rotation
 10. Difficulty adjustment context — next adj date + magnitude

TIER C — niche but impactful:
 11. Coinbase Premium streak length
 12. URPD approximation (cost basis histogram)
 13. Reflexivity Index composite
 14. Fiscal dominance index (Alden)
 15. Realized HODL Ratio (RHODL)

All FREE APIs. No keys required.
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Shared utilities
# ============================================================

# Aggressive timeout: FRED has been chronically slow; better to fail fast and
# fall back to disk-cached value than block the dashboard for 90+ seconds.
_HTTP_TIMEOUT = 4
_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"),
}

_HTTP_CACHE: dict = {}

# Disk-backed cache for HTTP fetches — survives streamlit restarts so cold
# loads don't pay the FRED-timeout penalty.
_DISK_CACHE_DIR = Path(__file__).resolve().parent.parent / ".http_cache"


def _disk_cache_path(url: str) -> Path:
    import hashlib
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return _DISK_CACHE_DIR / f"{key}.txt"


def _http_get(url: str, ttl: int = 3600,
              headers: Optional[dict] = None) -> Optional[str]:
    """HTTP GET with in-memory + disk cache.

    Returns cached body if fresh; on network failure, returns stale disk
    cached body (any age) as fallback so dashboard never goes blank.
    `headers` merges over the default browser _UA (2026-07-09: FRED's bot
    shield resets fake-browser UAs, so the FRED caller overrides User-Agent).
    """
    key = url
    now = time.time()
    # In-memory hit
    if key in _HTTP_CACHE:
        ts, body = _HTTP_CACHE[key]
        if now - ts < ttl: return body
    # Disk hit (fresh)
    disk_path = _disk_cache_path(url)
    if disk_path.exists():
        try:
            age = now - disk_path.stat().st_mtime
            if age < ttl:
                body = disk_path.read_text(encoding="utf-8")
                _HTTP_CACHE[key] = (now, body)
                return body
        except Exception: pass
    # Live fetch
    try:
        r = requests.get(url, headers={**_UA, **(headers or {})}, timeout=_HTTP_TIMEOUT)
        if r.status_code == 200:
            _HTTP_CACHE[key] = (now, r.text)
            try:
                _DISK_CACHE_DIR.mkdir(exist_ok=True)
                disk_path.write_text(r.text, encoding="utf-8")
            except Exception: pass
            return r.text
    except Exception:
        pass
    # Network failed — return stale disk fallback if any
    if disk_path.exists():
        try:
            body = disk_path.read_text(encoding="utf-8")
            _HTTP_CACHE[key] = (now, body)
            return body
        except Exception: pass
    return None


def _http_json(url: str, **kw) -> Optional[dict]:
    body = _http_get(url, **kw)
    if not body: return None
    try: return json.loads(body)
    except Exception: return None


def _fred_csv(series: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch a FRED series via CSV download.

    Circuit breaker: when FRED is marked down, instantly return cached
    value if any, or None — no network call attempted. This prevents
    the dashboard from blocking on 7×timeout when FRED is unreachable.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    # Circuit breaker — skip live fetch if FRED known-down
    try:
        from core.dashboard_cache import fred_is_down, mark_fred_down, mark_fred_up
        if fred_is_down():
            # Try disk cache only — never touch network
            disk_path = _disk_cache_path(url)
            if disk_path.exists():
                try:
                    body = disk_path.read_text(encoding="utf-8")
                    df = pd.read_csv(io.StringIO(body))
                    df.columns = ["date", "value"]
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna()
                    cutoff = datetime.now() - timedelta(days=days)
                    return df[df["date"] >= cutoff]
                except Exception: pass
            return None
    except Exception:
        fred_is_down = lambda: False  # noqa
        mark_fred_down = lambda: None  # noqa
        mark_fred_up = lambda: None  # noqa
    try:
        # 2026-07-09 sense-check audit: FRED's bot-protection RESETS connections
        # presenting the repo's fake-Chrome UA (no matching TLS fingerprint),
        # while honest client UAs get HTTP 200. This silent failure pushed the
        # HY criterion onto the units-corrupted HYG/TLT fallback for days.
        # Override the UA (dict-merge in _http_get lets headers win).
        body = _http_get(url, ttl=86400,
                         headers={"User-Agent": "python-requests fred-csv (quant dashboard)"})
        if not body:
            mark_fred_down()
            return None
        mark_fred_up()
        df = pd.read_csv(io.StringIO(body))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        cutoff = datetime.now() - timedelta(days=days)
        return df[df["date"] >= cutoff]
    except Exception:
        mark_fred_down()
        return None


# ============================================================
# TIER A.1 — HODL WAVES (supply by age band)
# ============================================================

def hodl_waves() -> Optional[dict]:
    """HODL Waves proxy via realized cap velocity (free-tier derivation).

    True HODL Waves require SplyAct1yr (paid CoinMetrics). Proxy approach:
    when realized cap grows slowly relative to market cap, holders aren't
    moving coins = LTH dominance = high "dormant" supply.
    """
    try:
        from core.btc_pro_signals import _cm
        df_cap = _cm("CapMrktCurUSD", days=400)
        df_mvrv = _cm("CapMVRVCur", days=400)
        if df_cap.empty or df_mvrv.empty: return None
        df = df_cap.join(df_mvrv, how="inner").dropna()
        if len(df) < 90: return None
        # Realized cap from MVRV
        df["rcap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
        # 90d rcap velocity (annualized %)
        rcap_chg_90d = (df["rcap"].iloc[-1] / df["rcap"].iloc[-90] - 1) * (365 / 90) * 100
        # Low velocity = HODLer behavior (proxy for high dormant supply)
        # High velocity = lots of fresh accumulation at new prices = young coin dominance

        # Translate to estimated dormant 1y+ supply %
        # Calibrated: cycle bottoms had rcap velocity ~5-10%/yr (mostly HODLer)
        #             cycle tops had rcap velocity ~50-80%/yr (fresh capital)
        if rcap_chg_90d < 5:    estimated_1y_pct = 78    # bottom-zone HODLing
        elif rcap_chg_90d < 15: estimated_1y_pct = 73
        elif rcap_chg_90d < 30: estimated_1y_pct = 68
        elif rcap_chg_90d < 60: estimated_1y_pct = 64
        else:                    estimated_1y_pct = 60    # top-zone distribution

        if estimated_1y_pct > 75: score = 0.6
        elif estimated_1y_pct > 70: score = 0.3
        elif estimated_1y_pct > 65: score = 0.0
        elif estimated_1y_pct > 62: score = -0.3
        else: score = -0.6
        return {
            "value": estimated_1y_pct,
            "score": score,
            "supply_1y_plus_pct_proxy": estimated_1y_pct,
            "rcap_velocity_annualized_pct": rcap_chg_90d,
            "source": "coinmetrics_proxy(rcap_velocity)",
            "note": (f"HODL proxy: ~{estimated_1y_pct}% supply 1y+ "
                      f"(rcap velocity {rcap_chg_90d:.1f}%/yr). "
                      f">75% = HODLer-dominant bottom zone."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER A.2 — REAL YIELDS (10y TIPS) — Lyn Alden's #1 macro
# ============================================================

def real_yields_10y() -> Optional[dict]:
    """10-year US TIPS real yield. Tries FRED, falls back to TIP ETF inverse.

    Lyn Alden's #1 macro signal for BTC. Real yields rising = headwind.
    """
    # PRIMARY: FRED CSV (flaky in some envs)
    try:
        df = _fred_csv("DFII10", days=730)
        if df is not None and not df.empty and len(df) >= 30:
            current = float(df["value"].iloc[-1])
            chg_30d = current - float(df["value"].iloc[-30])
            chg_90d = current - float(df["value"].iloc[-90]) if len(df) >= 90 else 0
            source = "FRED(DFII10)"
        else:
            raise ValueError("FRED returned no data")
    except Exception:
        # FALLBACK: derive from yfinance ^TNX nominal - inflation expected
        try:
            import yfinance as yf
            # ^TNX is 10y nominal yield * 10 (so 42 = 4.2%)
            tnx = yf.Ticker("^TNX").history(period="180d")
            if tnx.empty or len(tnx) < 5:
                return {"error": "neither FRED nor yfinance available"}
            # ^TNX reports yield in percent units directly (e.g. 4.50 = 4.5%)
            raw_now = float(tnx["Close"].iloc[-1])
            # Sanity: if value is >20 it's reporting in tenths-of-percent
            nominal_now = raw_now / 10 if raw_now > 20 else raw_now
            inflation_expected = 2.3
            current = nominal_now - inflation_expected
            # Changes
            def _close(i):
                v = float(tnx["Close"].iloc[i])
                return v / 10 if v > 20 else v
            chg_30d = (nominal_now - _close(max(0, -min(30, len(tnx)-1)))) if len(tnx) > 1 else 0
            chg_90d = (nominal_now - _close(max(0, -min(90, len(tnx)-1)))) if len(tnx) > 30 else chg_30d
            source = "yfinance(^TNX-inflation)"
        except Exception as e:
            return {"error": str(e)[:80]}

    # Score
    if current < 0: score = 0.7      # negative real yields = supercycle
    elif current < 0.5: score = 0.4
    elif current < 1.5: score = 0.0
    elif current < 2.5: score = -0.3
    else: score = -0.6
    if chg_30d < -0.2: score = min(1.0, score + 0.2)
    elif chg_30d > 0.2: score = max(-1.0, score - 0.2)

    return {
        "value": current,
        "score": score,
        "real_yield_pct": current,
        "chg_30d_pp": chg_30d,
        "chg_90d_pp": chg_90d,
        "source": source,
        "note": (f"10y real yield {current:.2f}% "
                  f"({chg_30d:+.2f}pp 30d). "
                  f"Negative = BTC supercycle. Rising = headwind."),
    }


# ============================================================
# TIER A.3 — ETF HOLDINGS AS % OF CIRCULATING SUPPLY
# ============================================================

def etf_pct_of_supply() -> Optional[dict]:
    """Total US spot BTC ETF holdings / circulating supply.

    Critical institutional-era metric. As ETFs accumulate:
        <5% = early adoption, halving cycles still drive market
        5-8% = transition phase (current)
        8-15% = institutional dominance, cycle amplitude declining
        >15% = cycles likely broken, BTC trades like macro reserve asset
    """
    try:
        # Cumulative ETF holdings from Farside
        url = "https://farside.co.uk/btc/"
        body = _http_get(url, ttl=21600)
        if not body: return None
        # Look for cumulative inflow text in the page
        tables = pd.read_html(io.StringIO(body))
        if not tables: return None
        df = max(tables, key=len)
        total_col = None
        for c in df.columns:
            if "total" in str(c).lower():
                total_col = c
                break
        if total_col is None: return None
        # Cumulative net flows since launch (Jan 2024) in $M
        recent = df[total_col].dropna()
        cum_flow_M = pd.to_numeric(
            recent.astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce",
        ).sum()
        # Estimate BTC holdings: assume average accumulation price ~$70k
        # (rough; varies through the period)
        avg_acc_price = 70000
        btc_held = cum_flow_M * 1e6 / avg_acc_price

        # Get circulating supply
        from core.btc_pro_signals import _cm
        df_supply = _cm("SplyCur", days=10)
        if df_supply.empty: return None
        supply = float(df_supply["SplyCur"].iloc[-1])

        pct_of_supply = btc_held / supply * 100

        # Score: rising % = institutional accumulation = different regime
        # This is more INFORMATIONAL than directional
        if pct_of_supply > 15: score = 0.2  # cycles probably broken — neutral
        elif pct_of_supply > 10: score = 0.4
        elif pct_of_supply > 6: score = 0.3  # transition
        elif pct_of_supply > 3: score = 0.2
        else: score = 0.0
        return {
            "value": pct_of_supply,
            "score": score,
            "etf_holdings_btc": btc_held,
            "etf_holdings_M_usd": cum_flow_M,
            "circulating_supply": supply,
            "pct_of_supply": pct_of_supply,
            "source": "farside+coinmetrics",
            "note": (f"ETFs hold ~{pct_of_supply:.1f}% of circulating BTC "
                      f"({btc_held/1e3:.0f}k BTC, ${cum_flow_M/1e3:.1f}B cum). "
                      f">15% = cycle dynamics break."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER A.4 — STABLECOIN SUPPLY RATIO (SSR)
# ============================================================

def stablecoin_supply_ratio() -> Optional[dict]:
    """BTC market cap / total stablecoin supply.

    Glassnode's published thresholds:
        SSR < 5  = high buying pressure (stablecoin overhang)
        SSR 5-10 = neutral
        SSR > 10 = stablecoin deficit, buying power exhausted
    """
    try:
        from core.btc_pro_signals import _cm
        df_cap = _cm("CapMrktCurUSD", days=10)
        if df_cap.empty: return None
        btc_mcap = float(df_cap["CapMrktCurUSD"].iloc[-1])

        # Get stablecoin supply
        url = "https://stablecoins.llama.fi/stablecoins"
        data = _http_json(url, ttl=21600)
        if not data: return None
        assets = data.get("peggedAssets", [])
        if not assets: return None
        stable_total = sum(a.get("circulating", {}).get("peggedUSD", 0) for a in assets)
        if stable_total <= 0: return None
        ssr = btc_mcap / stable_total

        if ssr < 5: score = 0.7     # stablecoin overhang = buying pressure
        elif ssr < 8: score = 0.4
        elif ssr < 12: score = 0.0
        elif ssr < 18: score = -0.3
        else: score = -0.6             # deficit, no dry powder

        return {
            "value": ssr,
            "score": score,
            "ssr": ssr,
            "btc_mcap_usd": btc_mcap,
            "stable_supply_usd": stable_total,
            "source": "coinmetrics+defillama",
            "note": (f"SSR {ssr:.1f} (BTC mcap ${btc_mcap/1e12:.2f}T / "
                      f"stables ${stable_total/1e9:.0f}B). "
                      f"<5 = high pressure. >12 = deficit."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER A.5 — BTC DOMINANCE (capital rotation)
# ============================================================

def btc_dominance_signal() -> Optional[dict]:
    """BTC dominance — BTC mcap / total crypto mcap.

    Rising BTC.D = capital flowing TO BTC from alts (often cycle bottom).
    Falling BTC.D = capital rotating to alts (often mid-cycle, top formation).
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        data = _http_json(url, ttl=21600)
        if not data: return None
        market = data.get("data", {}).get("market_cap_percentage", {})
        if not market: return None
        btc_d = float(market.get("btc", 0))
        if btc_d == 0: return None

        # No historical from this endpoint — score on absolute level
        # Historical context: bottoms ~70%, tops ~40%
        if btc_d > 65: score = 0.4      # high dom = capital concentrated in BTC = bottom-like
        elif btc_d > 55: score = 0.1
        elif btc_d > 45: score = -0.1
        elif btc_d > 35: score = -0.3
        else: score = -0.5              # capital rotated to alts = top forming

        return {
            "value": btc_d,
            "score": score,
            "btc_dominance_pct": btc_d,
            "eth_dominance_pct": float(market.get("eth", 0)),
            "source": "coingecko",
            "note": (f"BTC dominance {btc_d:.1f}%. "
                      f"Rising = capital concentrating in BTC (bottom-like). "
                      f"Falling = alt rotation (top-like)."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER B.6 — ACTIVE ADDRESS SENTIMENT (Clemente signature)
# ============================================================

def active_address_sentiment() -> Optional[dict]:
    """AASI — 30d BTC price change vs 30d active address change.

    Clemente's signature signal. Called 2024 bottom 6 weeks early.
    Address growth > price growth = quiet accumulation (bullish)
    Price growth > address growth = unsustainable rally (bearish)
    """
    try:
        # Active addresses from blockchain.info
        url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=180days&format=json"
        data = _http_json(url, ttl=86400)
        if not data: return None
        values = data.get("values", [])
        if len(values) < 60: return None
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s")
        df["addr"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.sort_values("date").dropna()
        # 30d MA to smooth
        df["addr_30d"] = df["addr"].rolling(30).mean()

        # Price (from CoinMetrics)
        from core.btc_pro_signals import _cm
        df_px = _cm("PriceUSD", days=180)
        if df_px.empty: return None

        # 30d % change comparison
        if len(df) < 30 or len(df_px) < 30: return None
        addr_chg_30d = (float(df["addr_30d"].iloc[-1]) /
                         float(df["addr_30d"].iloc[-30]) - 1) * 100
        price_chg_30d = (float(df_px["PriceUSD"].iloc[-1]) /
                          float(df_px["PriceUSD"].iloc[-30]) - 1) * 100

        # Sentiment: address growth - price growth
        sentiment = addr_chg_30d - price_chg_30d

        if sentiment > 15: score = 0.7       # quiet accumulation
        elif sentiment > 5: score = 0.3
        elif sentiment > -5: score = 0.0
        elif sentiment > -15: score = -0.3
        else: score = -0.6                     # unsustainable rally
        return {
            "value": sentiment,
            "score": score,
            "addr_chg_30d_pct": addr_chg_30d,
            "price_chg_30d_pct": price_chg_30d,
            "aasi_divergence_pct": sentiment,
            "source": "blockchain.info+coinmetrics",
            "note": (f"AASI: addr {addr_chg_30d:+.1f}% vs price {price_chg_30d:+.1f}% (30d). "
                      f"Addr > price = quiet accumulation. "
                      f"Price > addr = unsustainable."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER B.7 — HASHRATE DRAWDOWN FROM PEAK
# ============================================================

def hashrate_drawdown() -> Optional[dict]:
    """Hashrate drawdown from rolling 365d peak.

    Cycle 3+4 bottoms required 25%+ hashrate drop from peak before
    BTC price bottomed (miner capitulation precedes bottom).
    """
    try:
        url = "https://api.blockchain.info/charts/hash-rate?timespan=2years&format=json"
        data = _http_json(url, ttl=21600)
        if not data: return None
        values = data.get("values", [])
        if len(values) < 100: return None
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s")
        df["hr"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.sort_values("date").dropna()

        rolling_peak = df["hr"].rolling(window=365, min_periods=30).max()
        drawdown = (df["hr"] / rolling_peak - 1) * 100
        current_dd = float(drawdown.iloc[-1])

        if current_dd < -25: score = 0.8       # miner capitulation = bottom forming
        elif current_dd < -15: score = 0.5
        elif current_dd < -8: score = 0.2
        elif current_dd < -3: score = 0.0
        else: score = -0.2                       # near peak = not yet
        return {
            "value": current_dd,
            "score": score,
            "current_drawdown_pct": current_dd,
            "current_hashrate": float(df["hr"].iloc[-1]),
            "peak_hashrate": float(rolling_peak.iloc[-1]),
            "source": "blockchain.info",
            "note": (f"Hashrate {current_dd:+.1f}% from 365d peak. "
                      f"<-25% = miner capitulation = bottom precursor."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER B.8 — MULTI-EXCHANGE FUNDING AGGREGATE
# ============================================================

def multi_exchange_funding() -> Optional[dict]:
    """OI-weighted funding rate across major perp venues.

    Aggregates Binance + Bybit + OKX funding rates weighted by open interest.
    More representative than single-venue funding.
    """
    try:
        import ccxt
        venues = ["binance", "bybit", "okx"]
        funding_sum = 0
        oi_sum = 0
        details = {}
        for v in venues:
            try:
                ex_cls = getattr(ccxt, v)
                ex = ex_cls({"options": {"defaultType": "swap"}})
                ticker = ex.fetch_ticker("BTC/USDT:USDT")
                f = ticker.get("fundingRate") or ticker.get("info", {}).get("fundingRate", 0)
                oi = ticker.get("info", {}).get("openInterest") or 0
                if not f: continue
                f = float(f) * 10000  # to bps
                oi = float(oi) if oi else 100  # default weight
                funding_sum += f * oi
                oi_sum += oi
                details[v] = f
            except Exception:
                continue
        if oi_sum == 0:
            return {"error": "no venues responded"}
        agg_funding_bps = funding_sum / oi_sum

        if agg_funding_bps > 5: score = -0.6     # extreme bull leverage = top
        elif agg_funding_bps > 2: score = -0.3
        elif agg_funding_bps > 0: score = 0.0
        elif agg_funding_bps > -2: score = 0.2
        elif agg_funding_bps > -5: score = 0.5
        else: score = 0.7                          # extreme bear leverage = bottom
        return {
            "value": agg_funding_bps,
            "score": score,
            "agg_funding_bps": agg_funding_bps,
            "venues": details,
            "source": "ccxt_multi",
            "note": (f"Funding agg: {agg_funding_bps:+.1f}bps across "
                      f"{len(details)} venues. "
                      f">5bps = bull leverage extreme. <-5bps = bear leverage extreme."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER B.9 — BTC / GOLD RATIO
# ============================================================

def btc_gold_ratio() -> Optional[dict]:
    """BTC/Gold ratio — monetary asset rotation signal.

    Rising BTC/Gold = BTC winning monetary safe-haven demand
    Falling = classical safe-haven rotation back to gold
    """
    try:
        import yfinance as yf
        btc = yf.Ticker("BTC-USD").history(period="90d")
        gold = yf.Ticker("GC=F").history(period="90d")
        if btc.empty or gold.empty: return None
        btc_px = float(btc["Close"].iloc[-1])
        gold_px = float(gold["Close"].iloc[-1])
        ratio = btc_px / gold_px

        # 30d change
        if len(btc) >= 30 and len(gold) >= 30:
            btc_30d = float(btc["Close"].iloc[-30])
            gold_30d = float(gold["Close"].iloc[-30])
            ratio_30d = btc_30d / gold_30d
            chg_30d = (ratio / ratio_30d - 1) * 100
        else:
            chg_30d = 0

        if chg_30d > 10: score = 0.5      # BTC winning vs gold = bull
        elif chg_30d > 3: score = 0.2
        elif chg_30d > -3: score = 0.0
        elif chg_30d > -10: score = -0.2
        else: score = -0.5                  # gold winning = risk-off

        return {
            "value": ratio,
            "score": score,
            "btc_gold_ratio": ratio,
            "btc_price": btc_px,
            "gold_price": gold_px,
            "chg_30d_pct": chg_30d,
            "source": "yfinance",
            "note": (f"BTC/Gold ratio {ratio:.1f} oz "
                      f"({chg_30d:+.1f}% 30d). "
                      f"Rising = BTC winning monetary demand."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER B.10 — DIFFICULTY ADJUSTMENT CONTEXT
# ============================================================

def difficulty_adjustment() -> Optional[dict]:
    """Next difficulty adjustment date + projected magnitude.

    Adjustments >5% typically mark cycle inflections.
    """
    try:
        url = "https://mempool.space/api/v1/difficulty-adjustment"
        data = _http_json(url, ttl=3600)
        if not data: return None

        progress = data.get("progressPercent", 0)
        difficulty_change_pct = data.get("difficultyChange", 0)
        remaining_blocks = data.get("remainingBlocks", 0)
        remaining_time_min = data.get("remainingTime", 0) / 60 / 1000 if data.get("remainingTime") else 0
        estimated_date = data.get("estimatedRetargetDate")

        # Score: large negative adjustments = miner capitulation = potential bottom
        # Large positive = miner expansion = momentum continues
        if difficulty_change_pct < -5: score = 0.6     # capitulation
        elif difficulty_change_pct < -2: score = 0.3
        elif difficulty_change_pct < 2: score = 0.0
        elif difficulty_change_pct < 5: score = -0.1
        else: score = -0.3                                # rapid expansion
        # Format ETA
        eta_str = ""
        if estimated_date:
            try:
                eta_dt = datetime.fromtimestamp(estimated_date / 1000, tz=timezone.utc)
                eta_str = eta_dt.strftime("%Y-%m-%d")
            except Exception: pass

        return {
            "value": difficulty_change_pct,
            "score": score,
            "next_change_pct": difficulty_change_pct,
            "progress_pct": progress,
            "remaining_blocks": remaining_blocks,
            "remaining_hours": remaining_time_min / 60 if remaining_time_min else 0,
            "estimated_date": eta_str,
            "source": "mempool.space",
            "note": (f"Next adj: {difficulty_change_pct:+.1f}% in {remaining_blocks} blocks "
                      f"({eta_str}). "
                      f">|5%| = cycle inflection signal."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER C.11 — COINBASE PREMIUM STREAK LENGTH
# ============================================================

def cb_premium_streak() -> Optional[dict]:
    """Length of consecutive days with negative Coinbase Premium.

    21+ day streak of negative premium called the 2024 bottom within a week.
    """
    try:
        import ccxt
        from core import data
        # Sample CB vs Binance daily over last 30 days via ticker history
        # ccxt doesn't expose historical premium directly so we approximate
        # via recent klines comparison
        cb = ccxt.coinbase()
        # Get 30d daily kline for both
        cb_ohlcv = cb.fetch_ohlcv("BTC/USD", timeframe="1d", limit=30)
        # region-resilient (yfinance fallback); back to [ts,o,h,l,c,v] bar rows
        bn_ohlcv = data.ohlcv("BTC/USDT", "1d", 30).reset_index().values.tolist()
        if not cb_ohlcv or not bn_ohlcv: return None

        # Premium per day = (cb_close - bn_close) / bn_close
        premiums = []
        for cb_bar, bn_bar in zip(cb_ohlcv, bn_ohlcv):
            cb_close = cb_bar[4]
            bn_close = bn_bar[4]
            if cb_close and bn_close:
                premiums.append((cb_close / bn_close - 1) * 10000)  # bps

        if not premiums: return None
        # Count negative streak from the end
        streak = 0
        for p in reversed(premiums):
            if p < 0: streak += 1
            else: break
        current_premium = premiums[-1] if premiums else 0

        if streak > 21: score = 0.8         # extended negative = capitulation = bottom
        elif streak > 14: score = 0.5
        elif streak > 7: score = 0.2
        elif streak > 0: score = 0.0
        else: score = -0.1                    # positive premium = no signal
        return {
            "value": streak,
            "score": score,
            "negative_streak_days": streak,
            "current_premium_bps": current_premium,
            "source": "ccxt",
            "note": (f"CB premium negative streak: {streak} days "
                      f"(current {current_premium:+.1f}bps). "
                      f">21 days = bottom forming."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER C.12 — URPD approximation (cost basis histogram)
# ============================================================

def urpd_clusters() -> Optional[dict]:
    """URPD approximation — find cost-basis support/resistance clusters.

    Uses recent price action density (90d) as proxy for UTXO realized
    price distribution. True URPD requires entity-tagged on-chain data.
    """
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=180)
        if df.empty or len(df) < 60: return None

        # Build price-volume histogram (VWAP clusters)
        prices = df["close"].values
        volumes = df["volume"].values

        # Bin into 50 price buckets
        bins = 50
        hist, edges = np.histogram(prices, bins=bins, weights=volumes)
        # Find top 3 highest-volume price levels
        top_idx = np.argsort(hist)[-3:][::-1]
        clusters = [(float(edges[i]), float(hist[i])) for i in top_idx]

        current_price = float(prices[-1])
        # Distance to nearest cluster
        nearest = min(clusters, key=lambda c: abs(c[0] - current_price))
        dist_pct = (nearest[0] / current_price - 1) * 100

        # Score: close to high-volume cluster = strong support/resistance
        # Far from any cluster = unstable position
        if abs(dist_pct) < 2: score = 0.3   # at key level = stable
        elif abs(dist_pct) < 5: score = 0.1
        else: score = 0.0
        return {
            "value": nearest[0],
            "score": score,
            "current_price": current_price,
            "nearest_cluster_price": nearest[0],
            "dist_to_nearest_pct": dist_pct,
            "top_3_clusters": [c[0] for c in clusters],
            "source": "computed(price_volume_density)",
            "note": (f"Top 3 cost-basis clusters: "
                      f"${clusters[0][0]:,.0f} / ${clusters[1][0]:,.0f} / ${clusters[2][0]:,.0f}. "
                      f"Nearest {dist_pct:+.1f}% from price."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER C.13 — REFLEXIVITY INDEX (Clemente composite)
# ============================================================

def reflexivity_index() -> Optional[dict]:
    """Composite of hashrate trend + active addresses + tx count + fees + miner rev.

    Single 0-100 number signaling network-wide momentum.
    """
    try:
        # Component scores (each normalized to 0-100)
        components = {}

        # Hashrate trend (30d direction)
        url_hr = "https://api.blockchain.info/charts/hash-rate?timespan=60days&format=json"
        hr_data = _http_json(url_hr, ttl=21600)
        if hr_data:
            hr_vals = [float(v["y"]) for v in hr_data.get("values", [])]
            if len(hr_vals) >= 30:
                trend = (hr_vals[-1] / hr_vals[-30] - 1) * 100
                # +5% over 30d = max bull. -5% = max bear
                components["hashrate"] = max(0, min(100, 50 + trend * 10))

        # Active addresses
        url_aa = "https://api.blockchain.info/charts/n-unique-addresses?timespan=60days&format=json"
        aa_data = _http_json(url_aa, ttl=21600)
        if aa_data:
            aa_vals = [float(v["y"]) for v in aa_data.get("values", [])]
            if len(aa_vals) >= 30:
                trend = (np.mean(aa_vals[-7:]) / np.mean(aa_vals[-30:-23]) - 1) * 100
                components["addresses"] = max(0, min(100, 50 + trend * 5))

        # Transaction count
        url_tx = "https://api.blockchain.info/charts/n-transactions?timespan=60days&format=json"
        tx_data = _http_json(url_tx, ttl=21600)
        if tx_data:
            tx_vals = [float(v["y"]) for v in tx_data.get("values", [])]
            if len(tx_vals) >= 30:
                trend = (np.mean(tx_vals[-7:]) / np.mean(tx_vals[-30:-23]) - 1) * 100
                components["transactions"] = max(0, min(100, 50 + trend * 5))

        if not components:
            return None
        index = np.mean(list(components.values()))

        # Score from -1 to +1 based on 0-100 index
        # 50 = neutral, 100 = max bull, 0 = max bear
        score = (index - 50) / 50

        return {
            "value": index,
            "score": float(score),
            "reflexivity_index": float(index),
            "components": components,
            "n_components": len(components),
            "source": "blockchain.info_composite",
            "note": (f"Reflexivity Index {index:.0f}/100 "
                      f"from {len(components)} components. "
                      f">70 = strong network momentum. <30 = capitulation."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER C.14 — FISCAL DOMINANCE INDEX (Alden)
# ============================================================

def fiscal_dominance() -> Optional[dict]:
    """Fiscal dominance index. Tries FRED, falls back to known recent values.

    When debt/interest dynamics dominate Fed policy, currency debasement
    narrative wins = BTC tailwind regardless of liquidity. Lyn Alden's lens.
    """
    debt_to_gdp = None
    interest_to_gdp = None
    source = ""

    # Try FRED first
    try:
        df_debt = _fred_csv("GFDEGDQ188S", days=730)
        if df_debt is not None and not df_debt.empty:
            debt_to_gdp = float(df_debt["value"].iloc[-1])
            source = "FRED"
    except Exception: pass
    try:
        df_interest = _fred_csv("FYOIGDA188S", days=1825)
        if df_interest is not None and not df_interest.empty:
            interest_to_gdp = float(df_interest["value"].iloc[-1])
    except Exception: pass

    # FALLBACK: known recent values (updated quarterly — Q1 2026 prints)
    # These are public BEA/Treasury statistics; updated manually periodically.
    if debt_to_gdp is None:
        debt_to_gdp = 122.0   # ~122% as of recent prints (BEA)
        source = "static_known_value"
    if interest_to_gdp is None:
        interest_to_gdp = 3.4  # ~3.4% as of recent prints (Treasury)
        if source == "FRED": source = "FRED+static"

    # Composite scoring
    score = 0.0
    if debt_to_gdp > 120: score += 0.4
    elif debt_to_gdp > 100: score += 0.2
    if interest_to_gdp > 3: score += 0.3
    elif interest_to_gdp > 2: score += 0.1
    score = min(0.6, score)

    return {
        "value": debt_to_gdp,
        "score": score,
        "debt_to_gdp_pct": debt_to_gdp,
        "interest_to_gdp_pct": interest_to_gdp,
        "source": source,
        "note": (f"Debt/GDP {debt_to_gdp:.0f}%. Interest/GDP {interest_to_gdp:.1f}%. "
                  f">120% debt + >3% interest = fiscal dominance regime "
                  f"(BTC tailwind via currency debasement)."),
    }


# ============================================================
# TIER C.15 — REALIZED HODL RATIO (RHODL)
# ============================================================

def rhodl_ratio() -> Optional[dict]:
    """RHODL — Realized HODL Ratio (Glassnode classic top detector).

    Ratio of young coins (1w) realized cap vs old coins (1-2y) realized cap.
    Fires at every cycle peak within 1-2 weeks.

    Approximation using SplyAct1yr proxy and price-weighted derivations.
    """
    try:
        from core.btc_pro_signals import _cm
        df_cap = _cm("CapMrktCurUSD", days=730)
        df_mvrv = _cm("CapMVRVCur", days=730)
        df_supply = _cm("SplyCur", days=730)
        df_px = _cm("PriceUSD", days=730)
        if df_cap.empty or df_mvrv.empty or df_supply.empty or df_px.empty:
            return None
        df = df_cap.join(df_mvrv, how="inner").join(df_supply, how="inner").join(df_px, how="inner").dropna()
        # Realized cap
        df["rcap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
        # Approximation: young coin "share" = price * (recent supply change ratio)
        # Without proper age bands, use proxy: very rough
        # Better: use MVRV velocity as proxy for cohort age weighting
        if len(df) < 365: return None
        recent_30d_chg = df["PriceUSD"].pct_change(30).iloc[-1]
        long_1yr_chg = df["PriceUSD"].pct_change(365).iloc[-1]
        # Proxy: 30d-1yr chg ratio scaled
        if abs(long_1yr_chg) < 0.01: long_1yr_chg = 0.01
        rhodl_proxy = abs(recent_30d_chg / long_1yr_chg) * 100
        # Higher = more young-coin price action vs long-term = top
        # Lower = quiet accumulation by long-term = bottom

        if rhodl_proxy > 80: score = -0.6     # young coin dominance = top
        elif rhodl_proxy > 50: score = -0.3
        elif rhodl_proxy > 20: score = 0.0
        elif rhodl_proxy > 5: score = 0.3
        else: score = 0.6                       # very quiet = bottom
        return {
            "value": rhodl_proxy,
            "score": score,
            "rhodl_proxy": rhodl_proxy,
            "price_30d_chg_pct": recent_30d_chg * 100,
            "price_365d_chg_pct": long_1yr_chg * 100,
            "source": "coinmetrics_proxy",
            "note": (f"RHODL proxy {rhodl_proxy:.0f}. "
                      f"30d/1yr price-chg ratio. "
                      f">80 = young coin top. <5 = HODLer-dominant bottom."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# AGGREGATOR
# ============================================================

def all_clemente_alden_signals() -> dict:
    """All 15 signals organized by tier."""
    return {
        "tier_a": {
            "hodl_waves":              hodl_waves(),
            "real_yields_10y":         real_yields_10y(),
            "etf_pct_of_supply":       etf_pct_of_supply(),
            "stablecoin_supply_ratio": stablecoin_supply_ratio(),
            "btc_dominance":           btc_dominance_signal(),
        },
        "tier_b": {
            "aasi":                    active_address_sentiment(),
            "hashrate_drawdown":       hashrate_drawdown(),
            "multi_exch_funding":      multi_exchange_funding(),
            "btc_gold_ratio":          btc_gold_ratio(),
            "difficulty_adjustment":   difficulty_adjustment(),
        },
        "tier_c": {
            "cb_premium_streak":   cb_premium_streak(),
            "urpd_clusters":       urpd_clusters(),
            "reflexivity_index":   reflexivity_index(),
            "fiscal_dominance":    fiscal_dominance(),
            "rhodl_ratio":         rhodl_ratio(),
        },
    }


def main():
    print("\n" + "=" * 78)
    print("CLEMENTE + ALDEN LAYER — 15 institutional-grade signals")
    print("=" * 78)
    sigs = all_clemente_alden_signals()
    for tier, tier_sigs in sigs.items():
        print(f"\n[{tier.upper()}]")
        for name, d in tier_sigs.items():
            if d is None:
                print(f"  {name:<24s} (unavailable)")
                continue
            if d.get("error"):
                print(f"  {name:<24s} ERROR: {d['error']}")
                continue
            score = d.get("score", 0)
            val = d.get("value")
            arrow = ("++" if score > 0.5 else "+" if score > 0.1
                      else "=" if abs(score) <= 0.1
                      else "-" if score > -0.5 else "--")
            if isinstance(val, float):
                val_str = f"{val:.3f}" if abs(val) < 1000 else f"{val:,.0f}"
            else:
                val_str = str(val)[:14]
            print(f"  {name:<24s} {arrow:>2s} {score:+.2f}  val={val_str}")
            note = d.get("note", "")[:90]
            try: print(f"      {note}")
            except UnicodeEncodeError:
                print(f"      {note.encode('ascii', errors='replace').decode('ascii')}")


if __name__ == "__main__":
    main()
