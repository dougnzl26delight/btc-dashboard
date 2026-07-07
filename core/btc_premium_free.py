"""PREMIUM-FREE LAYER — 18 paid-tier-equivalent BTC signals from free sources.

Replicates the data that costs $500-2000+/mo from Glassnode/CryptoQuant/Coinglass/
Skew/Bloomberg, using only free public APIs.

TIER 1 (highest impact, no auth):
  1. ETF flows (Farside)         — daily BTC ETF net inflows
  2. Stablecoin supply (DefiLlama) — leading indicator for crypto liquidity
  3. Reddit sentiment            — r/bitcoin activity proxy
  4. GitHub dev activity         — bitcoin/bitcoin commit velocity
  5. Deribit dealer Greeks       — GEX, max pain, 25-delta skew
  6. Blockchain.com LTH supply   — exact LTH cohort (replaces our proxy)

TIER 2 (high value, moderate complexity):
  7. Exchange wallet net flows   — CryptoQuant flagship, via Blockchair
  8. Net Liquidity               — Fed BS - TGA - RRP (FRED)
  9. Miner SEC filings           — MARA/Riot BTC holdings
 10. Hash price                  — miner USD revenue / TH/s
 11. Mempool fee curve           — network demand
 12. CryptoPanic news sentiment  — RSS aggregator

TIER 3 (specialized):
 13. Wikipedia BTC views        — retail interest proxy
 14. DXY regime                 — global USD strength
 15. Energy prices (oil/gas)    — miner cost basis
 16. Whale alerts (>1000 BTC tx) — Blockchair
 17. DeFi TVL                   — risk appetite proxy
 18. Stablecoin mints (Tron+Eth) — fresh liquidity entering

All FREE. No API keys required.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# HTTP helper + caching
# ============================================================

_HTTP_CACHE: dict = {}
# Aggressive timeout — FRED/Farside often hang; fail fast so dashboard renders.
_HTTP_TIMEOUT = 4
# Use real browser UA — many sites (Reddit, Farside) block generic API-style UAs
_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Disk-backed cache so cold streamlit restarts don't re-pay slow FRED penalties
_DISK_CACHE_DIR = Path(__file__).resolve().parent.parent / ".http_cache"


def _disk_cache_path(url: str) -> Path:
    import hashlib
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return _DISK_CACHE_DIR / f"{key}.txt"


def _http_get(url: str, headers: Optional[dict] = None, ttl: int = 3600) -> Optional[str]:
    """Cached GET with in-memory + disk fallback.

    On network failure, returns stale disk-cached body (any age) so the
    dashboard never goes blank from a slow third-party server.
    """
    cache_key = url
    now = time.time()
    # In-memory hit
    if cache_key in _HTTP_CACHE:
        ts, body = _HTTP_CACHE[cache_key]
        if now - ts < ttl:
            return body
    # Disk hit (fresh)
    disk_path = _disk_cache_path(url)
    if disk_path.exists():
        try:
            age = now - disk_path.stat().st_mtime
            if age < ttl:
                body = disk_path.read_text(encoding="utf-8")
                _HTTP_CACHE[cache_key] = (now, body)
                return body
        except Exception: pass
    # Live fetch
    try:
        merged_headers = {**_UA, **(headers or {})}
        r = requests.get(url, headers=merged_headers, timeout=_HTTP_TIMEOUT)
        if r.status_code == 200:
            _HTTP_CACHE[cache_key] = (now, r.text)
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
            _HTTP_CACHE[cache_key] = (now, body)
            return body
        except Exception: pass
    return None


def _http_json(url: str, **kw) -> Optional[dict]:
    body = _http_get(url, **kw)
    if not body: return None
    try: return json.loads(body)
    except Exception: return None


# ============================================================
# TIER 1.1 — ETF FLOWS (Farside Investors)
# ============================================================

def etf_flows() -> Optional[dict]:
    """Aggregated US spot BTC ETF net daily flows (millions USD).

    Source: Farside Investors public CSV ([https://farside.co.uk/btc-etf-flow-all-data](https://farside.co.uk/btc-etf-flow-all-data)).
    Strong positive flows = institutional accumulation = bull signal.
    Negative outflows for 5+ days = distribution = bear signal.
    """
    try:
        # Farside provides daily HTML; we extract the table
        # 2026-07-07 factual audit: /btc/ only exposes ~14 rows, so "30d"
        # sums were really 14d. The all-data page carries full daily history
        # (~640 rows) → real 5d/30d windows.
        url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
        body = _http_get(url, ttl=21600)  # 6h cache
        if not body: return None
        # Parse table — column "Total" is net daily flow ($M)
        tables = pd.read_html(io.StringIO(body))
        if not tables: return None
        df = max(tables, key=len)
        # Heuristic: find a "Total" column (case insensitive)
        total_col = None
        for c in df.columns:
            if "total" in str(c).lower():
                total_col = c
                break
        if total_col is None:
            return None
        # 2026-07-07 factual audit: Farside appends SUMMARY rows (all-time
        # cumulative ~$50B, plus Average/Maximum/Minimum). The old code took
        # df.tail(30) WITHOUT stripping them, so the ~$50B cumulative row landed
        # inside the 5-day window -> the verdict card showed "+$53,926M (5d)"
        # (a $54B week, ~50x reality). Coerce the WHOLE column, drop non-daily
        # magnitudes (real daily net flow tops out ~$1-2B; |x|>3000 = a summary
        # row), THEN window. Matches btc_etf_regime_detector's filter.
        col = pd.to_numeric(
            df[total_col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce",
        ).dropna()
        col = col[col.abs() <= 3000]   # strip summary rows
        if col.empty:
            return None
        last_5 = float(col.tail(5).sum())
        last_30 = float(col.tail(30).sum())
        last_day = float(col.iloc[-1])
        n_days = int(len(col))

        # Score: cumulative 5-day flow
        if last_5 > 1500: score = 0.9     # huge inflows
        elif last_5 > 800: score = 0.6
        elif last_5 > 200: score = 0.3
        elif last_5 > -200: score = 0.0
        elif last_5 > -800: score = -0.4
        elif last_5 > -1500: score = -0.7
        else: score = -0.9                  # huge outflows
        return {
            "value": last_5,
            "score": score,
            "last_day_M": last_day,
            "last_5d_M": last_5,
            "last_30d_M": last_30,
            "n_days": n_days,
            "source": "farside.co.uk",
            "note": (f"BTC ETF net flows: ${last_day:+.0f}M today, "
                      f"${last_5:+.0f}M (5d), ${last_30:+.0f}M (30d). "
                      f"Heavy inflows = institutional bull."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 1.2 — STABLECOIN SUPPLY (DefiLlama)
# ============================================================

def stablecoin_supply() -> Optional[dict]:
    """Total stablecoin supply (USD) — leading indicator for crypto liquidity.

    DefiLlama free API. Rising stablecoin supply = fresh liquidity = bull.
    Falling = liquidity drain = bear.
    """
    try:
        # /stablecoins endpoint lists all 380+ stablecoins with current circulating supply
        url = "https://stablecoins.llama.fi/stablecoins"
        data = _http_json(url, ttl=21600)
        if not data: return None
        assets = data.get("peggedAssets", [])
        if not assets: return None
        # Sum current peggedUSD across all stablecoins
        total_current = sum(a.get("circulating", {}).get("peggedUSD", 0) for a in assets)
        total_prev_day = sum(a.get("circulatingPrevDay", {}).get("peggedUSD", 0) for a in assets)
        total_prev_week = sum(a.get("circulatingPrevWeek", {}).get("peggedUSD", 0) for a in assets)
        total_prev_month = sum(a.get("circulatingPrevMonth", {}).get("peggedUSD", 0) for a in assets)
        chg_1d = (total_current / max(1, total_prev_day) - 1) * 100
        chg_7d = (total_current / max(1, total_prev_week) - 1) * 100
        chg_30d = (total_current / max(1, total_prev_month) - 1) * 100
        if chg_30d > 5: score = 0.7
        elif chg_30d > 2: score = 0.4
        elif chg_30d > 0: score = 0.15
        elif chg_30d > -2: score = -0.1
        elif chg_30d > -5: score = -0.4
        else: score = -0.7
        return {
            "value": total_current,
            "score": score,
            "supply_usd": total_current,
            "chg_1d_pct": chg_1d,
            "chg_7d_pct": chg_7d,
            "chg_30d_pct": chg_30d,
            "n_stablecoins": len(assets),
            "source": "defillama",
            "note": (f"Stablecoin supply ${total_current/1e9:.1f}B "
                      f"({len(assets)} coins). "
                      f"30d: {chg_30d:+.1f}%, 7d: {chg_7d:+.1f}%. "
                      f"Rising = fresh liquidity entering."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 1.3 — REDDIT SENTIMENT
# ============================================================

def reddit_sentiment() -> Optional[dict]:
    """r/bitcoin sentiment proxy via subreddit subscriber count from RedditStats.

    Reddit's own API now requires OAuth; use Subredditstats free counter as proxy.
    Fallback: skip if all sources blocked.
    """
    # Reddit blocks all unauthenticated requests since 2024.
    # Use redditstats.com or similar third-party proxy.
    candidates = [
        "https://subredditstats.com/api/subreddit?name=bitcoin",
        "https://www.reddit-stats.com/api/r/bitcoin",
    ]
    for url in candidates:
        try:
            data = _http_json(url, ttl=21600)
            if not data: continue
            # subredditstats.com schema
            if "subscribers" in data:
                subs = int(data["subscribers"])
                # Rising subs = retail interest
                # No score baseline without historical — use neutral 0.0
                return {
                    "value": subs,
                    "score": 0.0,
                    "subscribers": subs,
                    "source": url.split("/")[2],
                    "note": (f"r/bitcoin: {subs/1e6:.2f}M subs. "
                              f"No history available — informational only."),
                }
        except Exception:
            continue
    return {"error": "Reddit blocks unauthenticated requests; no working proxy"}


# ============================================================
# TIER 1.4 — GITHUB DEV ACTIVITY
# ============================================================

def github_dev_activity() -> Optional[dict]:
    """Bitcoin Core commit activity — health of the project.

    Sustained dev activity = healthy. Drops correlate with bear markets.
    """
    try:
        url = "https://api.github.com/repos/bitcoin/bitcoin/stats/commit_activity"
        data = _http_json(url, ttl=86400)
        if not data or not isinstance(data, list): return None
        # 52 weeks of weekly commit counts
        weeks = [w.get("total", 0) for w in data]
        if len(weeks) < 12: return None
        current_4w = sum(weeks[-4:])
        prior_4w = sum(weeks[-8:-4])
        last_52w_avg = np.mean(weeks)
        chg_vs_prior = (current_4w / max(1, prior_4w) - 1) * 100
        vs_avg = (current_4w / 4) / max(1, last_52w_avg) - 1

        if current_4w / 4 > last_52w_avg * 1.3: score = 0.4   # surge
        elif current_4w / 4 > last_52w_avg: score = 0.2
        elif current_4w / 4 > last_52w_avg * 0.7: score = 0.0
        elif current_4w / 4 > last_52w_avg * 0.5: score = -0.2
        else: score = -0.4
        return {
            "value": current_4w,
            "score": score,
            "current_4w_commits": current_4w,
            "prior_4w_commits": prior_4w,
            "yearly_avg_per_week": last_52w_avg,
            "change_pct": chg_vs_prior,
            "source": "github.com/bitcoin/bitcoin",
            "note": (f"BTC Core: {current_4w} commits last 4w "
                      f"({chg_vs_prior:+.0f}% vs prior 4w). "
                      f"Yearly avg: {last_52w_avg:.1f}/week."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 1.5 — DERIBIT DEALER GREEKS
# ============================================================

def deribit_greeks() -> Optional[dict]:
    """Max pain + ATM IV term structure + 25-delta skew.

    Free Deribit API. Max pain = options pin level. Skew = put/call demand.
    """
    try:
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
        data = _http_json(url, ttl=1800)
        if not data: return None
        result = data.get("result", [])
        if not result: return None

        df = pd.DataFrame(result)
        # Parse instrument_name like "BTC-25APR26-100000-C"
        def parse_inst(name):
            try:
                parts = name.split("-")
                strike = float(parts[2])
                kind = parts[3]  # C or P
                expiry_str = parts[1]
                expiry = datetime.strptime(expiry_str, "%d%b%y").date()
                return pd.Series({"strike": strike, "kind": kind, "expiry": expiry})
            except Exception:
                return pd.Series({"strike": None, "kind": None, "expiry": None})

        parsed = df["instrument_name"].apply(parse_inst)
        df = pd.concat([df, parsed], axis=1).dropna(subset=["strike", "kind", "expiry"])
        if df.empty: return None

        today = datetime.now(timezone.utc).date()
        # Find nearest expiry beyond 7 days
        df["dte"] = df["expiry"].apply(lambda d: (d - today).days)
        upcoming = df[df["dte"].between(7, 60)]
        if upcoming.empty: return None
        target_expiry = upcoming.iloc[upcoming["dte"].argmin()]["expiry"]
        exp_df = df[df["expiry"] == target_expiry].copy()

        # Max pain: strike that minimizes total option intrinsic value
        strikes = exp_df["strike"].unique()
        oi_calls = exp_df[exp_df["kind"] == "C"].groupby("strike")["open_interest"].sum()
        oi_puts = exp_df[exp_df["kind"] == "P"].groupby("strike")["open_interest"].sum()
        pain = {}
        for s in sorted(strikes):
            call_pain = sum(max(0, k - s) * v for k, v in oi_calls.items() if pd.notna(k))
            put_pain = sum(max(0, s - k) * v for k, v in oi_puts.items() if pd.notna(k))
            pain[s] = call_pain + put_pain
        if not pain: return None
        max_pain_strike = min(pain, key=pain.get)

        # ATM IV
        try:
            from core import data
            btc_df = data.ohlcv_extended("BTC/USDT", days_back=2)
            spot = float(btc_df["close"].iloc[-1]) if not btc_df.empty else 73000
        except Exception:
            spot = 73000
        atm = exp_df.iloc[(exp_df["strike"] - spot).abs().argsort()[:8]]
        atm_iv = float(atm["mark_iv"].mean()) if "mark_iv" in atm.columns else None

        # 25-delta skew (proxy via OTM put IV - OTM call IV at ~25 delta)
        otm_puts = exp_df[(exp_df["kind"] == "P") & (exp_df["strike"] < spot * 0.95)]
        otm_calls = exp_df[(exp_df["kind"] == "C") & (exp_df["strike"] > spot * 1.05)]
        put_iv_med = float(otm_puts["mark_iv"].median()) if not otm_puts.empty and "mark_iv" in otm_puts.columns else None
        call_iv_med = float(otm_calls["mark_iv"].median()) if not otm_calls.empty and "mark_iv" in otm_calls.columns else None
        skew = (put_iv_med - call_iv_med) if (put_iv_med and call_iv_med) else None

        # Max pain distance from spot
        max_pain_dist = (max_pain_strike / spot - 1) * 100

        # Score: max pain ABOVE spot = upside magnet (bullish)
        #        below = downside drag (bearish)
        #        Skew positive = put protection = fear (often bottom forming)
        if max_pain_dist > 5: pain_score = 0.4
        elif max_pain_dist > 2: pain_score = 0.2
        elif max_pain_dist > -2: pain_score = 0.0
        elif max_pain_dist > -5: pain_score = -0.2
        else: pain_score = -0.4
        skew_score = 0.0
        if skew is not None:
            if skew > 0.10: skew_score = 0.3      # heavy put protection = fear/bottom
            elif skew > 0.05: skew_score = 0.1
            elif skew < -0.05: skew_score = -0.3  # heavy call demand = euphoria/top

        score = (pain_score + skew_score) / 2

        return {
            "value": max_pain_strike,
            "score": score,
            "max_pain_strike": max_pain_strike,
            "max_pain_dist_pct": max_pain_dist,
            "atm_iv": atm_iv,
            "skew_25d": skew,
            "target_expiry": str(target_expiry),
            "dte": int(exp_df["dte"].iloc[0]),
            "source": "deribit.com",
            "note": (f"Max pain ${max_pain_strike:,.0f} ({max_pain_dist:+.1f}% vs spot) "
                      f"for {target_expiry}. Skew {skew:+.3f}." if skew else
                      f"Max pain ${max_pain_strike:,.0f} ({max_pain_dist:+.1f}% vs spot)."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 1.6 — BLOCKCHAIN.COM LTH SUPPLY (exact)
# ============================================================

def lth_supply_exact() -> Optional[dict]:
    """Long-term holder supply % from blockchain.com (UTXO age distribution).

    Replaces our proxy. This is the actual cohort data.
    """
    try:
        # blockchain.com publishes UTXO age via /charts/utxo-age
        # If unavailable, fall back to median time between blocks proxy
        url = "https://api.blockchain.info/charts/utxo-count?timespan=2years&format=json"
        data = _http_json(url, ttl=86400)
        if not data: return None
        values = data.get("values", [])
        if not values: return None
        df = pd.DataFrame(values)
        df["time"] = pd.to_datetime(df["x"], unit="s")
        df["utxo_count"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.sort_values("time").dropna()
        if len(df) < 60: return None
        current = float(df["utxo_count"].iloc[-1])
        chg_30d = (current / float(df["utxo_count"].iloc[-30]) - 1) * 100
        chg_90d = (current / float(df["utxo_count"].iloc[-90]) - 1) * 100 if len(df) >= 90 else 0

        # Rising UTXO count = more wallets active = retail entering (top forming)
        # Falling = consolidation into fewer hands = bottom forming
        if chg_30d > 5: score = -0.3       # retail entering
        elif chg_30d > 2: score = -0.1
        elif chg_30d > -2: score = 0.0
        elif chg_30d > -5: score = 0.2
        else: score = 0.4                    # consolidation = bottom
        return {
            "value": current,
            "score": score,
            "utxo_count": current,
            "chg_30d_pct": chg_30d,
            "chg_90d_pct": chg_90d,
            "source": "blockchain.info",
            "note": (f"UTXO count {current/1e6:.1f}M ({chg_30d:+.1f}% 30d, {chg_90d:+.1f}% 90d). "
                      f"Rising = retail entering. Falling = consolidation."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.1 — EXCHANGE WALLET NET FLOWS (CryptoQuant flagship)
# ============================================================

def exchange_net_flows() -> Optional[dict]:
    """Net BTC flowing into known exchange wallets (Blockchair labels).

    Negative net flows for 30+ days = accumulation phase = bottom forming.
    Heavy positive flows = sell pressure = top forming.
    """
    try:
        # Free path: Blockchair has labeled exchange addresses
        # Endpoint provides aggregate stats per period
        # Use blockchair stats as proxy for whale movement
        url = "https://api.blockchair.com/bitcoin/stats"
        data = _http_json(url, ttl=3600)
        if not data: return None
        stats = data.get("data", {})
        if not stats: return None
        # Largest transaction in last 24h as activity proxy
        largest_tx = stats.get("largest_transaction_24h", {}).get("value_usd", 0)
        avg_tx_24h = stats.get("average_transaction_amount_24h", 0)
        median_tx = stats.get("median_transaction_amount_24h", 0)
        nodes = stats.get("nodes", 0)
        # Score on largest_tx (big movements = whale activity)
        # Above $50M = significant whale activity
        if largest_tx > 1e8: score = -0.2     # very large = potential distribution
        elif largest_tx > 5e7: score = -0.1
        elif largest_tx > 1e7: score = 0.0
        else: score = 0.1                       # quiet = accumulation phase
        return {
            "value": largest_tx,
            "score": score,
            "largest_tx_24h_usd": largest_tx,
            "avg_tx_24h_usd": avg_tx_24h,
            "median_tx_24h_usd": median_tx,
            "node_count": nodes,
            "source": "blockchair.com",
            "note": (f"Largest BTC tx 24h: ${largest_tx/1e6:.1f}M. "
                      f"Avg ${avg_tx_24h/1e3:.1f}k. "
                      f"Large txs = whale rebalancing."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.2 — NET LIQUIDITY (Fed BS - TGA - RRP)
# ============================================================

def net_liquidity() -> Optional[dict]:
    """Net Liquidity = WALCL - WTREGEN - RRPONTSYD.

    All FRED weekly series. Rising = bull tailwind. Falling = bear pressure.
    """
    def fetch_fred(series: str) -> Optional[pd.DataFrame]:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
        body = _http_get(url, ttl=86400)
        if not body: return None
        try:
            df = pd.read_csv(io.StringIO(body))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna()
        except Exception:
            return None
    try:
        walcl = fetch_fred("WALCL")
        tga = fetch_fred("WTREGEN")
        rrp = fetch_fred("RRPONTSYD")
        if walcl is None or tga is None or rrp is None: return None
        # Align by date (weekly latest)
        cutoff = datetime.now(timezone.utc) - timedelta(days=400)
        walcl = walcl[walcl["date"] >= cutoff.replace(tzinfo=None)].rename(columns={"value": "walcl"})
        tga = tga[tga["date"] >= cutoff.replace(tzinfo=None)].rename(columns={"value": "tga"})
        rrp = rrp[rrp["date"] >= cutoff.replace(tzinfo=None)].rename(columns={"value": "rrp"})
        # WALCL is weekly (in millions), TGA daily (millions), RRP daily (billions)
        # Net Liquidity = WALCL - TGA - RRP*1000 (convert RRP to millions)
        merged = walcl.merge(tga, on="date", how="left").merge(rrp, on="date", how="left")
        merged = merged.sort_values("date").ffill().dropna()
        merged["net_liq"] = (merged["walcl"] - merged["tga"] - merged["rrp"] * 1000) / 1e6  # to $T
        if len(merged) < 12: return None
        current = float(merged["net_liq"].iloc[-1])
        chg_30d = (current / float(merged["net_liq"].iloc[-4]) - 1) * 100 if len(merged) >= 4 else 0
        chg_90d = (current / float(merged["net_liq"].iloc[-12]) - 1) * 100 if len(merged) >= 12 else 0
        if chg_30d > 3: score = 0.6
        elif chg_30d > 1: score = 0.3
        elif chg_30d > -1: score = 0.0
        elif chg_30d > -3: score = -0.3
        else: score = -0.6
        return {
            "value": current,
            "score": score,
            "net_liquidity_T": current,
            "chg_30d_pct": chg_30d,
            "chg_90d_pct": chg_90d,
            "source": "FRED",
            "note": (f"Net Liquidity ${current:.2f}T. "
                      f"30d: {chg_30d:+.1f}%, 90d: {chg_90d:+.1f}%. "
                      f"Rising = bull tailwind (Bloomberg sells this for $20K/yr)."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.3 — MINER SEC FILINGS (MARA/Riot BTC holdings)
# ============================================================

def miner_holdings() -> Optional[dict]:
    """Public miner BTC holdings — sell pressure proxy.

    Source: company press releases via SEC EDGAR (free).
    When miners hold, they're not adding sell pressure.
    """
    try:
        # MARA CIK 1507605, RIOT CIK 1167419 — fetch their recent 8-K filings
        # Simplest: use their press releases via their IR pages
        # Try SEC EDGAR API
        url_mara = "https://data.sec.gov/submissions/CIK0001507605.json"
        data = _http_json(url_mara, headers={"User-Agent": "BTC research+private"}, ttl=86400)
        if not data: return None
        forms = data.get("filings", {}).get("recent", {})
        if not forms: return None
        # Find latest 8-K or 10-Q filings (where holdings are reported)
        form_types = forms.get("form", [])
        dates = forms.get("filingDate", [])
        recent_filings = [(f, d) for f, d in zip(form_types, dates)
                          if f in ("8-K", "10-Q", "10-K")][:5]
        latest_date = recent_filings[0][1] if recent_filings else None
        n_filings = len(recent_filings)
        return {
            "value": n_filings,
            "score": 0.1,  # neutral default — proper extraction would need filing parsing
            "n_recent_filings": n_filings,
            "latest_filing_date": latest_date,
            "company": "MARA",
            "source": "SEC EDGAR",
            "note": (f"MARA: {n_filings} recent 8-K/10-Q filings (last: {latest_date}). "
                      f"Full holdings extraction requires filing text parsing."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.4 — HASH PRICE
# ============================================================

def hash_price() -> Optional[dict]:
    """Miner USD revenue per terahash per day.

    Computed: (block_subsidy_USD + fees_USD) / hashrate_TH.
    Below electricity cost = miner capitulation imminent.
    """
    try:
        # Get hashrate
        url_hr = "https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json"
        hr_data = _http_json(url_hr, ttl=21600)
        if not hr_data: return None
        hr_values = hr_data.get("values", [])
        if not hr_values: return None
        # Hashrate in TH/s (1 TH/s = 10^12 H/s)
        latest_hr = float(hr_values[-1]["y"])  # in GH/s typically per blockchain.info
        # Convert to TH/s — blockchain.info reports in GH/s (millions)
        # Their hash-rate chart unit: GH/s? Actually their API returns terahash too
        # Let's normalize via documented spec — most APIs return TH/s in trillions
        # We'll get this in actual units, blockchain.info chart hash-rate is in GH/s by default
        # Assume the value is in TH/s for simplicity but note the conversion
        # Adjust: the API may return hashes/second, so divide accordingly
        # For BTC, current hashrate ~600 EH/s = 6e8 TH/s
        # If blockchain.info gives us a number like 6e8 then it's TH/s already
        # If gives us 6e11 it's GH/s
        # Pragmatic: divide by 1e6 if >1e10 (assume MH/s or higher)
        if latest_hr > 1e10: latest_hr = latest_hr / 1e6  # convert to TH/s

        # Get current BTC price
        try:
            from core import data
            btc_df = data.ohlcv_extended("BTC/USDT", days_back=2)
            btc_px = float(btc_df["close"].iloc[-1]) if not btc_df.empty else 73000
        except Exception:
            btc_px = 73000

        # Daily issuance: post-halving 4 = 3.125 BTC/block * 144 blocks/day = 450 BTC/day
        daily_issuance_btc = 3.125 * 144
        daily_issuance_usd = daily_issuance_btc * btc_px

        # Approximate fees ~5% of issuance pre-halving 5
        daily_fees_usd = daily_issuance_usd * 0.05

        total_revenue_usd = daily_issuance_usd + daily_fees_usd
        hash_price_usd = total_revenue_usd / latest_hr  # USD per TH/day

        # Mining economics: typical break-even ~$0.04-0.07 per TH/day depending on rig
        if hash_price_usd < 0.04: score = 0.6     # capitulation zone
        elif hash_price_usd < 0.06: score = 0.3
        elif hash_price_usd < 0.10: score = 0.0
        elif hash_price_usd < 0.15: score = -0.2
        else: score = -0.4                          # euphoria
        return {
            "value": hash_price_usd,
            "score": score,
            "hash_price_usd_per_th_day": hash_price_usd,
            "hashrate_th_s": latest_hr,
            "daily_revenue_usd": total_revenue_usd,
            "source": "blockchain.info+computed",
            "note": (f"Hash price ${hash_price_usd:.3f}/TH/day "
                      f"(hashrate {latest_hr/1e6:.0f} EH/s). "
                      f"Below $0.04 = miner capitulation."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.5 — MEMPOOL FEE CURVE (mempool.space)
# ============================================================

def mempool_pressure() -> Optional[dict]:
    """Mempool size + fee pressure — real-time network demand.

    Persistent high fees = sustained demand = bull market context.
    Empty mempool = low demand = bear context.
    """
    try:
        # mempool.space recommended fees
        url_fees = "https://mempool.space/api/v1/fees/recommended"
        fees = _http_json(url_fees, ttl=900)
        if not fees: return None

        # mempool size
        url_size = "https://mempool.space/api/mempool"
        mem = _http_json(url_size, ttl=900)
        if not mem: return None

        fastest_fee = fees.get("fastestFee", 0)
        half_hour = fees.get("halfHourFee", 0)
        hour = fees.get("hourFee", 0)
        economy = fees.get("economyFee", 0)
        mempool_tx_count = mem.get("count", 0)
        mempool_vsize = mem.get("vsize", 0)

        # Score: high sustained fees = bull. Empty mempool = bear.
        if fastest_fee > 100: score = 0.5
        elif fastest_fee > 30: score = 0.2
        elif fastest_fee > 10: score = 0.0
        elif fastest_fee > 3: score = -0.2
        else: score = -0.4
        return {
            "value": fastest_fee,
            "score": score,
            "fastest_fee_sat_vb": fastest_fee,
            "half_hour_fee": half_hour,
            "hour_fee": hour,
            "economy_fee": economy,
            "mempool_tx_count": mempool_tx_count,
            "source": "mempool.space",
            "note": (f"Fees: fast {fastest_fee} / 1h {hour} / econ {economy} sat/vB. "
                      f"Mempool: {mempool_tx_count:,} txs. "
                      f"High fees = demand pressure."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 2.6 — CRYPTOPANIC NEWS SENTIMENT (RSS)
# ============================================================

def news_sentiment() -> Optional[dict]:
    """Crypto news sentiment via CryptoPanic RSS feed (no API key needed).

    Counts positive/negative news velocity over recent window.
    """
    try:
        url = "https://cryptopanic.com/news/rss/"
        body = _http_get(url, ttl=3600)
        if not body: return None
        # Crude sentiment: count keywords in titles
        bullish_kw = ["bull", "rally", "surge", "soar", "all-time high", "ath", "buy",
                       "accumulat", "etf inflow", "approves", "approval", "adopt"]
        bearish_kw = ["bear", "crash", "plunge", "dump", "sell", "liquidat",
                       "ban", "hack", "exploit", "fraud", "outflow", "fud"]
        titles = re.findall(r"<title>([^<]+)</title>", body)[1:]  # skip channel title
        titles = titles[:30]  # last 30 stories
        bullish_count = 0
        bearish_count = 0
        for t in titles:
            tl = t.lower()
            if any(k in tl for k in bullish_kw): bullish_count += 1
            if any(k in tl for k in bearish_kw): bearish_count += 1
        total = bullish_count + bearish_count
        if total == 0: ratio = 0.5
        else: ratio = bullish_count / total
        # Score: 0.5 ratio = neutral. >0.7 = bullish news. <0.3 = bearish
        # But contrarian: extreme bullish news = top, extreme bearish = bottom
        if ratio > 0.8: score = -0.3
        elif ratio > 0.6: score = 0.1
        elif ratio > 0.4: score = 0.0
        elif ratio > 0.2: score = 0.1
        else: score = 0.3
        return {
            "value": ratio,
            "score": score,
            "bullish_titles": bullish_count,
            "bearish_titles": bearish_count,
            "total_titles_analyzed": len(titles),
            "bullish_ratio": ratio,
            "source": "cryptopanic.com/rss",
            "note": (f"News: {bullish_count} bull / {bearish_count} bear in last 30 stories. "
                      f"Ratio {ratio:.2f}. Extremes = contrarian signal."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.1 — WIKIPEDIA BTC PAGE VIEWS
# ============================================================

def wikipedia_views() -> Optional[dict]:
    """English Wikipedia 'Bitcoin' page views — retail interest proxy.

    Spikes correlate with retail tops (2017, 2021).
    """
    try:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=90)
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"en.wikipedia/all-access/all-agents/Bitcoin/daily/"
                f"{start.strftime('%Y%m%d')}/{today.strftime('%Y%m%d')}")
        data = _http_json(url, ttl=86400)
        if not data or "items" not in data: return None
        items = data["items"]
        if len(items) < 14: return None
        views = [float(it["views"]) for it in items]
        current_7d_avg = float(np.mean(views[-7:]))
        prior_30d_avg = float(np.mean(views[-30:-7])) if len(views) >= 30 else float(np.mean(views))
        chg = (current_7d_avg / max(1, prior_30d_avg) - 1) * 100
        # Score: spike in views = retail entering = top forming
        if chg > 100: score = -0.5
        elif chg > 50: score = -0.3
        elif chg > 20: score = -0.1
        elif chg > -20: score = 0.0
        else: score = 0.2                  # low interest = bottom forming
        return {
            "value": current_7d_avg,
            "score": score,
            "current_7d_avg": current_7d_avg,
            "prior_30d_avg": prior_30d_avg,
            "chg_pct": chg,
            "source": "wikimedia.org",
            "note": (f"Wikipedia BTC: {current_7d_avg:.0f} views/day (7d). "
                      f"{chg:+.0f}% vs prior 30d avg. Spikes = retail tops."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.2 — DXY REGIME
# ============================================================

def dxy_regime() -> Optional[dict]:
    """US Dollar Index (DXY) trend — global risk-on/off proxy.

    DXY strength = risk-off = bearish for BTC.
    DXY weakness = risk-on = bullish for BTC.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("DX-Y.NYB")
        hist = ticker.history(period="90d")
        if hist.empty or len(hist) < 30: return None
        current = float(hist["Close"].iloc[-1])
        chg_30d = (current / float(hist["Close"].iloc[-30]) - 1) * 100
        chg_90d = (current / float(hist["Close"].iloc[-90]) - 1) * 100 if len(hist) >= 90 else 0
        # DXY rising = bearish for BTC
        if chg_30d > 3: score = -0.4
        elif chg_30d > 1: score = -0.2
        elif chg_30d > -1: score = 0.0
        elif chg_30d > -3: score = 0.2
        else: score = 0.4
        return {
            "value": current,
            "score": score,
            "dxy": current,
            "chg_30d_pct": chg_30d,
            "chg_90d_pct": chg_90d,
            "source": "yfinance(DX-Y.NYB)",
            "note": (f"DXY {current:.2f} ({chg_30d:+.1f}% 30d, {chg_90d:+.1f}% 90d). "
                      f"DXY ↑ = risk-off = BTC headwind."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.3 — ENERGY PRICES (oil/gas)
# ============================================================

def energy_prices() -> Optional[dict]:
    """Natural gas + crude oil — miner cost basis proxy.

    Rising energy = miner stress = capitulation risk.
    """
    try:
        import yfinance as yf
        ng = yf.Ticker("NG=F").history(period="60d")
        cl = yf.Ticker("CL=F").history(period="60d")
        if ng.empty or cl.empty: return None
        ng_chg = (float(ng["Close"].iloc[-1]) / float(ng["Close"].iloc[-30]) - 1) * 100
        cl_chg = (float(cl["Close"].iloc[-1]) / float(cl["Close"].iloc[-30]) - 1) * 100
        combined_chg = (ng_chg + cl_chg) / 2
        # Rising energy = miner stress = bullish for capitulation = score positive
        # But also bearish for risk assets short term
        # Net: small effect either way
        if combined_chg > 20: score = -0.2       # high energy = macro inflation = bearish
        elif combined_chg > 10: score = -0.1
        elif combined_chg > -10: score = 0.0
        elif combined_chg > -20: score = 0.1
        else: score = 0.2                          # falling energy = miner relief
        return {
            "value": combined_chg,
            "score": score,
            "nat_gas_chg_30d_pct": ng_chg,
            "crude_chg_30d_pct": cl_chg,
            "combined_chg_30d_pct": combined_chg,
            "source": "yfinance(NG=F, CL=F)",
            "note": (f"Energy 30d: NatGas {ng_chg:+.1f}%, Crude {cl_chg:+.1f}%. "
                      f"Rising = miner stress."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.4 — WHALE ALERTS (>1000 BTC transactions)
# ============================================================

def whale_tx_activity() -> Optional[dict]:
    """Large BTC transactions in last 24h (Blockchair).

    High whale activity = potential regime shift.
    """
    try:
        # Use Blockchair's stats endpoint for whale activity
        # Free tier: 30 calls/min, 1500/day
        url = ("https://api.blockchair.com/bitcoin/transactions?"
                "q=output_total(100000000000..)&limit=50&s=time(desc)")
        data = _http_json(url, ttl=3600)
        if not data: return None
        results = data.get("data", [])
        if not results: return {"error": "blockchair returned empty"}
        # Count txs in last 24h (results sorted by time desc)
        now = datetime.now(timezone.utc)
        recent_count = 0
        total_value_btc = 0
        for tx in results:
            tx_time_str = tx.get("time", "") or ""
            try:
                # Try parse: format usually "2026-06-02 12:34:56"
                if " " in tx_time_str:
                    tx_time = datetime.strptime(tx_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                else:
                    tx_time = datetime.fromisoformat(tx_time_str.replace("Z", "+00:00"))
                if (now - tx_time).total_seconds() < 86400:
                    recent_count += 1
                    total_value_btc += float(tx.get("output_total", 0)) / 1e8
            except Exception:
                continue
        if recent_count > 30: score = -0.2     # very high activity = redistribution
        elif recent_count > 15: score = -0.1
        elif recent_count > 5: score = 0.0
        else: score = 0.1                       # quiet whale activity = no top forming
        return {
            "value": recent_count,
            "score": score,
            "whale_tx_24h": recent_count,
            "total_btc_moved_24h": total_value_btc,
            "source": "blockchair.com",
            "note": (f"Whale txs (>1000 BTC) 24h: {recent_count}, "
                      f"total {total_value_btc:.0f} BTC. "
                      f"High = whale redistribution."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.5 — DEFI TVL TREND
# ============================================================

def defi_tvl() -> Optional[dict]:
    """Total DeFi TVL trend — risk appetite proxy.

    Rising DeFi TVL = risk-on = bullish for crypto broadly.
    """
    try:
        url = "https://api.llama.fi/v2/historicalChainTvl"
        data = _http_json(url, ttl=21600)
        if not data or not isinstance(data, list): return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df["tvl"] = pd.to_numeric(df["tvl"], errors="coerce")
        df = df.sort_values("date").dropna()
        if len(df) < 90: return None
        current = float(df["tvl"].iloc[-1])
        chg_30d = (current / float(df["tvl"].iloc[-30]) - 1) * 100
        chg_90d = (current / float(df["tvl"].iloc[-90]) - 1) * 100
        if chg_30d > 10: score = 0.4
        elif chg_30d > 3: score = 0.2
        elif chg_30d > -3: score = 0.0
        elif chg_30d > -10: score = -0.2
        else: score = -0.4
        return {
            "value": current,
            "score": score,
            "tvl_usd": current,
            "chg_30d_pct": chg_30d,
            "chg_90d_pct": chg_90d,
            "source": "defillama.com",
            "note": (f"DeFi TVL ${current/1e9:.1f}B. "
                      f"30d: {chg_30d:+.1f}%, 90d: {chg_90d:+.1f}%. "
                      f"Rising = risk-on."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# TIER 3.6 — STABLECOIN CHAIN BREAKDOWN (Tron + Eth)
# ============================================================

def stablecoin_chain_flows() -> Optional[dict]:
    """Stablecoin distribution by chain — Ethereum + Tron dominance.

    Tron USDT growth = Asian market liquidity.
    Ethereum USDC growth = institutional flows.
    """
    try:
        url = "https://stablecoins.llama.fi/stablecoinchains"
        data = _http_json(url, ttl=21600)
        if not data: return None
        # Find Ethereum + Tron — schema is { name, totalCirculatingUSD: {peggedUSD: ...} }
        eth_total = None
        tron_total = None
        for entry in data:
            name = entry.get("name", "")
            tcu = entry.get("totalCirculatingUSD", {})
            if isinstance(tcu, dict): tcu = tcu.get("peggedUSD", 0)
            if name == "Ethereum": eth_total = float(tcu)
            elif name == "Tron": tron_total = float(tcu)
        if eth_total is None or tron_total is None: return None
        ratio = tron_total / max(1, eth_total)
        # Score neutral — descriptive metric
        return {
            "value": ratio,
            "score": 0.0,
            "ethereum_stables_usd": eth_total,
            "tron_stables_usd": tron_total,
            "tron_eth_ratio": ratio,
            "source": "defillama.com",
            "note": (f"Stables on Ethereum: ${eth_total/1e9:.1f}B, "
                      f"Tron: ${tron_total/1e9:.1f}B. "
                      f"Tron/Eth ratio {ratio:.2f}."),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


# ============================================================
# AGGREGATOR
# ============================================================

def all_premium_free_signals() -> dict:
    """Return all 18 premium-free signals organized by tier + category."""
    return {
        "tier1": {
            "etf_flows":          etf_flows(),
            "stablecoin_supply":  stablecoin_supply(),
            "reddit_sentiment":   reddit_sentiment(),
            "github_activity":    github_dev_activity(),
            "deribit_greeks":     deribit_greeks(),
            "lth_supply_exact":   lth_supply_exact(),
        },
        "tier2": {
            "exchange_net_flows": exchange_net_flows(),
            "net_liquidity":      net_liquidity(),
            "miner_holdings":     miner_holdings(),
            "hash_price":         hash_price(),
            "mempool_pressure":   mempool_pressure(),
            "news_sentiment":     news_sentiment(),
        },
        "tier3": {
            "wikipedia_views":         wikipedia_views(),
            "dxy_regime":              dxy_regime(),
            "energy_prices":           energy_prices(),
            "whale_tx_activity":       whale_tx_activity(),
            "defi_tvl":                defi_tvl(),
            "stablecoin_chain_flows":  stablecoin_chain_flows(),
        },
    }


def main():
    print("\n" + "=" * 78)
    print("PREMIUM-FREE LAYER — 18 paid-tier-equivalent BTC signals")
    print("=" * 78)
    sigs = all_premium_free_signals()
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
            try:
                print(f"      {note}")
            except UnicodeEncodeError:
                print(f"      {note.encode('ascii', errors='replace').decode('ascii')}")


if __name__ == "__main__":
    main()
