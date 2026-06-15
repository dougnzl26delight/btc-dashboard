"""PRO-TIER ON-CHAIN SIGNALS — Willy Woo + Top 1% Glassnode analyst layer.

These are the signals institutional crypto analysts (Checkmate, James Check,
Willy Woo, ARK Crypto) rely on for actual bottom calls. They complement the
halving clock (time anchor) with cost-basis confirmation and miner stress data.

All signals use FREE data sources:
    - CoinMetrics free tier: PriceUSD, CapMrktCurUSD, CapMVRVCur, SplyCur,
      HashRate, BlkCnt, IssTotUSD
    - blockchain.info charts API: difficulty, n-transactions, miners-revenue,
      estimated-transaction-volume-usd
    - ccxt (Coinbase + Binance spot)

For metrics gated behind paid tier (CapRealUSD, SAOPR, SplyAct1yr,
TxTfrValAdjUSD, DaysDestroyed), we DERIVE the relevant signal from the
available free metrics. Each proxy is documented inline.

Categorical mapping:
    onchain:       realized_cap_drawdown, reserve_risk, asopr,
                   lth_sth_supply_ratio, cdd_spikes, dormancy_flow, nvt_signal_woo
    fundamentals:  puell_multiple, difficulty_ribbon
    flows:         coinbase_premium_gap
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Data fetchers
# ============================================================

CM_BASE = "https://community-api.coinmetrics.io/v4"
CM_TIMEOUT = 12

_CM_CACHE: dict = {}


def _cm(metric: str, days: int = 730) -> pd.DataFrame:
    """Fetch a single CoinMetrics free-tier metric. Cached per-call."""
    cache_key = (metric, days)
    if cache_key in _CM_CACHE:
        return _CM_CACHE[cache_key]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = (f"{CM_BASE}/timeseries/asset-metrics?assets=btc"
            f"&metrics={metric}&frequency=1d"
            f"&start_time={start.isoformat()}&end_time={end.isoformat()}"
            f"&page_size=10000")
    try:
        r = requests.get(url, timeout=CM_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            _CM_CACHE[cache_key] = pd.DataFrame()
            return _CM_CACHE[cache_key]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[[metric]].dropna()
        _CM_CACHE[cache_key] = df
        return df
    except Exception:
        _CM_CACHE[cache_key] = pd.DataFrame()
        return pd.DataFrame()


_BC_CACHE: dict = {}

def _blockchain_info(chart: str, timespan: str = "2years") -> pd.DataFrame:
    """Fetch a chart from blockchain.info free API. Cached per-call.

    Returns UTC-aware DatetimeIndex to match CoinMetrics.
    """
    cache_key = (chart, timespan)
    if cache_key in _BC_CACHE:
        return _BC_CACHE[cache_key]
    url = f"https://api.blockchain.info/charts/{chart}?timespan={timespan}&format=json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        values = r.json().get("values", [])
        if not values:
            _BC_CACHE[cache_key] = pd.DataFrame()
            return _BC_CACHE[cache_key]
        df = pd.DataFrame(values)
        # Make UTC-aware to match CoinMetrics joins
        df["time"] = pd.to_datetime(df["x"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df["value"] = pd.to_numeric(df["y"], errors="coerce")
        df = df[["value"]].dropna()
        _BC_CACHE[cache_key] = df
        return df
    except Exception:
        _BC_CACHE[cache_key] = pd.DataFrame()
        return pd.DataFrame()


# ============================================================
# 1. REALIZED CAP DRAWDOWN — Checkmate's #1 bottom indicator
# ============================================================

def realized_cap_drawdown() -> Optional[dict]:
    """Realized Cap drawdown from rolling 365d max.

    DERIVATION: realized_cap = market_cap / MVRV_ratio
    Both inputs are free CoinMetrics tier.

    Historical pattern: every cycle bottom shows realized cap drop of 15-25%.
    Below -10% = bear market confirmed.
    Below -15% = bottom zone.
    """
    try:
        df_cap = _cm("CapMrktCurUSD", days=730)
        df_mvrv = _cm("CapMVRVCur", days=730)
        if df_cap.empty or df_mvrv.empty: return None
        df = df_cap.join(df_mvrv, how="inner").dropna()
        if len(df) < 100: return None
        df["rcap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
        rolling_max = df["rcap"].rolling(window=365, min_periods=30).max()
        drawdown = (df["rcap"] / rolling_max - 1) * 100
        current_dd = float(drawdown.iloc[-1])
        rcap_now = float(df["rcap"].iloc[-1])
        rcap_peak = float(rolling_max.iloc[-1])
        if current_dd < -20: score = 0.9
        elif current_dd < -15: score = 0.7
        elif current_dd < -10: score = 0.4
        elif current_dd < -5: score = 0.1
        elif current_dd < 0: score = -0.1
        elif current_dd < 2: score = -0.3
        else: score = -0.5
        return {
            "value": current_dd,
            "score": score,
            "rcap": rcap_now,
            "rcap_peak": rcap_peak,
            "source": "coinmetrics_derived(CapMrktCurUSD/CapMVRVCur)",
            "note": (f"Realized Cap {current_dd:+.1f}% from rolling peak. "
                      f"Bottoms historically at -20% to -25%."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 2. RESERVE RISK — Glassnode classic LTH conviction indicator
# ============================================================

def reserve_risk() -> Optional[dict]:
    """Reserve Risk proxy via price vs MVRV-Z dynamics.

    Original Reserve Risk = price / HODL bank (sum of unspent UTXOs by age).
    PROXY: low MVRV-Z with positive forward momentum = high LTH conviction
    at low price = same signal as low Reserve Risk.

    Implementation: rolling 4yr percentile of (price × inv_mvrv_z normalized).
    Below 5th percentile = generational buy zone.
    Above 95th percentile = peak euphoria.
    """
    try:
        df_px = _cm("PriceUSD", days=1460)
        df_mvrv = _cm("CapMVRVCur", days=1460)
        if df_px.empty or df_mvrv.empty: return None
        df = df_px.join(df_mvrv, how="inner").dropna()
        if len(df) < 200: return None
        # MVRV Z-score using 4y rolling window
        rmean = df["CapMVRVCur"].rolling(1460, min_periods=200).mean()
        rstd = df["CapMVRVCur"].rolling(1460, min_periods=200).std()
        z = (df["CapMVRVCur"] - rmean) / rstd
        # Reserve risk proxy: when Z is low AND price is rising, conviction is high
        # Mathematically: RR ~ price / (Z-implied HODL strength)
        rr_raw = df["PriceUSD"] * np.maximum(0.5, 1 + z)
        # Normalize via rolling 4y percentile rank
        rolling_window = rr_raw.iloc[-1460:] if len(rr_raw) >= 1460 else rr_raw
        pct_rank = float((rolling_window < rr_raw.iloc[-1]).sum() / len(rolling_window))
        scaled = pct_rank * 0.05
        current_z = float(z.iloc[-1]) if not z.empty else 0
        # 2026-06 RECALIBRATION (Glassnode top-1% review):
        # Reserve Risk should track ACTUAL cost-basis capitulation, not price
        # percentile. Historical generational bottoms had MVRV-Z below -1.5
        # (2018) or below -2.0 (2015). At MVRV-Z = -0.67, we're at "mild value"
        # not "deep value." Score must reflect that.
        if scaled < 0.003 and current_z < -1.5:
            score = 0.95   # actual generational buy (2018/2015 magnitudes)
        elif scaled < 0.005 and current_z < -1.0:
            score = 0.7    # deep value zone (historical bottom magnitudes)
        elif scaled < 0.010 and current_z < -0.5:
            score = 0.20   # mild value forming — NOT generational yet
        elif scaled < 0.010:
            score = 0.05   # low percentile but Z not confirming
        elif scaled < 0.020:
            score = -0.05
        elif scaled < 0.035 and current_z > 1:
            score = -0.5
        elif scaled < 0.035:
            score = -0.2
        elif current_z > 2:
            score = -0.85  # peak euphoria
        else:
            score = -0.4
        return {
            "value": scaled,
            "score": score,
            "pct_rank_4y": pct_rank,
            "current_z": float(z.iloc[-1]) if not z.empty else None,
            "source": "coinmetrics_proxy(price+MVRV-Z)",
            "note": (f"Reserve Risk proxy {scaled:.4f} (pct rank {pct_rank*100:.0f}%). "
                      f"Below 0.003 = generational buy."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 3. PUELL MULTIPLE — miner revenue stress
# ============================================================

def puell_multiple() -> Optional[dict]:
    """Puell Multiple = daily issuance USD / 365d MA of daily issuance USD.

    Uses CoinMetrics IssTotUSD (free tier).
    Sub 0.5 = miner capitulation = bottom zone.
    Above 4.0 = peak euphoria.
    """
    try:
        df = _cm("IssTotUSD", days=730)
        if df.empty or len(df) < 365: return None
        iss = df["IssTotUSD"]
        ma365 = iss.rolling(window=365, min_periods=200).mean()
        puell = iss / ma365
        current = float(puell.iloc[-1])
        ma7 = float(puell.rolling(window=7, min_periods=2).mean().iloc[-1])
        if current < 0.5: score = 0.9
        elif current < 0.7: score = 0.6
        elif current < 1.0: score = 0.2
        elif current < 1.5: score = -0.1
        elif current < 2.5: score = -0.3
        elif current < 4.0: score = -0.6
        else: score = -0.9
        return {
            "value": current,
            "score": score,
            "ma7": ma7,
            "daily_iss_usd": float(iss.iloc[-1]),
            "ma365_iss_usd": float(ma365.iloc[-1]),
            "source": "coinmetrics_IssTotUSD",
            "note": (f"Puell {current:.2f} (7d MA {ma7:.2f}). "
                      f"<0.5 = miner capitulation. >4.0 = euphoria peak."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 4. COINBASE PREMIUM GAP — Woo's US institutional flow signal
# ============================================================

def coinbase_premium_gap() -> Optional[dict]:
    """Coinbase Premium Gap = (CB BTC/USD - Binance BTC/USDT) / Binance.

    Persistent positive premium = US institutional buying = bottom signal.
    Persistent negative premium = US institutional selling.
    """
    try:
        import ccxt
        from core import data
        cb = ccxt.coinbase()
        cb_t = cb.fetch_ticker("BTC/USD")
        bn_t = data.btc_ticker()  # region-resilient; same shape as ccxt fetch_ticker
        cb_px = cb_t.get("last")
        bn_px = bn_t.get("last")
        if not cb_px or not bn_px: return None
        premium_bps = (cb_px / bn_px - 1) * 10000
        premium_pct = premium_bps / 100
        if premium_bps > 5: score = 0.8
        elif premium_bps > 2: score = 0.5
        elif premium_bps > 0: score = 0.2
        elif premium_bps > -2: score = -0.2
        elif premium_bps > -5: score = -0.5
        else: score = -0.8
        return {
            "value": premium_pct,
            "score": score,
            "premium_bps": premium_bps,
            "cb_price": cb_px,
            "binance_price": bn_px,
            "source": "ccxt_spot",
            "note": (f"CB premium {premium_bps:+.1f}bps ({premium_pct:+.3f}%). "
                      f"Positive = US institutional buying."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 5. DIFFICULTY RIBBON — Woo's miner capitulation indicator
# ============================================================

def difficulty_ribbon() -> Optional[dict]:
    """Difficulty Ribbon — 8 SMAs of difficulty (Willy Woo's signal).

    Compressed / inverted ribbon = miner capitulation.
    Cross UP after compression = historical bottom marker.
    """
    try:
        df = _blockchain_info("difficulty", timespan="2years")
        if df.empty or len(df) < 200: return None
        diff = df["value"]
        windows = [9, 14, 25, 40, 60, 90, 128, 200]
        smas = {w: diff.rolling(window=w, min_periods=max(2, w//2)).mean() for w in windows}
        latest = {w: float(smas[w].iloc[-1]) for w in windows}
        shortest = latest[windows[0]]
        longest = latest[windows[-1]]
        spread_pct = (shortest / longest - 1) * 100

        if spread_pct < -8: score = 0.85
        elif spread_pct < -4: score = 0.6
        elif spread_pct < -1: score = 0.3
        elif spread_pct < 2: score = 0.0
        elif spread_pct < 5: score = -0.2
        elif spread_pct < 10: score = -0.4
        else: score = -0.6

        # Cross-up detection: 20d window
        ratio_recent = smas[windows[0]].iloc[-20:] / smas[windows[-1]].iloc[-20:]
        cross_up = (ratio_recent.iloc[-1] > 1.0) and (ratio_recent.iloc[0] < 1.0)
        if cross_up: score = min(0.95, score + 0.4)
        return {
            "value": spread_pct,
            "score": score,
            "ribbon_spread_pct": spread_pct,
            "shortest_sma": shortest,
            "longest_sma": longest,
            "recent_cross_up": bool(cross_up),
            "source": "blockchain.info_difficulty",
            "note": (f"Ribbon spread {spread_pct:+.2f}%. "
                      f"<0 = miner capitulation. Cross-up after = bottom marker."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 6. aSOPR PROXY — MVRV-1.0 cross dynamics
# ============================================================

def asopr() -> Optional[dict]:
    """aSOPR proxy via MVRV-1.0 cross dynamics.

    Original aSOPR = avg UTXO spent profit/loss ratio (Glassnode paid tier).
    PROXY: when MVRV crosses 1.0 from above and STAYS below = bear confirmed
    (same signal as aSOPR rejecting 1.0). Reclaim from below = recovery.

    Use 30d MVRV vs 90d MVRV around the 1.0 line.
    """
    try:
        df = _cm("CapMVRVCur", days=365)
        if df.empty or len(df) < 100: return None
        mvrv = df["CapMVRVCur"]
        current = float(mvrv.iloc[-1])
        ma7 = float(mvrv.rolling(window=7, min_periods=2).mean().iloc[-1])
        ma30 = float(mvrv.rolling(window=30, min_periods=10).mean().iloc[-1])

        recent = mvrv.iloc[-60:]
        # Rejections at 1.0 from below
        rejections_at_1 = ((recent.shift(1) > 1.0) & (recent.shift(-1) < 1.0)).sum()
        below_count = int((recent < 1.0).sum())

        if ma7 > 1.05 and ma30 > 1.0: score = 0.4    # sustained above = bull
        elif ma7 > 1.0 and ma30 > 1.0: score = 0.2
        elif ma7 < 0.95 and ma30 < 0.98: score = 0.7  # deep below = bottom forming
        elif ma7 < 1.0 and below_count > 30: score = 0.5
        elif rejections_at_1 >= 2: score = -0.5
        else: score = 0.0
        return {
            "value": current,
            "score": score,
            "ma7": ma7,
            "ma30": ma30,
            "rejections_at_1": int(rejections_at_1),
            "days_below_1": below_count,
            "source": "coinmetrics_proxy(MVRV-cross-1)",
            "note": (f"aSOPR proxy via MVRV: now {current:.3f} (7d MA {ma7:.3f}). "
                      f"<1 sustained = bear/bottom forming. {below_count}/60d below 1.0."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 7. LTH/STH SUPPLY DYNAMICS PROXY — realized price stability
# ============================================================

def lth_sth_supply_ratio() -> Optional[dict]:
    """LTH/STH supply proxy via realized price stability.

    Original requires SplyAct1yr (paid tier).
    PROXY: When realized price (rcap/supply) stays FLAT while market price
    drops = HODLers not selling (LTH supply rising) = bottom positioning.
    When realized price RISES while market price drops = LTHs taking profit
    on capitulation = distribution = top.

    Measures 90d slope of realized price vs market price.
    """
    try:
        df_cap = _cm("CapMrktCurUSD", days=400)
        df_mvrv = _cm("CapMVRVCur", days=400)
        df_supply = _cm("SplyCur", days=400)
        df_px = _cm("PriceUSD", days=400)
        if df_cap.empty or df_mvrv.empty or df_supply.empty or df_px.empty: return None
        df = (df_cap.join(df_mvrv, how="inner")
                       .join(df_supply, how="inner")
                       .join(df_px, how="inner")
                       .dropna())
        if len(df) < 100: return None
        # Realized price = realized cap / supply
        df["rcap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
        df["realized_price"] = df["rcap"] / df["SplyCur"]
        # 90d slopes
        if len(df) < 90: return None
        recent = df.iloc[-90:]
        rp_slope = (recent["realized_price"].iloc[-1] /
                     recent["realized_price"].iloc[0] - 1) * 100
        mp_slope = (recent["PriceUSD"].iloc[-1] /
                     recent["PriceUSD"].iloc[0] - 1) * 100

        # divergence: realized rising while market falling = LTH distribution
        # realized flat while market falling = LTH holding (bullish)
        divergence = rp_slope - mp_slope

        if mp_slope < -5 and abs(rp_slope) < 3:
            # market falling, realized flat = HODLers holding through pain
            score = 0.7
        elif mp_slope < -5 and rp_slope > 5:
            # market falling AND realized rising = LTH distribution = TOP
            score = -0.8
        elif mp_slope > 10 and rp_slope < 0:
            # market rising, realized falling = STH driving rally = unstable
            score = -0.2
        elif mp_slope > 10 and rp_slope > 5:
            # both rising = healthy bull
            score = 0.3
        else:
            score = 0.0
        return {
            "value": divergence,
            "score": score,
            "realized_price_90d_chg_pct": rp_slope,
            "market_price_90d_chg_pct": mp_slope,
            "divergence_pct": divergence,
            "current_realized_price": float(df["realized_price"].iloc[-1]),
            "source": "coinmetrics_proxy(realized_price_slope)",
            "note": (f"Realized price {rp_slope:+.1f}% vs market {mp_slope:+.1f}% (90d). "
                      f"Divergence {divergence:+.1f}%. Flat realized + falling market = LTH HOLD."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 8. CDD SPIKES PROXY — MVRV velocity
# ============================================================

def cdd_spikes() -> Optional[dict]:
    """CDD spike proxy via MVRV velocity.

    Original Coin Days Destroyed requires DaysDestroyed (paid tier).
    PROXY: rapid MVRV change implies large UTXO movement (the same
    underlying phenomenon CDD measures). Z-score the 7d MVRV change.

    Z > 2.5 = large LTH/whale movement
    Z < -1 sustained = HODLers holding (low movement)
    """
    try:
        df = _cm("CapMVRVCur", days=365)
        if df.empty or len(df) < 100: return None
        mvrv = df["CapMVRVCur"]
        mvrv_chg7 = mvrv.pct_change(7).fillna(0)
        ma = mvrv_chg7.rolling(window=90, min_periods=30).mean()
        std = mvrv_chg7.rolling(window=90, min_periods=30).std()
        z = (mvrv_chg7 - ma) / std
        current_z = float(z.iloc[-1])
        ma7_z = float(z.rolling(window=7, min_periods=2).mean().iloc[-1])

        if abs(current_z) > 3: score = -0.3
        elif current_z > 2: score = -0.2
        elif current_z < -1: score = 0.3
        else: score = 0.0
        return {
            "value": current_z,
            "score": score,
            "ma7_z": ma7_z,
            "mvrv_7d_chg_pct": float(mvrv_chg7.iloc[-1]) * 100,
            "source": "coinmetrics_proxy(MVRV-velocity)",
            "note": (f"CDD proxy z-score {current_z:+.2f} (7d MA {ma7_z:+.2f}). "
                      f"|Z|>3 = significant movement. Z<-1 sustained = holding."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 9. DORMANCY FLOW PROXY — market cap vs network throughput
# ============================================================

def dormancy_flow() -> Optional[dict]:
    """Dormancy flow proxy.

    Original = market cap / (price × CDD × 365). Paid metric.
    PROXY: market cap / (price × hash_rate × time_factor)
    Higher = price is supported by network security = healthy
    Lower = price ahead of network = unstable

    Use rolling percentile rank.
    """
    try:
        df_cap = _cm("CapMrktCurUSD", days=365)
        df_hash = _cm("HashRate", days=365)
        df_px = _cm("PriceUSD", days=365)
        if df_cap.empty or df_hash.empty or df_px.empty: return None
        df = df_cap.join(df_hash, how="inner").join(df_px, how="inner").dropna()
        if len(df) < 90: return None
        # network value ratio: cap / (price × hash_rate)
        df["dfm"] = df["CapMrktCurUSD"] / (df["PriceUSD"] * df["HashRate"])
        recent = df["dfm"].iloc[-365:]
        if len(recent) < 30: return None
        current = float(df["dfm"].iloc[-1])
        pct_rank = float((recent < current).sum() / len(recent))

        if pct_rank > 0.85: score = 0.5
        elif pct_rank > 0.65: score = 0.2
        elif pct_rank < 0.15: score = 0.3
        elif pct_rank < 0.30: score = 0.1
        else: score = 0.0
        return {
            "value": pct_rank,
            "score": score,
            "raw_value": current,
            "pct_rank_365d": pct_rank,
            "source": "coinmetrics_proxy(cap/hash)",
            "note": (f"Dormancy proxy pct rank {pct_rank*100:.0f}%. "
                      f"High = network supports price. Low = price ahead of utility."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 10. NVT SIGNAL (Woo) — market cap / 90d MA tx volume
# ============================================================

def nvt_signal_woo() -> Optional[dict]:
    """NVT Signal (Willy Woo's smoothed NVT).

    Original = market cap / 90d MA of TxTfrValAdjUSD. Paid tier.
    PROXY: use blockchain.info estimated-transaction-volume-usd (free).

    Below 40 = bottom undervaluation.
    Above 150 = top overvaluation.
    """
    try:
        df_cap = _cm("CapMrktCurUSD", days=365)
        df_tx = _blockchain_info("estimated-transaction-volume-usd", timespan="1year")
        if df_cap.empty or df_tx.empty: return None
        # Align frequencies
        df = df_cap.join(df_tx.rename(columns={"value": "tx_usd"}), how="inner").dropna()
        if len(df) < 90: return None
        tx_ma90 = df["tx_usd"].rolling(window=90, min_periods=30).mean()
        nvt = df["CapMrktCurUSD"] / tx_ma90
        nvt = nvt.dropna()
        if len(nvt) < 100: return None
        current = float(nvt.iloc[-1])
        # Use percentile rank vs trailing 365d — blockchain.info tx volume is
        # lower than CoinMetrics' adjusted version so absolute thresholds
        # don't match Woo's published 40/150 scale. Rank approach is robust.
        window = nvt.iloc[-365:] if len(nvt) >= 365 else nvt
        pct = float((window < current).sum() / len(window))
        if pct < 0.10: score = 0.85   # at bottom of recent range = undervalued
        elif pct < 0.25: score = 0.5
        elif pct < 0.60: score = 0.0
        elif pct < 0.80: score = -0.3
        elif pct < 0.95: score = -0.6  # at top of recent range = overvalued
        else: score = -0.85
        return {
            "value": current,
            "score": score,
            "pct_rank_365d": pct,
            "tx_volume_usd_90d_ma": float(tx_ma90.iloc[-1]),
            "source": "coinmetrics+blockchain.info(rank)",
            "note": (f"NVT Signal (Woo proxy): {current:.0f} (pct rank {pct*100:.0f}%). "
                      f"Low rank = bottom value. High rank = overvalued."),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Aggregator
# ============================================================

def all_pro_signals() -> dict:
    """Return all 10 pro-tier signals routed to categories."""
    return {
        "onchain": {
            "realized_cap_drawdown": realized_cap_drawdown(),
            "reserve_risk":          reserve_risk(),
            "asopr":                 asopr(),
            "lth_sth_supply_ratio":  lth_sth_supply_ratio(),
            "cdd_spikes":            cdd_spikes(),
            "dormancy_flow":         dormancy_flow(),
            "nvt_signal_woo":        nvt_signal_woo(),
        },
        "fundamentals": {
            "puell_multiple":     puell_multiple(),
            "difficulty_ribbon":  difficulty_ribbon(),
        },
        "flows": {
            "coinbase_premium_gap": coinbase_premium_gap(),
        },
    }


def main():
    print("\n" + "=" * 76)
    print("PRO-TIER ON-CHAIN SIGNALS (Woo + Glassnode top-1% layer)")
    print("=" * 76)
    sigs = all_pro_signals()
    for cat, cat_sigs in sigs.items():
        print(f"\n[{cat.upper()}]")
        for name, d in cat_sigs.items():
            if d is None:
                print(f"  {name:<24s} (unavailable)")
                continue
            if d.get("error"):
                print(f"  {name:<24s} ERROR: {d['error'][:60]}")
                continue
            score = d.get("score", 0)
            val = d.get("value")
            arrow = ("++" if score > 0.5 else "+" if score > 0.1
                      else "=" if abs(score) <= 0.1
                      else "-" if score > -0.5 else "--")
            val_str = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)[:10]
            print(f"  {name:<24s} {arrow:>2s} {score:+.2f}  val={val_str}")
            print(f"    {d.get('note', '')[:90]}")


if __name__ == "__main__":
    main()
