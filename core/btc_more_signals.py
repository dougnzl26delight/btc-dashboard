"""8 more FREE top-tier BTC signals.

THE BIG ONES:
    1. MVRV Z-Score — most predictive cycle indicator ever published
    2. Hayes Liquidity Trinity (Fed BS - TGA - RRP) — macro liquidity gauge
    3. Cross-asset ratios (BTC/Gold, BTC/SPX, BTC/NDX, BTC/TLT)
    4. Google Trends "bitcoin" — retail attention proxy
    5. CME BTC Futures Basis — institutional positioning
    6. NVT Signal — network value vs transactions
    7. Stock-to-Flow Deflection — fair value gap
    8. Yield Curve + Real Rates — recession/liquidity regime

All from FREE sources: CoinMetrics, FRED, yfinance, Google Trends, Binance.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _http_json(url: str, timeout: int = 15, headers: Optional[dict] = None):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _http_text(url: str, timeout: int = 15):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _score_threshold(value, bull_threshold, bear_threshold):
    if value is None: return None
    if bull_threshold > bear_threshold:
        if value >= bull_threshold: return 1.0
        if value <= bear_threshold: return -1.0
        return (value - bear_threshold) / (bull_threshold - bear_threshold) * 2 - 1
    if value <= bull_threshold: return 1.0
    if value >= bear_threshold: return -1.0
    return -((value - bull_threshold) / (bear_threshold - bull_threshold) * 2 - 1)


# ============================================================
# 1. MVRV Z-SCORE — the gold standard cycle indicator
# ============================================================

def _cm_single_metric(metric: str, start_time: str = "2014-01-01") -> Optional[pd.DataFrame]:
    """Fetch single CoinMetrics metric (multi-metric requires paid tier)."""
    url = (f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
           f"?assets=btc&metrics={metric}&start_time={start_time}&page_size=10000")
    d = _http_json(url, timeout=30)
    if not d or "data" not in d: return None
    rows = []
    for r in d["data"]:
        try:
            rows.append({"date": pd.to_datetime(r["time"]).date(),
                         metric: float(r[metric])})
        except Exception: continue
    return pd.DataFrame(rows) if rows else None


def mvrv_z_score() -> Optional[dict]:
    """MVRV Z-Score = (Market Cap - Realized Cap) / std(Market Cap).

    Universally regarded as the most predictive Bitcoin cycle indicator.
        Z > 7   : EUPHORIA (cycle peak) — sell zone
        Z < 0   : DEEP VALUE (cycle bottom) — buy zone

    Free CoinMetrics tier limits to ONE metric per call, so we fetch separately.
    """
    try:
        # Use MVRV ratio (free) and compute Z-score against historical distribution.
        # Equivalent in signal value to the original mcap/rcap formula since
        # MVRV ratio = mcap / rcap.
        df = _cm_single_metric("CapMVRVCur", start_time="2018-01-01")
        if df is None or len(df) < 200: return None
        df = df.rename(columns={"CapMVRVCur": "mvrv"}).sort_values("date").reset_index(drop=True)

        rolling_mean = df["mvrv"].rolling(1460, min_periods=200).mean()
        rolling_std = df["mvrv"].rolling(1460, min_periods=200).std()
        df["z_score"] = (df["mvrv"] - rolling_mean) / rolling_std

        current = df.iloc[-1]
        z = float(current["z_score"]) if pd.notna(current["z_score"]) else None
        if z is None: return None

        # 2026-06-01 RECALIBRATED for muted institutional cycles.
        # Cycle 5 peak Z-Score was +0.99 (cycle 4 was +2.85, cycle 3 was +7).
        # Each cycle's peak Z is decaying ~50%. New threshold: Z >= +0.8 = bear.
        score = _score_threshold(z, -1.0, 0.8)

        return {
            "value": z,
            "score": score,
            "mvrv_current": float(current["mvrv"]),
            "mvrv_4y_mean": float(rolling_mean.iloc[-1]),
            "mvrv_4y_std": float(rolling_std.iloc[-1]),
            "source": "coinmetrics",
            "note": "Z < -1 = bull; Z > +0.8 = bear (recalibrated for muted cycle 6)",
        }
    except Exception:
        return None


# ============================================================
# 2. HAYES LIQUIDITY TRINITY — the macro thesis core
# ============================================================

def hayes_liquidity_trinity() -> Optional[dict]:
    """Net USD Liquidity = Fed Balance Sheet - Treasury General Account - Reverse Repo.

    This is Hayes' central thesis: BTC tracks this metric with 3-6 month lag.
    Rising net liquidity = bull setup. Falling = bearish for BTC.

    All free via FRED CSV endpoints.
    """
    try:
        # Fetch each FRED series
        def _fred(series_id):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2020-01-01"
            text = _http_text(url, timeout=20)
            if not text: return None
            rows = []
            for line in text.split("\n")[1:]:
                parts = line.strip().split(",")
                if len(parts) < 2: continue
                # FRED uses "." for missing values — skip these
                if parts[1].strip() in (".", "", "NaN"): continue
                try:
                    rows.append({"date": pd.to_datetime(parts[0]).date(),
                                 "val": float(parts[1])})
                except Exception: continue
            return pd.DataFrame(rows) if rows else None

        # WALCL = Fed Total Assets (weekly, millions)
        fed_bs = _fred("WALCL")
        # WTREGEN = Treasury General Account (weekly, billions)
        tga = _fred("WTREGEN")
        # RRPONTSYD = Reverse Repo Op Total (daily, billions)
        rrp = _fred("RRPONTSYD")

        if fed_bs is None or tga is None or rrp is None: return None

        # Get latest for each
        latest_fed = float(fed_bs["val"].iloc[-1]) * 1e6   # millions -> dollars
        latest_tga = float(tga["val"].iloc[-1]) * 1e9      # billions -> dollars
        latest_rrp = float(rrp["val"].iloc[-1]) * 1e9      # billions -> dollars

        net_liquidity = latest_fed - latest_tga - latest_rrp
        # Compare to 90 days ago for trend
        if len(fed_bs) >= 14:  # weekly data ~13 weeks = 90d
            fed_90d_ago = float(fed_bs["val"].iloc[-14]) * 1e6
            tga_90d_ago = float(tga["val"].iloc[-14]) * 1e9 if len(tga) >= 14 else latest_tga
            rrp_90d_ago = float(rrp["val"].iloc[-min(90, len(rrp))]) * 1e9
            net_90d_ago = fed_90d_ago - tga_90d_ago - rrp_90d_ago
            change_90d_pct = (net_liquidity / net_90d_ago - 1) * 100
        else:
            change_90d_pct = 0

        # Score: rising net liquidity = bull
        score = _score_threshold(change_90d_pct, 2.0, -2.0)

        return {
            "value": net_liquidity / 1e12,   # in $T
            "score": score,
            "fed_balance_sheet_t": latest_fed / 1e12,
            "tga_balance_b": latest_tga / 1e9,
            "rrp_balance_b": latest_rrp / 1e9,
            "change_90d_pct": change_90d_pct,
            "source": "FRED",
            "note": "Net liquidity = Fed BS - TGA - RRP. Hayes thesis: BTC follows this with 3-6m lag",
        }
    except Exception:
        return None


# ============================================================
# 3. CROSS-ASSET RATIOS — regime context from outside crypto
# ============================================================

def cross_asset_ratios() -> Optional[dict]:
    """BTC's relationship to traditional risk assets.

    BTC/Gold rising = digital gold thesis winning.
    BTC/SPX rising = BTC outperforming equity.
    BTC/NDX rising = outperforming tech (decoupling from AI flow).
    BTC/TLT inverse = BTC as risk-on (when bonds fall, BTC rises).
    """
    try:
        import yfinance as yf
        # Fetch each separately to avoid MultiIndex issues
        closes = {}
        for t in ["BTC-USD", "GLD", "SPY", "QQQ", "TLT"]:
            try:
                h = yf.Ticker(t).history(period="180d", interval="1d")
                if not h.empty and len(h) >= 30:
                    closes[t] = h["Close"]
            except Exception:
                continue
        if "BTC-USD" not in closes or len(closes) < 2: return None

        btc_now = float(closes["BTC-USD"].iloc[-1])
        btc_30d = float(closes["BTC-USD"].iloc[-30])

        ratios = {}
        scores = []
        for asset in ["GLD", "SPY", "QQQ", "TLT"]:
            if asset not in closes: continue
            asset_now = float(closes[asset].iloc[-1])
            asset_30d = float(closes[asset].iloc[-30])
            if asset_now <= 0 or asset_30d <= 0: continue
            ratio_now = btc_now / asset_now
            ratio_30d = btc_30d / asset_30d
            change_pct = (ratio_now / ratio_30d - 1) * 100
            ratios[f"btc_vs_{asset.lower()}"] = {
                "ratio_now": ratio_now,
                "change_30d_pct": change_pct,
            }
            scores.append(_score_threshold(change_pct, 5, -5))

        if not scores: return None
        scores = [s for s in scores if s is not None]
        if not scores: return None
        avg_score = float(np.mean(scores))

        return {
            "value": avg_score,
            "score": avg_score,
            "ratios": ratios,
            "source": "yfinance",
            "note": "BTC outperforming other assets (rising ratios) = bull setup",
        }
    except Exception:
        return None


# ============================================================
# 4. GOOGLE TRENDS — retail attention
# ============================================================

def google_trends_btc() -> Optional[dict]:
    """Google Trends 'bitcoin' search interest.

    Spike to 80-100 = retail FOMO = late-cycle.
    Cool to 10-30 = no interest = accumulation phase.
    Free via pytrends.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(["bitcoin"], timeframe="today 3-m", geo="")
        df = pytrends.interest_over_time()
        if df is None or df.empty: return None

        values = df["bitcoin"].values
        current = float(values[-1])
        avg_30d = float(np.mean(values[-30:])) if len(values) >= 30 else current
        avg_90d = float(np.mean(values))
        # Trend: current vs 90-day average
        trend_pct = (current / avg_90d - 1) * 100 if avg_90d > 0 else 0

        # Score: HIGH trends = retail attention = late cycle (bear contrarian)
        # LOW trends = no interest = accumulation phase (bull contrarian)
        # Note: doesn't always work — extreme low can also = capitulation
        if current < 20: score = 0.5    # low attention = bull setup
        elif current < 40: score = 0.2
        elif current < 60: score = 0.0
        elif current < 80: score = -0.3
        else: score = -0.6              # peak attention = sell zone

        return {
            "value": current,
            "score": score,
            "current_score": current,
            "avg_30d": avg_30d,
            "avg_90d": avg_90d,
            "trend_pct_vs_90d": trend_pct,
            "source": "google_trends",
            "note": "high search (80+) = retail FOMO (bear); low (<20) = accumulation (bull)",
        }
    except Exception:
        return None


# ============================================================
# 5. CME BITCOIN FUTURES BASIS — institutional positioning
# ============================================================

def cme_basis() -> Optional[dict]:
    """CME BTC Futures basis vs spot.

    Persistent contango (futures > spot) = institutional bullish positioning.
    Backwardation (futures < spot) = institutional bearish / hedging.
    Free via yfinance BTC=F (front-month futures).
    """
    try:
        import yfinance as yf
        cme = yf.Ticker("BTC=F")
        spot_hist = yf.Ticker("BTC-USD").history(period="5d", interval="1d")
        cme_hist = cme.history(period="5d", interval="1d")
        if cme_hist.empty or spot_hist.empty: return None

        cme_last = float(cme_hist["Close"].iloc[-1])
        spot_last = float(spot_hist["Close"].iloc[-1])
        basis_pct = (cme_last / spot_last - 1) * 100
        basis_annualized = basis_pct * 12   # approximate, assuming monthly contract

        # Score: positive annualized basis = bull (contango)
        # negative = backwardation (institutional hedge)
        score = _score_threshold(basis_annualized, 3, -3)

        return {
            "value": basis_annualized,
            "score": score,
            "cme_close": cme_last,
            "spot_close": spot_last,
            "basis_pct": basis_pct,
            "source": "yfinance:BTC=F",
            "note": "positive basis = institutional contango (bull); backwardation = hedging (bear)",
        }
    except Exception:
        return None


# ============================================================
# 6. NVT SIGNAL — network value to transactions
# ============================================================

def nvt_signal() -> Optional[dict]:
    """Network Value to Transactions ratio.

    Like P/E for Bitcoin. High NVT = overvalued vs usage (bear).
    Low NVT = undervalued vs usage (bull).
    """
    try:
        df_raw = _cm_single_metric("NVTAdj", start_time="2020-01-01")
        if df_raw is None or len(df_raw) < 30: return None
        df = df_raw.rename(columns={"NVTAdj": "nvt"}).sort_values("date").reset_index(drop=True)
        current = float(df["nvt"].iloc[-1])
        # 90-day percentile
        window = df["nvt"].tail(1460) if len(df) >= 1460 else df["nvt"]
        pct_rank = (window <= current).sum() / len(window) * 100

        # NVT: high = overvalued (bear), low = undervalued (bull)
        score = _score_threshold(pct_rank, 25, 90)  # inverted: high rank = bear

        return {
            "value": current,
            "score": score,
            "percentile_rank": pct_rank,
            "source": "coinmetrics",
            "note": "low NVT = network undervalued (bull); high NVT = overvalued (bear)",
        }
    except Exception:
        return None


# ============================================================
# 7. STOCK-TO-FLOW DEFLECTION — fair value gap
# ============================================================

def stock_to_flow_deflection() -> Optional[dict]:
    """S2F model deflection — how far price is from PlanB's S2F fair value.

    S2F = stock / flow. For BTC: total supply / annual new issuance.
    PlanB model: log(price) = 3.3 * log(S2F) + constant.

    Deflection > 0 = price above S2F fair value (bear/distribution).
    Deflection < 0 = price below S2F fair value (bull/accumulation).
    """
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=10)
        if df.empty: return None
        current_price = float(df["close"].iloc[-1])

        # Compute S2F: total supply ~19.7M, annual issuance after 2024 halving = ~164k
        total_supply = 19_700_000
        days_post_halving = (datetime.now(timezone.utc).date() -
                             datetime(2024, 4, 20).date()).days
        if days_post_halving >= 0:
            annual_issuance = 164_000   # 450/day * 365 = 164k (post-2024 halving)
        else:
            annual_issuance = 328_000
        s2f = total_supply / annual_issuance

        # PlanB formula: log10(price_USD) ≈ -1.84 + 3.36 * log10(s2f)
        log_fair = -1.84 + 3.36 * math.log10(s2f)
        fair_value = 10 ** log_fair
        deflection = current_price / fair_value - 1

        # 2026-06-01 DEPRECATED — backtest showed S2F deflection was BULLISH
        # at the 2025 peak (deflection -11%). PlanB's S2F model has degraded
        # since cycle 4. Reduce score magnitude by 50% so it doesn't dominate
        # the composite. Keep available as context but don't trust it.
        score = _score_threshold(deflection, -0.3, 0.5)
        if score is not None:
            score *= 0.5  # halve impact

        return {
            "value": deflection * 100,
            "score": score,
            "current_price": current_price,
            "s2f_fair_value": fair_value,
            "s2f_ratio": s2f,
            "deflection_pct": deflection * 100,
            "source": "computed (PlanB S2F — degraded model, weight halved)",
            "note": "PlanB S2F is degraded. Score halved. Provided for context only.",
        }
    except Exception:
        return None


# ============================================================
# 8. YIELD CURVE + REAL RATES — recession / liquidity regime
# ============================================================

def yield_curve_real_rates() -> Optional[dict]:
    """Yield curve slope + real fed funds rate.

    Inverted curve (10Y-2Y < 0) historically predicts recession 6-18m ahead.
    Real rate (FFR - CPI) > 2% = restrictive monetary policy (bear).
    Real rate < 0% = accommodative (bull).
    """
    try:
        def _fred(series_id):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2020-01-01"
            text = _http_text(url, timeout=20)
            if not text: return None
            rows = []
            for line in text.split("\n")[1:]:
                parts = line.strip().split(",")
                if len(parts) < 2: continue
                try:
                    rows.append({"date": pd.to_datetime(parts[0]).date(),
                                  "val": float(parts[1])})
                except Exception: continue
            return pd.DataFrame(rows)

        ten_y = _fred("DGS10")    # 10-year Treasury yield
        two_y = _fred("DGS2")     # 2-year Treasury yield
        ffr = _fred("FEDFUNDS")   # Fed funds rate
        cpi = _fred("CPIAUCSL")   # CPI all urban consumers

        if any(x is None for x in [ten_y, two_y, ffr, cpi]): return None

        latest_10y = float(ten_y["val"].iloc[-1])
        latest_2y = float(two_y["val"].iloc[-1])
        latest_ffr = float(ffr["val"].iloc[-1])

        # CPI YoY
        if len(cpi) >= 13:
            cpi_now = float(cpi["val"].iloc[-1])
            cpi_yr_ago = float(cpi["val"].iloc[-13])
            cpi_yoy = (cpi_now / cpi_yr_ago - 1) * 100
        else:
            cpi_yoy = 3.0

        curve_slope = latest_10y - latest_2y  # > 0 = normal, < 0 = inverted
        real_rate = latest_ffr - cpi_yoy

        # Combined score:
        # Normal curve + neg real rate = bull
        # Inverted curve + pos real rate = bear (recession risk)
        curve_score = _score_threshold(curve_slope, 1.0, -0.5)
        real_rate_score = _score_threshold(real_rate, -1.0, 2.5)  # inverted: low/neg = bull
        avg_score = (curve_score + real_rate_score) / 2 if curve_score and real_rate_score else None

        return {
            "value": curve_slope,
            "score": avg_score,
            "yield_10y": latest_10y,
            "yield_2y": latest_2y,
            "curve_slope_pct": curve_slope,
            "fed_funds_rate": latest_ffr,
            "cpi_yoy_pct": cpi_yoy,
            "real_rate_pct": real_rate,
            "source": "FRED",
            "note": "curve > 1% + real rate < 0 = liquidity bull; inverted curve + high real rate = recession risk",
        }
    except Exception:
        return None


# ============================================================
# AGGREGATOR
# ============================================================

def all_more_signals() -> dict:
    """Pull all 8 additional signals. Returns dict (None for failures)."""
    return {
        "mvrv_z_score":         mvrv_z_score(),
        "hayes_liquidity":      hayes_liquidity_trinity(),
        "cross_asset_ratios":   cross_asset_ratios(),
        "google_trends":        google_trends_btc(),
        "cme_basis":            cme_basis(),
        "nvt_signal":           nvt_signal(),
        "s2f_deflection":       stock_to_flow_deflection(),
        "yield_curve":          yield_curve_real_rates(),
    }


def main():
    print("\n" + "=" * 76)
    print("8 MORE FREE TOP-TIER BTC SIGNALS")
    print("=" * 76)
    sigs = all_more_signals()
    for name, d in sigs.items():
        if d is None:
            print(f"\n[{name.upper()}]  (unavailable)")
            continue
        print(f"\n[{name.upper()}]  value={d.get('value')}  score={d.get('score')}")
        for k, v in d.items():
            if k in ("value", "score", "source", "note"): continue
            print(f"  {k}: {v}")
        print(f"  source: {d.get('source')}")
        print(f"  note:   {d.get('note')}")


if __name__ == "__main__":
    main()
