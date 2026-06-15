"""Phillip Swift / LookIntoBitcoin indicator suite.

Implements the signature visual indicators Swift popularized:
  1. Golden Ratio Multiplier  — 350d MA × N multipliers (1.6/2/3/5/8/13/21)
  2. 2-Year MA Multiplier     — price/2y MA with 5x top band
  3. Logarithmic Regression   — Rainbow chart bands
  4. Bottom Cap / Top Cap     — Realized × 0.2 / Average × 35
  5. HODL Waves proxy         — supply distribution by age (free-tier approximation)
  6. LTH-NUPL zone            — Capitulation/Hope-Fear/Optimism/Belief/Euphoria

Each returns the value + cycle zone label + visual context.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _btc_history(period: str = "max") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker("BTC-USD").history(period=period)
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def _cm(metric: str, days: int = 1460) -> Optional[pd.Series]:
    try:
        from core.btc_pro_signals import _cm as _coinmetrics
        df = _coinmetrics(metric, days=days)
        if df is None or df.empty: return None
        return df.iloc[:, 0]
    except Exception:
        return None


def _live_btc_price() -> float:
    try:
        from core import data
        return data.btc_spot()  # region-resilient (Kraken/Coinbase/Binance/Bitstamp)
    except Exception:
        return 0.0


# ============================================================
# 1. GOLDEN RATIO MULTIPLIER — Swift's signature work
# ============================================================

GR_MULTIPLIERS = [
    (1.6, "Acceleration",       "🟢"),
    (2.0, "Resistance",          "🟢"),
    (3.0, "Bull Market",         "🟡"),
    (5.0, "Late Bull",           "🟡"),
    (8.0, "Top Cap",             "🟠"),
    (13.0, "Major Top",          "🔴"),
    (21.0, "Maximum Bubble",     "🔴"),
]


def golden_ratio_multiplier() -> dict:
    """Price vs 350d MA × Fibonacci multipliers.

    Swift's most important visual: shows where in the bull cycle you are.
    Bull market tops historically clustered at 21× 350d MA.
    Bear bottoms at < 1.0× 350d MA.
    """
    df = _btc_history("4y")
    if df is None or len(df) < 350:
        return {"error": "insufficient data"}
    ma_350 = float(df["Close"].rolling(350).mean().iloc[-1])
    if pd.isna(ma_350):
        return {"error": "350d MA NaN"}
    price = _live_btc_price() or float(df["Close"].iloc[-1])
    multiplier = price / ma_350

    # Find current zone
    current_zone = "Sub-MA (bottom zone)"
    current_emoji = "🟢"
    next_band_mult = 1.6
    for mult, label, emoji in GR_MULTIPLIERS:
        if multiplier < mult:
            next_band_mult = mult
            break
        current_zone = label
        current_emoji = emoji

    bands = []
    for mult, label, emoji in GR_MULTIPLIERS:
        bands.append({
            "multiplier": mult,
            "label": label,
            "price": ma_350 * mult,
            "current": multiplier >= mult and multiplier < (GR_MULTIPLIERS[
                GR_MULTIPLIERS.index((mult, label, emoji)) + 1
            ][0] if GR_MULTIPLIERS.index((mult, label, emoji)) < len(GR_MULTIPLIERS) - 1 else mult * 10),
        })

    return {
        "price":           price,
        "ma_350":          ma_350,
        "multiplier":      multiplier,
        "current_zone":    current_zone,
        "current_emoji":   current_emoji,
        "next_band":       {"multiplier": next_band_mult, "price": ma_350 * next_band_mult},
        "bands":           bands,
        "interpretation":  ("Below 350d MA — capitulation zone" if multiplier < 1
                              else f"In '{current_zone}' zone — {multiplier:.2f}× 350d MA"),
    }


# ============================================================
# 2. 2-YEAR MA MULTIPLIER — Swift's other classic
# ============================================================

def two_year_ma_multiplier() -> dict:
    """Price vs 2y MA — visualizes 5× band for top.

    Historical: cycle tops cluster at 5× the 2y MA. Bottoms below 1×.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 730:
        return {"error": "insufficient data"}
    ma_2y = float(df["Close"].rolling(730).mean().iloc[-1])
    if pd.isna(ma_2y):
        return {"error": "2y MA NaN"}
    price = _live_btc_price() or float(df["Close"].iloc[-1])
    multiplier = price / ma_2y

    # Bands
    bands_2y = [
        (0.6, "Deep Capitulation", "🟢"),
        (1.0, "Below 2y MA",       "🟢"),
        (1.5, "Recovery",           "🟡"),
        (2.5, "Mid-Cycle",          "🟡"),
        (3.5, "Late Cycle",         "🟠"),
        (5.0, "TOP ZONE",            "🔴"),
    ]
    current_zone = "Sub-bottom"
    current_emoji = "🟢"
    for mult, label, emoji in bands_2y:
        if multiplier >= mult:
            current_zone = label
            current_emoji = emoji

    return {
        "price":          price,
        "ma_2y":          ma_2y,
        "multiplier":     multiplier,
        "current_zone":   current_zone,
        "current_emoji":  current_emoji,
        "top_band_price": ma_2y * 5,
        "bottom_band":    ma_2y * 0.6,
        "interpretation": (f"{multiplier:.2f}× 2y MA — {current_zone}"
                            f" (top zone at 5× = ${ma_2y*5:,.0f})"),
    }


# ============================================================
# 3. LOGARITHMIC REGRESSION / RAINBOW CHART
# ============================================================

def logarithmic_regression() -> dict:
    """BTC price vs log regression model — Swift's Rainbow chart math.

    Model: log10(price) = a + b*log10(days_since_genesis)
    Approximates with empirical constants. Returns zone label.
    """
    df = _btc_history("10y")
    if df is None or len(df) < 100:
        return {"error": "insufficient data"}

    # Genesis: Jan 3 2009
    GENESIS = pd.Timestamp("2009-01-03")
    days_since = (datetime.now() - GENESIS).days
    log_days = np.log10(days_since)

    # Empirical regression (LookIntoBitcoin coefficients approx)
    # log10(price) = 5.84 * log10(days) - 17.01 (rough fit, updates slowly)
    log_price_model = 5.84 * log_days - 17.01
    model_price = 10 ** log_price_model

    price = _live_btc_price() or float(df["Close"].iloc[-1])
    deviation = (price / model_price - 1) * 100

    # Rainbow zones
    if deviation < -50:    zone, emoji = "Fire Sale (BUY)", "🔥"
    elif deviation < -25:   zone, emoji = "BUY!",            "🟢"
    elif deviation < 0:     zone, emoji = "Accumulate",      "🟢"
    elif deviation < 50:    zone, emoji = "Cheap",            "🟢"
    elif deviation < 100:   zone, emoji = "Fair Value",       "🟡"
    elif deviation < 200:   zone, emoji = "Still Cheap",      "🟡"
    elif deviation < 300:   zone, emoji = "Resistance",       "🟠"
    elif deviation < 500:   zone, emoji = "FOMO",              "🔴"
    else:                    zone, emoji = "MAXIMUM BUBBLE",   "🔴"

    return {
        "price":          price,
        "model_price":    float(model_price),
        "deviation_pct":  float(deviation),
        "zone":           zone,
        "emoji":          emoji,
        "interpretation": (f"BTC ${price:,.0f} vs model ${model_price:,.0f}  "
                            f"{deviation:+.0f}% — {zone}"),
    }


# ============================================================
# 4. BOTTOM CAP / TOP CAP MODELS
# ============================================================

def cap_models() -> dict:
    """BTC price models based on Realized Cap.

    Bottom Cap = Realized Cap × 0.2
    Top Cap    = Average Cap × 35
    """
    rc = _cm("CapRealUSD", days=400)
    mc = _cm("CapMrktCurUSD", days=400)
    if rc is None or mc is None:
        return {"error": "data unavailable"}
    realized = float(rc.iloc[-1])
    market = float(mc.iloc[-1])
    # Average Cap = (Realized + Market) / 2 approx
    avg_cap = (realized + market) / 2
    # Supply approximation
    SUPPLY = 19_700_000  # ~current circulating BTC
    bottom_cap_price = (realized * 0.2) / SUPPLY
    top_cap_price = (avg_cap * 35) / SUPPLY
    price = _live_btc_price()

    return {
        "price":            price,
        "bottom_cap":       bottom_cap_price,
        "top_cap":          top_cap_price,
        "pct_to_bottom":    ((price / bottom_cap_price) - 1) * 100 if bottom_cap_price > 0 else 0,
        "pct_to_top":       ((price / top_cap_price) - 1) * 100 if top_cap_price > 0 else 0,
        "interpretation":   (f"Bottom Cap ${bottom_cap_price:,.0f}  Top Cap ${top_cap_price:,.0f}"),
    }


# ============================================================
# 5. HODL WAVES proxy
# ============================================================

def hodl_waves_proxy() -> dict:
    """Free-tier proxy for HODL Waves using realized cap velocity.

    True HODL Waves require paid CoinMetrics (SplyAct1yr etc).
    Proxy: realized cap velocity = how slowly coins are moving.
    Low velocity = LTHs holding = bullish cycle structure.
    """
    rc = _cm("CapRealUSD", days=400)
    mc = _cm("CapMrktCurUSD", days=400)
    if rc is None or mc is None:
        return {"error": "data unavailable"}
    df = pd.concat([rc, mc], axis=1).dropna()
    df.columns = ["realized", "market"]
    # MVRV ratio
    mvrv = df["market"] / df["realized"]
    # Velocity proxy: 30d change in realized cap / realized cap
    velocity = df["realized"].pct_change(30) * 100
    velocity_now = float(velocity.iloc[-1]) if not pd.isna(velocity.iloc[-1]) else 0
    mvrv_now = float(mvrv.iloc[-1])

    # Lower velocity = more dormant supply = LTH dominance
    if velocity_now < 2:    pct_lth, label = 75, "LTH DOMINANT (bullish structure)"
    elif velocity_now < 5:  pct_lth, label = 65, "LTH-leaning"
    elif velocity_now < 8:  pct_lth, label = 55, "Mixed"
    elif velocity_now < 12: pct_lth, label = 45, "STH-leaning"
    else:                     pct_lth, label = 35, "STH/Speculation (top region)"

    return {
        "velocity_30d":  velocity_now,
        "lth_pct_est":   pct_lth,
        "sth_pct_est":   100 - pct_lth,
        "mvrv":          mvrv_now,
        "label":         label,
        "interpretation": f"30d realized cap velocity {velocity_now:+.1f}% — {label} (est ~{pct_lth}% LTH supply)",
    }


# ============================================================
# 6. LTH-NUPL zone bands
# ============================================================

LTH_NUPL_ZONES = [
    (-0.25, "Capitulation",    "🔴", "DEEP VALUE — bottoms historically here"),
    (0.0,    "Hope-Fear",        "🟠", "Recovery begins"),
    (0.25,   "Optimism",         "🟡", "Early bull"),
    (0.50,   "Belief",            "🟢", "Mid-bull"),
    (0.75,   "Euphoria",          "🔴", "TOP ZONE — distribution"),
]


def lth_nupl_zone() -> dict:
    """LTH-NUPL proxy using MVRV with bias toward older coins.

    NUPL = 1 - 1/MVRV (proxy). LTH-NUPL = NUPL with smoothing for older supply.
    """
    s = _cm("CapMVRVCur", days=400)
    if s is None or s.empty:
        return {"error": "data unavailable"}
    s = s.dropna()
    mvrv_now = float(s.iloc[-1])
    nupl_proxy = 1 - 1 / mvrv_now if mvrv_now > 0 else -1
    # LTH-NUPL slightly smoother than NUPL
    lth_nupl = nupl_proxy * 0.9

    zone = "Sub-capitulation"
    emoji = "🔴"
    interp = "Below all bands — deep value"
    for threshold, label, em, descr in LTH_NUPL_ZONES:
        if lth_nupl >= threshold:
            zone = label
            emoji = em
            interp = descr
    return {
        "lth_nupl":       lth_nupl,
        "nupl":            nupl_proxy,
        "mvrv":            mvrv_now,
        "zone":            zone,
        "emoji":           emoji,
        "interpretation":  f"LTH-NUPL {lth_nupl:.2f} — {zone}: {interp}",
    }


# ============================================================
# Aggregator
# ============================================================

def all_swift_indicators() -> dict:
    """Run all 6 LookIntoBitcoin-style indicators."""
    return {
        "asof":                    datetime.now(timezone.utc).isoformat(),
        "golden_ratio_multiplier": golden_ratio_multiplier(),
        "two_year_ma_multiplier":  two_year_ma_multiplier(),
        "log_regression":          logarithmic_regression(),
        "cap_models":              cap_models(),
        "hodl_waves":              hodl_waves_proxy(),
        "lth_nupl":                lth_nupl_zone(),
    }


def main():
    r = all_swift_indicators()
    print("=" * 70)
    print("SWIFT / LOOKINTOBITCOIN INDICATOR SUITE")
    print("=" * 70)
    for key, info in r.items():
        if key == "asof": continue
        if info.get("error"):
            print(f"\n  {key.upper()}: {info['error']}")
            continue
        emoji = info.get("emoji") or info.get("current_emoji", "")
        interp = info.get("interpretation", "")
        try: print(f"\n  {emoji} {key.upper():25s} {interp[:80]}")
        except UnicodeEncodeError:
            print(f"\n     {key.upper():25s} {interp.encode('ascii','replace').decode()[:80]}")


if __name__ == "__main__":
    main()
