"""Phillip Swift Watch — extended Swift framework + content monitoring.

Phillip Swift is the creator of LookIntoBitcoin.com. He's a UK-based BTC
cycle analyst with a strong track record:
  - Called Apr 2021 cycle 3 top via Pi Cycle Top (cross was Apr 13, peak Apr 14)
  - Called Dec 2018 cycle 2 bottom via his suite
  - Has been more cautious about cycle 5 (correctly predicting "muted cycle")

This module adds the remaining indicators from his framework not yet built:
  1. Bitcoin Risk Index    — his composite meta-indicator (0=buy, 1=sell)
  2. Thermocap Multiple    — total cumulative miner revenue vs market cap
  3. Profitable Days       — % of days an investor would be in profit
  4. 200WMA Heatmap        — price vs 200-week MA color zones

Plus content monitoring:
  - Twitter handle: @PositiveCrypto
  - LookIntoBitcoin.com cycle indicators page
  - Bitcoin Magazine Pro author page
  - Coin Bureau / Robin Seyr / Pomp Podcast appearance links

He runs his own newsletter and frequently appears on BTC-focused podcasts.
Worth monitoring his Twitter for short-form takes between podcasts.
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


def _cm(metric: str, days: int = 3650) -> Optional[pd.Series]:
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
# 1. BITCOIN RISK INDEX — Swift's composite meta-indicator
# ============================================================

def bitcoin_risk_index() -> dict:
    """Phillip Swift's Bitcoin Risk Index — 0 = max BUY, 1 = max SELL.

    Composite of his key indicators (his approximate weights):
      40% MVRV Z-Score
      15% Mayer Multiple
      15% NUPL proxy
      10% Pi Cycle progress
      10% Golden Ratio Multiplier rank
      10% Reserve Risk proxy

    Each component normalized to [0, 1] where 1 = bubble/euphoria.
    Output banding: 0.0-0.2 BUY, 0.2-0.4 ACCUMULATE, 0.4-0.6 NEUTRAL,
    0.6-0.8 RAISE CASH, 0.8-1.0 SELL.
    """
    df = _btc_history("8y")
    if df is None or len(df) < 730:
        return {"error": "insufficient data"}
    closes = df["Close"]
    price = _live_btc_price() or float(closes.iloc[-1])

    # 1. MVRV Z-Score (40% weight) — proxy via raw MVRV
    mvrv_raw = _cm("CapMVRVCur", days=1460)
    if mvrv_raw is not None and len(mvrv_raw) > 200:
        rmean = mvrv_raw.rolling(1460, min_periods=200).mean()
        rstd = mvrv_raw.rolling(1460, min_periods=200).std()
        mvrv_z = (mvrv_raw - rmean) / rstd
        z_now = float(mvrv_z.iloc[-1]) if not pd.isna(mvrv_z.iloc[-1]) else 0
        # Map z from [-1.5, 7] to [0, 1]
        mvrv_score = max(0, min(1, (z_now + 1.5) / 8.5))
    else:
        mvrv_score = 0.5  # neutral fallback

    # 2. Mayer Multiple (15%)
    ma_200 = closes.rolling(200).mean()
    mayer = float((closes / ma_200).iloc[-1]) if not pd.isna(ma_200.iloc[-1]) else 1.0
    # Map mayer from [0.5, 2.5] to [0, 1]
    mayer_score = max(0, min(1, (mayer - 0.5) / 2.0))

    # 3. NUPL proxy (15%) via 1-1/MVRV
    if mvrv_raw is not None and len(mvrv_raw) > 0:
        m = float(mvrv_raw.iloc[-1])
        nupl = 1 - 1/m if m > 0 else 0
        # Map nupl from [-0.25, 0.75] to [0, 1]
        nupl_score = max(0, min(1, (nupl + 0.25) / 1.0))
    else:
        nupl_score = 0.5

    # 4. Pi Cycle ratio (10%)
    ma_111 = closes.rolling(111).mean()
    ma_350x2 = closes.rolling(350).mean() * 2
    pi_ratio = float((ma_111 / ma_350x2).iloc[-1]) if not pd.isna(ma_350x2.iloc[-1]) else 0.5
    pi_score = max(0, min(1, pi_ratio))  # already in 0-1 range mostly

    # 5. Golden Ratio Multiplier rank (10%)
    ma_350 = closes.rolling(350).mean()
    gr_mult = closes / ma_350
    gr_now = float(gr_mult.iloc[-1]) if not pd.isna(gr_mult.iloc[-1]) else 1.0
    # Map gr_mult from [0.5, 13] to [0, 1]
    gr_score = max(0, min(1, (gr_now - 0.5) / 12.5))

    # 6. Reserve Risk proxy (10%) — use MVRV-Z as approximation
    rr_score = mvrv_score  # tightly correlated proxy

    # Weighted composite
    risk_index = (
        0.40 * mvrv_score +
        0.15 * mayer_score +
        0.15 * nupl_score +
        0.10 * pi_score +
        0.10 * gr_score +
        0.10 * rr_score
    )

    if risk_index < 0.2:    zone, emoji, action = "MAX BUY",       "🟢", "Deploy aggressively"
    elif risk_index < 0.4:  zone, emoji, action = "ACCUMULATE",     "🟢", "DCA in"
    elif risk_index < 0.6:  zone, emoji, action = "NEUTRAL",         "🟡", "Hold"
    elif risk_index < 0.8:  zone, emoji, action = "RAISE CASH",     "🟠", "Trim 25-50%"
    else:                    zone, emoji, action = "MAX SELL",         "🔴", "Exit 75%+"

    return {
        "risk_index":      risk_index,
        "zone":            zone,
        "emoji":           emoji,
        "action":          action,
        "components": {
            "mvrv_score":   mvrv_score,
            "mayer_score":  mayer_score,
            "nupl_score":   nupl_score,
            "pi_score":     pi_score,
            "gr_score":     gr_score,
            "rr_score":     rr_score,
        },
        "interpretation":  f"Bitcoin Risk Index {risk_index:.2f} — {zone}: {action}",
    }


# ============================================================
# 2. THERMOCAP MULTIPLE — cumulative miner revenue vs market cap
# ============================================================

def thermocap_multiple() -> dict:
    """Thermocap = total cumulative miner revenue (since genesis).
    Thermocap Multiple = Market Cap / Thermocap.

    > 16 = TOP zone (historically). Cycle peaks: ~32 in 2013, ~16 in 2017,
    ~10 in 2021 (each cycle compresses).
    """
    rev = _cm("RevUSD", days=4000)
    cap = _cm("CapMrktCurUSD", days=4000)
    if rev is None or cap is None:
        return {"error": "data unavailable"}
    df = pd.concat([rev, cap], axis=1).dropna()
    if df.empty: return {"error": "alignment failed"}
    df.columns = ["rev", "cap"]
    thermocap = df["rev"].cumsum()
    multiplier = df["cap"] / thermocap
    cur = float(multiplier.iloc[-1])

    if cur > 16:       zone, emoji = "TOP ZONE",          "🔴"
    elif cur > 10:     zone, emoji = "Elevated",            "🟠"
    elif cur > 5:      zone, emoji = "Bull market",         "🟡"
    elif cur > 2:      zone, emoji = "Recovery",            "🟢"
    else:               zone, emoji = "BOTTOM ZONE",         "🟢"

    return {
        "multiplier":     cur,
        "zone":           zone,
        "emoji":          emoji,
        "interpretation": f"Thermocap Multiple {cur:.1f}× — {zone} (>16 = top, <2 = bottom)",
    }


# ============================================================
# 3. PROFITABLE DAYS — Swift's "% days BTC investor in profit"
# ============================================================

def profitable_days() -> dict:
    """% of days BTC was lower than current price (= % of investors in profit).

    Cycle bottoms: < 70% (lots of underwater holders)
    Cycle tops: > 99% (almost everyone profitable)
    """
    df = _btc_history("max")
    if df is None or len(df) < 100:
        return {"error": "insufficient data"}
    closes = df["Close"]
    price = _live_btc_price() or float(closes.iloc[-1])
    profitable_pct = float((closes < price).mean() * 100)

    if profitable_pct < 70:    zone, emoji = "Bottom region",   "🟢"
    elif profitable_pct < 85:  zone, emoji = "Mid-cycle",        "🟡"
    elif profitable_pct < 95:  zone, emoji = "Late cycle",       "🟠"
    elif profitable_pct < 99:  zone, emoji = "Top zone",         "🔴"
    else:                       zone, emoji = "EXTREME TOP",      "🔴"

    return {
        "profitable_pct": profitable_pct,
        "zone":           zone,
        "emoji":          emoji,
        "interpretation": (f"{profitable_pct:.1f}% of all days were below current price = "
                            f"% investors in profit — {zone}"),
    }


# ============================================================
# 4. 200-WEEK MA HEATMAP color zone
# ============================================================

def two_hundred_week_ma_heatmap() -> dict:
    """200-week MA heatmap — color based on % above/below 200wMA.

    Bitcoin has NEVER closed a week below 200wMA at any cycle bottom and
    recovered — sub-200wMA = generational opportunity.
    """
    df = _btc_history("8y")
    if df is None or len(df) < 1400:
        return {"error": "insufficient data"}
    # 200 weeks ≈ 1400 days
    closes = df["Close"]
    ma_200w = closes.rolling(1400).mean()
    price = _live_btc_price() or float(closes.iloc[-1])
    ma_now = float(ma_200w.iloc[-1])
    if pd.isna(ma_now) or ma_now == 0: return {"error": "MA NaN"}
    pct_vs_ma = (price / ma_now - 1) * 100

    if pct_vs_ma < 0:        zone, emoji = "BELOW 200wMA (GEN BOTTOM)", "🟢"
    elif pct_vs_ma < 50:     zone, emoji = "Accumulation",                "🟢"
    elif pct_vs_ma < 100:    zone, emoji = "Fair value",                  "🟡"
    elif pct_vs_ma < 200:    zone, emoji = "Cyclical bull",               "🟡"
    elif pct_vs_ma < 400:    zone, emoji = "Late cycle",                  "🟠"
    else:                     zone, emoji = "TOP ZONE",                   "🔴"

    return {
        "price":          price,
        "ma_200w":        ma_now,
        "pct_vs_ma":      pct_vs_ma,
        "zone":           zone,
        "emoji":          emoji,
        "interpretation": (f"BTC ${price:,.0f} vs 200wMA ${ma_now:,.0f}  "
                            f"({pct_vs_ma:+.1f}%) — {zone}"),
    }


# ============================================================
# CONTENT MONITORING — Swift's online presence
# ============================================================

SWIFT_CONTENT = {
    "twitter": {
        "handle": "@PositiveCrypto",
        "url":    "https://twitter.com/PositiveCrypto",
        "embed_url": "https://twitter.com/PositiveCrypto",
        "description": "Phillip Swift's primary Twitter account — short-form takes between podcasts",
    },
    "lookintobitcoin": {
        "name": "LookIntoBitcoin.com — Cycle Indicators",
        "url":   "https://www.lookintobitcoin.com/cycle-indicators",
        "description": "His master dashboard of all cycle indicators",
    },
    "bitcoin_magazine_pro": {
        "name": "Bitcoin Magazine Pro — Phillip Swift author page",
        "url":   "https://www.bitcoinmagazinepro.com/profile/phillipswift/",
        "description": "His articles + paid research access",
    },
    "newsletter": {
        "name": "LookIntoBitcoin Newsletter",
        "url":   "https://www.lookintobitcoin.com/newsletter",
        "description": "Weekly cycle update — sign up for email",
    },
    "youtube_channels": [
        ("Coin Bureau",            "https://www.youtube.com/@CoinBureau",
            "Guy interviews Swift regularly during major cycle events"),
        ("Robin Seyr",             "https://www.youtube.com/@RobinSeyr",
            "Frequent Phillip Swift interviews"),
        ("Pomp Podcast",           "https://www.youtube.com/@AnthonyPompliano",
            "Anthony Pompliano interviews on cycle calls"),
        ("Bitcoin Magazine",       "https://www.youtube.com/@BitcoinMagazine",
            "Phillip writes for them; appears regularly"),
        ("The Wolf Of All Streets","https://www.youtube.com/@TheWolfOfAllStreets",
            "Frequent Swift appearances"),
    ],
}


# ============================================================
# Aggregator
# ============================================================

def all_swift_watch() -> dict:
    return {
        "asof":              datetime.now(timezone.utc).isoformat(),
        "risk_index":        bitcoin_risk_index(),
        "thermocap":         thermocap_multiple(),
        "profitable_days":   profitable_days(),
        "two_hundred_wma":   two_hundred_week_ma_heatmap(),
        "content":           SWIFT_CONTENT,
    }


def main():
    r = all_swift_watch()
    print("=" * 70)
    print("PHILLIP SWIFT WATCH")
    print("=" * 70)
    for key in ["risk_index", "thermocap", "profitable_days", "two_hundred_wma"]:
        info = r[key]
        if info.get("error"):
            print(f"\n  {key}: {info['error']}")
            continue
        emoji = info.get("emoji", "")
        interp = info.get("interpretation", "")
        try: print(f"\n  {emoji} {key.upper():25s} {interp[:80]}")
        except UnicodeEncodeError:
            print(f"\n     {key.upper():25s} {interp.encode('ascii','replace').decode()[:80]}")

    print(f"\n\nCONTENT TO MONITOR:")
    c = r["content"]
    print(f"  Twitter:    {c['twitter']['handle']} -> {c['twitter']['url']}")
    print(f"  Master:     {c['lookintobitcoin']['url']}")
    print(f"  BM Pro:     {c['bitcoin_magazine_pro']['url']}")
    print(f"  Newsletter: {c['newsletter']['url']}")
    print(f"  Podcast/YT channels with frequent Swift interviews:")
    for name, url, descr in c["youtube_channels"]:
        print(f"    - {name:30s}  {url}")


if __name__ == "__main__":
    main()
