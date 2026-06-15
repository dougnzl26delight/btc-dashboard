"""Jesse Olson's equity-side technical layer — QQQ.

Mirrors btc_jesse_olson.py but for the Nasdaq-100 ETF. Key signals:
  - Distance to Olson's 589 'gap' level (pivotal trapdoor)
  - Distance to 200-day SMA (~620, near-term support)
  - Distance to 200-week SMA (~454, generational support)
  - 3-week MACD on QQQ (Olson's primary timing)
  - Weekly Heikin Ashi color streak (trend confirmation)
  - Weekly RSI divergence (momentum exhaustion)

Tiered verdicts:
  🟢 SAFE: above 589, above 200dMA, MACD bullish
  🟡 CAUTION: weakening (MACD bear OR below 200dMA)
  🟡 WATCH: within 5% of 589
  🔴 GAP_BROKEN: below 589 — 200wMA retest in play
  🔴 RETESTING: below 200wMA — generational move underway

USE: feed into EQUITY TOP WATCH on dashboard. NZ user holds 70% equity, so
a 200wMA test = -36% drawdown on ~NZ$91k = ~NZ$32k hit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Olson's pivotal trapdoor level for QQQ — "gap below 589 → 200wMA test in play"
OLSON_GAP_LEVEL = 589

# Olson's mapped downside gaps / sniper targets for QQQ (from his June 2026 posts).
# Price tends to fill these in sequence on the way down. Used to show "how far
# through Olson's roadmap" the breakdown is. Update as he posts new levels.
OLSON_QQQ_GAPS = [672.78, 589.0]   # next gap, then the 200wMA trapdoor


def _yahoo_direct_qqq(period_y: int = 6):
    """Fallback QQQ daily history via Yahoo's chart JSON API directly.

    Different code path than yfinance's scraping layer — when yfinance returns
    empty/NaN frames this raw endpoint usually still works. No API key.
    (Stooq was tried as fallback but added a JS bot-wall in 2026.)
    """
    try:
        import json as _json
        import urllib.request
        import pandas as pd
        rng = f"{min(period_y, 10)}y"
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/QQQ"
               f"?interval=1d&range={rng}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = _json.loads(r.read().decode("utf-8", errors="ignore"))
        result = payload.get("chart", {}).get("result", [None])[0]
        if not result: return None
        ts = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quote.get("close"): return None
        df = pd.DataFrame({
            "Open":   quote.get("open", []),
            "High":   quote.get("high", []),
            "Low":    quote.get("low", []),
            "Close":  quote.get("close", []),
        }, index=pd.to_datetime(ts, unit="s"))
        df = df.dropna(subset=["Close"]).sort_index()
        return df if not df.empty else None
    except Exception:
        return None


def _qqq_history(period_y: int = 6):
    """Pull QQQ daily history — yfinance primary, Stooq fallback.

    The equity-priority rotation trigger depends on this feed being fresh;
    yfinance intermittently returns empty frames or NaN tails, so a dead or
    NaN-tailed primary fails over to Stooq.
    """
    h = None
    try:
        import yfinance as yf
        qqq = yf.Ticker("QQQ")
        h = qqq.history(period=f"{period_y}y", interval="1d")
        if h is not None and not h.empty:
            # If the last 3 sessions are ALL NaN, treat the feed as broken
            tail = h["Close"].tail(3)
            if tail.dropna().empty:
                h = None
    except Exception:
        h = None

    if h is None or h.empty:
        h = _yahoo_direct_qqq(period_y)
    return h if (h is not None and not h.empty) else None


def qqq_levels() -> dict:
    """Distance from QQQ to 200wMA, 200dMA, and Olson's 589 gap."""
    h = _qqq_history(6)
    if h is None or h.empty:
        return {"error": "no data"}

    import math
    import pandas as pd

    # yfinance sometimes returns NaN for the most recent (partial) session.
    # Use the last VALID close — a NaN here previously cascaded into a false
    # RETESTING tier (nan > wma200 == False) and a false rotation signal.
    closes_valid = h["Close"].dropna()
    if closes_valid.empty:
        return {"error": "no valid closes"}
    last_close = float(closes_valid.iloc[-1])
    last_date = closes_valid.index[-1]
    if not math.isfinite(last_close) or last_close <= 0:
        return {"error": "invalid last close"}

    # 200-day SMA
    dma200 = float(closes_valid.tail(200).mean()) if len(closes_valid) >= 200 else None

    # 200-week SMA — resample to weekly then 200-week mean
    wkly = h["Close"].resample("W").last().dropna()
    wma200 = float(wkly.tail(200).mean()) if len(wkly) >= 200 else None

    def _pct(target):
        if target is None: return None
        return (target / last_close - 1) * 100

    return {
        "last_close":      round(last_close, 2),
        "last_date":       str(last_date.date()),
        "wma200":          round(wma200, 2) if wma200 else None,
        "dma200":          round(dma200, 2) if dma200 else None,
        "gap_level":       OLSON_GAP_LEVEL,
        "pct_to_wma200":   round(_pct(wma200), 2) if wma200 else None,
        "pct_to_dma200":   round(_pct(dma200), 2) if dma200 else None,
        "pct_to_gap":      round(_pct(OLSON_GAP_LEVEL), 2),
        "above_gap":       last_close > OLSON_GAP_LEVEL,
        "above_dma200":    last_close > dma200 if dma200 else None,
        "above_wma200":    last_close > wma200 if wma200 else None,
    }


def three_week_macd_qqq() -> dict:
    """3-week MACD on QQQ — Olson's primary timing signal."""
    h = _qqq_history(3)
    if h is None or h.empty:
        return {"error": "no data"}

    import pandas as pd

    # Resample to 3-week bars
    closes = h["Close"].resample("3W").last().dropna()
    if len(closes) < 30:
        return {"error": "insufficient data"}

    # MACD: 12 EMA - 26 EMA, signal = 9 EMA of MACD
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    macd_now = float(macd.iloc[-1])
    signal_now = float(signal.iloc[-1])
    hist_now = float(histogram.iloc[-1])
    hist_prev = float(histogram.iloc[-2]) if len(histogram) >= 2 else 0

    cross_up = (hist_now > 0 and hist_prev <= 0)
    cross_down = (hist_now < 0 and hist_prev >= 0)
    bullish = macd_now > signal_now

    return {
        "macd":            round(macd_now, 2),
        "signal":          round(signal_now, 2),
        "histogram":       round(hist_now, 2),
        "histogram_prev":  round(hist_prev, 2),
        "cross_up":        cross_up,
        "cross_down":      cross_down,
        "bullish":         bullish,
        "trend":           "BULLISH" if bullish else "BEARISH",
    }


def weekly_heikin_ashi_qqq() -> dict:
    """QQQ weekly Heikin Ashi — Olson uses for trend confirmation."""
    h = _qqq_history(3)
    if h is None or h.empty:
        return {"error": "no data"}

    import pandas as pd

    wkly = h[["Open", "High", "Low", "Close"]].resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    if len(wkly) < 10:
        return {"error": "insufficient data"}

    # Heikin Ashi
    ha_close = (wkly["Open"] + wkly["High"] + wkly["Low"] + wkly["Close"]) / 4
    ha_open = [float(wkly.iloc[0]["Open"])]
    for i in range(1, len(wkly)):
        ha_open.append((ha_open[-1] + float(ha_close.iloc[i - 1])) / 2)
    ha_open = pd.Series(ha_open, index=wkly.index)

    # Color: green if ha_close > ha_open
    colors = (ha_close > ha_open).astype(int)

    # Current streak
    current_color = int(colors.iloc[-1])
    streak = 1
    for i in range(len(colors) - 2, -1, -1):
        if int(colors.iloc[i]) == current_color:
            streak += 1
        else:
            break

    return {
        "current_color":   "GREEN" if current_color else "RED",
        "streak_weeks":    streak,
        "recent_colors":   ["G" if c else "R" for c in colors.iloc[-10:].tolist()],
        "bullish":         bool(current_color),
        "ha_close_now":    round(float(ha_close.iloc[-1]), 2),
        "ha_open_now":     round(float(ha_open.iloc[-1]), 2),
    }


def weekly_rsi_divergence_qqq() -> dict:
    """QQQ weekly RSI(14) — momentum exhaustion + divergence detection."""
    h = _qqq_history(3)
    if h is None or h.empty:
        return {"error": "no data"}

    import pandas as pd

    wkly = h["Close"].resample("W").last().dropna()
    if len(wkly) < 30:
        return {"error": "insufficient data"}

    # RSI 14
    delta = wkly.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_now = float(rsi.iloc[-1])

    # Look-back window for divergence
    look = 26
    last_n_price = wkly.iloc[-look:]
    last_n_rsi = rsi.iloc[-look:]

    if last_n_price.empty or last_n_rsi.empty:
        return {"error": "insufficient lookback"}

    price_high_idx = last_n_price.idxmax()
    rsi_high_idx = last_n_rsi.idxmax()
    price_low_idx = last_n_price.idxmin()
    rsi_low_idx = last_n_rsi.idxmin()

    # Bearish: price made HH but RSI made LH (price peak AFTER RSI peak)
    bearish_div = price_high_idx > rsi_high_idx + pd.Timedelta(weeks=4)
    # Bullish: price made LL but RSI made HL (price low AFTER RSI low)
    bullish_div = price_low_idx > rsi_low_idx + pd.Timedelta(weeks=4)

    return {
        "rsi":                  round(rsi_now, 1),
        "rsi_overbought":       rsi_now > 70,
        "rsi_oversold":         rsi_now < 30,
        "bearish_divergence":   bool(bearish_div),
        "bullish_divergence":   bool(bullish_div),
        "trend":                ("OVERBOUGHT" if rsi_now > 70
                                  else "OVERSOLD" if rsi_now < 30
                                  else "NEUTRAL"),
    }


def all_qqq_olson_signals() -> dict:
    """Compute every QQQ Olson signal in one call (for precompute caching)."""
    return {
        "levels":         qqq_levels(),
        "macd_3w":         three_week_macd_qqq(),
        "heikin_ashi":     weekly_heikin_ashi_qqq(),
        "rsi_divergence":  weekly_rsi_divergence_qqq(),
    }


def qqq_olson_verdict() -> dict:
    """Composite QQQ Olson verdict — tiered alert.

    Tiers:
      🟢 SAFE         : above 589, above 200dMA, MACD bullish
      🟡 CAUTION      : tech weakening
      🟡 WATCH        : within 5% of 589 trapdoor
      🔴 GAP_BROKEN   : below 589 — 200wMA test in play
      🔴 RETESTING    : below 200wMA — generational move underway
    """
    sigs = all_qqq_olson_signals()
    lvl = sigs.get("levels", {}) or {}
    macd = sigs.get("macd_3w", {}) or {}
    ha = sigs.get("heikin_ashi", {}) or {}
    rsi = sigs.get("rsi_divergence", {}) or {}

    # Data-integrity gate: a failed/NaN price feed must NEVER produce a severe
    # tier. Return DATA_GAP (neutral) so downstream triggers treat it as no-signal.
    if lvl.get("error") or not lvl.get("last_close"):
        return {
            "tier":          "DATA_GAP",
            "tier_emoji":     "⚪",
            "color":          "#888",
            "action":         ("QQQ price feed unavailable (yfinance NaN/empty). "
                                "No tier assigned — treat as no-signal, retry next cycle."),
            "last_close":     None,
            "wma200":         lvl.get("wma200"),
            "dma200":         lvl.get("dma200"),
            "gap_level":      OLSON_GAP_LEVEL,
            "pct_to_gap":     None,
            "pct_to_dma200":  None,
            "pct_to_wma200":  None,
            "summary":        "QQQ data unavailable — no verdict",
            "signals":        sigs,
        }

    last_close = lvl.get("last_close", 0) or 0
    above_gap = bool(lvl.get("above_gap", True))
    above_dma200 = bool(lvl.get("above_dma200", True))
    above_wma200 = bool(lvl.get("above_wma200", True))
    macd_bullish = bool(macd.get("bullish", True))
    pct_to_gap = lvl.get("pct_to_gap", 0) or 0
    pct_to_wma200 = lvl.get("pct_to_wma200", 0) or 0
    pct_to_dma200 = lvl.get("pct_to_dma200", 0) or 0

    # Tier logic — most severe first
    if not above_wma200:
        tier = "RETESTING"
        color = "#ef4444"
        emoji = "🔴"
        action = ("QQQ below 200-week SMA. Generational support test underway. "
                  "Equity DEFENSIVE positioning warranted.")
    elif not above_gap:
        tier = "GAP_BROKEN"
        color = "#ef4444"
        emoji = "🔴"
        action = (f"QQQ broke below Olson's 589 gap. "
                  f"Next genuine support: 200wMA at ${lvl.get('wma200', 0):,.0f} "
                  f"({pct_to_wma200:+.1f}%). SCALE OUT 25-50% of equities.")
    elif abs(pct_to_gap) < 5:
        tier = "WATCH"
        color = "#f0b90b"
        emoji = "🟡"
        action = (f"Within 5% of 589 trapdoor (currently {pct_to_gap:+.1f}%). "
                  f"Set scaled-exit triggers; watch for daily close below 589.")
    elif not macd_bullish or not above_dma200:
        tier = "CAUTION"
        color = "#f0b90b"
        emoji = "🟡"
        reasons = []
        if not macd_bullish: reasons.append("3w MACD bearish")
        if not above_dma200: reasons.append(f"below 200dMA ({pct_to_dma200:+.1f}%)")
        action = f"Tech weakening: {', '.join(reasons)}. Monitor for 589 break."
    else:
        tier = "SAFE"
        color = "#22c55e"
        emoji = "🟢"
        action = (f"All clear. QQQ ${last_close:,.0f}, "
                  f"{abs(pct_to_gap):.1f}% above 589 trapdoor, "
                  f"3w MACD bullish, above 200dMA.")

    # Bearish divergence amplifier
    if rsi.get("bearish_divergence") and tier == "SAFE":
        tier = "CAUTION"
        color = "#f0b90b"
        emoji = "🟡"
        action += " + Bearish RSI divergence detected on weekly."

    # Olson's downside gap roadmap — next unfilled gap below price + progress.
    next_gap = None
    next_gap_pct = None
    gaps_below = [g for g in OLSON_QQQ_GAPS if g < last_close]
    if gaps_below:
        next_gap = max(gaps_below)          # nearest gap below current price
        next_gap_pct = (next_gap / last_close - 1) * 100
    gaps_filled = len([g for g in OLSON_QQQ_GAPS if g >= last_close])

    return {
        "tier":          tier,
        "tier_emoji":     emoji,
        "color":          color,
        "action":         action,
        "last_close":     last_close,
        "wma200":         lvl.get("wma200"),
        "dma200":         lvl.get("dma200"),
        "gap_level":      OLSON_GAP_LEVEL,
        "pct_to_gap":     pct_to_gap,
        "pct_to_dma200":  pct_to_dma200,
        "pct_to_wma200":  pct_to_wma200,
        # Olson gap roadmap
        "next_gap":       next_gap,
        "next_gap_pct":   round(next_gap_pct, 2) if next_gap_pct is not None else None,
        "gaps_total":     len(OLSON_QQQ_GAPS),
        "gaps_filled":    gaps_filled,
        "summary":        (f"QQQ ${last_close:,.0f} · "
                            f"gap@${OLSON_GAP_LEVEL} ({pct_to_gap:+.1f}%) · "
                            f"200wMA@${lvl.get('wma200', 0):,.0f} ({pct_to_wma200:+.1f}%)"),
        "signals":        sigs,
    }


def main():
    v = qqq_olson_verdict()
    print(f"{v['tier_emoji']} QQQ Olson — {v['tier']}")
    print(f"  {v['summary']}")
    print(f"  Action: {v['action']}")
    print()
    print("Details:")
    for k, s in v["signals"].items():
        print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
