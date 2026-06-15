"""JESSE OLSON TECHNICAL LAYER — multi-week TA framework.

Jesse Olson (@JesseOlson on X) is a crypto technical analyst whose signature
framework is multi-week timeframe analysis. He successfully called the
cycle 5 BTC top via 3-week MACD bearish crossover + bearish histogram
divergence. He also called the October 10, 2025 Binance cascade early.

His three signature signals (all free OHLCV-based):

    1. THREE-WEEK MACD — his #1 timing tool
       Bullish cross + histogram positive = bottom confirmation
       Bearish cross + histogram divergence = top confirmation

    2. WEEKLY HEIKIN ASHI — color + wick pattern
       Sustained red w/ no lower wick = downtrend confirmed
       Green flip + lower wick after extended red = bottom reversal

    3. WEEKLY RSI DIVERGENCE — bullish divergence at lows
       Lower price low + higher RSI low = momentum reversal forming

Track record (per public press coverage):
    - Sep 2024 ($40k bottom call): WRONG (BTC went to $126k peak instead)
    - 2025 cycle 5 TOP call:        RIGHT (peak Oct 6 2025 $126,296)
    - Oct 10 2025 Binance cascade:  RIGHT (called early)

His strength = top identification.
Bottom calling has weaker historical accuracy.

Complements our on-chain layer (Glassnode/Woo) which is cost-basis driven.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Data loading
# ============================================================

_OHLCV_CACHE: dict = {}


def _load_ohlcv(days: int = 1825) -> Optional[pd.DataFrame]:
    """5 years of daily OHLCV. Cached per-process."""
    if days in _OHLCV_CACHE:
        return _OHLCV_CACHE[days]
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=days)
        if df.empty: return None
        # Ensure datetime index
        if "date" in df.columns:
            df = df.set_index("date")
        elif df.index.name is None:
            df.index.name = "date"
        df = df.sort_index()
        _OHLCV_CACHE[days] = df
        return df
    except Exception:
        return None


# ============================================================
# 1. THREE-WEEK MACD — his #1 timing tool
# ============================================================

def three_week_macd() -> Optional[dict]:
    """3-week MACD — Jesse Olson's #1 cycle timing tool.

    He called the cycle 5 TOP using:
        - Bearish MACD crossover on 3-week chart
        - Bearish histogram divergence

    Inverse signal = bottom call:
        - Bullish MACD crossover on 3-week chart
        - Bullish histogram divergence
    """
    try:
        df = _load_ohlcv(days=1825)
        if df is None or len(df) < 500: return None

        # Resample to 3-week candles (21 days)
        df3w = df["close"].resample("21D").last().dropna()
        if len(df3w) < 60: return None

        # Standard MACD: EMA12 - EMA26, signal = EMA9 of MACD
        ema_fast = df3w.ewm(span=12, adjust=False).mean()
        ema_slow = df3w.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0
        prev_macd = float(macd_line.iloc[-2]) if len(macd_line) > 1 else 0
        prev_signal = float(signal_line.iloc[-2]) if len(signal_line) > 1 else 0

        # Cross detection
        bullish_cross = (current_macd > current_signal) and (prev_macd <= prev_signal)
        bearish_cross = (current_macd < current_signal) and (prev_macd >= prev_signal)

        # Histogram bullish/bearish divergence vs price
        # Look at last 6 swing points (~6 candles = 4-5 months)
        recent_close = df3w.iloc[-6:]
        recent_hist = histogram.iloc[-6:]
        price_higher_low = recent_close.iloc[-1] < recent_close.min() * 1.02
        hist_higher_low = recent_hist.iloc[-1] > recent_hist.min() * 0.85
        bullish_divergence = price_higher_low and hist_higher_low and current_hist < 0
        # Bearish divergence (top signal)
        price_higher_high = recent_close.iloc[-1] > recent_close.max() * 0.98
        hist_lower_high = recent_hist.iloc[-1] < recent_hist.max() * 0.85
        bearish_divergence = price_higher_high and hist_lower_high and current_hist > 0

        # Phase + score
        if bullish_cross:
            phase = "BULLISH_CROSS"
            score = 0.85
            note = ("3-week MACD bullish crossover — Jesse Olson bottom signal. "
                    "Strongest BTC cycle timing tool in his framework.")
        elif bearish_cross:
            phase = "BEARISH_CROSS"
            score = -0.85
            note = ("3-week MACD bearish crossover — Jesse Olson top signal. "
                    "Called cycle 5 top via this signal.")
        elif bullish_divergence:
            phase = "BULLISH_DIVERGENCE"
            score = 0.55
            note = "Histogram bullish divergence — bottom forming on 3w chart"
        elif bearish_divergence:
            phase = "BEARISH_DIVERGENCE"
            score = -0.55
            note = "Histogram bearish divergence — top forming on 3w chart"
        elif current_macd > current_signal and current_hist > 0:
            phase = "BULLISH_TREND"
            score = 0.35
            note = "MACD above signal, histogram positive — uptrend intact"
        elif current_macd > current_signal:
            phase = "BULL_WEAKENING"
            score = 0.10
            note = "MACD above signal but histogram weakening"
        elif current_macd < current_signal and current_hist < 0 and current_hist > prev_hist:
            phase = "BEAR_WEAKENING"
            score = 0.15
            note = "MACD below signal but histogram improving — early bull pivot?"
        elif current_macd < current_signal:
            phase = "BEARISH_TREND"
            score = -0.35
            note = "MACD below signal, downtrend intact"
        else:
            phase = "NEUTRAL"
            score = 0.0
            note = "No clear MACD signal"

        return {
            "value": current_hist,
            "score": score,
            "phase": phase,
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_hist,
            "bullish_cross": bool(bullish_cross),
            "bearish_cross": bool(bearish_cross),
            "bullish_divergence": bool(bullish_divergence),
            "bearish_divergence": bool(bearish_divergence),
            "source": "3w_MACD_Olson",
            "note": note,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 2. WEEKLY HEIKIN ASHI — color + wick pattern
# ============================================================

def weekly_heikin_ashi() -> Optional[dict]:
    """Weekly Heikin Ashi color + wick pattern.

    Jesse Olson watches:
        - Sustained red with no lower wick = strong downtrend
        - Sustained green with no upper wick = strong uptrend
        - Green flip + lower wick after extended red = bottom reversal
        - Red flip + upper wick after extended green = top reversal
    """
    try:
        df = _load_ohlcv(days=1095)
        if df is None or len(df) < 60: return None

        # Resample to weekly OHLC
        df_w = pd.DataFrame({
            "open":  df["open"].resample("7D").first(),
            "high":  df["high"].resample("7D").max(),
            "low":   df["low"].resample("7D").min(),
            "close": df["close"].resample("7D").last(),
        }).dropna()
        if len(df_w) < 20: return None

        # Heikin Ashi computation
        ha = pd.DataFrame(index=df_w.index)
        ha["ha_close"] = (df_w["open"] + df_w["high"] + df_w["low"] + df_w["close"]) / 4
        ha["ha_open"] = np.nan
        ha.loc[ha.index[0], "ha_open"] = (df_w["open"].iloc[0] + df_w["close"].iloc[0]) / 2
        for i in range(1, len(ha)):
            ha.loc[ha.index[i], "ha_open"] = (
                (ha["ha_open"].iloc[i-1] + ha["ha_close"].iloc[i-1]) / 2
            )
        ha["ha_high"] = np.maximum.reduce([df_w["high"].values, ha["ha_open"].values, ha["ha_close"].values])
        ha["ha_low"] = np.minimum.reduce([df_w["low"].values, ha["ha_open"].values, ha["ha_close"].values])

        # Current candle properties
        last = ha.iloc[-1]
        is_green = last["ha_close"] > last["ha_open"]
        body_top = max(last["ha_open"], last["ha_close"])
        body_bot = min(last["ha_open"], last["ha_close"])
        upper_wick = last["ha_high"] - body_top
        lower_wick = body_bot - last["ha_low"]
        body_size = abs(last["ha_close"] - last["ha_open"])
        upper_wick_pct = (upper_wick / body_size * 100) if body_size > 0 else 0
        lower_wick_pct = (lower_wick / body_size * 100) if body_size > 0 else 0

        # Recent color sequence (last 8 candles)
        recent = ha.iloc[-8:]
        colors = ["G" if r["ha_close"] > r["ha_open"] else "R" for _, r in recent.iterrows()]
        recent_green = sum(1 for c in colors if c == "G")
        recent_red = sum(1 for c in colors if c == "R")

        # Pattern detection — Jesse Olson style
        # 1. Reversal signals (key)
        prev_2_colors = colors[-3:-1]  # 2 candles before current
        flip_to_green_after_red = (is_green and prev_2_colors == ["R", "R"] and
                                    lower_wick_pct > 20)
        flip_to_red_after_green = (not is_green and prev_2_colors == ["G", "G"] and
                                    upper_wick_pct > 20)
        # 2. Strong trend signals
        sustained_red_no_lower_wick = (recent_red >= 4 and not is_green and lower_wick_pct < 10)
        sustained_green_no_upper_wick = (recent_green >= 4 and is_green and upper_wick_pct < 10)

        # Score + phase
        if flip_to_green_after_red:
            phase = "BOTTOM_REVERSAL_FLIP"
            score = 0.75
            note = ("Green Heikin Ashi after extended red + lower wick. "
                    "Jesse Olson bottom reversal pattern.")
        elif flip_to_red_after_green:
            phase = "TOP_REVERSAL_FLIP"
            score = -0.75
            note = "Red HA after extended green + upper wick — top reversal pattern"
        elif sustained_red_no_lower_wick:
            phase = "STRONG_DOWNTREND"
            score = -0.65
            note = ("Sustained red HA with no lower wick — "
                    "Jesse Olson strong downtrend signal")
        elif sustained_green_no_upper_wick:
            phase = "STRONG_UPTREND"
            score = 0.65
            note = "Sustained green HA with no upper wick — strong uptrend"
        elif is_green and lower_wick_pct > 30:
            phase = "BULL_WITH_REJECTION"
            score = 0.4
            note = "Green HA with significant lower wick = buyers absorbing dips"
        elif not is_green and upper_wick_pct > 30:
            phase = "BEAR_WITH_REJECTION"
            score = -0.4
            note = "Red HA with significant upper wick = sellers absorbing rallies"
        elif is_green:
            phase = "MILD_BULL"
            score = 0.2
            note = "Green HA candle, no special pattern"
        else:
            phase = "MILD_BEAR"
            score = -0.2
            note = "Red HA candle, no special pattern"

        return {
            "value": 1.0 if is_green else -1.0,
            "score": score,
            "phase": phase,
            "is_green": bool(is_green),
            "upper_wick_pct": upper_wick_pct,
            "lower_wick_pct": lower_wick_pct,
            "recent_sequence": "".join(colors),
            "ha_open": float(last["ha_open"]),
            "ha_close": float(last["ha_close"]),
            "ha_high": float(last["ha_high"]),
            "ha_low": float(last["ha_low"]),
            "source": "weekly_HA_Olson",
            "note": note,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 3. WEEKLY RSI DIVERGENCE
# ============================================================

def weekly_rsi_divergence() -> Optional[dict]:
    """Weekly RSI bullish/bearish divergence detector.

    Jesse Olson watches weekly RSI for momentum divergence vs price:
        - Lower price low + higher RSI low = bullish divergence (bottom)
        - Higher price high + lower RSI high = bearish divergence (top)
    """
    try:
        df = _load_ohlcv(days=1095)
        if df is None or len(df) < 200: return None

        # Resample to weekly close
        df_w = df["close"].resample("7D").last().dropna()
        if len(df_w) < 50: return None

        # 14-period weekly RSI
        delta = df_w.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        current_rsi = float(rsi.iloc[-1])

        # Find recent price swing lows/highs in last 30 weeks
        recent_price = df_w.iloc[-30:]
        recent_rsi = rsi.iloc[-30:]
        if len(recent_price) < 20: return None

        # Detect divergence: split window into two halves
        first_half_idx = len(recent_price) // 2
        first_half_price = recent_price.iloc[:first_half_idx]
        second_half_price = recent_price.iloc[first_half_idx:]
        first_half_rsi = recent_rsi.iloc[:first_half_idx]
        second_half_rsi = recent_rsi.iloc[first_half_idx:]

        # Bullish divergence: second half has lower price low but higher RSI low
        first_price_low = first_half_price.min()
        second_price_low = second_half_price.min()
        first_rsi_low = first_half_rsi.min()
        second_rsi_low = second_half_rsi.min()
        bullish_div = (second_price_low < first_price_low * 0.98 and
                        second_rsi_low > first_rsi_low * 1.05 and
                        current_rsi < 50)

        # Bearish divergence: second half has higher price high but lower RSI high
        first_price_high = first_half_price.max()
        second_price_high = second_half_price.max()
        first_rsi_high = first_half_rsi.max()
        second_rsi_high = second_half_rsi.max()
        bearish_div = (second_price_high > first_price_high * 1.02 and
                        second_rsi_high < first_rsi_high * 0.95 and
                        current_rsi > 50)

        # RSI zone
        if current_rsi < 30: zone = "OVERSOLD"
        elif current_rsi < 40: zone = "NEAR_OVERSOLD"
        elif current_rsi < 60: zone = "NEUTRAL"
        elif current_rsi < 70: zone = "NEAR_OVERBOUGHT"
        else: zone = "OVERBOUGHT"

        # Score + phase
        if bullish_div:
            phase = "BULLISH_DIVERGENCE"
            score = 0.7
            note = ("Weekly RSI bullish divergence — momentum reversing at lows. "
                    "Jesse Olson bottom signal.")
        elif bearish_div:
            phase = "BEARISH_DIVERGENCE"
            score = -0.7
            note = ("Weekly RSI bearish divergence — momentum exhausting at highs. "
                    "Jesse Olson top signal.")
        elif current_rsi < 30:
            phase = "OVERSOLD"
            score = 0.4
            note = f"Weekly RSI {current_rsi:.0f} — oversold zone, bottom-forming territory"
        elif current_rsi > 70:
            phase = "OVERBOUGHT"
            score = -0.4
            note = f"Weekly RSI {current_rsi:.0f} — overbought zone, top-forming territory"
        elif current_rsi > 60:
            phase = "BULL_MOMENTUM"
            score = 0.15
            note = f"Weekly RSI {current_rsi:.0f} — bull momentum"
        elif current_rsi > 50:
            phase = "MILD_BULL"
            score = 0.05
            note = f"Weekly RSI {current_rsi:.0f} — mild bull bias"
        elif current_rsi > 40:
            phase = "MILD_BEAR"
            score = -0.05
            note = f"Weekly RSI {current_rsi:.0f} — mild bear bias"
        else:
            phase = "BEAR_MOMENTUM"
            score = -0.25
            note = f"Weekly RSI {current_rsi:.0f} — bear momentum"

        return {
            "value": current_rsi,
            "score": score,
            "phase": phase,
            "zone": zone,
            "current_rsi": current_rsi,
            "bullish_divergence": bool(bullish_div),
            "bearish_divergence": bool(bearish_div),
            "first_price_low": float(first_price_low),
            "second_price_low": float(second_price_low),
            "first_rsi_low": float(first_rsi_low),
            "second_rsi_low": float(second_rsi_low),
            "source": "weekly_RSI_Olson",
            "note": note,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Aggregator
# ============================================================

def all_olson_signals() -> dict:
    """Return all three Jesse Olson TA signals."""
    return {
        "three_week_macd":       three_week_macd(),
        "weekly_heikin_ashi":    weekly_heikin_ashi(),
        "weekly_rsi_divergence": weekly_rsi_divergence(),
    }


def olson_combined_verdict() -> dict:
    """Combined verdict across his 3 signals.

    All 3 bullish = high-conviction bottom call.
    All 3 bearish = high-conviction top call.
    Mixed = wait.
    """
    sigs = all_olson_signals()
    valid_scores = []
    bullish_count = 0
    bearish_count = 0
    for name, d in sigs.items():
        if d is None or d.get("error"): continue
        s = d.get("score")
        if s is None: continue
        valid_scores.append(s)
        if s > 0.3: bullish_count += 1
        elif s < -0.3: bearish_count += 1

    avg = sum(valid_scores) / max(1, len(valid_scores))
    if bullish_count >= 2 and avg > 0.4:
        verdict = "BOTTOM SIGNAL FORMING (Olson framework)"
        verdict_level = "BULLISH"
    elif bearish_count >= 2 and avg < -0.4:
        verdict = "TOP SIGNAL FORMING (Olson framework)"
        verdict_level = "BEARISH"
    elif bullish_count >= 1 and bearish_count == 0:
        verdict = "Early bull bias (Olson framework)"
        verdict_level = "MILD_BULL"
    elif bearish_count >= 1 and bullish_count == 0:
        verdict = "Early bear bias (Olson framework)"
        verdict_level = "MILD_BEAR"
    else:
        verdict = "No clear signal (Olson framework)"
        verdict_level = "NEUTRAL"

    return {
        "signals":       sigs,
        "avg_score":     avg,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "n_valid":       len(valid_scores),
        "verdict":       verdict,
        "verdict_level": verdict_level,
    }


def main():
    print("\n" + "=" * 76)
    print("JESSE OLSON TECHNICAL LAYER — multi-week TA framework")
    print("=" * 76)
    print()
    result = olson_combined_verdict()
    sigs = result["signals"]
    for name, d in sigs.items():
        if d is None:
            print(f"[{name.upper()}]  unavailable")
            continue
        if d.get("error"):
            print(f"[{name.upper()}]  ERROR: {d['error']}")
            continue
        print(f"[{name.upper()}]")
        print(f"  phase: {d.get('phase')}")
        print(f"  score: {d.get('score'):+.2f}")
        print(f"  note:  {d.get('note', '')[:90]}")
        print()
    print("=" * 76)
    print(f"VERDICT: {result['verdict']}")
    print(f"Avg score: {result['avg_score']:+.2f}  "
          f"({result['bullish_count']} bull / {result['bearish_count']} bear of {result['n_valid']})")
    print("=" * 76)


if __name__ == "__main__":
    main()
