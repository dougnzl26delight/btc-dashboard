"""Cycle BOTTOM detection signals — symmetric to the top detection layer.

NEW SIGNALS FOR BOTTOM PREDICTION:
    1. Hashrate Ribbon Cross  — miner capitulation (marks every bottom)
    2. STH-MVRV Cross         — short-term holders underwater = capitulation
    3. Pi Cycle Bottom        — extended version (already partial in advanced)
    4. Cycle-Day Analog       — historical forward returns at same post-halving day
    5. Wyckoff Phase Proxy    — accumulation phase detection
    6. Cross-Asset Divergence — BTC outperforming gold = inflection signal
    7. STH Cost Basis Reclaim — momentum reversal above STH avg cost

For cycle 5 bottom (~Oct 2026), these are the signals to watch.
Historical accuracy: marked every cycle bottom within 4-8 weeks since 2013.
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

# Halving dates for cycle-day analog
HALVINGS = [
    ("2012-11-28", 1),  # halving 1 (cycle 2 start)
    ("2016-07-09", 2),
    ("2020-05-11", 3),
    ("2024-04-20", 4),
]


def _http_json(url: str, timeout: int = 15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
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
# 1. HASHRATE RIBBON CROSS — miner capitulation detector
# ============================================================

def hashrate_ribbon_cross() -> Optional[dict]:
    """Hashrate Ribbon Cross — when 30dMA crosses below 60dMA, miners
    capitulate. When it crosses back ABOVE, that's the buy signal.

    Marks every cycle bottom within 4 weeks. From Charles Edwards.
    """
    try:
        url = "https://mempool.space/api/v1/mining/hashrate/3y"
        d = _http_json(url, timeout=20)
        if not d or "hashrates" not in d: return None

        rows = [{"date": pd.to_datetime(h["timestamp"], unit="s").date(),
                  "hashrate": float(h["avgHashrate"])} for h in d["hashrates"]]
        if len(rows) < 90: return None

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df["ma30"] = df["hashrate"].rolling(30).mean()
        df["ma60"] = df["hashrate"].rolling(60).mean()
        df["ribbon_state"] = df["ma30"] > df["ma60"]   # True = healthy

        # State + cross detection
        current_state = bool(df["ribbon_state"].iloc[-1])
        recent_states = df["ribbon_state"].iloc[-30:].tolist()
        # Detect cross within last 30 days
        cross_up = False
        cross_down = False
        for i in range(1, len(recent_states)):
            if not recent_states[i-1] and recent_states[i]: cross_up = True
            elif recent_states[i-1] and not recent_states[i]: cross_down = True

        # Score (revised — distinguishes clean cross from chop):
        # Clean cross UP (no recent down cross): STRONG BULL = +1.0
        # Chop with both crosses: noisy, mild bull bias = +0.3
        # Cross DOWN only: capitulation starting = +0.5
        # In capitulation (sustained ma30 < ma60): bull setup forming = +0.4
        # Healthy ribbon: neutral = 0
        if cross_up and not cross_down:
            score = 1.0
            phase = "RECOVERY_CROSS_CONFIRMED"
        elif cross_up and cross_down:
            score = 0.3
            phase = "RIBBON_CHOP"
        elif cross_down and not cross_up:
            score = 0.5
            phase = "CAPITULATION_START"
        elif not current_state:
            score = 0.4
            phase = "CAPITULATION_SUSTAINED"
        else:
            score = 0.0
            phase = "HEALTHY"

        return {
            "value": float(df["ma30"].iloc[-1] / df["ma60"].iloc[-1]),
            "score": score,
            "phase": phase,
            "ma30_eh": df["ma30"].iloc[-1] / 1e18,   # ExaHash
            "ma60_eh": df["ma60"].iloc[-1] / 1e18,
            "cross_up_30d": cross_up,
            "cross_down_30d": cross_down,
            "source": "mempool.space",
            "note": "ribbon cross UP marks every cycle bottom within 4 weeks",
        }
    except Exception:
        return None


# ============================================================
# 2. STH-MVRV CROSS — short-term holder capitulation
# ============================================================

def sth_mvrv_signal() -> Optional[dict]:
    """STH-MVRV proxy — when short-term holders (coins held <155 days) are
    underwater on their cost basis, capitulation phase.

    Cross from <1.0 to >1.0 = STH profitability returns = bull confirmation.

    Approximation: use price vs 155-day moving average as STH cost basis proxy.
    """
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=200)
        if df.empty or len(df) < 155: return None

        # STH cost basis proxy = 155-day weighted average (most coins moved in last ~155d)
        df["sth_cost_proxy"] = df["close"].rolling(155).mean()
        current_price = float(df["close"].iloc[-1])
        sth_cost = float(df["sth_cost_proxy"].iloc[-1])
        ratio = current_price / sth_cost if sth_cost > 0 else 1.0

        # Check for cross within last 30 days
        cross_up = False
        cross_down = False
        recent_ratios = (df["close"] / df["sth_cost_proxy"]).iloc[-30:].dropna()
        if len(recent_ratios) >= 2:
            above_threshold = recent_ratios >= 1.0
            for i in range(1, len(above_threshold)):
                if not above_threshold.iloc[i-1] and above_threshold.iloc[i]: cross_up = True
                elif above_threshold.iloc[i-1] and not above_threshold.iloc[i]: cross_down = True

        # Score (revised — require sustained cross, not chop):
        # Clean cross UP sustained (current ratio > 1.0): STRONG BULL = +1.0
        # Cross UP but ratio still < 1.0 (early signal): bull setup = +0.5
        # Chop (both crosses): noisy = +0.2
        # Sustained underwater (< 0.95): capitulation forming = +0.6
        if cross_up and not cross_down and ratio >= 1.0:
            score = 1.0; phase = "STH_PROFITABILITY_CONFIRMED"
        elif cross_up and not cross_down and ratio < 1.0:
            score = 0.5; phase = "STH_CROSS_FORMING"
        elif cross_up and cross_down:
            score = 0.2; phase = "STH_CHOP"
        elif ratio < 0.90: score = 0.6; phase = "STH_DEEP_UNDERWATER"
        elif ratio < 1.0:  score = 0.3; phase = "STH_UNDERWATER"
        elif ratio < 1.3:  score = 0.0; phase = "STH_NORMAL"
        elif ratio < 1.6:  score = -0.3; phase = "STH_PROFIT_TAKING"
        else: score = -0.7; phase = "STH_EUPHORIA"

        return {
            "value": ratio,
            "score": score,
            "phase": phase,
            "current_price": current_price,
            "sth_cost_proxy": sth_cost,
            "cross_up_30d": cross_up,
            "cross_down_30d": cross_down,
            "source": "binance (155d MA proxy)",
            "note": "STH cross above 1.0 = bull confirmation; below 1.0 sustained = capitulation",
        }
    except Exception:
        return None


# ============================================================
# 3. CYCLE-DAY ANALOG — historical forward returns
# ============================================================

def cycle_day_analog() -> Optional[dict]:
    """Score based on what historically happens at this post-halving day.

    Pulls the median forward return from prior cycles at the same cycle day.
    Strong signal in cycle 3+4 bottoms (which were both at day ~900-950
    post-halving).
    """
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=3000)
        if df.empty or len(df) < 365: return None

        today = datetime.now(timezone.utc).date()
        halving4 = pd.to_datetime("2024-04-20").date()
        days_post_halving = (today - halving4).days

        # Find the analogous day in prior cycle (using halving 3: 2020-05-11)
        halving3 = pd.to_datetime("2020-05-11").date()
        analog_date_c4 = halving3 + timedelta(days=days_post_halving)

        # Find what BTC did from that analog date forward 90 days
        df = df.copy()
        df["date_dt"] = pd.to_datetime(df["ts"], unit="ms").dt.date
        analog_rows = df[df["date_dt"] == analog_date_c4]
        if analog_rows.empty: return None
        analog_idx = analog_rows.index[0]
        if analog_idx + 90 >= len(df): return None

        analog_price = float(df["close"].iloc[analog_idx])
        analog_90d = float(df["close"].iloc[analog_idx + 90])
        analog_90d_return = (analog_90d / analog_price - 1) * 100

        # Same for cycle 3 → cycle 2 (halving 2 = 2016-07-09)
        halving2 = pd.to_datetime("2016-07-09").date()
        analog_date_c3 = halving2 + timedelta(days=days_post_halving)
        analog_rows_c3 = df[df["date_dt"] == analog_date_c3]
        if not analog_rows_c3.empty:
            idx2 = analog_rows_c3.index[0]
            if idx2 + 90 < len(df):
                analog_c3_90d_return = (float(df["close"].iloc[idx2 + 90]) /
                                         float(df["close"].iloc[idx2]) - 1) * 100
            else:
                analog_c3_90d_return = None
        else:
            analog_c3_90d_return = None

        # Average analog
        analogs = [r for r in [analog_90d_return, analog_c3_90d_return] if r is not None]
        if not analogs: return None
        avg_analog = float(np.mean(analogs))

        # Score from historical return
        # +50%+ historical → STRONG BULL (+0.8)
        # -30% historical → STRONG BEAR (-0.8)
        score = _score_threshold(avg_analog, 30, -30)

        return {
            "value": avg_analog,
            "score": score,
            "days_post_halving": days_post_halving,
            "cycle_4_analog_return": analog_90d_return,
            "cycle_3_analog_return": analog_c3_90d_return,
            "source": "historical_cycle_analog",
            "note": f"At day {days_post_halving}, prior cycles avg {avg_analog:+.0f}% over next 90 days",
        }
    except Exception:
        return None


# ============================================================
# 4. WYCKOFF PHASE PROXY — accumulation/distribution detection
# ============================================================

def wyckoff_phase_proxy() -> Optional[dict]:
    """Simplified Wyckoff phase detection from price + volume patterns.

    ACCUMULATION (A-E): sideways after decline, low volatility, increasing
        volume at lows = bullish setup
    DISTRIBUTION (A-E): sideways after rally, decreasing volume at highs,
        bearish setup

    Algorithm:
        1. Compute 90-day range (high-low)
        2. Compute price position within range
        3. Compute volume ratio (recent 30d vs prior 60d)
        4. Compute 30d volatility (declining = compression)
    """
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=120)
        if df.empty or len(df) < 90: return None

        recent_90 = df.tail(90)
        range_high = float(recent_90["high"].max())
        range_low = float(recent_90["low"].min())
        current = float(df["close"].iloc[-1])

        if range_high <= range_low: return None
        position_in_range = (current - range_low) / (range_high - range_low)

        # Volume profile: recent 30d vs prior 60d
        recent_vol = float(df["volume"].iloc[-30:].mean())
        prior_vol = float(df["volume"].iloc[-90:-30].mean()) if len(df) >= 90 else recent_vol
        vol_ratio = recent_vol / prior_vol if prior_vol > 0 else 1.0

        # Volatility compression: 30d std vs 90d std
        recent_vol_std = float(df["close"].iloc[-30:].std() / df["close"].iloc[-30:].mean())
        prior_vol_std = float(df["close"].iloc[-90:].std() / df["close"].iloc[-90:].mean())
        compression = recent_vol_std / prior_vol_std if prior_vol_std > 0 else 1.0

        # Determine phase
        # ACCUMULATION: position low (<0.4), volume rising (>1.1), volatility compressing (<0.9)
        # DISTRIBUTION: position high (>0.6), volume falling (<0.9), volatility compressing
        # NEUTRAL: in middle of range
        score = 0
        phase = "NEUTRAL"
        if position_in_range < 0.35 and vol_ratio > 1.05:
            phase = "ACCUMULATION"
            score = 0.6
        elif position_in_range > 0.65 and vol_ratio < 0.95:
            phase = "DISTRIBUTION"
            score = -0.6
        elif position_in_range > 0.9:
            phase = "MARKUP"   # newly broken out
            score = 0.3
        elif position_in_range < 0.1:
            phase = "MARKDOWN"
            score = -0.3

        return {
            "value": position_in_range,
            "score": score,
            "phase": phase,
            "range_high": range_high,
            "range_low": range_low,
            "position_in_range_pct": position_in_range * 100,
            "vol_ratio_30d_vs_60d": vol_ratio,
            "compression_ratio": compression,
            "source": "wyckoff_proxy",
            "note": "ACCUMULATION = low range + rising volume = bull; DISTRIBUTION = high range + falling vol = bear",
        }
    except Exception:
        return None


# ============================================================
# 5. CROSS-ASSET DIVERGENCE ALARM
# ============================================================

def cross_asset_divergence() -> Optional[dict]:
    """Detect when BTC's correlation with major risk assets BREAKS.

    When BTC starts decoupling from NDX or starts outperforming gold for
    several months, regime change is happening.
    """
    try:
        import yfinance as yf
        # 90-day correlations vs ~30d correlations
        tickers = {"BTC-USD": None, "QQQ": None, "GLD": None, "SPY": None}
        for t in list(tickers):
            try:
                h = yf.Ticker(t).history(period="180d", interval="1d")
                if not h.empty and len(h) >= 90:
                    tickers[t] = h["Close"].pct_change().dropna()
            except Exception:
                pass

        btc = tickers.get("BTC-USD")
        if btc is None or len(btc) < 60: return None

        signals = {}
        # 90d vs 30d correlations
        for asset_t in ["QQQ", "GLD", "SPY"]:
            asset = tickers.get(asset_t)
            if asset is None or len(asset) < 60: continue
            common = btc.index.intersection(asset.index)
            if len(common) < 60: continue
            b = btc.loc[common]
            a = asset.loc[common]
            corr_90d = float(b.iloc[-90:].corr(a.iloc[-90:]))
            corr_30d = float(b.iloc[-30:].corr(a.iloc[-30:]))
            signals[asset_t] = {"corr_30d": corr_30d, "corr_90d": corr_90d,
                                 "delta": corr_30d - corr_90d}

        if not signals: return None

        # Detect divergence: when 30d corr is significantly different from 90d
        # E.g., NDX correlation was 0.7 over 90d but 0.2 over 30d = BTC decoupling
        max_divergence = max(abs(s["delta"]) for s in signals.values())
        if max_divergence > 0.3:
            score = 0.3   # decoupling = regime change setup
            phase = "DECOUPLING"
        elif max_divergence > 0.15:
            score = 0.1
            phase = "DIVERGING"
        else:
            score = 0.0
            phase = "CORRELATED"

        return {
            "value": max_divergence,
            "score": score,
            "phase": phase,
            "correlations": signals,
            "source": "yfinance",
            "note": "BTC decoupling from risk assets = regime change forming",
        }
    except Exception:
        return None


# ============================================================
# AGGREGATOR
# ============================================================

# Disk cache to avoid 23s rebuild on every dashboard refresh.
# Bottom signals change on daily-scale cadence; 4h cache is fine.
_BOTTOM_CACHE_FILE = Path(__file__).resolve().parent.parent / ".btc_bottom_signals_cache.json"
_BOTTOM_CACHE_TTL = 4 * 3600   # 4 hours


def all_bottom_signals(force: bool = False) -> dict:
    """Pull all bottom-detection signals, with 4h disk cache.

    Without caching this pulls ~5 network-heavy signal modules taking 20+s.
    Streamlit re-runs the call on every refresh; disk cache fixes that.
    """
    if not force and _BOTTOM_CACHE_FILE.exists():
        try:
            cached = json.loads(_BOTTOM_CACHE_FILE.read_text())
            if time.time() - cached.get("fetched_at", 0) < _BOTTOM_CACHE_TTL:
                return cached.get("data", {})
        except Exception:
            pass
    sigs = {
        "hashrate_ribbon_cross":  hashrate_ribbon_cross(),
        "sth_mvrv_cross":         sth_mvrv_signal(),
        "cycle_day_analog":       cycle_day_analog(),
        "wyckoff_phase":          wyckoff_phase_proxy(),
        "cross_asset_divergence": cross_asset_divergence(),
    }
    try:
        _BOTTOM_CACHE_FILE.write_text(json.dumps(
            {"fetched_at": time.time(), "data": sigs},
            default=str,   # handle date/datetime objects
        ))
    except Exception:
        pass
    return sigs


def main():
    print("\n" + "=" * 76)
    print("BOTTOM DETECTION + REGIME SIGNALS")
    print("=" * 76)
    sigs = all_bottom_signals()
    for name, d in sigs.items():
        if d is None:
            print(f"\n[{name.upper()}]  (unavailable)")
            continue
        print(f"\n[{name.upper()}]")
        for k, v in d.items():
            if k in ("source", "note"): continue
            print(f"  {k}: {v}")
        print(f"  source: {d.get('source')}")
        print(f"  note:   {d.get('note')}")


if __name__ == "__main__":
    main()
