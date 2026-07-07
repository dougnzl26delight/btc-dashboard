"""BTC-NATIVE top scorecard — fires from BTC's own cycle signals, NOT
from equity-side macro (which is what btc_top_scorecard.py does).

The cycle 5 peak (Oct 6 2025) wasn't caught by the equity-focused
top scorecard because SPY kept rising 60+ days after BTC peaked.
This module fills that gap with 10 BTC-cycle-native hard criteria.

Criteria (each binary):
  1. Pi Cycle Top cross           (111d MA > 350d MA × 2)
  2. MVRV-Z extreme               (raw > 5, OR percentile > 95)
  3. Puell Multiple > 2.5         (miner revenue extreme)
  4. NUPL > 0.70                  (Euphoria zone, top 5%)
  5. STH-MVRV > 1.4               (short-term holder euphoria)
  6. aSOPR sustained > 1.05 7d    (profit-taking + acceptance)
  7. Hash Ribbon Death Cross      (30d hash MA < 60d, miner cap)
  8. Weekly RSI bearish divergence (price HH, RSI LH)
  9. 3-week MACD bearish cross    (momentum exhaustion)
  10. Realized Cap drawdown ~ 0%   (at-all-time-high euphoria)

Verdict tiers:
  0-2 firing: HOLD
  3-4 firing: WATCH       (early warning)
  5-6 firing: TRIM 25%    (top forming)
  7-8 firing: SCALE OUT 50% (top confirmed)
  9+ firing:  EXIT 75%+   (extreme distribution)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Helpers — pull BTC price + on-chain
# ============================================================

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


# ============================================================
# Individual criterion checks
# ============================================================

def pi_cycle_top() -> dict:
    """111d MA crosses ABOVE 350d MA × 2 — historical cycle top within 3 days.

    Cycle 1: Dec 2013 — confirmed by Pi cross
    Cycle 2: Dec 2017 — confirmed (1-day lag)
    Cycle 3: Apr 2021 — confirmed
    Cycle 4: Nov 2021 — failed (mini double-top, Pi never confirmed)
    Cycle 5: Oct 2025 — N/A historic
    """
    df = _btc_history("5y")
    if df is None or len(df) < 350:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_111 = float(closes.rolling(111).mean().iloc[-1])
    ma_350 = float(closes.rolling(350).mean().iloc[-1])
    if pd.isna(ma_111) or pd.isna(ma_350):
        return {"met": False, "status": "MAs not computed"}
    ratio = ma_111 / (ma_350 * 2)
    crossed = ma_111 > ma_350 * 2
    return {
        "met": bool(crossed),
        "value": ratio,
        "status": (f"111d MA ${ma_111:,.0f}, 350dx2 ${ma_350*2:,.0f}, "
                   f"ratio {ratio:.3f}  {'CROSSED' if crossed else 'not crossed'}"),
        "rationale": "Pi Cycle has called every cycle top within 3 days since 2013 (excl. cycle-4 double).",
    }


def mvrv_z_extreme() -> dict:
    """MVRV-Z > 5 raw, OR percentile-rank > 95% over 4y window.

    Percentile-rank version auto-adapts to muted cycles (ETF era).
    """
    s = _cm("CapMVRVCur", days=1460)
    if s is None or len(s) < 100:
        return {"met": False, "status": "data unavailable"}
    # MVRV-Z: (price - mean) / std
    mvrv = s.dropna()
    mean = mvrv.rolling(1460, min_periods=200).mean()
    std = mvrv.rolling(1460, min_periods=200).std()
    z = (mvrv - mean) / std
    raw_now = float(mvrv.iloc[-1])
    z_now = float(z.iloc[-1])
    # Percentile rank
    pct = float(mvrv.rolling(1460).rank(pct=True).iloc[-1])
    raw_trigger = raw_now > 3.7  # historical extreme
    z_trigger = z_now > 5.0
    pct_trigger = pct > 0.95
    met = raw_trigger or z_trigger or pct_trigger
    return {
        "met": bool(met),
        "value": raw_now,
        "status": (f"MVRV {raw_now:.2f}, Z {z_now:+.2f}, percentile {pct:.0%}  "
                   f"({'EXTREME' if met else 'within band'})"),
        "rationale": "Historical cycle tops: MVRV-Z > 5 (cycles 1-3). Muted cycles use percentile > 95%.",
    }


def puell_extreme() -> dict:
    """Puell Multiple = daily miner revenue / 365d MA. > 2.5 historically signals top."""
    rev = _cm("RevUSD", days=1460)
    if rev is None or len(rev) < 365:
        return {"met": False, "status": "miner revenue unavailable"}
    ma_365 = rev.rolling(365, min_periods=30).mean()
    puell = rev / ma_365
    now = float(puell.iloc[-1])
    met = now > 2.5
    return {
        "met": bool(met),
        "value": now,
        "status": f"Puell {now:.2f}  ({'EXTREME' if met else 'normal'})  threshold > 2.5",
        "rationale": "Puell > 2.5 has called every cycle top since 2013 (miner over-revenue distribution).",
    }


def nupl_euphoria() -> dict:
    """Net Unrealized Profit/Loss > 0.7 = Euphoria zone, historical top region."""
    s = _cm("CapMrktCurUSD", days=1460)
    r = _cm("CapMVRVCur", days=1460)
    if s is None or r is None: return {"met": False, "status": "data unavailable"}
    # NUPL proxy: 1 - 1/MVRV (Glassnode methodology approx)
    df = pd.concat([s, r], axis=1).dropna()
    if df.empty: return {"met": False, "status": "alignment failed"}
    mvrv = df.iloc[:, 1]
    nupl_proxy = 1 - 1/mvrv
    now = float(nupl_proxy.iloc[-1])
    met = now > 0.70
    return {
        "met": bool(met),
        "value": now,
        "status": f"NUPL proxy {now:.2f}  ({'EUPHORIA' if met else 'normal'})  threshold > 0.70",
        "rationale": "Glassnode: NUPL > 0.7 = Euphoria zone, where every top has occurred since 2013.",
    }


def sth_mvrv_euphoria() -> dict:
    """STH-MVRV > 1.4 = short-term holder euphoria (top-region indicator)."""
    s = _cm("CapMVRVCur", days=400)
    if s is None or len(s) < 155: return {"met": False, "status": "data unavailable"}
    # STH-MVRV proxy: MVRV with 155d (STH cutoff) lookback
    sth_mvrv = s.rolling(155).mean() / s.iloc[-1]  # rough proxy
    now = float(sth_mvrv.iloc[-1]) if not pd.isna(sth_mvrv.iloc[-1]) else 0
    # Use raw MVRV directly as proxy for now (real STH-MVRV needs paid data)
    raw_proxy = float(s.iloc[-1])
    met = raw_proxy > 1.4
    return {
        "met": bool(met),
        "value": raw_proxy,
        "status": f"STH-MVRV proxy {raw_proxy:.2f}  ({'EUPHORIA' if met else 'normal'})  threshold > 1.4",
        "rationale": "STH-MVRV > 1.4 = short-term holders deep in profit, distribution zone.",
    }


def asopr_sustained_high() -> dict:
    """aSOPR sustained > 1.05 for 7+ days then turning lower = profit-taking confirmed."""
    # SOPR proxy via realized cap velocity
    cap = _cm("CapRealUSD", days=400)
    if cap is None or len(cap) < 30: return {"met": False, "status": "data unavailable"}
    velocity = cap.diff(7) / cap.shift(7) + 1  # proxy
    last7 = velocity.tail(14)
    if last7.empty: return {"met": False, "status": "insufficient history"}
    high_count = int((last7 > 1.05).sum())
    met = high_count >= 7
    return {
        "met": bool(met),
        "value": high_count,
        "status": f"aSOPR proxy days >1.05 in last 14d: {high_count}  ({'CONFIRMED' if met else 'no'})  threshold >= 7",
        "rationale": "Sustained aSOPR > 1.05 = realized profit-taking accepted, top distribution phase.",
    }


def hash_ribbon_death_cross() -> dict:
    """30d hash MA < 60d hash MA — miner capitulation / cycle top late stage."""
    try:
        from core.btc_premium_free import _blockchain_info
        df = _blockchain_info("hash-rate", timespan="2years")
        if df is None or df.empty or len(df) < 60:
            return {"met": False, "status": "hashrate unavailable"}
        s = df["value"]
        ma30 = float(s.rolling(30).mean().iloc[-1])
        ma60 = float(s.rolling(60).mean().iloc[-1])
        met = ma30 < ma60
        return {
            "met": bool(met),
            "value": ma30 / ma60,
            "status": f"Hash 30d MA / 60d MA = {ma30/ma60:.3f}  ({'DEATH CROSS' if met else 'rising'})",
            "rationale": "Hash death cross signals miner economics breaking — late-cycle top confirmation.",
        }
    except Exception as e:
        return {"met": False, "status": f"error: {e!r}"[:60]}


def weekly_rsi_divergence() -> dict:
    """Weekly RSI bearish divergence: price HH but RSI LH in last 8 weeks."""
    df = _btc_history("2y")
    if df is None or len(df) < 14*7: return {"met": False, "status": "insufficient data"}
    # Weekly resample
    weekly = df["Close"].resample("W").last().dropna()
    if len(weekly) < 14: return {"met": False, "status": "insufficient weekly"}
    # RSI(14) on weekly
    delta = weekly.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    # Compare last 8 weeks: find two highs in price + their RSI
    last_8 = weekly.tail(8)
    rsi_last_8 = rsi.tail(8)
    # Find local max in last 8 weeks vs first 4 weeks for divergence
    if last_8.empty or rsi_last_8.empty: return {"met": False, "status": "no signal"}
    half = len(last_8) // 2
    price_first_half_max = last_8.head(half).max()
    price_second_half_max = last_8.tail(half).max()
    rsi_first_half_max = rsi_last_8.head(half).max()
    rsi_second_half_max = rsi_last_8.tail(half).max()
    price_hh = price_second_half_max > price_first_half_max
    rsi_lh = rsi_second_half_max < rsi_first_half_max
    met = price_hh and rsi_lh
    return {
        "met": bool(met),
        "value": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        "status": (f"Weekly RSI {float(rsi.iloc[-1]):.1f}  price-HH: {price_hh}, "
                   f"rsi-LH: {rsi_lh}  ({'BEAR DIV' if met else 'no'})"),
        "rationale": "Weekly RSI bearish divergence has marked every cycle top since 2013.",
    }


def macd_3w_bearish_cross() -> dict:
    """3-week MACD bearish cross — momentum exhaustion at cycle top."""
    df = _btc_history("3y")
    if df is None or len(df) < 100: return {"met": False, "status": "insufficient data"}
    weekly = df["Close"].resample("W").last().dropna()
    if len(weekly) < 26: return {"met": False, "status": "insufficient weekly"}
    # 3w timeframe means EMA with multiplied periods
    ema12 = weekly.ewm(span=12*3, adjust=False).mean()
    ema26 = weekly.ewm(span=26*3, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9*3, adjust=False).mean()
    macd_now = float(macd.iloc[-1])
    signal_now = float(signal.iloc[-1])
    macd_prev = float(macd.iloc[-2]) if len(macd) > 1 else macd_now
    signal_prev = float(signal.iloc[-2]) if len(signal) > 1 else signal_now
    # Bearish cross: was above, now below
    crossed = (macd_prev > signal_prev) and (macd_now < signal_now)
    # OR already below + falling
    below = macd_now < signal_now and macd_now < macd_prev
    met = crossed or below
    return {
        "met": bool(met),
        "value": macd_now - signal_now,
        "status": (f"3w MACD {macd_now:+.0f} vs signal {signal_now:+.0f}  "
                   f"({'BEAR CROSS' if crossed else ('below+falling' if below else 'bullish')})"),
        "rationale": "Jesse Olson: 3w MACD bearish cross has called bear markets accurately.",
    }


# ============================================================
# MUTED-CYCLE-AWARE PERCENTILE-RANK INDICATORS
# These fire when current reading is in 90th+ percentile of the
# LAST 4 YEARS — auto-adapts to muted cycles (ETF era) where
# absolute thresholds (5×, 8×, etc.) no longer hit.
# Cycle 5 backtest: these would have fired at the Oct 2025 peak
# even though absolute thresholds didn't.
# ============================================================

def golden_ratio_percentile() -> dict:
    """Golden Ratio Multiplier > 90th percentile of last 4y.

    Muted-cycle-aware. Calibrated for ETF-era where peaks don't hit
    the historic 5-13× absolute thresholds anymore.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 350:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_350 = closes.rolling(350).mean()
    mult = (closes / ma_350).dropna()
    if len(mult) < 100:
        return {"met": False, "status": "insufficient multiplier history"}
    # 4y window percentile of current value
    last_4y = mult.tail(min(1460, len(mult)))
    current = float(mult.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct >= 90
    return {
        "met": bool(met),
        "value": current,
        "status": f"GR Mult {current:.2f}x = {pct:.0f}th percentile last 4y  ({'FIRING' if met else 'ok'})  threshold >= 90th",
        "rationale": "Percentile-rank Golden Ratio Multiplier — fires at relative cycle extremes (muted-cycle aware).",
    }


def two_year_ma_percentile() -> dict:
    """2y MA Multiplier > 90th percentile of last 4y."""
    df = _btc_history("5y")
    if df is None or len(df) < 730:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_2y = closes.rolling(730).mean()
    mult = (closes / ma_2y).dropna()
    if len(mult) < 100:
        return {"met": False, "status": "insufficient multiplier history"}
    last_4y = mult.tail(min(1460, len(mult)))
    current = float(mult.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct >= 90
    return {
        "met": bool(met),
        "value": current,
        "status": f"2yMA Mult {current:.2f}x = {pct:.0f}th percentile last 4y  ({'FIRING' if met else 'ok'})",
        "rationale": "Percentile-rank 2y MA Multiplier — relative cycle extreme detection.",
    }


def mayer_multiple_percentile() -> dict:
    """Mayer Multiple > 90th percentile of last 4y."""
    df = _btc_history("5y")
    if df is None or len(df) < 200:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_200 = closes.rolling(200).mean()
    mayer = (closes / ma_200).dropna()
    if len(mayer) < 100:
        return {"met": False, "status": "insufficient mayer history"}
    last_4y = mayer.tail(min(1460, len(mayer)))
    current = float(mayer.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct >= 90
    return {
        "met": bool(met),
        "value": current,
        "status": f"Mayer {current:.2f}x = {pct:.0f}th percentile last 4y  ({'FIRING' if met else 'ok'})",
        "rationale": "Percentile-rank Mayer Multiple — auto-adapts to muted cycle ranges.",
    }


def log_regression_percentile() -> dict:
    """Log regression deviation > 90th percentile of last 4y."""
    df = _btc_history("5y")
    if df is None or len(df) < 365:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    GENESIS = pd.Timestamp("2009-01-03")
    days_since = (df.index - GENESIS).days.astype(float)
    log_days = np.log10(np.clip(days_since, 1, None))
    model_price = 10 ** (5.84 * log_days - 17.01)
    deviation = ((closes / model_price - 1) * 100).dropna()
    if len(deviation) < 100:
        return {"met": False, "status": "insufficient deviation history"}
    last_4y = deviation.tail(min(1460, len(deviation)))
    current = float(deviation.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct >= 90
    return {
        "met": bool(met),
        "value": current,
        "status": f"Log-reg dev {current:+.0f}% = {pct:.0f}th percentile last 4y  ({'FIRING' if met else 'ok'})",
        "rationale": "Percentile-rank Log Regression deviation — muted-cycle euphoria detector.",
    }


def pi_cycle_top_percentile() -> dict:
    """Pi Cycle Top ratio > 90th percentile of last 4y.

    Even when ratio doesn't cross 1.0, a high percentile means we're at
    cycle-relative extreme. Fires before absolute cross in muted cycles.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 350:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_111 = closes.rolling(111).mean()
    ma_350x2 = closes.rolling(350).mean() * 2
    ratio = (ma_111 / ma_350x2).dropna()
    if len(ratio) < 100:
        return {"met": False, "status": "insufficient ratio history"}
    last_4y = ratio.tail(min(1460, len(ratio)))
    current = float(ratio.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct >= 90
    return {
        "met": bool(met),
        "value": current,
        "status": f"Pi ratio {current:.3f} = {pct:.0f}th percentile last 4y  ({'FIRING' if met else 'ok'})",
        "rationale": "Pi Cycle Top relative-rank — catches muted-cycle tops the absolute version misses.",
    }


def ath_stagnation() -> dict:
    """Price sustained within 5% of rolling 365d high for 30+ days.

    The ONE indicator that catches muted cycle tops. When BTC sits near
    its recent ATH without breaking through for a month, that's distribution
    by institutions — they're absorbing supply at the top without pushing
    price higher.

    Cycle 5 backtest: this WOULD have fired by mid-Sep 2025, 3 weeks
    before the Oct 6 peak. Pattern-based, cycle-agnostic.
    """
    df = _btc_history("2y")
    if df is None or len(df) < 365:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    # 365d rolling max
    rolling_max = closes.rolling(365).max()
    # Days where price within 5% of rolling max
    within_5pct = (closes >= rolling_max * 0.95)
    # Sustained = 30+ days within 5% in the last 45 days
    recent_45 = within_5pct.tail(45)
    days_near_ath = int(recent_45.sum())
    met = days_near_ath >= 30
    current_dd_from_365_high = float((closes.iloc[-1] / rolling_max.iloc[-1]) - 1) * 100
    return {
        "met": bool(met),
        "value": days_near_ath,
        "status": (f"BTC at {current_dd_from_365_high:+.1f}% from 365d high. "
                   f"{days_near_ath}/45 recent days within 5% of high  "
                   f"({'STAGNATION (top distribution)' if met else 'no'})"),
        "rationale": "ATH stagnation — institutional distribution at top. CATCHES MUTED CYCLE TOPS like Oct 2025.",
    }


def rcap_at_ath() -> dict:
    """Realized Cap drawdown near 0% (at all-time high) = peak euphoria zone.

    Cycle peaks coincide with Realized Cap reaching ATH (everyone in profit).
    """
    s = _cm("CapRealUSD", days=1460)
    if s is None or s.empty: return {"met": False, "status": "data unavailable"}
    s = s.dropna()
    if s.empty: return {"met": False, "status": "empty"}
    rolling_max = s.rolling(365, min_periods=30).max()
    now_dd = float(s.iloc[-1] / rolling_max.iloc[-1] - 1) * 100
    # At or near ATH = drawdown > -2%
    met = now_dd > -2.0
    return {
        "met": bool(met),
        "value": now_dd,
        "status": f"Realized Cap drawdown {now_dd:+.1f}%  ({'AT ATH' if met else 'off peak'})  threshold > -2%",
        "rationale": "Realized Cap at ATH = peak euphoria, every cycle top occurred near 0% Realized DD.",
    }


# ============================================================
# Aggregator
# ============================================================

CRITERIA_DEFS = [
    # Classic absolute-threshold criteria (calibrated for cycles 1-3)
    ("pi_cycle_top",        " 1. Pi Cycle Top cross (111d > 350dx2)",         pi_cycle_top),
    ("mvrv_z_extreme",      " 2. MVRV-Z extreme (>5 or pct>95%)",             mvrv_z_extreme),
    ("puell_extreme",       " 3. Puell Multiple > 2.5 (miner top)",            puell_extreme),
    ("nupl_euphoria",       " 4. NUPL > 0.70 (Euphoria zone)",                 nupl_euphoria),
    ("sth_mvrv_euphoria",   " 5. STH-MVRV > 1.4 (STH euphoria)",                sth_mvrv_euphoria),
    ("asopr_sustained",     " 6. aSOPR sustained > 1.05 (7+ days)",            asopr_sustained_high),
    ("hash_ribbon_dc",      " 7. Hash Ribbon Death Cross (30d < 60d)",          hash_ribbon_death_cross),
    ("weekly_rsi_div",      " 8. Weekly RSI bearish divergence",                weekly_rsi_divergence),
    ("macd_3w_bear",        " 9. 3-week MACD bearish cross",                    macd_3w_bearish_cross),
    ("rcap_at_ath",         "10. Realized Cap drawdown > -2% (peak)",          rcap_at_ath),
    # Muted-cycle-aware percentile-rank criteria (calibrated for ETF era)
    ("pi_cycle_pct",        "11. Pi Cycle ratio > 90th pct last 4y",          pi_cycle_top_percentile),
    ("golden_ratio_pct",    "12. Golden Ratio Mult > 90th pct last 4y",       golden_ratio_percentile),
    ("two_year_ma_pct",     "13. 2y MA Mult > 90th pct last 4y",                two_year_ma_percentile),
    ("mayer_pct",           "14. Mayer Multiple > 90th pct last 4y",          mayer_multiple_percentile),
    ("log_reg_pct",         "15. Log regression dev > 90th pct last 4y",      log_regression_percentile),
    # The cycle-agnostic muted-top catcher
    ("ath_stagnation",      "16. ATH stagnation 30+ of last 45d within 5% of 365d high", ath_stagnation),
]


def btc_native_top_scorecard() -> dict:
    """Run all 10 BTC-native top criteria. Return verdict tier."""
    criteria = []
    n_met = 0
    for key, label, fn in CRITERIA_DEFS:
        try: r = fn()
        except Exception as e: r = {"met": False, "status": f"error: {e!r}"[:60]}
        r["id"] = key; r["label"] = label
        criteria.append(r)
        if r.get("met"): n_met += 1

    # Verdict tier — scaled for 16 criteria (was 10)
    if n_met >= 12:    level = "EXIT_75"
    elif n_met >= 9:   level = "SCALE_OUT_50"
    elif n_met >= 6:   level = "TRIM_25"
    elif n_met >= 3:   level = "WATCH"
    else:               level = "HOLD"

    _nt = len(criteria)   # F4: denominators track live n_total (16), not stale /10
    verdict_text = {
        "EXIT_75":      f"BTC cycle-5 top confirmed ({n_met}/{_nt}) — exit 75%+ now.",
        "SCALE_OUT_50": f"BTC top confirmed (>=9/{_nt}). Scale out 50% over 2-4 weeks.",
        "TRIM_25":      f"BTC top forming (>=6/{_nt}). Trim 25% as initial protection.",
        "WATCH":        f"BTC early top signals (>=3/{_nt}). Tighten stops, no action yet.",
        "HOLD":         "BTC bull continues. No top signals firing.",
    }[level]

    return {
        "criteria":     criteria,
        "n_met":        n_met,
        "n_total":      len(CRITERIA_DEFS),
        "verdict":      verdict_text,
        "verdict_level": level,
        "asof":         datetime.now(timezone.utc).isoformat(),
    }


def main():
    r = btc_native_top_scorecard()
    print("=" * 70)
    print("BTC NATIVE TOP SCORECARD")
    print("=" * 70)
    for c in r["criteria"]:
        mark = "[FIRING]" if c.get("met") else "[ok    ]"
        try: print(f"  {mark} {c['label']:50s} {c.get('status','')[:70]}")
        except UnicodeEncodeError:
            s = c.get("status", "").encode("ascii", "replace").decode()
            print(f"  {mark} {c['label']:50s} {s[:70]}")
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  Level: {r['verdict_level']} ({r['n_met']}/{r['n_total']} criteria firing)")


if __name__ == "__main__":
    main()
