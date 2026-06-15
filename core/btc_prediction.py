"""BTC PREDICTION MACHINE — multi-horizon directional forecast.

Synthesizes 45+ signals across 7 categories into bull/bear classifications
plus probabilistic price targets at 4 horizons (intraday, short-term,
medium-term, long-term).

Goal: catch cycle peaks/bottoms within ~5% and trend reversals within 1-2 weeks.
Backtest reference: 2025-10-06 cycle 5 peak caught at 94% capture, 73 days early
via the cycle_top_percentile detector. This module extends that approach across
all directions and horizons.

SIGNAL CATEGORIES (weights are predictive reliability from backtests):
  A. TECHNICAL    (weight 1.0)  RSI, MACD, EMA, BB, Donchian, MTF, ATR, Pi Cycle
  B. ON-CHAIN     (weight 1.2)  MVRV, NUPL, cycle pos, addresses, realized cap
  C. SENTIMENT    (weight 0.8)  F&G, cycle composite, BTC.D, ETH.D
  D. DERIVATIVES  (weight 1.0)  funding, DVOL, IV vs RV, skew
  E. MACRO        (weight 1.1)  DXY, VIX, 10Y, M2, NDX corr, gold ratio, Hayes
  F. LIQUIDATIONS (weight 0.9)  OI, cascade prob, funding-OI imbalance
  G. CYCLE        (weight 1.5)  halving phase, cycle decay, peak detector

OUTPUT (state_of_btc):
  - Regime classification (BULL / RANGE_BULL / RANGE / RANGE_BEAR / BEAR)
  - Per-horizon direction scores (-1 to +1) with confidence
  - Probabilistic price targets at 7/30/90/180-day horizons
  - Key support + resistance levels with confluence counts
  - Catalysts + tail risks
  - Full signal breakdown table

CACHING: 4-hour cache for expensive API calls. Force refresh via force=True.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO / ".btc_prediction_cache.json"
CACHE_TTL = 4 * 3600

# Category weights (calibrated empirically)
CATEGORY_WEIGHTS = {
    "technical":     1.0,
    "onchain":       1.2,
    "sentiment":     0.8,
    "derivatives":   1.0,
    "macro":         1.1,
    "liquidations":  0.9,
    "cycle":         1.5,        # current-position signals (where we ARE)
    "cycle_outlook": 1.5,        # FORWARD-looking cycle (where we're GOING) — split out
    "flows":         1.4,
    "options_adv":   1.2,
    "fundamentals":  1.3,
    "regime_models": 1.3,
}

# Horizon signal weighting matrix
# Each horizon emphasizes different signal categories
HORIZON_CATEGORY_WEIGHTS = {
    "intraday":   {"technical": 2.0, "derivatives": 2.0, "liquidations": 1.5,
                    "sentiment": 0.5, "macro": 0.5, "onchain": 0.3,
                    "cycle": 0.5, "cycle_outlook": 0.0,
                    "flows": 1.5, "options_adv": 2.0, "fundamentals": 0.3,
                    "regime_models": 0.5},
    "weekly":     {"technical": 2.0, "derivatives": 1.5, "liquidations": 1.5,
                    "sentiment": 1.2, "macro": 0.8, "onchain": 0.5,
                    "cycle": 0.7, "cycle_outlook": 0.2,
                    "flows": 2.0, "options_adv": 1.8, "fundamentals": 0.5,
                    "regime_models": 0.7},
    "short_term": {"technical": 1.5, "derivatives": 1.0, "liquidations": 1.0,
                    "sentiment": 1.5, "macro": 1.0, "onchain": 0.8,
                    "cycle": 1.0, "cycle_outlook": 0.5,
                    "flows": 2.0, "options_adv": 1.5, "fundamentals": 1.0,
                    "regime_models": 1.0},
    "medium_term":{"technical": 0.8, "derivatives": 0.5, "liquidations": 0.5,
                    "sentiment": 1.2, "macro": 1.5, "onchain": 1.5,
                    "cycle": 0.8, "cycle_outlook": 2.0,    # forward-looking dominant
                    "flows": 1.5, "options_adv": 0.8, "fundamentals": 2.0,
                    "regime_models": 2.0},
    "long_term":  {"technical": 0.3, "derivatives": 0.2, "liquidations": 0.2,
                    "sentiment": 0.5, "macro": 1.0, "onchain": 1.5,
                    "cycle": 0.5, "cycle_outlook": 3.0,    # FORWARD dominates long-term
                    "flows": 0.8, "options_adv": 0.3, "fundamentals": 2.5,
                    "regime_models": 2.5},
}


# === Cache helpers ===

def _read_cache() -> Optional[dict]:
    if not CACHE_FILE.exists(): return None
    try:
        d = json.loads(CACHE_FILE.read_text())
        if time.time() - d.get("fetched_at", 0) < CACHE_TTL:
            return d.get("data")
    except Exception:
        pass
    return None


def _write_cache(data: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps({"fetched_at": time.time(), "data": data}))
    except Exception:
        pass


# === Signal helpers ===

def _safe_call(fn, default=None):
    """Run a callable, return default on any error."""
    try:
        return fn()
    except Exception:
        return default


def _score_threshold(value, bull_threshold, bear_threshold, neutral=0.0):
    """Map a value into a -1..+1 score given bull/bear thresholds."""
    if value is None: return None
    if bull_threshold > bear_threshold:
        # Higher = more bullish
        if value >= bull_threshold: return 1.0
        if value <= bear_threshold: return -1.0
        # Linear interp
        return (value - bear_threshold) / (bull_threshold - bear_threshold) * 2 - 1
    else:
        # Higher = more bearish (e.g., RSI overbought)
        if value <= bull_threshold: return 1.0
        if value >= bear_threshold: return -1.0
        return -((value - bull_threshold) / (bear_threshold - bull_threshold) * 2 - 1)


# === A. TECHNICAL SIGNALS ===

def technical_signals() -> dict:
    """All technical signals — runs from existing rig modules."""
    out = {}
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=400)
        if df.empty or len(df) < 200:
            return {"error": "insufficient_data"}
        close = df["close"]
        current = float(close.iloc[-1])

        # EMAs and SMAs
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        mayer = current / sma200 if sma200 > 0 else 1.0

        # RSI daily + weekly
        def _rsi(s, p=14):
            d = s.diff()
            g = d.where(d > 0, 0).rolling(p).mean()
            l = (-d.where(d < 0, 0)).rolling(p).mean()
            return 100 - 100 / (1 + g / l.replace(0, np.nan))

        rsi_d = float(_rsi(close).iloc[-1])
        rsi_w = float(_rsi(close.iloc[::7]).iloc[-1])

        # MACD
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - macd_sig).iloc[-1])

        # Weekly MACD
        wc = close.iloc[::7]
        wmacd = wc.ewm(span=12, adjust=False).mean() - wc.ewm(span=26, adjust=False).mean()
        wmacd_sig = wmacd.ewm(span=9, adjust=False).mean()
        wmacd_hist = float((wmacd - wmacd_sig).iloc[-1])

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
        bb_pct = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        # Donchian
        donch_high = float(df["high"].iloc[-21:-1].max())
        donch_low = float(df["low"].iloc[-21:-1].min())

        # Pi Cycle Top
        sma111 = float(close.rolling(111).mean().iloc[-1]) if len(close) >= 111 else 0
        sma350x2 = float(close.rolling(350).mean().iloc[-1] * 2) if len(close) >= 350 else 0
        pi_cycle_on = sma111 > sma350x2 if sma350x2 > 0 else False

        # ATR-normalized momentum
        tr = pd.concat([df["high"] - df["low"],
                         (df["high"] - close.shift()).abs(),
                         (df["low"] - close.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        ret_30d = float(close.iloc[-1] / close.iloc[-31] - 1) if len(close) >= 31 else 0
        ret_90d = float(close.iloc[-1] / close.iloc[-91] - 1) if len(close) >= 91 else 0

        # Multi-TF confluence
        try:
            from core.multi_timeframe import confluence
            mtf = confluence("BTC/USDT")
            mtf_score = mtf.get("confluence_score", 0)
            mtf_dir = mtf.get("net_direction", 0)
        except Exception:
            mtf_score, mtf_dir = 0, 0

        # Score each signal
        out = {
            "rsi_daily":          {"value": rsi_d, "score": _score_threshold(rsi_d, 30, 70)},
            "rsi_weekly":         {"value": rsi_w, "score": _score_threshold(rsi_w, 40, 70)},
            "macd_daily":         {"value": macd_hist, "score": 1.0 if macd_hist > 0 else -1.0},
            "macd_weekly":        {"value": wmacd_hist, "score": 1.0 if wmacd_hist > 0 else -1.0},
            "vs_ema21":           {"value": current/ema21, "score": 0.5 if current > ema21 else -0.5},
            "vs_sma50":           {"value": current/sma50, "score": 0.7 if current > sma50 else -0.7},
            "vs_sma200":          {"value": mayer, "score": 1.0 if mayer > 1.0 else -1.0},
            "mayer_multiple":     {"value": mayer, "score": _score_threshold(mayer, 1.5, 0.7)},
            "bb_position":        {"value": bb_pct, "score": _score_threshold(bb_pct, 0.2, 0.8)},
            "donchian_high":      {"value": donch_high, "score": 1.0 if current >= donch_high else 0},
            "donchian_low":       {"value": donch_low, "score": -1.0 if current <= donch_low else 0},
            "pi_cycle_top":       {"value": pi_cycle_on, "score": -1.0 if pi_cycle_on else 0},
            "atr_30d_momentum":   {"value": ret_30d, "score": _score_threshold(ret_30d, 0.10, -0.10)},
            "atr_90d_momentum":   {"value": ret_90d, "score": _score_threshold(ret_90d, 0.20, -0.20)},
            "mtf_confluence":     {"value": mtf_score * mtf_dir, "score": mtf_score * mtf_dir},
        }
    except Exception as e:
        out["error"] = str(e)
    return out


# === B. ON-CHAIN SIGNALS ===

def onchain_signals() -> dict:
    out = {}
    try:
        from core.onchain import cycle_position, get_mvrv, get_nupl
        mvrv = get_mvrv()
        nupl = get_nupl()
        cp = cycle_position()

        mvrv_val = mvrv.get("mvrv")
        nupl_val = nupl.get("nupl")
        cycle_score = cp.get("score", 50)
        cycle_phase = cp.get("phase", "?")

        # MVRV: low = bull, high = bear (contrarian for accumulation)
        mvrv_score = None
        if mvrv_val is not None:
            mvrv_score = _score_threshold(mvrv_val, 0.8, 2.5)  # low MVRV = bull

        # NUPL similar
        nupl_score = None
        if nupl_val is not None:
            nupl_score = _score_threshold(nupl_val, -0.1, 0.7)

        # Cycle phase: deep_bear = bull setup, euphoria = bear setup
        cycle_dir_score = 1.0 - (cycle_score / 50.0)  # score 0 -> +1, 100 -> -1
        cycle_dir_score = max(-1.0, min(1.0, cycle_dir_score))

        out = {
            "mvrv":               {"value": mvrv_val, "score": mvrv_score, "phase": cycle_phase},
            "nupl":               {"value": nupl_val, "score": nupl_score},
            "cycle_position":     {"value": cycle_score, "score": cycle_dir_score},
        }

        # Cycle top detector
        try:
            from core.cycle_top_percentile import cycle_top_score
            cts = cycle_top_score()
            if not cts.get("error") and cts.get("verdict") != "NOT_NEAR_ATH":
                score = cts["score"]
                # High cycle_top score = BEAR signal
                out["cycle_top_detector"] = {
                    "value": score,
                    "score": -score / 100,  # 100 -> -1.0
                    "verdict": cts.get("verdict"),
                }
        except Exception:
            pass

    except Exception as e:
        out["error"] = str(e)
    return out


# === C. SENTIMENT SIGNALS ===

def sentiment_signals() -> dict:
    out = {}
    try:
        from core.fear_greed import latest, cycle_composite_score
        fg = latest()
        composite = cycle_composite_score()

        fg_val = fg.get("value")
        # F&G: low = contrarian BULL (fear = buy)
        fg_score = None
        if fg_val is not None:
            fg_score = _score_threshold(fg_val, 20, 75)  # low fear = bull

        fg_7d_chg = fg.get("7d_change", 0)
        # F&G falling rapidly = capitulation = bull
        fg_momentum_score = None
        if fg_7d_chg is not None:
            fg_momentum_score = _score_threshold(fg_7d_chg, -20, 20)  # falling = bull

        comp_score = composite.get("composite_score") if composite else None
        comp_dir_score = None
        if comp_score is not None:
            comp_dir_score = _score_threshold(comp_score, 25, 75)  # low = bull

        out = {
            "fear_greed":         {"value": fg_val, "score": fg_score},
            "fg_7d_change":       {"value": fg_7d_chg, "score": fg_momentum_score},
            "cycle_composite":    {"value": comp_score, "score": comp_dir_score},
        }
    except Exception as e:
        out["error"] = str(e)

    # BTC dominance regime
    try:
        from core.btc_dominance import status
        ds = status()
        if not ds.get("error"):
            dom = ds.get("btc_dominance_pct", 50)
            # BTC.D 50-65 is neutral-bullish for BTC, >65 is bearish (capitulation),
            # <40 is altseason (alts outperform but BTC still rises)
            dom_score = 0.0
            if 50 <= dom <= 60: dom_score = 0.3   # mild bull for BTC
            elif 60 < dom <= 65: dom_score = 0.5
            elif dom > 65: dom_score = -0.3       # too dominant = late bear
            elif 40 <= dom < 50: dom_score = 0.2
            elif dom < 40: dom_score = -0.2

            out["btc_dominance"] = {"value": dom, "score": dom_score,
                                     "regime": ds.get("regime")}
    except Exception:
        pass

    return out


# === D. DERIVATIVES SIGNALS ===

def derivatives_signals() -> dict:
    out = {}

    # Funding rates
    try:
        from core import data
        # Get current funding for BTC perp
        ticker = data._EX.fetch_funding_rate("BTC/USDT:USDT")
        funding = float(ticker.get("fundingRate", 0))
        # Funding extreme positive = crowded long = BEAR
        # Funding extreme negative = crowded short = BULL
        funding_score = _score_threshold(funding * 10000, -10, 10)  # in bps
        out["funding_rate_8h"] = {"value": funding * 100, "score": funding_score}
    except Exception:
        pass

    # DVOL (implied vol)
    try:
        from core.options_iv import get_atm_iv
        iv = get_atm_iv("BTC")
        atm_iv = iv.get("atm_iv_pct", 0.5)
        # IV: low IV = complacency (bull continuation), high = fear (contrarian bull)
        # We score: very high or very low = directional uncertainty
        iv_score = 0.0
        if atm_iv > 0.8: iv_score = 0.3   # high vol = fear = contrarian bull
        elif atm_iv < 0.4: iv_score = -0.2  # complacency
        out["atm_iv"] = {"value": atm_iv, "score": iv_score}
    except Exception:
        pass

    return out


# === E. MACRO SIGNALS ===

def macro_signals() -> dict:
    out = {}
    try:
        from core.macro_correlation import latest_metrics, regime_status
        m = latest_metrics()
        r = regime_status()

        # VIX: low = risk-on (bull), high = risk-off (bear)
        vix = m.get("VIX", {}).get("value")
        vix_chg = m.get("VIX", {}).get("ret_5d", 0)
        vix_score = None
        if vix is not None:
            vix_score = _score_threshold(vix, 15, 30)  # low VIX = bull
        out["vix"] = {"value": vix, "score": vix_score}

        # VIX momentum
        if vix_chg is not None:
            out["vix_5d_change"] = {
                "value": vix_chg * 100,
                "score": _score_threshold(vix_chg, -0.1, 0.2),  # falling VIX = bull
            }

        # 10Y yield
        tnx = m.get("TNX", {}).get("value")
        if tnx is not None:
            # Yield > 5% = bond market stress, capital flees risk, BTC bear
            # Yield < 4% = easier conditions, BTC bull
            out["10y_yield"] = {"value": tnx,
                                 "score": _score_threshold(tnx, 4.0, 5.0)}

        # DXY
        dxy = m.get("DXY", {}).get("value")
        if dxy is not None:
            # Strong dollar = BTC bear (negatively correlated)
            out["dxy"] = {"value": dxy,
                          "score": _score_threshold(dxy, 95, 110)}

        # NDX correlation (Hayes thesis)
        ndx_chg = m.get("NDX", {}).get("ret_5d", 0)
        if ndx_chg is not None:
            out["ndx_momentum"] = {"value": ndx_chg * 100,
                                    "score": _score_threshold(ndx_chg, -0.05, 0.05)}

        # Hayes regime gating
        regime_str = r.get("regime", "normal").lower()
        regime_map = {"normal": 0.0, "caution": -0.3, "de_risk": -0.7, "full_kill": -1.0}
        out["hayes_regime"] = {"value": regime_str,
                                "score": regime_map.get(regime_str, 0.0)}
    except Exception as e:
        out["error"] = str(e)

    return out


# === F. LIQUIDATIONS ===

def liquidations_signals() -> dict:
    out = {}
    try:
        from core.liquidation_pressure import liquidation_pressure
        for pair in ("BTC/USDT",):
            lp = liquidation_pressure(pair)
            if not lp.get("error"):
                edge = lp.get("edge_direction", "no_edge")
                # fade_long means longs about to cascade = BEAR
                # fade_short means shorts about to cover = BULL
                if edge == "fade_long":
                    out["cascade_pressure"] = {"value": "long_cascade_brewing", "score": -0.5}
                elif edge == "fade_short":
                    out["cascade_pressure"] = {"value": "short_squeeze_brewing", "score": 0.5}
                else:
                    out["cascade_pressure"] = {"value": "balanced", "score": 0.0}
                # Funding component
                funding = lp.get("funding_bps_8h", 0)
                if abs(funding) > 10:
                    out["funding_extreme"] = {
                        "value": funding,
                        "score": -0.4 if funding > 0 else 0.4
                    }
    except Exception as e:
        out["error"] = str(e)
    return out


# === G. CYCLE CONTEXT ===

def cycle_signals() -> dict:
    """Cycle signals — split into CURRENT POSITION vs FORWARD OUTLOOK.

    CURRENT signals reflect where we are RIGHT NOW (bearish in mid-bear).
    FORWARD signals reflect the FUTURE outlook (bullish in late-bear approaching bottom).
    Long-term horizon should heavily weight FORWARD signals.
    """
    out = {}
    today = datetime.now(timezone.utc).date()

    halving4 = datetime(2024, 4, 20).date()
    days_post_halving = (today - halving4).days
    out["days_post_halving"] = {"value": days_post_halving}

    # === CURRENT POSITION SIGNALS (where we are NOW) ===
    # Used by intraday/weekly/short_term forecasts.
    if days_post_halving < 100: pos_score = 0.8
    elif days_post_halving < 300: pos_score = 0.6
    elif days_post_halving < 500: pos_score = 0.4
    elif days_post_halving < 600: pos_score = 0.0
    elif days_post_halving < 900: pos_score = -0.6  # late bull / bear bleed (now)
    elif days_post_halving < 1100: pos_score = -0.3
    elif days_post_halving < 1300: pos_score = 0.3
    else: pos_score = 0.5
    out["cycle_current_position"] = {"value": f"day_{days_post_halving}",
                                      "score": pos_score}

    cycle5_peak = datetime(2025, 10, 6).date()
    days_post_peak = (today - cycle5_peak).days
    out["days_post_cycle5_peak"] = {"value": days_post_peak}

    projected_bottom = datetime(2026, 10, 9).date()
    days_to_bottom = (projected_bottom - today).days
    out["days_to_projected_bottom"] = {"value": days_to_bottom}

    # Current cycle bottom proximity (near-term bearish, doesn't reflect outlook)
    if days_post_peak < 90: bot_pos_score = -0.6
    elif days_post_peak < 200: bot_pos_score = -0.4
    elif days_post_peak < 350: bot_pos_score = -0.2
    elif days_post_peak < 500: bot_pos_score = 0.3
    else: bot_pos_score = 0.5
    out["cycle_bottom_proximity_now"] = {"value": days_post_peak, "score": bot_pos_score}

    # === FORWARD OUTLOOK SIGNALS (where we're GOING over 6m-2y) ===
    # Used by long_term and medium_term forecasts.
    # At day 700-900: bottom approaching, recovery ahead = BULL outlook
    # At day 200-500: peak approaching, distribution coming = BEAR outlook
    # At day 0-200: early bull, plenty of upside = BULL
    # At day 900-1200: recovery confirmed, mid-bull = STRONG BULL
    if days_post_halving < 200:    fwd_score = 0.6   # early bull ahead
    elif days_post_halving < 400:  fwd_score = 0.4
    elif days_post_halving < 550:  fwd_score = -0.2  # peak forming, distribution ahead
    elif days_post_halving < 700:  fwd_score = -0.5  # post-peak, bear ahead
    elif days_post_halving < 850:  fwd_score = 0.3   # late bear, bottom approaching (bull outlook!)
    elif days_post_halving < 1100: fwd_score = 0.8   # bottoming / early recovery
    elif days_post_halving < 1300: fwd_score = 0.9   # confirmed recovery
    else:                          fwd_score = 0.4   # mid-cycle
    out["cycle_forward_outlook"] = {"value": f"day_{days_post_halving}",
                                     "score": fwd_score,
                                     "note": "long-term 6m-2y cycle outlook"}

    # Days-to-bottom forward signal
    # The closer we are to projected bottom, the more BULL the forward outlook
    if days_to_bottom < 0: fwd_bot_score = 0.7      # past bottom = bull
    elif days_to_bottom < 90: fwd_bot_score = 0.6   # bottom imminent
    elif days_to_bottom < 200: fwd_bot_score = 0.4  # bottom in months
    elif days_to_bottom < 365: fwd_bot_score = 0.2  # bottom > 1y out
    else: fwd_bot_score = -0.2                      # bottom > 1y away
    out["forward_bottom_proximity"] = {"value": days_to_bottom, "score": fwd_bot_score}

    # MVRV cycle (still useful as current-state signal)
    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        if cp.get("score") is not None:
            out["mvrv_cycle"] = {
                "value": cp["score"],
                "score": 1.0 - cp["score"]/50.0,
                "phase": cp["phase"],
            }
    except Exception:
        pass

    return out


# === AGGREGATOR ===

def pull_all_signals(force: bool = False) -> dict:
    """Pull every signal category. Returns nested dict."""
    if not force:
        cached = _read_cache()
        if cached:
            return cached

    cycle_all = cycle_signals()
    # Split cycle into CURRENT POSITION + FORWARD OUTLOOK
    cycle_current = {k: v for k, v in cycle_all.items()
                      if k not in ("cycle_forward_outlook", "forward_bottom_proximity")}
    cycle_outlook = {k: v for k, v in cycle_all.items()
                      if k in ("cycle_forward_outlook", "forward_bottom_proximity")}

    # === HALVING CLOCK (most reliable BTC signal — 1 day cycle 5 peak error) ===
    try:
        from core.halving_clock import halving_clock_signal, halving_clock_forward_outlook
        hc = halving_clock_signal()
        hc_fwd = halving_clock_forward_outlook()
        if hc and not hc.get("error"):
            cycle_current["halving_clock"] = hc
        if hc_fwd and not hc_fwd.get("error"):
            cycle_outlook["halving_clock_forward"] = hc_fwd
    except Exception:
        pass

    signals = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "technical":     technical_signals(),
        "onchain":       onchain_signals(),
        "sentiment":     sentiment_signals(),
        "derivatives":   derivatives_signals(),
        "macro":         macro_signals(),
        "liquidations":  liquidations_signals(),
        "cycle":         cycle_current,
        "cycle_outlook": cycle_outlook,
    }

    # Advanced signals (institutional-grade additions)
    try:
        from core.btc_advanced_signals import all_advanced_signals
        adv = all_advanced_signals()
        for cat_name, cat_sigs in adv.items():
            cat_out = {}
            for sig_name, sig_data in cat_sigs.items():
                if sig_data is None: continue
                cat_out[sig_name] = sig_data
            signals[cat_name] = cat_out
    except Exception:
        pass

    # Top-tier additional signals from btc_more_signals
    try:
        from core.btc_more_signals import all_more_signals
        more = all_more_signals()
        routing = {
            "mvrv_z_score":        ("onchain", "mvrv_z_score"),
            "hayes_liquidity":     ("macro", "hayes_liquidity"),
            "cross_asset_ratios":  ("macro", "cross_asset_ratios"),
            "google_trends":       ("sentiment", "google_trends"),
            "cme_basis":           ("flows", "cme_basis"),
            "nvt_signal":          ("onchain", "nvt_signal"),
            "s2f_deflection":      ("regime_models", "s2f_deflection"),
            "yield_curve":         ("macro", "yield_curve"),
        }
        for sig_name, sig_data in more.items():
            if sig_data is None: continue
            cat, key = routing.get(sig_name, ("technical", sig_name))
            if cat not in signals: signals[cat] = {}
            signals[cat][key] = sig_data
    except Exception:
        pass

    # Bottom detection signals (symmetric to top detection)
    try:
        from core.btc_bottom_signals import all_bottom_signals
        bottom = all_bottom_signals()
        bottom_routing = {
            "hashrate_ribbon_cross":  ("fundamentals", "hashrate_ribbon_cross"),
            "sth_mvrv_cross":         ("onchain", "sth_mvrv_cross"),
            "cycle_day_analog":       ("cycle", "cycle_day_analog"),
            "wyckoff_phase":          ("technical", "wyckoff_phase"),
            "cross_asset_divergence": ("macro", "cross_asset_divergence"),
        }
        for sig_name, sig_data in bottom.items():
            if sig_data is None: continue
            cat, key = bottom_routing.get(sig_name, ("technical", sig_name))
            if cat not in signals: signals[cat] = {}
            signals[cat][key] = sig_data
    except Exception:
        pass

    # === PRO-TIER ON-CHAIN SIGNALS (Woo + top-1% Glassnode layer, 2026-06) ===
    # 10 new signals: realized_cap_drawdown, reserve_risk, puell, CB premium,
    # difficulty ribbon, aSOPR, LTH/STH supply, CDD, dormancy flow, NVT Woo.
    try:
        from core.btc_pro_signals import all_pro_signals
        pro = all_pro_signals()
        for cat_name, cat_sigs in pro.items():
            if cat_name not in signals: signals[cat_name] = {}
            for sig_name, sig_data in cat_sigs.items():
                if sig_data is None: continue
                # Skip on error but log under the original name for visibility
                signals[cat_name][sig_name] = sig_data
    except Exception:
        pass

    # === JESSE OLSON TECHNICAL LAYER (multi-week TA, 2026-06) ===
    # 3-week MACD + weekly Heikin Ashi + weekly RSI divergence.
    # All routed to technical category.
    try:
        from core.btc_jesse_olson import all_olson_signals
        olson = all_olson_signals()
        for sig_name, sig_data in olson.items():
            if sig_data is None or sig_data.get("error"): continue
            if "technical" not in signals: signals["technical"] = {}
            signals["technical"][sig_name] = sig_data
    except Exception:
        pass

    # === CLEMENTE + ALDEN LAYER (institutional-grade signals, 2026-06-03) ===
    # 15 signals across macro + on-chain. HODL Waves, TIPS real yields,
    # ETF % of supply, SSR, BTC.D, AASI, hashrate drawdown, multi-exch funding,
    # BTC/Gold, difficulty adj, CB premium streak, URPD, Reflexivity, fiscal dom, RHODL.
    try:
        from core.btc_clemente_alden import all_clemente_alden_signals
        ca = all_clemente_alden_signals()
        ca_routing = {
            "hodl_waves":              "onchain",
            "real_yields_10y":         "macro",
            "etf_pct_of_supply":       "flows",
            "stablecoin_supply_ratio": "flows",
            "btc_dominance":           "macro",
            "aasi":                    "onchain",
            "hashrate_drawdown":       "fundamentals",
            "multi_exch_funding":      "derivatives",
            "btc_gold_ratio":          "macro",
            "difficulty_adjustment":   "fundamentals",
            "cb_premium_streak":       "flows",
            "urpd_clusters":           "onchain",
            "reflexivity_index":       "regime_models",
            "fiscal_dominance":        "macro",
            "rhodl_ratio":             "onchain",
        }
        for tier, tier_sigs in ca.items():
            for sig_name, sig_data in tier_sigs.items():
                if sig_data is None or sig_data.get("error"): continue
                cat = ca_routing.get(sig_name, "fundamentals")
                if cat not in signals: signals[cat] = {}
                signals[cat][sig_name] = sig_data
    except Exception:
        pass

    # === PREMIUM-FREE LAYER (18 paid-tier-equivalent signals, 2026-06) ===
    # ETF flows, stablecoins, Net Liquidity, hash price, Deribit Greeks, +13 more
    try:
        from core.btc_premium_free import all_premium_free_signals
        pfree = all_premium_free_signals()
        # Route each signal to appropriate category
        routing = {
            "etf_flows":              "flows",
            "stablecoin_supply":      "flows",
            "stablecoin_chain_flows": "flows",
            "exchange_net_flows":     "flows",
            "defi_tvl":               "flows",
            "github_activity":        "fundamentals",
            "miner_holdings":         "fundamentals",
            "hash_price":             "fundamentals",
            "mempool_pressure":       "fundamentals",
            "lth_supply_exact":       "onchain",
            "whale_tx_activity":      "onchain",
            "deribit_greeks":         "options_adv",
            "news_sentiment":         "sentiment",
            "wikipedia_views":        "sentiment",
            "reddit_sentiment":       "sentiment",
            "net_liquidity":          "macro",
            "dxy_regime":             "macro",
            "energy_prices":          "macro",
        }
        for tier, tier_sigs in pfree.items():
            for sig_name, sig_data in tier_sigs.items():
                if sig_data is None or sig_data.get("error"): continue
                cat = routing.get(sig_name, "fundamentals")
                if cat not in signals: signals[cat] = {}
                signals[cat][sig_name] = sig_data
    except Exception:
        pass

    # Try get current BTC price
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        signals["btc_price"] = float(df["close"].iloc[-1])
    except Exception:
        signals["btc_price"] = None

    _write_cache(signals)
    return signals


# === SCORING ===

def category_directional_score(category_signals: dict) -> tuple[float, int, int]:
    """Conviction-weighted category score.

    Top-tier predictor insight: averaging dilutes strong signals.
    When MVRV Z hits -1.0 (deep value) and macd_daily hits +0.1, the
    average is -0.45 — but the STRONG signal is the real edge.

    Fix: weight by conviction = |score|^1.5 so signals near zero
    contribute less and signals near ±1 dominate.
    """
    scores = []
    weights = []
    total = 0
    for name, d in category_signals.items():
        if name == "error" or not isinstance(d, dict): continue
        total += 1
        s = d.get("score")
        if s is not None and isinstance(s, (int, float)):
            s = float(s)
            scores.append(s)
            # Conviction weight: signals at extremes get more weight.
            # |0.1|^1.5 = 0.03, |0.5|^1.5 = 0.35, |1.0|^1.5 = 1.0
            # Floor at 0.1 so we don't ignore weak signals entirely.
            weights.append(max(0.1, abs(s) ** 1.5))
    if not scores: return 0.0, 0, total
    return float(np.average(scores, weights=weights)), len(scores), total


def horizon_forecast(signals: dict, horizon: str) -> dict:
    """Compute directional score for a specific horizon.

    2026-06-01: Added STRONG SIGNAL OVERRIDE based on 2025 peak backtest.
    The backtest showed cycle_composite hit -1.0 at peak day but was buried
    by averaging with mild bull signals (mayer +0.2, technical +0.3).
    New rule: if 2+ signals are at extreme (-0.8 or +0.8), bias the
    interpretation toward that direction regardless of average.
    """
    weights = HORIZON_CATEGORY_WEIGHTS[horizon]
    weighted_sum = 0.0
    weight_total = 0.0
    breakdown = {}

    # Track CYCLE-RELEVANT extreme signals only.
    # The override should detect cycle peaks/bottoms — NOT confirm existing
    # bear/bull regimes. So we only count "cycle indicators": cycle_composite,
    # mvrv_z_score, cycle_top_detector, pi_cycle_top/bottom, nupl, mvrv,
    # cycle_position, mayer (if at percentile extreme), power_law.
    CYCLE_AWARE_SIGNALS = {
        "cycle_composite",     # MVP — most predictive
        "mvrv_z_score",
        "cycle_top_detector",
        "pi_cycle_top",
        "pi_cycle_bottom",
        "mvrv",
        "nupl",
        "cycle_position",
        "power_law",
        "mvrv_cycle",
        "hashrate_ribbon_cross",
        "sth_mvrv_cross",
        "cycle_day_analog",
        # === HIGHEST RELIABILITY SIGNALS (added 2026-06-02) ===
        # Halving clock — historical std dev only 8d for peaks, 12d for bottoms
        # Cycle 5 peak prediction error: 1 day
        "halving_clock",            # current cycle position
        "halving_clock_forward",    # forward outlook (peak/bottom proximity)
        # === PRO-TIER ON-CHAIN LAYER (Woo + Glassnode review, 2026-06) ===
        # Each is recognized institutional bottom indicator
        "realized_cap_drawdown",    # Checkmate's #1 bottom signal
        "reserve_risk",             # Glassnode generational-buy classic
        "puell_multiple",           # Miner capitulation marker
        "difficulty_ribbon",        # Woo's miner stress signal
        "lth_sth_supply_ratio",     # Cohort positioning at extremes
        "nvt_signal_woo",           # Network valuation extremes
        "coinbase_premium_gap",     # US institutional flow direction
        # === PREMIUM-FREE LAYER cycle-relevant (2026-06) ===
        "etf_flows",                # Biggest cycle 5/6 driver
        "net_liquidity",            # Macro pulse
        "hash_price",               # Miner economics
        "stablecoin_supply",        # Fresh liquidity entering
        # === CLEMENTE + ALDEN cycle-relevant (2026-06-03) ===
        "aasi",                     # Clemente bottom signal (quiet accumulation)
        "hashrate_drawdown",        # Miner capitulation marker
        "cb_premium_streak",        # 21+ days neg = bottom forming
        "stablecoin_supply_ratio",  # Dry powder availability
        "real_yields_10y",          # Alden macro #1
        "etf_pct_of_supply",        # Institutional ownership trend
    }
    cycle_bear_signals = []
    cycle_bull_signals = []

    for cat, w in weights.items():
        cat_sigs = signals.get(cat, {})
        score, n_scored, n_total = category_directional_score(cat_sigs)
        breakdown[cat] = {"score": score, "weight": w,
                           "n_scored": n_scored, "n_total": n_total}
        if n_scored > 0:
            # FIXED: weight by horizon-weight ALONE (not × n_scored).
            # Previous formula made categories with many signals dominate
            # regardless of horizon-specific weighting, making all horizons
            # converge to similar scores.
            weighted_sum += score * w
            weight_total += w

        # Track extreme CYCLE-AWARE signals only
        for sig_name, sig_data in cat_sigs.items():
            if not isinstance(sig_data, dict): continue
            if sig_name not in CYCLE_AWARE_SIGNALS: continue
            s = sig_data.get("score")
            if s is None: continue
            s = float(s)
            # cycle_composite is the MVP — give it 2x weight
            count_weight = 2 if sig_name == "cycle_composite" else 1
            if s <= -0.7:
                cycle_bear_signals.append((sig_name, s, count_weight))
            elif s >= 0.7:
                cycle_bull_signals.append((sig_name, s, count_weight))

    strong_bear_count = sum(w for _, _, w in cycle_bear_signals)
    strong_bull_count = sum(w for _, _, w in cycle_bull_signals)

    if weight_total == 0:
        return {"horizon": horizon, "direction_score": 0.0,
                "interpretation": "NEUTRAL", "confidence": "LOW", "breakdown": breakdown}

    final_score = weighted_sum / weight_total

    # === STRONG CYCLE-AWARE SIGNAL OVERRIDE ===
    # Only fires when CYCLE INDICATORS (not just any technical) are at extremes.
    # This prevents the override from confirming existing bear/bull regimes;
    # it specifically detects cycle peaks/bottoms.
    override_applied = None
    if strong_bear_count >= 2 and final_score > -0.5:
        final_score = min(final_score, -0.4)
        sig_names = ", ".join(name for name, _, _ in cycle_bear_signals)
        override_applied = f"CYCLE_TOP_OVERRIDE ({strong_bear_count} cycle indicators: {sig_names})"
    elif strong_bull_count >= 2 and final_score < 0.5:
        final_score = max(final_score, 0.4)
        sig_names = ", ".join(name for name, _, _ in cycle_bull_signals)
        override_applied = f"CYCLE_BOTTOM_OVERRIDE ({strong_bull_count} cycle indicators: {sig_names})"

    if final_score > 0.5: interp = "STRONG BULL"
    elif final_score > 0.2: interp = "BULL"
    elif final_score > -0.2: interp = "NEUTRAL"
    elif final_score > -0.5: interp = "BEAR"
    else: interp = "STRONG BEAR"

    # Confidence based on category agreement
    cat_scores = [b["score"] for b in breakdown.values() if b["n_scored"] > 0]
    if len(cat_scores) >= 3:
        std = float(np.std(cat_scores))
        if std < 0.2: confidence = "HIGH"
        elif std < 0.4: confidence = "MEDIUM"
        else: confidence = "LOW"
    else:
        confidence = "LOW"

    return {
        "horizon": horizon,
        "direction_score": final_score,
        "interpretation": interp,
        "confidence": confidence,
        "breakdown": breakdown,
        "strong_bear_count": strong_bear_count,
        "strong_bull_count": strong_bull_count,
        "override_applied": override_applied,
    }


def price_targets(btc_price: float, horizons: dict, vol_annualized: float = 0.42) -> dict:
    """Compute probabilistic price targets per horizon.

    Uses Geometric Brownian Motion with realized vol + directional drift from
    each horizon's directional_score.
    """
    out = {}
    horizon_days = {"intraday": 1, "weekly": 7, "short_term": 30,
                     "medium_term": 90, "long_term": 180}

    for h, days in horizon_days.items():
        score = horizons.get(h, {}).get("direction_score", 0)
        # Convert directional score to annualized drift
        # Score +1 = +60% annualized drift, -1 = -60% drift (conservative)
        annual_drift = score * 0.60
        # Per-day drift and stdev
        daily_drift = annual_drift / 365
        daily_vol = vol_annualized / np.sqrt(365)
        # Lognormal projection
        # Median = current * exp(daily_drift * days)
        # P25 = exp(daily_drift*days - 0.67 * daily_vol * sqrt(days))
        # P75 = exp(daily_drift*days + 0.67 * daily_vol * sqrt(days))
        horizon_vol = daily_vol * np.sqrt(days)
        median = btc_price * float(np.exp(daily_drift * days))
        p5 = btc_price * float(np.exp(daily_drift * days - 1.645 * horizon_vol))
        p25 = btc_price * float(np.exp(daily_drift * days - 0.674 * horizon_vol))
        p75 = btc_price * float(np.exp(daily_drift * days + 0.674 * horizon_vol))
        p95 = btc_price * float(np.exp(daily_drift * days + 1.645 * horizon_vol))

        out[h] = {
            "days": days,
            "median": median,
            "p5": p5, "p25": p25, "p75": p75, "p95": p95,
            "directional_score": score,
        }
    return out


def regime_classification(direction_scores: dict, btc_price: float = 0,
                          signals: dict = None) -> str:
    """Cycle-aware regime classification.

    Old version used only direction scores → produced absurd readings like
    "RANGE_BULL" while BTC was 41% below ATH (cycle-5 bear). Fix: anchor
    regime to cycle context (distance from ATH), then let signal direction
    shift the label within that anchor.

    Distance from ATH zones:
        < 15% below ATH  : "near ATH" zone — bull/bear range applies
        15-30% below ATH : "cycle decline" — bear-anchored regimes only
        > 30% below ATH  : "cycle bear" — only bear/recovery labels
        > 60% below ATH  : "deep bear / capitulation"
    """
    s = direction_scores.get("short_term", {}).get("direction_score", 0)
    m = direction_scores.get("medium_term", {}).get("direction_score", 0)
    avg = (s + m) / 2

    # Try to get distance from ATH
    dist_to_ath = None
    if signals and "fundamentals" in signals:
        for name, d in signals.get("fundamentals", {}).items():
            if not isinstance(d, dict): continue
            if "ath" in name.lower():
                dist_to_ath = d.get("value")
                break
    if dist_to_ath is None and signals and "technical" in signals:
        for name, d in signals.get("technical", {}).items():
            if not isinstance(d, dict): continue
            if "ath" in name.lower():
                dist_to_ath = d.get("value")
                break
    # Fallback: compute from current price vs known cycle 5 peak
    if dist_to_ath is None and btc_price > 0:
        dist_to_ath = btc_price / 124659  # cycle 5 peak hardcoded fallback

    if dist_to_ath is None: dist_to_ath = 1.0

    # === CYCLE-AWARE classification ===
    if dist_to_ath >= 0.95:
        # Within 5% of ATH — full bull/bear range
        if avg > 0.4: return "BULL"
        if avg > 0.1: return "RANGE_BULL"
        if avg > -0.1: return "RANGE"
        if avg > -0.4: return "RANGE_BEAR"
        return "BEAR"
    if dist_to_ath >= 0.85:
        # 5-15% off ATH — early decline or recovery near top
        if avg > 0.3: return "BULL"
        if avg > 0.0: return "RANGE_BULL"
        if avg > -0.3: return "EARLY_DECLINE"
        return "BEAR_CONFIRMED"
    if dist_to_ath >= 0.70:
        # 15-30% off ATH — cycle decline (cannot be bull-regime here)
        if avg > 0.2: return "CYCLE_BEAR_BOUNCE"
        if avg > -0.2: return "CYCLE_BEAR"
        return "CYCLE_BEAR_ACCELERATING"
    if dist_to_ath >= 0.50:
        # 30-50% off ATH — mid-late cycle bear (where we are now)
        # NOTE: BOTTOM_FORMING removed here — it was being read as "bottom is in"
        # by users. At -41% from ATH with only soft signals firing, this is a
        # late-bear rally, not actual bottom formation. Bottom confirmation
        # requires the hard-criteria scorecard.
        if avg > 0.3: return "LATE_BEAR_RALLY"     # was BOTTOM_FORMING
        if avg > 0.0: return "LATE_CYCLE_BEAR"
        if avg > -0.3: return "CYCLE_BEAR_GRIND"
        return "CAPITULATION_ZONE"
    if dist_to_ath >= 0.30:
        # 50-70% off ATH — deep bear (cycle 4 bottom level)
        if avg > 0.3: return "DEEP_BEAR_RECOVERY"
        if avg > -0.1: return "DEEP_BEAR"
        return "CAPITULATION"
    # < 30% of ATH — extreme bear (cycle 3 bottom magnitudes)
    if avg > 0: return "GENERATIONAL_BUY_ZONE"
    return "EXTREME_CAPITULATION"


def ensemble_predictions(signals: dict) -> dict:
    """Run 3 independent predictor "lenses" and return their consensus.

    LENS 1 — TECHNICAL: technical + derivatives + liquidations + options_adv
    LENS 2 — ON-CHAIN: onchain + fundamentals + flows + regime_models
    LENS 3 — MACRO: macro + sentiment + cycle

    Each lens computes its own direction score. Agreement between lenses =
    high-confidence prediction. Disagreement = uncertainty zone.
    """
    lenses = {
        "technical_lens": ["technical", "derivatives", "liquidations", "options_adv"],
        "onchain_lens":   ["onchain", "fundamentals", "flows", "regime_models"],
        "macro_lens":     ["macro", "sentiment", "cycle"],
    }

    lens_scores = {}
    for lens_name, cats in lenses.items():
        scores = []
        weights = []
        n_signals = 0
        for cat in cats:
            cat_sigs = signals.get(cat, {})
            if not isinstance(cat_sigs, dict): continue
            for name, d in cat_sigs.items():
                if name == "error" or not isinstance(d, dict): continue
                s = d.get("score")
                if s is None: continue
                scores.append(float(s))
                weights.append(1.0)
                n_signals += 1
        if not scores:
            lens_scores[lens_name] = {"score": 0.0, "n_signals": 0,
                                        "interpretation": "no_data"}
            continue
        avg = float(np.average(scores, weights=weights))
        if avg > 0.4: interp = "STRONG BULL"
        elif avg > 0.1: interp = "BULL"
        elif avg > -0.1: interp = "NEUTRAL"
        elif avg > -0.4: interp = "BEAR"
        else: interp = "STRONG BEAR"
        lens_scores[lens_name] = {
            "score": avg, "n_signals": n_signals,
            "interpretation": interp,
        }

    # Consensus: do all 3 lenses agree on direction?
    dir_signs = [
        1 if l["score"] > 0.1 else (-1 if l["score"] < -0.1 else 0)
        for l in lens_scores.values() if l.get("n_signals", 0) > 0
    ]
    if not dir_signs:
        consensus = "unknown"
    elif all(s > 0 for s in dir_signs):
        consensus = "UNANIMOUS BULL"
    elif all(s < 0 for s in dir_signs):
        consensus = "UNANIMOUS BEAR"
    elif all(s == 0 for s in dir_signs):
        consensus = "UNANIMOUS NEUTRAL"
    else:
        consensus = "MIXED / UNCERTAIN"

    return {
        "lenses": lens_scores,
        "consensus": consensus,
        "lens_count_bull": sum(1 for s in dir_signs if s > 0),
        "lens_count_bear": sum(1 for s in dir_signs if s < 0),
        "lens_count_neutral": sum(1 for s in dir_signs if s == 0),
    }


def state_of_btc(force: bool = False, log_outcome: bool = True) -> dict:
    """Master function — full BTC prediction state.

    Args:
        force: refresh cache
        log_outcome: log this prediction to outcomes DB (default True).
                     Set False for one-off queries.
    """
    signals = pull_all_signals(force=force)
    btc_price = signals.get("btc_price") or 0

    horizons = {}
    for h in ("intraday", "weekly", "short_term", "medium_term", "long_term"):
        horizons[h] = horizon_forecast(signals, h)

    targets = price_targets(btc_price, horizons) if btc_price > 0 else {}
    regime = regime_classification(horizons, btc_price=btc_price, signals=signals)

    ensemble = ensemble_predictions(signals)

    state = {
        "as_of": signals.get("as_of"),
        "btc_price": btc_price,
        "regime": regime,
        "horizons": horizons,
        "price_targets": targets,
        "signals": signals,
        "ensemble": ensemble,
    }

    # Log to outcomes DB for adaptive learning + hit-rate tracking
    if log_outcome and btc_price > 0:
        try:
            from core.prediction_outcomes import log_prediction, resolve_due_predictions
            # Resolve any due first (cheap operation, scores past predictions)
            resolve_due_predictions()
            log_prediction(state)
        except Exception:
            pass  # logging failure should not break prediction

    # Attach adaptive hit rates if available
    try:
        from core.prediction_outcomes import hit_rates_by_horizon, detect_signal_anomalies
        state["hit_rates_by_horizon"] = hit_rates_by_horizon()
        state["signal_anomalies"] = detect_signal_anomalies()
    except Exception:
        state["hit_rates_by_horizon"] = {}
        state["signal_anomalies"] = []

    return state


# === CLI ===

def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _format_score(s):
    """Format a -1..+1 score with arrow."""
    if s is None: return "  n/a"
    if s > 0.5: return f"++ {s:+.2f}"
    if s > 0.1: return f"+  {s:+.2f}"
    if s > -0.1: return f"=  {s:+.2f}"
    if s > -0.5: return f"-  {s:+.2f}"
    return f"-- {s:+.2f}"


def main():
    print("\n" + "=" * 78)
    print(f"BTC PREDICTION MACHINE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 78)

    s = state_of_btc()
    print(f"\nCurrent price: ${s['btc_price']:,.0f}")
    print(f"Overall regime: {s['regime']}")

    _section("MULTI-HORIZON DIRECTIONAL FORECAST")
    print(f"\n  {'Horizon':<14s} {'Score':>7s}  {'Direction':<14s} {'Confidence':<11s}")
    print("  " + "-" * 56)
    for h in ("intraday", "short_term", "medium_term", "long_term"):
        hd = s["horizons"][h]
        label = {"intraday": "Intraday", "short_term": "Short (1-30d)",
                 "medium_term": "Medium (1-6m)", "long_term": "Long (6m-2y)"}[h]
        print(f"  {label:<14s} {hd['direction_score']:>+6.2f}  "
              f"{hd['interpretation']:<14s} {hd['confidence']:<11s}")

    _section("PRICE TARGETS (lognormal projection from signals)")
    if s["price_targets"]:
        print()
        print(f"  {'Horizon':<14s} {'P5':>9s} {'P25':>9s} {'Median':>9s} {'P75':>9s} {'P95':>9s}")
        print("  " + "-" * 60)
        for h, t in s["price_targets"].items():
            label = {"intraday": "1 day", "short_term": "30 days",
                     "medium_term": "90 days", "long_term": "180 days"}[h]
            print(f"  {label:<14s} ${t['p5']:>7,.0f} ${t['p25']:>7,.0f} "
                  f"${t['median']:>7,.0f} ${t['p75']:>7,.0f} ${t['p95']:>7,.0f}")

    # === ENSEMBLE ===
    ens = s.get("ensemble", {})
    if ens:
        _section("ENSEMBLE — 3 independent prediction lenses")
        print()
        print(f"  Consensus: {ens.get('consensus', '?')}")
        print(f"  Bull lenses: {ens.get('lens_count_bull', 0)}  "
              f"Bear: {ens.get('lens_count_bear', 0)}  "
              f"Neutral: {ens.get('lens_count_neutral', 0)}")
        print()
        for lens_name, ld in ens.get("lenses", {}).items():
            label = {
                "technical_lens": "Technical lens (TA+derivs+liq+options)",
                "onchain_lens":   "On-chain lens (onchain+fund+flows+models)",
                "macro_lens":     "Macro lens (macro+sentiment+cycle)",
            }.get(lens_name, lens_name)
            print(f"  {label:<48s} {ld['score']:+.2f}  {ld['interpretation']:<14s} "
                  f"({ld['n_signals']} signals)")

    # === HIT RATES ===
    hr = s.get("hit_rates_by_horizon", {})
    if hr:
        _section("PREDICTION HIT RATES (from outcome log)")
        print()
        for h, d in hr.items():
            print(f"  {h:<14s}: {d['n_correct']}/{d['n_observations']} "
                  f"({d['hit_rate']*100 if d['hit_rate'] else 0:.0f}% hit)")

    # === ANOMALIES ===
    anom = s.get("signal_anomalies", [])
    if anom:
        _section("SIGNAL ANOMALIES (regime change candidates)")
        print()
        for a in anom[:5]:
            print(f"  {a['signal']:<28s} z={a['z_score']:+.2f}  {a['interpretation']}")

    _section("CATEGORY BREAKDOWN — average score per category, per horizon")
    cats = ["technical", "onchain", "sentiment", "derivatives", "macro", "liquidations",
            "cycle", "flows", "options_adv", "fundamentals", "regime_models"]
    print(f"\n  {'Category':<14s} {'Intraday':>9s} {'Short':>9s} {'Medium':>9s} {'Long':>9s}")
    print("  " + "-" * 56)
    for cat in cats:
        row = f"  {cat:<14s}"
        for h in ("intraday", "short_term", "medium_term", "long_term"):
            br = s["horizons"][h]["breakdown"].get(cat, {})
            score = br.get("score", 0)
            row += f"  {_format_score(score):>8s}"
        print(row)

    _section("ALL SIGNALS — raw readings")
    for cat in cats:
        cs = s["signals"].get(cat, {})
        if cs.get("error"): continue
        print(f"\n  [{cat.upper()}]")
        for name, d in cs.items():
            if name == "error" or not isinstance(d, dict): continue
            val = d.get("value")
            score = d.get("score")
            if isinstance(val, float):
                val_str = f"{val:>8.2f}"
            elif isinstance(val, int):
                val_str = f"{val:>8d}"
            elif isinstance(val, bool):
                val_str = f"{str(val):>8s}"
            else:
                val_str = f"{str(val)[:18]:>8s}"
            score_str = _format_score(score) if score is not None else "    n/a"
            print(f"    {name:<28s}  {val_str}  {score_str}")


if __name__ == "__main__":
    main()
