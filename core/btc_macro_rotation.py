"""Macro Rotation Tracker — equities → BTC rotation timing.

Based on the Howell/Alden framework: global liquidity drives the rotation
order. Equities and BTC both respond to liquidity, but BTC is the higher-beta
sensor that bottoms first (except crypto-contagion years like 2022).

Practical use case: "I'm holding equities and want to know WHEN to rotate
capital into BTC."

Methodology:
    1. Track S&P 500 drawdown from peak
    2. Track BTC drawdown from peak
    3. Compute BTC-SPY 30d rolling correlation
    4. Read global liquidity phase (via net_liquidity signal)
    5. Combine into a rotation phase + recommended % allocation

Rotation phases:
    PRE-ROTATION   — equities + BTC both elevated; rotation premature
    WATCH          — equities holding up while BTC bottoming; small rotation
    ACTIVE         — equities cracking + BTC accumulation zone; rotate 25-50%
    AGGRESSIVE     — both bottomed/turning + liquidity tailwind; rotate 75-100%
    COMPLETE       — recovery confirmed; finish rotation
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Cache for yfinance results (5 min)
_YF_CACHE: dict = {}


def _yf_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Cached yfinance history fetch."""
    key = (ticker, period)
    now = time.time()
    if key in _YF_CACHE:
        ts, df = _YF_CACHE[key]
        if now - ts < 300: return df
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        _YF_CACHE[key] = (now, df)
        return df
    except Exception:
        return None


# ============================================================
# 1. SPY drawdown from peak
# ============================================================

def spy_drawdown() -> Optional[dict]:
    """S&P 500 drawdown from rolling 365d peak."""
    df = _yf_history("SPY", "2y")
    if df is None or df.empty: return None
    df = df.copy()
    df["roll_peak"] = df["Close"].rolling(window=365, min_periods=30).max()
    df["dd"] = (df["Close"] / df["roll_peak"] - 1) * 100
    current_dd = float(df["dd"].iloc[-1])
    current_price = float(df["Close"].iloc[-1])
    peak_price = float(df["roll_peak"].iloc[-1])
    chg_30d_pct = (current_price / float(df["Close"].iloc[-30]) - 1) * 100 if len(df) >= 30 else 0
    chg_90d_pct = (current_price / float(df["Close"].iloc[-90]) - 1) * 100 if len(df) >= 90 else 0

    return {
        "current_price": current_price,
        "peak_price": peak_price,
        "drawdown_pct": current_dd,
        "chg_30d_pct": chg_30d_pct,
        "chg_90d_pct": chg_90d_pct,
    }


# ============================================================
# 2. BTC drawdown from peak
# ============================================================

def btc_drawdown() -> Optional[dict]:
    """BTC drawdown from cycle 5 ATH ($124,659 on Oct 6, 2025)."""
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=400)
        if df.empty: return None
        cycle5_peak = 124659
        current_price = float(df["close"].iloc[-1])
        # Drawdown from cycle 5 peak
        dd_from_ath = (current_price / cycle5_peak - 1) * 100
        # Also rolling 365d
        df["roll_peak"] = df["close"].rolling(window=365, min_periods=30).max()
        df["dd"] = (df["close"] / df["roll_peak"] - 1) * 100
        current_dd = float(df["dd"].iloc[-1])
        chg_30d_pct = (current_price / float(df["close"].iloc[-30]) - 1) * 100 if len(df) >= 30 else 0
        chg_90d_pct = (current_price / float(df["close"].iloc[-90]) - 1) * 100 if len(df) >= 90 else 0
        return {
            "current_price": current_price,
            "peak_price": cycle5_peak,
            "drawdown_pct": dd_from_ath,
            "rolling_dd_pct": current_dd,
            "chg_30d_pct": chg_30d_pct,
            "chg_90d_pct": chg_90d_pct,
        }
    except Exception:
        return None


# ============================================================
# 3. BTC-SPY 30d rolling correlation
# ============================================================

def btc_spy_correlation() -> Optional[dict]:
    """30-day and 90-day rolling correlation between BTC and SPY."""
    try:
        from core import data
        btc_df = data.ohlcv_extended("BTC/USDT", days_back=180)
        spy_df = _yf_history("SPY", "6mo")
        if btc_df.empty or spy_df is None or spy_df.empty: return None

        # Align dates
        btc_df = btc_df.copy()
        btc_df.index = btc_df.index.tz_localize(None) if btc_df.index.tz else btc_df.index
        spy_df = spy_df.copy()
        spy_df.index = spy_df.index.tz_localize(None) if spy_df.index.tz else spy_df.index

        btc_returns = btc_df["close"].pct_change().dropna()
        spy_returns = spy_df["Close"].pct_change().dropna()

        # Align on common dates
        common = btc_returns.index.intersection(spy_returns.index)
        if len(common) < 30: return None
        b = btc_returns.loc[common]
        s = spy_returns.loc[common]

        corr_30d = float(b.iloc[-30:].corr(s.iloc[-30:])) if len(common) >= 30 else None
        corr_90d = float(b.iloc[-90:].corr(s.iloc[-90:])) if len(common) >= 90 else None

        return {
            "corr_30d": corr_30d,
            "corr_90d": corr_90d,
            "n_days_available": len(common),
        }
    except Exception:
        return None


# ============================================================
# 3b. HIGH YIELD CREDIT SPREADS (Druckenmiller / Howell #1 signal)
# ============================================================

def hy_credit_spreads() -> Optional[dict]:
    """HY-Treasury credit spread — the bear-market alarm.

    When HY spreads widen >100bps in 30 days, equity bear is starting.
    When HY spreads compress after widening, that's the bottom signal.
    Druckenmiller bought BTC in 2020 the week HY spreads peaked and started compressing.
    """
    # PRIMARY: FRED BAMLH0A0HYM2 (HY-treasury OAS)
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("BAMLH0A0HYM2", days=180)
        if df is not None and not df.empty and len(df) >= 30:
            current = float(df["value"].iloc[-1])
            chg_30d = current - float(df["value"].iloc[-30])
            chg_90d = current - float(df["value"].iloc[-90]) if len(df) >= 90 else 0
            phase = _hy_phase(current, chg_30d)
            score = _hy_score(current, chg_30d)
            return {
                "value": current,
                "score": score,
                "spread_pct": current,
                "chg_30d_pp": chg_30d,
                "chg_90d_pp": chg_90d,
                "phase": phase,
                "source": "FRED(BAMLH0A0HYM2)",
                "note": (f"HY-Treasury spread {current:.2f}% "
                          f"({chg_30d:+.2f}pp 30d). {phase}"),
            }
    except Exception:
        pass

    # FALLBACK: HYG ETF price inversely correlates with HY spreads
    try:
        hyg = _yf_history("HYG", "6mo")
        tlt = _yf_history("TLT", "6mo")
        if hyg is not None and tlt is not None and not hyg.empty and not tlt.empty:
            # HYG/TLT ratio proxy
            ratio_now = float(hyg["Close"].iloc[-1]) / float(tlt["Close"].iloc[-1])
            ratio_30d = float(hyg["Close"].iloc[-30]) / float(tlt["Close"].iloc[-30]) if len(hyg) >= 30 else ratio_now
            chg_pct = (ratio_now / ratio_30d - 1) * 100
            # Higher ratio = compressed spreads (risk-on)
            # Rising ratio = HY winning vs treasuries = spreads compressing = bullish
            phase = ("COMPRESSING" if chg_pct > 1
                     else "WIDENING" if chg_pct < -1
                     else "STABLE")
            # Approximate spread value (calibrated)
            approx_spread = 4.0 + (1.0 - ratio_now / 1.0) * 5  # rough
            score = _hy_score(approx_spread, -chg_pct * 0.05)
            return {
                "value": approx_spread,
                "score": score,
                "spread_pct_estimated": approx_spread,
                "hyg_tlt_ratio": ratio_now,
                "chg_30d_pct": chg_pct,
                "phase": phase,
                "source": "yfinance(HYG/TLT)",
                "note": (f"HY-Treasury proxy: HYG/TLT ratio {ratio_now:.3f} "
                          f"({chg_pct:+.1f}% 30d). Spreads {phase.lower()}."),
            }
    except Exception:
        pass

    return {"error": "no HY data available"}


def _hy_phase(spread: float, chg_30d: float) -> str:
    if chg_30d > 1.0: return "ALARM_WIDENING"  # bear forming
    if chg_30d > 0.3: return "WIDENING"
    if spread > 6: return "ELEVATED"
    if chg_30d < -0.5 and spread < 4: return "COMPRESSING_FAST"  # bottom signal!
    if chg_30d < -0.2: return "COMPRESSING"
    if spread < 3.5: return "TIGHT (risk-on)"
    return "STABLE"


def _hy_score(spread: float, chg_30d: float) -> float:
    """Score: positive when HY compressing (bullish for risk assets/BTC)."""
    if chg_30d < -0.5 and spread < 4: return 0.8     # peak fear releasing = bottom signal
    if chg_30d < -0.2: return 0.4
    if spread < 3.5 and chg_30d < 0.1: return 0.2     # tight + stable = risk-on
    if chg_30d > 0.5: return -0.6                       # widening = bear forming
    if chg_30d > 0.2: return -0.3
    return 0.0


# ============================================================
# 3c. VIX TERM STRUCTURE (fear vs complacency)
# ============================================================

def vix_term_structure() -> Optional[dict]:
    """VIX9D/VIX/VIX3M ratio analysis.

    VIX9D/VIX3M < 0.85 = complacency (bad time to rotate)
    VIX9D/VIX3M > 1.15 = panic (the buying opportunity)
    """
    try:
        vix9d = _yf_history("^VIX9D", "1mo")
        vix = _yf_history("^VIX", "1mo")
        vix3m = _yf_history("^VIX3M", "1mo")
        if vix is None or vix.empty:
            return {"error": "VIX data unavailable"}

        v_current = float(vix["Close"].iloc[-1])
        v9d_current = float(vix9d["Close"].iloc[-1]) if vix9d is not None and not vix9d.empty else None
        v3m_current = float(vix3m["Close"].iloc[-1]) if vix3m is not None and not vix3m.empty else None

        # Term structure ratio (front / back)
        if v9d_current and v3m_current:
            ratio = v9d_current / v3m_current
        elif v_current and v3m_current:
            ratio = v_current / v3m_current
        else:
            ratio = 1.0

        # Phase
        if ratio > 1.15: phase = "PANIC (buying opportunity)"
        elif ratio > 1.05: phase = "ELEVATED FEAR"
        elif ratio > 0.95: phase = "NORMAL"
        elif ratio > 0.85: phase = "MILD COMPLACENCY"
        else: phase = "EXTREME COMPLACENCY (top forming)"

        # Score: panic = bullish for buying (positive), complacency = bearish (negative)
        if ratio > 1.20: score = 0.7
        elif ratio > 1.10: score = 0.4
        elif ratio > 0.95: score = 0.0
        elif ratio > 0.85: score = -0.3
        else: score = -0.6

        return {
            "value": ratio,
            "score": score,
            "vix9d": v9d_current,
            "vix": v_current,
            "vix3m": v3m_current,
            "term_ratio": ratio,
            "phase": phase,
            "source": "yfinance(^VIX9D,^VIX,^VIX3M)",
            "note": (f"VIX {v_current:.1f}, term ratio "
                      f"{ratio:.2f}. {phase}."),
        }
    except Exception as e:
        return {"error": str(e)[:60]}


# ============================================================
# 4. Global liquidity phase
# ============================================================

def liquidity_phase() -> dict:
    """Read global liquidity direction via Net Liquidity (or fallback).

    Howell framework: liquidity peaked Sep 2025, currently in downswing.
    """
    # Try the existing Net Liquidity signal
    try:
        from core.btc_premium_free import net_liquidity
        nl = net_liquidity()
        if nl and not nl.get("error"):
            chg_30d = nl.get("chg_30d_pct", 0)
            chg_90d = nl.get("chg_90d_pct", 0)
            if chg_30d > 1.5: phase = "EXPANDING"
            elif chg_30d > 0: phase = "MILD_EXPANSION"
            elif chg_30d > -1.5: phase = "FLAT"
            elif chg_30d > -3: phase = "MILD_CONTRACTION"
            else: phase = "CONTRACTING"
            return {
                "phase": phase,
                "chg_30d_pct": chg_30d,
                "chg_90d_pct": chg_90d,
                "net_liquidity_T": nl.get("net_liquidity_T", 0),
                "source": nl.get("source", "FRED"),
            }
    except Exception:
        pass

    # FALLBACK: Use yfinance proxies — DXY (inverse) + 2y yield (inverse)
    try:
        dxy = _yf_history("DX-Y.NYB", "6mo")
        if dxy is not None and not dxy.empty:
            dxy_chg_30d = (float(dxy["Close"].iloc[-1]) / float(dxy["Close"].iloc[-30]) - 1) * 100 if len(dxy) >= 30 else 0
            # Rising DXY = tightening liquidity (inverse)
            implied_liq_chg = -dxy_chg_30d
            if implied_liq_chg > 1.5: phase = "EXPANDING"
            elif implied_liq_chg > 0: phase = "MILD_EXPANSION"
            elif implied_liq_chg > -1.5: phase = "FLAT"
            elif implied_liq_chg > -3: phase = "MILD_CONTRACTION"
            else: phase = "CONTRACTING"
            return {
                "phase": phase,
                "chg_30d_pct": implied_liq_chg,
                "chg_90d_pct": None,
                "source": "DXY_inverse_proxy",
            }
    except Exception:
        pass

    return {"phase": "UNKNOWN", "chg_30d_pct": 0, "source": "no data"}


# ============================================================
# 5. ROTATION PHASE DETERMINATION
# ============================================================

# ============================================================
# 3d. EARNINGS / VALUATION CYCLE (Druckenmiller)
# ============================================================

def earnings_valuation() -> Optional[dict]:
    """SPY trailing P/E + earnings yield vs 10y treasury.

    Druckenmiller framework: earnings yield (E/P) vs 10y yield tells you
    if equities have a risk premium worth holding. Negative ERP = equities
    expensive vs bonds = rotate out.
    """
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        info = spy.info or {}
        pe = info.get("trailingPE") or info.get("forwardPE")
        if not pe: return {"error": "P/E unavailable"}

        # Earnings yield
        earnings_yield = 100 / float(pe)

        # 10y treasury yield via ^TNX
        tnx_hist = _yf_history("^TNX", "5d")
        treasury_10y = None
        if tnx_hist is not None and not tnx_hist.empty:
            raw = float(tnx_hist["Close"].iloc[-1])
            treasury_10y = raw / 10 if raw > 20 else raw

        # Equity Risk Premium = E/P - 10y yield
        erp = earnings_yield - treasury_10y if treasury_10y else None

        # Shiller CAPE approximation (need historical earnings; rough proxy)
        # Use trailing P/E with cyclical adjustment factor of ~1.3 for current cycle
        cape_proxy = float(pe) * 1.3

        # Score: high ERP = equities cheap relative to bonds (don't rotate)
        # Low/negative ERP = equities expensive (rotate to BTC)
        if erp is not None:
            if erp > 4: score = -0.3        # equities very cheap, don't rotate
            elif erp > 2: score = -0.1
            elif erp > 0: score = 0.1        # mild rotation case
            elif erp > -1: score = 0.3       # rotation case forming
            else: score = 0.5                  # equities expensive = strong rotation
        else: score = 0.0

        return {
            "value": earnings_yield,
            "score": score,
            "trailing_pe": float(pe),
            "earnings_yield_pct": earnings_yield,
            "treasury_10y_pct": treasury_10y,
            "equity_risk_premium_pp": erp,
            "shiller_cape_proxy": cape_proxy,
            "source": "yfinance(SPY+^TNX)",
            "note": (f"SPY P/E {pe:.1f}, E-yield {earnings_yield:.1f}%, "
                      f"10y {treasury_10y:.1f}%, ERP {erp:+.1f}pp." if erp else
                      f"SPY P/E {pe:.1f}, E-yield {earnings_yield:.1f}%."),
        }
    except Exception as e:
        return {"error": str(e)[:60]}


# ============================================================
# 3e. YIELD CURVE (recession indicator)
# ============================================================

def yield_curve() -> Optional[dict]:
    """2y/10y treasury yield spread — recession indicator.

    Inverted (< 0) for 12+ months historically precedes recession.
    Re-steepening from inversion = recession imminent (3-6 months).
    """
    try:
        # Try FRED T10Y2Y
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("T10Y2Y", days=730)
        if df is not None and not df.empty:
            current = float(df["value"].iloc[-1])
            chg_30d = current - float(df["value"].iloc[-30]) if len(df) >= 30 else 0
            chg_90d = current - float(df["value"].iloc[-90]) if len(df) >= 90 else 0
            # Days inverted (consecutive)
            inverted_recent = (df["value"].iloc[-180:] < 0).sum() if len(df) >= 180 else 0
            source = "FRED(T10Y2Y)"
        else:
            raise ValueError("FRED unavailable")
    except Exception:
        # Fallback via yfinance: ^TNX (10y) - ^IRX (3m treasury) ~ proxy
        try:
            tnx = _yf_history("^TNX", "6mo")
            irx = _yf_history("^IRX", "6mo")
            if tnx is None or tnx.empty: return None
            t10 = float(tnx["Close"].iloc[-1]) / 10
            t3m = float(irx["Close"].iloc[-1]) / 10 if irx is not None and not irx.empty else t10 - 0.5
            current = t10 - t3m  # 10y - 3m proxy
            chg_30d = 0
            chg_90d = 0
            inverted_recent = 0
            source = "yfinance(^TNX-^IRX)"
        except Exception as e:
            return {"error": str(e)[:60]}

    # Phase
    if current < -0.5: phase = "DEEPLY_INVERTED"
    elif current < 0: phase = "INVERTED"
    elif current < 0.5 and chg_30d > 0.1: phase = "RE_STEEPENING (recession imminent)"
    elif current < 1.0: phase = "FLAT"
    else: phase = "NORMAL"

    # Score for rotation: re-steepening from inversion = recession = bottom forming for risk assets
    if "RE_STEEPENING" in phase: score = 0.4
    elif phase == "INVERTED": score = -0.2
    elif phase == "DEEPLY_INVERTED": score = -0.4
    elif phase == "NORMAL": score = 0.2
    else: score = 0.0

    return {
        "value": current,
        "score": score,
        "spread_pp": current,
        "chg_30d_pp": chg_30d,
        "chg_90d_pp": chg_90d,
        "days_inverted_last_180": int(inverted_recent),
        "phase": phase,
        "source": source,
        "note": (f"2y/10y spread {current:+.2f}pp "
                  f"({chg_30d:+.2f}pp 30d). {phase}."),
    }


# ============================================================
# 3f. CURRENCY DYNAMICS (DXY trend)
# ============================================================

def currency_dynamics() -> Optional[dict]:
    """DXY trend — global USD strength.

    Druckenmiller, Alden: DXY rising = risk-off = BTC headwind.
    DXY falling = risk-on = BTC tailwind.
    """
    try:
        dxy = _yf_history("DX-Y.NYB", "6mo")
        if dxy is None or dxy.empty: return None
        current = float(dxy["Close"].iloc[-1])
        chg_30d = (current / float(dxy["Close"].iloc[-30]) - 1) * 100 if len(dxy) >= 30 else 0
        chg_90d = (current / float(dxy["Close"].iloc[-90]) - 1) * 100 if len(dxy) >= 90 else 0

        if chg_30d > 3: score = -0.4
        elif chg_30d > 1: score = -0.2
        elif chg_30d > -1: score = 0.0
        elif chg_30d > -3: score = 0.2
        else: score = 0.4

        phase = ("STRONG_DOLLAR (risk-off)" if chg_30d > 1
                 else "WEAK_DOLLAR (risk-on)" if chg_30d < -1
                 else "STABLE")

        return {
            "value": current,
            "score": score,
            "dxy": current,
            "chg_30d_pct": chg_30d,
            "chg_90d_pct": chg_90d,
            "phase": phase,
            "source": "yfinance(DX-Y.NYB)",
            "note": f"DXY {current:.2f} ({chg_30d:+.1f}% 30d). {phase}.",
        }
    except Exception as e:
        return {"error": str(e)[:60]}


# ============================================================
# 3g. RISK MANAGEMENT OVERLAY
# ============================================================

def risk_management(deploy_pct: int, btc_drawdown: float, vix_value: Optional[float],
                     total_stake_nzd: float = 130000) -> dict:
    """Risk management recommendations: position cap, stop loss, cash buffer.

    Druckenmiller/PTJ wisdom:
    - Never deploy 100% even on highest conviction (always keep cash buffer)
    - Stop loss = 1.5x recent ATR (Paul Tudor Jones rule)
    - Maximum per-asset allocation = 50% of liquid net worth
    """
    # Position cap: 50% of stake regardless of signal (PTJ rule)
    max_position_cap = 0.50

    # Stop loss: if BTC drops another -25% from current, exit 50% of position
    # If BTC drops -35%, exit 100%
    suggested_stop_25pct = "Exit 50% if BTC drops another -25%"
    suggested_stop_35pct = "Exit 100% if BTC drops another -35%"

    # Cash buffer: never go below 10% cash (Druckenmiller rule)
    min_cash_buffer_pct = 10

    # Actual recommended deploy (respects cap)
    actual_deploy_pct = min(deploy_pct, int(max_position_cap * 100))
    deploy_nzd = total_stake_nzd * actual_deploy_pct / 100
    remaining_for_btc = total_stake_nzd * max_position_cap - deploy_nzd
    cash_buffer_nzd = total_stake_nzd * min_cash_buffer_pct / 100

    # Volatility-adjusted stop
    if vix_value:
        # Higher VIX = wider stop (don't get stopped by noise)
        stop_pct = max(15, min(35, vix_value * 1.5))
    else:
        stop_pct = 25

    return {
        "max_position_cap_pct": max_position_cap * 100,
        "min_cash_buffer_pct": min_cash_buffer_pct,
        "actual_deploy_pct": actual_deploy_pct,
        "deploy_nzd": deploy_nzd,
        "remaining_btc_capacity_nzd": remaining_for_btc,
        "cash_buffer_nzd": cash_buffer_nzd,
        "stop_loss_pct": stop_pct,
        "suggested_stop_25pct": suggested_stop_25pct,
        "suggested_stop_35pct": suggested_stop_35pct,
        "note": (f"Deploy ${deploy_nzd:,.0f} NZD now. Reserve ${remaining_for_btc:,.0f} "
                  f"NZD for additional BTC tranches. Keep ${cash_buffer_nzd:,.0f} NZD as "
                  f"untouched cash buffer. Stop loss: -{stop_pct:.0f}% from entry."),
    }


# ============================================================
# 3h. WHAT TO SELL (high-beta first)
# ============================================================

def what_to_sell_guidance() -> dict:
    """Guidance on which equity exposure to sell first to fund BTC rotation.

    PTJ framework: sell highest-beta / most overvalued positions first.
    Avoid selling defensives/dividend stocks (your portfolio hedge).
    """
    return {
        "sell_first": [
            {"category": "High-beta tech (QQQ, ARKK, semis)",
             "rationale": "Highest correlation with risk-on; most exposed to liquidity drain",
             "examples": "QQQ, SOXX, ARKK, TSLA, NVDA"},
            {"category": "Growth/profitless tech",
             "rationale": "Highest beta in drawdowns, lowest fundamental support",
             "examples": "ARK funds, recent IPOs, SPACs"},
            {"category": "Crypto-adjacent stocks",
             "rationale": "Holding both BTC and BTC-stocks = double exposure",
             "examples": "COIN, MSTR, MARA, RIOT — sell these BEFORE buying BTC"},
        ],
        "sell_last": [
            {"category": "Dividend aristocrats / defensives",
             "rationale": "Portfolio ballast in drawdowns; sell only if forced",
             "examples": "PG, KO, JNJ, XLP"},
            {"category": "Cash-flow positive utilities",
             "rationale": "Negative correlation with rate cycles",
             "examples": "XLU"},
            {"category": "Energy (selectively)",
             "rationale": "Inflation hedge; complement to BTC thesis",
             "examples": "XLE, oil majors"},
        ],
        "tax_loss_harvest_first": (
            "If you have positions UNDERWATER in your stocks rig, sell those "
            "FIRST to harvest losses (offsets future BTC gains). Then sell winners "
            "from the 'sell first' category."
        ),
    }


# ============================================================
# 3i. NZ TAX AWARENESS
# ============================================================

def nz_tax_considerations(deploy_nzd: float) -> dict:
    """NZ-specific tax considerations for BTC ETF rotation.

    User mentioned 'buying ETF no NZ tax' — but FIF regime applies above
    NZ$50k cost basis on overseas ETFs. Adding visibility.
    """
    fif_threshold = 50000
    over_fif = deploy_nzd > fif_threshold

    return {
        "deploy_nzd": deploy_nzd,
        "fif_threshold_nzd": fif_threshold,
        "over_fif_threshold": over_fif,
        "considerations": [
            ("FIF regime applies above NZ$50k cost basis on overseas ETFs",
             "REQUIRED" if over_fif else "below threshold — exempt"),
            ("FDR method: 5% deemed return × marginal rate (~1.65% effective annual)",
             "Applies if over threshold"),
            ("CV method: lower of FDR or actual capital change × marginal rate",
             "Choose annually"),
            ("Direct BTC holding: IRD treats crypto as property — taxable on disposal if trading intent",
             "Different framework than ETF"),
            ("PIE fund wrapper: caps tax at 28% PIR (better than 33% trader)",
             "Consider if structured this way"),
        ],
        "action_items": [
            "Verify with accountant whether your specific ETF + structure triggers FIF",
            "Document holding intent (investor vs trader) in writing",
            "If holding >NZ$50k cost basis, set up FDR/CV calculation",
            "Tax harvest LOSSES from equities before realizing BTC gains",
        ],
        "note": (f"At ${deploy_nzd:,.0f} NZD deploy: "
                  + ("OVER FIF threshold — get written tax advice first." if over_fif else
                      "Under FIF threshold — exempt for now.")),
    }


def kelly_position_size(base_pct: int, btc_drawdown_pct: float,
                         vix_value: Optional[float],
                         confirming_signals: int) -> dict:
    """Kelly-criterion vol-adjusted position size.

    Adjusts the base recommendation by:
    - BTC drawdown depth (deeper = larger position justified)
    - Current implied volatility (higher = smaller position)
    - Number of confirming bull signals
    """
    # Volatility multiplier
    # Normal VIX is ~16. Extreme is 40+. Use inverse scaling.
    if vix_value is not None and vix_value > 0:
        vol_mult = 16.0 / max(8.0, vix_value)  # 0.4 at VIX 40, 1.0 at VIX 16
    else:
        vol_mult = 0.85  # mild conservatism without VIX data

    # Drawdown multiplier (deeper drawdown = more conviction to size up)
    if btc_drawdown_pct < -55: dd_mult = 1.3
    elif btc_drawdown_pct < -45: dd_mult = 1.15
    elif btc_drawdown_pct < -35: dd_mult = 1.0
    else: dd_mult = 0.85

    # Signal confluence multiplier
    sig_mult = 0.7 + 0.1 * min(5, confirming_signals)  # 0.7 at 0 signals, 1.2 at 5+

    kelly_pct = base_pct * vol_mult * dd_mult * sig_mult
    kelly_pct = max(0, min(100, int(round(kelly_pct))))

    return {
        "base_pct": base_pct,
        "kelly_adjusted_pct": kelly_pct,
        "vol_multiplier": vol_mult,
        "drawdown_multiplier": dd_mult,
        "signal_multiplier": sig_mult,
        "vix_used": vix_value,
        "btc_dd_used": btc_drawdown_pct,
    }


def dca_pace(deploy_pct: int, urgency: str) -> dict:
    """Recommend a DCA pace based on urgency level.

    urgency: LOW / MEDIUM / HIGH / IMMEDIATE
    """
    if urgency == "IMMEDIATE":
        weeks = 1; tranches = 3; freq = "every 2-3 days"
    elif urgency == "HIGH":
        weeks = 2; tranches = 4; freq = "twice per week"
    elif urgency == "MEDIUM":
        weeks = 4; tranches = 4; freq = "weekly"
    else:  # LOW
        weeks = 8; tranches = 8; freq = "weekly"

    pct_per_tranche = round(deploy_pct / tranches, 1)
    return {
        "weeks": weeks,
        "tranches": tranches,
        "frequency": freq,
        "pct_per_tranche": pct_per_tranche,
        "urgency": urgency,
        "recommendation": (f"Deploy {deploy_pct}% over {weeks} weeks "
                            f"in {tranches} tranches "
                            f"({pct_per_tranche}% each, {freq}). "
                            f"Reassess after each tranche."),
    }


def _bottom_confirmation_cap() -> tuple:
    """2026-07-07 logic audit (F2): the bottom-confirmation scorecard is the
    TIMING authority. Macro relative-value says WHERE (equities rich vs BTC);
    the scorecard says WHEN. Returns (max_deploy_pct, label, n_met, n_total).
    The rotation tree keys "BTC bottom zone" off price drawdown alone, so at
    -49% it fired "ROTATE 20%" while the hard scorecard read 2/10 (no bottom).
    This caps the macro deploy % at what the scorecard actually clears. Fail
    safe: if confirmation is unavailable, cap to a scout tranche — never full
    macro deployment on price drawdown alone."""
    try:
        from core.dashboard_cache import get_cached
        bc = get_cached("bottom_confirmation") or {}
        if not bc:
            return 15, "UNCONFIRMED (data n/a)", 0, 10
        n_met = int(bc.get("n_met", 0))
        n_total = int(bc.get("n_total", 10)) or 10
        if n_met >= 7:  return 100, "CONFIRMED", n_met, n_total
        if n_met >= 5:  return 75, "SCALE-IN", n_met, n_total
        if n_met >= 3:  return 20, "EARLY", n_met, n_total
        return 0, "NO_BOTTOM", n_met, n_total   # hard gate not met -> hold
    except Exception:
        return 15, "UNCONFIRMED (data n/a)", 0, 10


def rotation_phase() -> dict:
    """Combine SPY drawdown + BTC drawdown + correlation + liquidity +
    HY spreads + VIX term structure into a rotation phase + Kelly-sized
    recommendation + DCA pace.

    Phases:
        PRE-ROTATION  — both elevated; rotation premature; recommend 0%
        WATCH         — equities holding up but BTC bottoming; recommend 10-25%
        ACTIVE        — equities cracking + BTC at accumulation zone; 25-50%
        AGGRESSIVE    — both bottomed/turning + liquidity tailwind; 75-100%
        COMPLETE      — recovery confirmed; finish rotation
    """
    spy = spy_drawdown()
    btc = btc_drawdown()
    corr = btc_spy_correlation()
    liq = liquidity_phase()
    hy = hy_credit_spreads()
    vix = vix_term_structure()
    val = earnings_valuation()
    yc = yield_curve()
    dxy = currency_dynamics()

    if not spy or not btc:
        return {"error": "missing SPY or BTC data"}

    spy_dd = spy["drawdown_pct"]
    btc_dd = btc["drawdown_pct"]
    corr_30 = corr.get("corr_30d") if corr else None

    # Phase determination
    spy_at_peak = spy_dd > -5
    spy_pulling_back = -15 <= spy_dd <= -5
    spy_correcting = -25 <= spy_dd < -15
    spy_bear = spy_dd < -25

    btc_in_bottom_zone = btc_dd < -40
    btc_in_capitulation = btc_dd < -55
    btc_recovered = btc_dd > -25

    liq_phase = liq.get("phase", "UNKNOWN")
    liq_expansion = liq_phase in ("EXPANDING", "MILD_EXPANSION")
    liq_contraction = liq_phase in ("CONTRACTING", "MILD_CONTRACTION")

    # Decision tree
    if spy_at_peak and btc_recovered:
        phase_id = "PRE_ROTATION"
        deploy_pct = 0
        rationale = (f"Both elevated: SPY {spy_dd:.1f}% from peak, BTC {btc_dd:.1f}% from peak. "
                     "Rotation premature — wait for SPY pullback or BTC capitulation.")
        action = "HOLD EQUITIES"

    elif spy_at_peak and btc_in_bottom_zone:
        # Classic rotation entry: equities expensive, BTC discounted
        phase_id = "WATCH"
        deploy_pct = 20
        rationale = (f"SPY near peak ({spy_dd:.1f}%) while BTC discounted ({btc_dd:.1f}%). "
                      "Classic rotation entry — equities overvalued relative to BTC. "
                      "Begin gradual rotation 20% of equity stake.")
        action = "BEGIN ROTATION (20%)"

    elif spy_pulling_back and btc_in_bottom_zone:
        phase_id = "ACTIVE"
        deploy_pct = 40
        rationale = (f"SPY pulling back ({spy_dd:.1f}%) + BTC in bottom zone ({btc_dd:.1f}%). "
                      "Both correcting — rotation active. Move 40% of equity stake into BTC. "
                      "Watch for liquidity turn to accelerate.")
        action = "ROTATE 40%"

    elif spy_correcting and btc_in_capitulation:
        if liq_expansion:
            phase_id = "AGGRESSIVE"
            deploy_pct = 85
            rationale = (f"SPY correcting ({spy_dd:.1f}%) + BTC capitulation ({btc_dd:.1f}%) + "
                          f"liquidity {liq_phase}. All conditions met. Aggressive rotation "
                          "into BTC. This is the high-conviction rotation window.")
            action = "ROTATE 85% AGGRESSIVELY"
        else:
            phase_id = "ACTIVE_DEEP"
            deploy_pct = 60
            rationale = (f"SPY correcting + BTC capitulating but liquidity still "
                          f"{liq_phase.lower()}. Strong rotation case (60%) but hold some "
                          "powder for liquidity confirmation.")
            action = "ROTATE 60%"

    elif spy_bear and btc_in_capitulation:
        phase_id = "AGGRESSIVE"
        deploy_pct = 90
        rationale = (f"Both in serious bear territory (SPY {spy_dd:.1f}%, BTC {btc_dd:.1f}%). "
                      "Generational rotation opportunity. Move 90% into BTC.")
        action = "ROTATE 90%"

    elif btc_recovered and not spy_at_peak:
        # BTC already recovered, equities lagging
        phase_id = "COMPLETE"
        deploy_pct = 100
        rationale = (f"BTC recovering ({btc_dd:.1f}%) while SPY still soft ({spy_dd:.1f}%). "
                      "Rotation thesis playing out as predicted by Howell framework. "
                      "Finish remaining rotation.")
        action = "FINISH ROTATION (100%)"

    else:
        phase_id = "WATCH"
        deploy_pct = 15
        rationale = (f"Mixed signals: SPY {spy_dd:.1f}%, BTC {btc_dd:.1f}%, "
                      f"liquidity {liq_phase}. Default to 15% rotation pace.")
        action = "ROTATE 15%"

    # ── 2026-07-07 logic audit (F2): BOTTOM-CONFIRMATION GATE ─────────────
    # Reconciles this panel (which said "BEGIN ROTATION 20%" off a pure price
    # drawdown proxy) with the 3 scorecard panels that said "far from bottom".
    # The bottom scorecard is the timing authority and caps how much the macro
    # case is cleared to actually deploy.
    _conf_cap, _conf_label, _conf_m, _conf_t = _bottom_confirmation_cap()
    _gate_active = deploy_pct > _conf_cap
    if _gate_active:
        _macro_pct = deploy_pct
        deploy_pct = _conf_cap
        if _conf_cap <= 0:
            action = f"ARMED - HOLD (bottom {_conf_label} {_conf_m}/{_conf_t})"
        else:
            action = f"ROTATE {_conf_cap}% (confirmation-gated {_conf_m}/{_conf_t})"
        rationale = (
            f"Macro relative-value favours rotation (~{_macro_pct}% on price), "
            f"BUT the bottom-confirmation scorecard is {_conf_label} "
            f"({_conf_m}/{_conf_t}) - the timing gate caps deployment at "
            f"{_conf_cap}%. Macro says WHERE, the scorecard says WHEN; wait for "
            f"confirmation to build. [macro basis: {rationale}]"
        )

    # Modifiers + confirming signal count
    notes = []
    if _gate_active:
        notes.append(f"Deploy % gated by bottom-confirmation scorecard "
                     f"({_conf_label} {_conf_m}/{_conf_t}) - macro alone does not "
                     f"time the entry.")
    confirming_signals = 0
    if corr_30 is not None and corr_30 > 0.7:
        notes.append(f"BTC-SPY correlation high ({corr_30:.2f}) — they're moving together")
    elif corr_30 is not None and corr_30 < 0.2:
        notes.append(f"BTC-SPY correlation low ({corr_30:.2f}) — divergence regime")
        confirming_signals += 1
    if liq_expansion:
        notes.append("Liquidity tailwind active")
        confirming_signals += 1
    if liq_contraction:
        notes.append("Liquidity headwind — be patient")

    # HY credit spreads (Druckenmiller's #1 signal)
    if hy and not hy.get("error"):
        hy_phase_str = hy.get("phase", "")
        if "COMPRESSING" in hy_phase_str or "TIGHT" in hy_phase_str:
            notes.append(f"HY spreads {hy_phase_str.lower()} — risk-on confirmed")
            confirming_signals += 1
        elif "WIDENING" in hy_phase_str or "ALARM" in hy_phase_str:
            notes.append(f"HY spreads {hy_phase_str.lower()} — equity bear forming")

    # VIX term structure
    if vix and not vix.get("error"):
        vix_phase_str = vix.get("phase", "")
        if "PANIC" in vix_phase_str:
            notes.append(f"VIX term structure: {vix_phase_str} — high-conviction buy window")
            confirming_signals += 2  # extra weight
        elif "ELEVATED FEAR" in vix_phase_str:
            notes.append(f"VIX: {vix_phase_str} — rotation opportunity forming")
            confirming_signals += 1
        elif "EXTREME COMPLACENCY" in vix_phase_str:
            notes.append(f"VIX: {vix_phase_str} — wait, top likely forming")

    # Kelly sizing
    vix_value = vix.get("vix") if vix and not vix.get("error") else None
    kelly = kelly_position_size(deploy_pct, btc_dd, vix_value, confirming_signals)

    # Urgency for DCA pace
    if phase_id == "AGGRESSIVE":   urgency = "HIGH"
    elif phase_id == "ACTIVE":     urgency = "MEDIUM"
    elif phase_id == "WATCH":      urgency = "LOW"
    elif phase_id == "COMPLETE":   urgency = "IMMEDIATE"
    else:                           urgency = "LOW"

    # Valuation, yield curve, DXY signals also feed into confirming count
    if val and not val.get("error"):
        if val.get("score", 0) > 0.2:
            notes.append(f"SPY ERP {val.get('equity_risk_premium_pp', 0):+.1f}pp — equities expensive vs bonds")
            confirming_signals += 1
        elif val.get("score", 0) < -0.2:
            notes.append(f"SPY ERP {val.get('equity_risk_premium_pp', 0):+.1f}pp — equities cheap vs bonds; wait")
    if yc and not yc.get("error"):
        yc_phase = yc.get("phase", "")
        if "RE_STEEPENING" in yc_phase:
            notes.append(f"Yield curve {yc_phase} — recession imminent, BTC bottom forming")
            confirming_signals += 1
        elif "INVERTED" in yc_phase:
            notes.append(f"Yield curve {yc_phase} — recession risk elevated")
    if dxy and not dxy.get("error"):
        if dxy.get("score", 0) > 0.2:
            notes.append(f"DXY {dxy.get('phase', '')} — BTC tailwind")
            confirming_signals += 1
        elif dxy.get("score", 0) < -0.2:
            notes.append(f"DXY {dxy.get('phase', '')} — BTC headwind")

    final_pct = kelly["kelly_adjusted_pct"]
    dca = dca_pace(final_pct, urgency)

    # Risk management overlay + tax + sell guidance
    risk = risk_management(final_pct, btc_dd, vix_value, total_stake_nzd=130000)
    tax = nz_tax_considerations(risk["deploy_nzd"])
    sells = what_to_sell_guidance()

    return {
        "phase_id": phase_id,
        "deploy_pct": deploy_pct,
        "kelly_pct": final_pct,
        "kelly_details": kelly,
        "dca": dca,
        "action": action,
        "rationale": rationale,
        "notes": notes,
        "confirming_signals": confirming_signals,
        "spy": spy,
        "btc": btc,
        "correlation": corr,
        "liquidity": liq,
        "hy_credit_spreads": hy,
        "vix_term_structure": vix,
        "earnings_valuation": val,
        "yield_curve": yc,
        "currency_dynamics": dxy,
        "risk_management": risk,
        "nz_tax_considerations": tax,
        "what_to_sell": sells,
    }


# ============================================================
# 6. HISTORICAL BACKTEST
# ============================================================

def _classify_phase_at_dd(spy_dd: float, btc_dd: float) -> str:
    """Simplified phase classification used for backtest."""
    spy_at_peak = spy_dd > -5
    spy_correcting = -25 <= spy_dd < -15
    spy_pulling_back = -15 <= spy_dd <= -5
    spy_bear = spy_dd < -25

    btc_in_bottom = btc_dd < -40
    btc_capitulation = btc_dd < -55
    btc_recovered = btc_dd > -25

    if spy_at_peak and btc_recovered: return "PRE_ROTATION"
    if spy_at_peak and btc_in_bottom: return "WATCH"
    if spy_pulling_back and btc_in_bottom: return "ACTIVE"
    if spy_correcting and btc_capitulation: return "AGGRESSIVE"
    if spy_bear and btc_capitulation: return "AGGRESSIVE"
    if btc_recovered and not spy_at_peak: return "COMPLETE"
    return "WATCH"


def historical_backtest() -> dict:
    """Backtest the rotation indicator at known historical bottoms.

    Tests whether the indicator would have correctly flagged the 3 major
    BTC bottoms (2018, 2020, 2022) before they happened.
    """
    test_periods = [
        # Cycle 3 bottom: BTC bottomed Dec 15, 2018 at $3,200
        {
            "label": "Cycle 3 bottom (Dec 2018)",
            "btc_peak_date": datetime(2017, 12, 17).date(),
            "btc_peak_price": 19783,
            "btc_bottom_date": datetime(2018, 12, 15).date(),
            "btc_bottom_price": 3200,
            "spy_peak_date": datetime(2018, 9, 20).date(),
            "spy_peak_price": 293.58,  # SPY at the 2018 peak
            "spy_bottom_date": datetime(2018, 12, 24).date(),
            "spy_bottom_price": 234.34,
            "test_dates": [
                ("90d before BTC bottom",  datetime(2018, 9, 15).date()),
                ("60d before",              datetime(2018, 10, 15).date()),
                ("30d before",              datetime(2018, 11, 15).date()),
                ("AT BTC bottom",           datetime(2018, 12, 15).date()),
            ],
        },
        # COVID bottom: BTC bottomed Mar 13, 2020 at $3,850
        {
            "label": "COVID bottom (Mar 2020)",
            "btc_peak_date": datetime(2019, 6, 26).date(),
            "btc_peak_price": 13796,
            "btc_bottom_date": datetime(2020, 3, 13).date(),
            "btc_bottom_price": 3850,
            "spy_peak_date": datetime(2020, 2, 19).date(),
            "spy_peak_price": 339.08,
            "spy_bottom_date": datetime(2020, 3, 23).date(),
            "spy_bottom_price": 222.95,
            "test_dates": [
                ("90d before",  datetime(2019, 12, 14).date()),
                ("30d before",  datetime(2020, 2, 12).date()),
                ("7d before",   datetime(2020, 3, 6).date()),
                ("AT bottom",   datetime(2020, 3, 13).date()),
            ],
        },
        # Cycle 4 bottom: BTC bottomed Nov 9, 2022 at $15,500
        {
            "label": "Cycle 4 bottom (Nov 2022)",
            "btc_peak_date": datetime(2021, 11, 10).date(),
            "btc_peak_price": 68789,
            "btc_bottom_date": datetime(2022, 11, 9).date(),
            "btc_bottom_price": 15500,
            "spy_peak_date": datetime(2022, 1, 4).date(),
            "spy_peak_price": 477.71,
            "spy_bottom_date": datetime(2022, 10, 12).date(),  # SPY bottomed BEFORE BTC this cycle
            "spy_bottom_price": 357.04,
            "test_dates": [
                ("90d before BTC bottom",  datetime(2022, 8, 11).date()),
                ("60d before",              datetime(2022, 9, 10).date()),
                ("AT SPY bottom",           datetime(2022, 10, 12).date()),
                ("AT BTC bottom",           datetime(2022, 11, 9).date()),
            ],
        },
    ]

    results = []
    for tp in test_periods:
        # Use yfinance for SPY + BTC history
        try:
            import yfinance as yf
            # Get long enough history
            start = (tp["btc_peak_date"] - timedelta(days=60))
            end = (tp["btc_bottom_date"] + timedelta(days=30))
            spy_hist = yf.Ticker("SPY").history(start=start.isoformat(), end=end.isoformat())
            btc_hist = yf.Ticker("BTC-USD").history(start=start.isoformat(), end=end.isoformat())
            if spy_hist.empty or btc_hist.empty: continue

            # Normalize indices
            spy_hist = spy_hist.copy()
            spy_hist.index = spy_hist.index.tz_localize(None) if spy_hist.index.tz else spy_hist.index
            btc_hist = btc_hist.copy()
            btc_hist.index = btc_hist.index.tz_localize(None) if btc_hist.index.tz else btc_hist.index

            tp_results = {"label": tp["label"],
                          "btc_peak": tp["btc_peak_price"],
                          "btc_bottom": tp["btc_bottom_price"],
                          "spy_peak": tp["spy_peak_price"],
                          "spy_bottom": tp["spy_bottom_price"],
                          "tests": []}

            for label, test_date in tp["test_dates"]:
                # Get closest dates' prices
                try:
                    spy_at = spy_hist.asof(pd.Timestamp(test_date))
                    btc_at = btc_hist.asof(pd.Timestamp(test_date))
                    if pd.isna(spy_at["Close"]) or pd.isna(btc_at["Close"]): continue
                    spy_price = float(spy_at["Close"])
                    btc_price = float(btc_at["Close"])

                    # Drawdowns from prior peaks
                    spy_dd = (spy_price / tp["spy_peak_price"] - 1) * 100
                    btc_dd = (btc_price / tp["btc_peak_price"] - 1) * 100

                    phase = _classify_phase_at_dd(spy_dd, btc_dd)
                    tp_results["tests"].append({
                        "label": label,
                        "date": test_date.isoformat(),
                        "spy_price": spy_price,
                        "btc_price": btc_price,
                        "spy_dd": spy_dd,
                        "btc_dd": btc_dd,
                        "phase_signaled": phase,
                    })
                except Exception:
                    continue
            results.append(tp_results)
        except Exception:
            continue

    # Summary verdict
    summary_lines = []
    for r in results:
        if r["tests"]:
            phases = [t["phase_signaled"] for t in r["tests"]]
            at_bottom = next((t for t in r["tests"] if "AT" in t["label"] and "BTC" in t["label"]), None)
            if at_bottom:
                summary_lines.append(
                    f"{r['label']}: signaled {at_bottom['phase_signaled']} AT bottom "
                    f"(SPY {at_bottom['spy_dd']:.1f}%, BTC {at_bottom['btc_dd']:.1f}%)"
                )

    return {
        "periods": results,
        "n_periods": len(results),
        "summary_lines": summary_lines,
    }


# ============================================================


def main():
    print("\n" + "=" * 78)
    print("MACRO ROTATION TRACKER — equities -> BTC")
    print("=" * 78)
    r = rotation_phase()
    if r.get("error"):
        print(f"ERROR: {r['error']}")
        return
    print()
    print(f"PHASE:       {r['phase_id']}")
    print(f"ACTION:      {r['action']}")
    print(f"DEPLOY %:    {r['deploy_pct']}% of equity stake -> BTC")
    print()
    print(f"RATIONALE:")
    print(f"  {r['rationale']}")
    print()
    if r["notes"]:
        print("NOTES:")
        for n in r["notes"]: print(f"  - {n}")
    print()
    spy = r["spy"]; btc = r["btc"]; liq = r["liquidity"]
    print(f"SPY:  ${spy['current_price']:,.2f}  drawdown {spy['drawdown_pct']:+.1f}%  "
          f"30d {spy['chg_30d_pct']:+.1f}%")
    print(f"BTC:  ${btc['current_price']:,.0f}    drawdown {btc['drawdown_pct']:+.1f}%  "
          f"30d {btc['chg_30d_pct']:+.1f}%")
    corr = r["correlation"]
    if corr:
        print(f"Correlation: 30d {corr.get('corr_30d', 0):.2f}, 90d {corr.get('corr_90d', 0):.2f}")
    print(f"Liquidity:   {liq.get('phase', '?')} ({liq.get('chg_30d_pct', 0):+.1f}% 30d)")


if __name__ == "__main__":
    main()
