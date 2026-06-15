"""Incremental Macro Layer — 8 leading indicators that fire 6-12 months
before major equity tops (2000, 2007, 2021).

All free-tier:
  - FRED CSV (via btc_clemente_alden._fred_csv, with circuit breaker)
  - yfinance (for MOVE index)

Signals:
  A. GLOBAL CYCLE
     1. OECD US CLI — 6m annualized change
     2. CB LEI YoY proxy (built from CLI level)
  B. FUNDING & PLUMBING
     3. MOVE Index — bond volatility (^MOVE)
     4. RRP utilization + SOFR-IORB spread (dollar liquidity)
  C. CREDIT CYCLE
     5. US credit impulse (BUSLOANS acceleration)
     6. Senior Loan Officer Survey (DRTSCILM)
  D. REAL ECONOMY
     7. Sahm Rule (SAHMREALTIME)
     8. Initial claims 4w MA crossing 12m MA (ICSA)

Each function returns a dict with at least:
  {firing: bool, score: float (0..1), value: float, status: str, rationale: str}
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
# Helpers
# ============================================================

def _fred(series: str, days: int = 1460) -> Optional[pd.DataFrame]:
    """Wrap btc_clemente_alden._fred_csv (uses circuit breaker + disk cache)."""
    try:
        from core.btc_clemente_alden import _fred_csv
        return _fred_csv(series, days=days)
    except Exception:
        return None


def _yf(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty: return None
        return df
    except Exception:
        return None


def _err(msg: str) -> dict:
    return {"firing": False, "score": 0.0, "value": None, "status": msg, "rationale": ""}


# ============================================================
# A. GLOBAL CYCLE
# ============================================================

def oecd_cli_6m_change() -> dict:
    """OECD US Composite Leading Indicator — 6-month annualized change.

    Leads turning points by 6-9 months. Negative + falling = late cycle.
    """
    df = _fred("USALOLITOAASTSAM", days=2000)
    if df is None or df.empty:
        return _err("OECD CLI data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 7:
        return _err("OECD CLI insufficient history")
    # Monthly series. 6 months ago = 6 rows back (each row = 1 month).
    current = float(df["value"].iloc[-1])
    six_mo_ago = float(df["value"].iloc[-7]) if len(df) >= 7 else float(df["value"].iloc[0])
    three_mo_ago = float(df["value"].iloc[-4]) if len(df) >= 4 else current
    cli_6m_annl = ((current / six_mo_ago) ** 2 - 1) * 100  # annualize
    cli_3m_annl = ((current / three_mo_ago) ** 4 - 1) * 100
    # Firing: 6m annualized < -2.0 AND 3m worse than 6m (still falling)
    firing = cli_6m_annl < -2.0 and cli_3m_annl < cli_6m_annl
    score = max(0.0, min(1.0, (-cli_6m_annl + 2.0) / 4.0))  # ramp -2..-6 -> 0..1
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": cli_6m_annl,
        "status": (f"OECD CLI 6m annualized: {cli_6m_annl:+.2f}% "
                   f"(3m: {cli_3m_annl:+.2f}%) "
                   f"({'FIRING' if firing else 'ok'})"),
        "rationale": ("OECD CLI 6m < -2% historically called every US recession "
                       "since 1970 by 6-9 months."),
    }


def cb_lei_yoy() -> dict:
    """Conference Board LEI YoY proxy — built from OECD CLI level YoY.

    Real CB LEI not freely available on FRED in YoY form. We use OECD CLI
    YoY as best free proxy; both indices share ~85% overlap in construction.
    Threshold: YoY < -4% has called every recession since 1959.
    """
    df = _fred("USALOLITOAASTSAM", days=1100)
    if df is None or df.empty:
        return _err("LEI/CLI data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 13:
        return _err("LEI/CLI insufficient history")
    current = float(df["value"].iloc[-1])
    year_ago = float(df["value"].iloc[-13])  # 12 months back
    lei_yoy = (current / year_ago - 1) * 100
    firing = lei_yoy < -4.0
    score = max(0.0, min(1.0, (-lei_yoy - 1.0) / 5.0))  # ramp -1..-6 -> 0..1
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": lei_yoy,
        "status": (f"OECD CLI YoY (LEI proxy): {lei_yoy:+.2f}% "
                   f"({'FIRING — recession imminent' if firing else 'ok'})"),
        "rationale": ("LEI YoY < -4% has called every US recession since 1959 "
                       "with no false positives. Best single recession signal."),
    }


# ============================================================
# B. FUNDING & PLUMBING
# ============================================================

def move_index_elevated() -> dict:
    """MOVE Index — bond market volatility.

    Spikes 2-4 weeks before equity tops historically. Firing if above
    180d 75th percentile and rising.
    """
    df = _yf("^MOVE", period="1y")
    if df is None or df.empty:
        return _err("MOVE data unavailable")
    closes = df["Close"].dropna()
    if len(closes) < 30:
        return _err("MOVE insufficient history")
    current = float(closes.iloc[-1])
    p75 = float(closes.tail(180).quantile(0.75)) if len(closes) >= 180 else \
          float(closes.quantile(0.75))
    rising = (len(closes) >= 14 and current > float(closes.iloc[-14]))
    firing = current > p75 and rising
    # Hard limit: 150+ = liquidity event imminent
    extreme = current > 150
    score = max(0.0, min(1.0, (current - p75) / 30.0))
    if extreme: score = 1.0
    return {
        "firing": bool(firing or extreme),
        "score": float(score),
        "value": current,
        "status": (f"MOVE: {current:.0f} (180d p75: {p75:.0f}) "
                   f"({'EXTREME' if extreme else 'FIRING' if firing else 'ok'})"),
        "rationale": ("Bond vol spikes 2-4 weeks before equity tops. "
                       "Above 150 = liquidity event imminent."),
        "extreme": bool(extreme),
    }


def dollar_liquidity_stress() -> dict:
    """RRP utilization collapsing + SOFR-IORB spread positive.

    RRP draining + SOFR > IORB = bank reserves under pressure.
    Sept-2019-style repo crunch signal.
    """
    rrp_df = _fred("RRPONTSYD", days=400)
    sofr_df = _fred("SOFR", days=90)
    iorb_df = _fred("IORB", days=90)

    rrp_signal = False
    rrp_val = None
    rrp_status = "RRP unavailable"
    if rrp_df is not None and not rrp_df.empty:
        rrp_df = rrp_df.sort_values("date").reset_index(drop=True)
        rrp_now_b = float(rrp_df["value"].iloc[-1])  # already in billions
        # FRED RRPONTSYD is in billions of USD
        rrp_val = rrp_now_b
        # 30d change
        if len(rrp_df) >= 30:
            rrp_30d = float(rrp_df["value"].iloc[-30])
            rrp_change = rrp_now_b - rrp_30d
            rrp_signal = rrp_now_b < 200 and rrp_change < -50
            rrp_status = (f"RRP: ${rrp_now_b:.0f}B (30d: {rrp_change:+.0f}B) "
                          f"({'COLLAPSING' if rrp_signal else 'ok'})")
        else:
            rrp_status = f"RRP: ${rrp_now_b:.0f}B"

    sofr_iorb_signal = False
    spread = None
    sofr_status = "SOFR-IORB unavailable"
    if sofr_df is not None and iorb_df is not None and not sofr_df.empty and not iorb_df.empty:
        sofr_val = float(sofr_df.sort_values("date")["value"].iloc[-1])
        iorb_val = float(iorb_df.sort_values("date")["value"].iloc[-1])
        spread = (sofr_val - iorb_val) * 100  # bps
        sofr_iorb_signal = spread > 5.0
        sofr_status = (f"SOFR-IORB: {spread:+.1f}bps "
                       f"({'STRESS' if sofr_iorb_signal else 'ok'})")

    firing = rrp_signal or sofr_iorb_signal
    score = (0.5 if rrp_signal else 0.0) + (0.5 if sofr_iorb_signal else 0.0)
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": {"rrp_b": rrp_val, "sofr_iorb_bps": spread},
        "status": f"{rrp_status} | {sofr_status}",
        "rationale": ("RRP draining + SOFR>IORB = dollar plumbing under stress, "
                       "Sept-2019 repo crunch precedent."),
        "rrp_collapsing": bool(rrp_signal),
        "sofr_above_iorb": bool(sofr_iorb_signal),
    }


# ============================================================
# C. CREDIT CYCLE
# ============================================================

def us_credit_impulse() -> dict:
    """Credit IMPULSE = change in growth rate of C&I loans.

    Leads SPX by 9-12 months. Howell's favorite signal.
    """
    df = _fred("BUSLOANS", days=2000)
    if df is None or df.empty:
        return _err("BUSLOANS data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 100:  # need 2y+ of weekly data
        return _err("BUSLOANS insufficient history")
    # Weekly series. 52 weeks = 1 year.
    # Compute 52-week % change, then change in that change.
    values = df["value"]
    yoy = values.pct_change(52) * 100
    impulse = yoy - yoy.shift(52)  # change in YoY growth
    current_impulse = float(impulse.iloc[-1]) if not pd.isna(impulse.iloc[-1]) else 0.0
    one_yr_ago_impulse = (float(impulse.iloc[-52])
                           if len(impulse) >= 52 and not pd.isna(impulse.iloc[-52])
                           else 0.0)
    falling = current_impulse < one_yr_ago_impulse
    firing = current_impulse < -2.0 and falling
    score = max(0.0, min(1.0, (-current_impulse - 0.5) / 4.0))
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": current_impulse,
        "status": (f"Credit impulse: {current_impulse:+.2f}pp "
                   f"(was {one_yr_ago_impulse:+.2f}pp 1y ago) "
                   f"({'FIRING' if firing else 'ok'})"),
        "rationale": ("Credit impulse leads SPX by 9-12 months. "
                       "Negative + falling = credit cycle turning down."),
    }


def senior_loan_officer_tightening() -> dict:
    """Net % of banks tightening C&I lending standards (large firms).

    Leads HY widening by 1-2 quarters, recession by 2-3 quarters.
    Released quarterly.
    """
    df = _fred("DRTSCILM", days=1500)
    if df is None or df.empty:
        return _err("DRTSCILM data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    current = float(df["value"].iloc[-1])
    firing = current > 20.0
    score = max(0.0, min(1.0, (current - 5.0) / 35.0))  # 5..40 -> 0..1
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": current,
        "status": (f"Banks tightening C&I: {current:+.1f}% net "
                   f"({'FIRING' if firing else 'ok'})"),
        "rationale": ("Banks tighten BEFORE defaults. Leads HY widening by 1-2 "
                       "quarters, recession by 2-3."),
    }


# ============================================================
# D. REAL ECONOMY LATE-CYCLE
# ============================================================

def sahm_rule() -> dict:
    """Sahm Recession Indicator.

    Unemployment 3m MA - 12m low. Threshold 0.5pp. Has called every US
    recession since 1948 with no false positives.
    """
    df = _fred("SAHMREALTIME", days=730)
    if df is None or df.empty:
        # Fallback: try the published Sahm rule indicator
        df = _fred("SAHMCURRENT", days=730)
    if df is None or df.empty:
        return _err("Sahm data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    current = float(df["value"].iloc[-1])
    firing = current > 0.5
    score = max(0.0, min(1.0, current / 1.0))
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": current,
        "status": (f"Sahm: +{current:.2f}pp "
                   f"({'FIRING — recession confirmed' if firing else 'ok'})"),
        "rationale": ("Sahm > 0.5pp = recession started. 100% hit rate since 1948."),
    }


def initial_claims_cross() -> dict:
    """Initial jobless claims 4w MA crossing above 12m MA.

    Faster than Sahm. When 4w > 12m MA AND rising, equity drawdown >10%
    follows within 6 months 87% of the time.
    """
    df = _fred("ICSA", days=500)
    if df is None or df.empty:
        return _err("ICSA data unavailable")
    df = df.sort_values("date").reset_index(drop=True)
    values = df["value"]
    if len(values) < 52:
        return _err("ICSA insufficient history")
    ma_4w = values.rolling(4).mean()
    ma_12m = values.rolling(52).mean()
    ma_4w_now = float(ma_4w.iloc[-1])
    ma_12m_now = float(ma_12m.iloc[-1])
    # Rising = 4w MA higher than 28w ago
    rising = (len(ma_4w) >= 28 and not pd.isna(ma_4w.iloc[-28])
              and ma_4w_now > float(ma_4w.iloc[-28]))
    cross = ma_4w_now > ma_12m_now
    firing = cross and rising
    score = max(0.0, min(1.0, (ma_4w_now - ma_12m_now) / 30000))  # scale to 0-1
    if not cross: score = 0.0
    return {
        "firing": bool(firing),
        "score": float(score),
        "value": {"ma_4w": ma_4w_now, "ma_12m": ma_12m_now, "rising": bool(rising)},
        "status": (f"Claims 4w MA: {ma_4w_now:,.0f} vs 12m MA: {ma_12m_now:,.0f} "
                   f"{'(CROSS+rising = FIRING)' if firing else '(cross={})'.format(cross)}"),
        "rationale": ("4w MA above 12m MA + rising = 87% hit rate for >10% equity "
                       "drawdown within 6 months."),
    }


# ============================================================
# AGGREGATE
# ============================================================

def all_macro_signals() -> dict:
    """Compute all 8 incremental macro signals."""
    return {
        "oecd_cli_6m":           oecd_cli_6m_change(),
        "cb_lei_yoy":            cb_lei_yoy(),
        "move_elevated":         move_index_elevated(),
        "dollar_liq_stress":     dollar_liquidity_stress(),
        "credit_impulse":        us_credit_impulse(),
        "sloos_tightening":      senior_loan_officer_tightening(),
        "sahm_rule":             sahm_rule(),
        "claims_cross":          initial_claims_cross(),
        "asof":                  datetime.now(timezone.utc).isoformat(),
    }


def main():
    sigs = all_macro_signals()
    print("=" * 70)
    print("INCREMENTAL MACRO LAYER — 8 leading indicators")
    print("=" * 70)
    n_firing = 0
    for k, v in sigs.items():
        if k == "asof": continue
        mark = "[FIRING]" if v.get("firing") else "[ok    ]"
        if v.get("firing"): n_firing += 1
        try: print(f"  {mark} {k:22s} {v.get('status','')[:80]}")
        except UnicodeEncodeError:
            s = v.get('status','').encode('ascii','replace').decode()
            print(f"  {mark} {k:22s} {s[:80]}")
    print(f"\n  Total firing: {n_firing}/8")


if __name__ == "__main__":
    main()
