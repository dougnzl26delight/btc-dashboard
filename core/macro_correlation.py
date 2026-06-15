"""Macro correlation monitor — detect AI-crash regime per Arthur Hayes thesis.

Hayes' thesis (multiple 2026 essays): BTC bottomed at $60k. AI-credit-crisis
drives Fed liquidity response which drives BTC to $200k. Risk scenario:
mega-AI-IPO/merger bust that markets can't absorb → BTC drops alongside NQ.

This module detects that regime in REAL TIME so the rig can de-risk before
the worst of the crash propagates to crypto.

Watches:
    - NQ (Nasdaq-100 futures) — AI bubble proxy
    - SPY (S&P 500) — broad risk
    - VIX — vol regime
    - DXY (US dollar index) — Hayes' "liquidity tide" indicator
    - 10Y yield — Fed pivot probability

Signals:
    NQ_24h < -5%      → AI-crash regime; de-risk all longs
    VIX > 35           → fear regime; pause new entries
    DXY breakout       → liquidity tightening; reduce crypto exposure
    10Y < 3.5%         → Fed pivot priced in; increase crypto exposure
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
import pandas as pd


CACHE_DIR = Path(__file__).resolve().parent.parent / ".macro_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 1800  # 30 min freshness for macro data

SYMBOLS = {
    "NQ": "QQQ",       # Nasdaq-100 proxy (QQQ ETF — yfinance reliable)
    "SPY": "SPY",       # S&P 500
    "VIX": "^VIX",      # vol index
    "DXY": "DX-Y.NYB",  # dollar index
    "TNX": "^TNX",      # 10Y yield (×10 in display)
}

# === Regime thresholds (per Hayes + macro practitioner consensus) ===
NQ_CRASH_THRESHOLD_24H = -0.05    # NQ -5% in 24h = AI crash regime
NQ_HARD_CRASH_24H = -0.10          # NQ -10% in 24h = full kill switch
VIX_FEAR_THRESHOLD = 35             # VIX > 35 = pause new entries
VIX_EXTREME_THRESHOLD = 50          # VIX > 50 = de-risk
DXY_TIGHTENING_THRESHOLD_5D = 0.03  # DXY +3% in 5d = liquidity tightening
TNX_FED_PIVOT_THRESHOLD = 3.5      # 10Y < 3.5% = pivot priced in (yfinance returns actual yield)


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "").replace(".", "_")
    return CACHE_DIR / f"{safe}.json"


def _cached_history(symbol: str, days: int = 30) -> pd.DataFrame:
    cp = _cache_path(symbol)
    if cp.exists() and time.time() - cp.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            return pd.read_json(cp, orient="split")
        except Exception:
            pass
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{max(days, 30)}d")
        if not hist.empty:
            hist.to_json(cp, orient="split")
        return hist
    except Exception:
        return pd.DataFrame()


def latest_metrics() -> dict:
    """Latest readings for all macro indicators."""
    out = {}
    for name, sym in SYMBOLS.items():
        df = _cached_history(sym, days=30)
        if df.empty or len(df) < 5:
            out[name] = {"value": None, "available": False}
            continue
        latest = float(df["Close"].iloc[-1])
        ret_1d = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) if len(df) > 1 else 0
        ret_5d = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) if len(df) > 5 else 0
        ret_20d = float(df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) if len(df) > 20 else 0
        out[name] = {
            "value": latest,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "available": True,
        }
    return out


def regime_status() -> dict:
    """Classify current macro regime + triggered alerts."""
    m = latest_metrics()
    flags = []
    de_risk_level = 0  # 0=normal, 1=caution, 2=de-risk, 3=full-kill

    if m.get("NQ", {}).get("available"):
        nq_1d = m["NQ"]["ret_1d"]
        if nq_1d < NQ_HARD_CRASH_24H:
            flags.append(f"NQ HARD CRASH: {nq_1d*100:+.1f}% in 24h (threshold {NQ_HARD_CRASH_24H*100:.0f}%)")
            de_risk_level = max(de_risk_level, 3)
        elif nq_1d < NQ_CRASH_THRESHOLD_24H:
            flags.append(f"NQ CRASH: {nq_1d*100:+.1f}% in 24h (Hayes AI-bust regime)")
            de_risk_level = max(de_risk_level, 2)

    if m.get("VIX", {}).get("available"):
        vix = m["VIX"]["value"]
        if vix > VIX_EXTREME_THRESHOLD:
            flags.append(f"VIX EXTREME: {vix:.1f} > {VIX_EXTREME_THRESHOLD}")
            de_risk_level = max(de_risk_level, 2)
        elif vix > VIX_FEAR_THRESHOLD:
            flags.append(f"VIX FEAR: {vix:.1f} > {VIX_FEAR_THRESHOLD}")
            de_risk_level = max(de_risk_level, 1)

    if m.get("DXY", {}).get("available"):
        dxy_5d = m["DXY"]["ret_5d"]
        if dxy_5d > DXY_TIGHTENING_THRESHOLD_5D:
            flags.append(f"DXY TIGHTENING: {dxy_5d*100:+.1f}% in 5d (liquidity headwind)")
            de_risk_level = max(de_risk_level, 1)

    if m.get("TNX", {}).get("available"):
        tnx = m["TNX"]["value"]
        if tnx < TNX_FED_PIVOT_THRESHOLD:
            flags.append(f"10Y YIELD LOW: {tnx:.2f}% < {TNX_FED_PIVOT_THRESHOLD}% (Fed pivot priced in — bullish crypto)")

    regime_label = {0: "normal", 1: "caution", 2: "de-risk", 3: "FULL_KILL"}[de_risk_level]
    return {
        "regime": regime_label,
        "de_risk_level": de_risk_level,
        "flags": flags,
        "metrics": m,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_position_size_scale() -> float:
    """Macro-driven scale multiplier for any sleeve [0.0, 1.0]."""
    s = regime_status()
    return {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.0}[s["de_risk_level"]]


def main():
    s = regime_status()
    print("=" * 80)
    print(f"MACRO REGIME — {s['regime'].upper()}  (de_risk_level={s['de_risk_level']})")
    print("=" * 80)
    print()
    print("Latest readings:")
    for name, d in s["metrics"].items():
        if not d.get("available"):
            print(f"  {name:<6s} unavailable")
            continue
        print(f"  {name:<6s} val={d['value']:>9.2f}  1d={d['ret_1d']*100:>+5.1f}%  "
              f"5d={d['ret_5d']*100:>+5.1f}%  20d={d['ret_20d']*100:>+5.1f}%")
    print()
    if s["flags"]:
        print("Regime flags:")
        for f in s["flags"]:
            print(f"  ! {f}")
    else:
        print("No regime flags — normal trading conditions.")
    print()
    print(f"Position size scale for sleeves: {get_position_size_scale():.2f}x")


if __name__ == "__main__":
    main()
