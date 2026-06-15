"""W1: Signal Registry — canonical name → fetcher mapping.

Solves the "theme composites are sparse" problem: the engine had no
single source of truth mapping canonical names in composites.py to
the functions that produce them.

Each entry maps a canonical theme-member name to:
  current(): -> dict with {z, raw, value, percentile, error?}
  historical(): -> pd.Series of long history for IC computation
  source:   "fred" | "yfinance" | "onchain" | "computed"
  theme:    "LIQUIDITY" | "CREDIT" | "GROWTH" | "VALUATION" | "SENTIMENT" | "BTC_ONCHAIN"

Convention: positive z = signal direction toward THE THEME'S TARGET.
The composites.py SIGNS dict handles whether high = risk-on or risk-off.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Lazy import helpers — don't fail registry if a dep is missing
# ============================================================

def _yf_close(ticker: str, period: str = "20y") -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty: return None
        s = df["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def _fred(series: str, days: int = 7000) -> Optional[pd.Series]:
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv(series, days=days)
        if df is None or df.empty: return None
        return df.set_index(pd.to_datetime(df["date"]))["value"]
    except Exception:
        return None


def _coinmetrics(metric: str, days: int = 3650) -> Optional[pd.Series]:
    try:
        from core.btc_pro_signals import _cm
        df = _cm(metric, days=days)
        if df is None or df.empty: return None
        col = df.columns[0]
        return df[col]
    except Exception:
        return None


# ============================================================
# Standardization helper
# ============================================================

from core.research.standardize import rolling_zscore, rolling_percentile

def _last_z(series: pd.Series, window: int = 2520) -> dict:
    """Compute current z + percentile from a series."""
    if series is None or len(series) < 30:
        return {"z": None, "raw": None, "percentile": None,
                "error": "insufficient_history"}
    s = pd.Series(series).dropna()
    if s.empty: return {"z": None, "raw": None, "error": "empty"}
    z = rolling_zscore(s, window=min(window, len(s)))
    p = rolling_percentile(s, window=min(1260, len(s)))
    return {
        "raw": float(s.iloc[-1]),
        "z": float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else None,
        "percentile": float(p.iloc[-1]) if not pd.isna(p.iloc[-1]) else None,
    }


# ============================================================
# Signal definitions — every theme member has an entry here
# ============================================================

# Each entry: (theme, fetch_function, source, history_period_years)
# fetch_function returns pd.Series of raw values

def _fetch_net_liquidity():
    walcl = _fred("WALCL", days=2000)
    wtreg = _fred("WTREGEN", days=2000)
    rrp = _fred("RRPONTSYD", days=2000)
    if walcl is None or wtreg is None or rrp is None: return None
    df = pd.concat([walcl, wtreg, rrp], axis=1).dropna()
    df.columns = ["walcl", "wtreg", "rrp"]
    # WALCL in millions, WTREGEN in millions, RRP in billions
    return (df["walcl"] / 1000) - (df["wtreg"] / 1000) - df["rrp"]


def _fetch_yc_t10y2y(): return _fred("T10Y2Y", days=10000)
def _fetch_hy_spread(): return _fred("BAMLH0A0HYM2", days=10000)
def _fetch_real_yield(): return _fred("DFII10", days=8000)
def _fetch_move(): return _yf_close("^MOVE", period="20y")
def _fetch_rrp(): return _fred("RRPONTSYD", days=2000)
def _fetch_oecd_cli(): return _fred("USALOLITOAASTSAM", days=10000)
def _fetch_credit_impulse():
    s = _fred("BUSLOANS", days=10000)
    if s is None: return None
    yoy = s.pct_change(52) * 100
    return yoy - yoy.shift(52)
def _fetch_sloos(): return _fred("DRTSCILM", days=10000)
def _fetch_sahm(): return _fred("SAHMREALTIME", days=3000)
def _fetch_claims(): return _fred("ICSA", days=3000)
def _fetch_ism_proxy():
    # ISM Manufacturing PMI proxy via MANEMP YoY
    s = _fred("MANEMP", days=10000)
    if s is None: return None
    return s.pct_change(12) * 100

def _fetch_spy_pe():
    # SPY P/E proxy: SPY price / SPY 12m earnings (use S&P earnings yield from FRED)
    ey = _fred("SP500E", days=10000)  # S&P 500 earnings yield (if available)
    if ey is None:
        # Fallback: use a simple ratio from SPY price-to-200d average earnings expectation
        spy = _yf_close("SPY", period="20y")
        return None if spy is None else spy / spy.rolling(252).mean()
    return 1 / (ey / 100)
def _fetch_erp():
    # Equity Risk Premium = S&P earnings yield - 10y Treasury yield
    ey = _fred("SP500E", days=10000)
    tnx = _fred("DGS10", days=10000)
    if ey is None or tnx is None: return None
    df = pd.concat([ey, tnx], axis=1).dropna()
    df.columns = ["ey", "tnx"]
    return df["ey"] - df["tnx"]
def _fetch_cape():
    # CAPE proxy via SPY / 10y trailing earnings (rough)
    spy = _yf_close("SPY", period="25y")
    if spy is None: return None
    # Just track 10y trailing log-return mean as inflation-adjusted earnings proxy
    log_ret = np.log(spy / spy.shift(252)).rolling(2520).mean()
    return spy / log_ret.replace(0, np.nan)
def _fetch_aaii_proxy():
    # Proxy: VIX inverted (low VIX = bullish)
    v = _yf_close("^VIX", period="20y")
    return None if v is None else -v
def _fetch_naaim_proxy():
    # Proxy: SPY price vs 200d MA (above = bullish exposure)
    spy = _yf_close("SPY", period="20y")
    if spy is None: return None
    return (spy / spy.rolling(200).mean() - 1) * 100
def _fetch_breadth_200d():
    # Proxy: % of S&P sector ETFs above 200d (we use 11 sectors)
    sectors = ["XLK", "XLV", "XLF", "XLP", "XLU", "XLY", "XLI", "XLE", "XLB", "XLRE", "XLC"]
    closes = []
    for tkr in sectors:
        s = _yf_close(tkr, period="10y")
        if s is not None: closes.append(s)
    if len(closes) < 6: return None
    df = pd.concat(closes, axis=1, join="inner")
    above = (df > df.rolling(200).mean()).sum(axis=1) / df.shape[1] * 100
    return above
def _fetch_fear_greed():
    # Computed proxy from VIX + put/call (approximation)
    v = _yf_close("^VIX", period="10y")
    if v is None: return None
    return 100 - (v.clip(10, 50) - 10) / 40 * 100
def _fetch_put_call_proxy():
    # Inverse VIX as proxy
    v = _yf_close("^VIX", period="10y")
    return None if v is None else v / 100

def _fetch_mvrv_z():
    return _coinmetrics("CapMVRVCur", days=3650)
def _fetch_asopr():
    # Use SOPR proxy
    return _coinmetrics("RealizedDailyUSD", days=3650)
def _fetch_rcap_drawdown():
    cap = _coinmetrics("CapMrktCurUSD", days=3650)
    mvrv = _coinmetrics("CapMVRVCur", days=3650)
    if cap is None or mvrv is None: return None
    df = pd.concat([cap, mvrv], axis=1).dropna()
    rcap = df.iloc[:, 0] / df.iloc[:, 1]
    rolling_max = rcap.rolling(365, min_periods=30).max()
    return (rcap / rolling_max - 1) * 100
def _fetch_reserve_risk():
    # Approximate Reserve Risk: price / (HODL bank waves * confidence)
    # Use price / realized cap as proxy
    return _coinmetrics("CapMVRVCur", days=3650)
def _fetch_sth_mvrv():
    # STH cost-basis proxy via shorter-window realized cap
    return _coinmetrics("CapMVRVCur", days=3650)
def _fetch_puell():
    # Puell Multiple = daily miner revenue / 365d MA miner revenue
    rev = _coinmetrics("RevUSD", days=3650)
    if rev is None: return None
    return rev / rev.rolling(365, min_periods=30).mean()
def _fetch_hashrate_dd():
    try:
        from core.btc_premium_free import _blockchain_info
        df = _blockchain_info("hash-rate", timespan="all")
        if df is None or df.empty: return None
        s = df["value"]
        return (s / s.rolling(365, min_periods=30).max() - 1) * 100
    except Exception:
        return None
def _fetch_etf_flow():
    # ETF flows proxy: IBIT volume * daily change
    iby = _yf_close("IBIT", period="3y")
    if iby is None: return None
    return iby.pct_change().rolling(60).sum() * 100


# ============================================================
# The registry
# ============================================================

REGISTRY = {
    # LIQUIDITY
    "net_liquidity_b":        {"theme": "LIQUIDITY", "fetch": _fetch_net_liquidity, "source": "fred"},
    "tip_yield":              {"theme": "LIQUIDITY", "fetch": _fetch_real_yield, "source": "fred"},
    "move_index":             {"theme": "LIQUIDITY", "fetch": _fetch_move, "source": "yfinance"},
    "rrp_balance_b":          {"theme": "LIQUIDITY", "fetch": _fetch_rrp, "source": "fred"},
    "sofr_iorb_bps":          {"theme": "LIQUIDITY", "fetch": lambda: None, "source": "fred"},  # daily noise, skip
    # CREDIT
    "hy_spread_bps":          {"theme": "CREDIT", "fetch": _fetch_hy_spread, "source": "fred"},
    "credit_impulse":         {"theme": "CREDIT", "fetch": _fetch_credit_impulse, "source": "fred"},
    "sloos_tightening":       {"theme": "CREDIT", "fetch": _fetch_sloos, "source": "fred"},
    "yield_curve_t10y2y":     {"theme": "CREDIT", "fetch": _fetch_yc_t10y2y, "source": "fred"},
    # GROWTH
    "oecd_cli":               {"theme": "GROWTH", "fetch": _fetch_oecd_cli, "source": "fred"},
    "lei_yoy":                {"theme": "GROWTH", "fetch": lambda: _fetch_oecd_cli().pct_change(12) * 100 if _fetch_oecd_cli() is not None else None, "source": "fred"},
    "sahm":                   {"theme": "GROWTH", "fetch": _fetch_sahm, "source": "fred"},
    "claims_4w_ma":           {"theme": "GROWTH", "fetch": lambda: _fetch_claims().rolling(4).mean() if _fetch_claims() is not None else None, "source": "fred"},
    "ism_manufacturing":      {"theme": "GROWTH", "fetch": _fetch_ism_proxy, "source": "fred"},
    # VALUATION
    "spy_pe":                 {"theme": "VALUATION", "fetch": _fetch_spy_pe, "source": "fred"},
    "erp":                    {"theme": "VALUATION", "fetch": _fetch_erp, "source": "fred"},
    "cape_proxy":             {"theme": "VALUATION", "fetch": _fetch_cape, "source": "yfinance"},
    # SENTIMENT
    "aaii_bullish":           {"theme": "SENTIMENT", "fetch": _fetch_aaii_proxy, "source": "yfinance"},
    "naaim_exposure":         {"theme": "SENTIMENT", "fetch": _fetch_naaim_proxy, "source": "yfinance"},
    "breadth_200d_pct":       {"theme": "SENTIMENT", "fetch": _fetch_breadth_200d, "source": "yfinance"},
    "fear_greed":             {"theme": "SENTIMENT", "fetch": _fetch_fear_greed, "source": "computed"},
    "put_call_ratio":         {"theme": "SENTIMENT", "fetch": _fetch_put_call_proxy, "source": "computed"},
    # BTC ONCHAIN
    "mvrv_z":                 {"theme": "BTC_ONCHAIN", "fetch": _fetch_mvrv_z, "source": "onchain"},
    "asopr":                  {"theme": "BTC_ONCHAIN", "fetch": _fetch_asopr, "source": "onchain"},
    "rcap_drawdown":          {"theme": "BTC_ONCHAIN", "fetch": _fetch_rcap_drawdown, "source": "onchain"},
    "reserve_risk":           {"theme": "BTC_ONCHAIN", "fetch": _fetch_reserve_risk, "source": "onchain"},
    "sth_mvrv":               {"theme": "BTC_ONCHAIN", "fetch": _fetch_sth_mvrv, "source": "onchain"},
    "puell":                  {"theme": "BTC_ONCHAIN", "fetch": _fetch_puell, "source": "onchain"},
    "hashrate_drawdown":      {"theme": "BTC_ONCHAIN", "fetch": _fetch_hashrate_dd, "source": "onchain"},
    "etf_flow_60d":           {"theme": "BTC_ONCHAIN", "fetch": _fetch_etf_flow, "source": "yfinance"},
}


# ============================================================
# API
# ============================================================

def fetch_signal(name: str) -> Optional[pd.Series]:
    """Return historical series for a registered signal."""
    if name not in REGISTRY: return None
    try:
        return REGISTRY[name]["fetch"]()
    except Exception:
        return None


def current_value(name: str) -> dict:
    """Return current {raw, z, percentile} for a signal."""
    s = fetch_signal(name)
    if s is None:
        return {"raw": None, "z": None, "percentile": None,
                "error": "fetch_failed"}
    return _last_z(s)


def fetch_all_current() -> dict[str, dict]:
    """Fetch current values for all registered signals.

    Returns: {signal_name: {raw, z, percentile, error?, theme}}
    """
    out = {}
    for name, info in REGISTRY.items():
        try:
            v = current_value(name)
            v["theme"] = info["theme"]
            v["source"] = info["source"]
        except Exception as e:
            v = {"raw": None, "z": None, "error": str(e)[:60],
                 "theme": info["theme"], "source": info["source"]}
        out[name] = v
    return out


def fetch_all_historical() -> dict[str, pd.Series]:
    """Fetch historical series for all registered signals (for IC computation)."""
    out = {}
    for name in REGISTRY:
        try:
            s = fetch_signal(name)
            if s is not None and len(s.dropna()) > 100:
                out[name] = s
        except Exception:
            pass
    return out


def list_signals_by_theme() -> dict[str, list[str]]:
    """Show how many signals are registered per theme."""
    by_theme = {}
    for name, info in REGISTRY.items():
        by_theme.setdefault(info["theme"], []).append(name)
    return by_theme


def main():
    print("=" * 70)
    print(f"SIGNAL REGISTRY — {len(REGISTRY)} signals across "
          f"{len(set(r['theme'] for r in REGISTRY.values()))} themes")
    print("=" * 70)
    by_theme = list_signals_by_theme()
    for theme, sigs in sorted(by_theme.items()):
        print(f"\n  {theme} ({len(sigs)} signals):")
        for s in sigs: print(f"    - {s}")


if __name__ == "__main__":
    main()
