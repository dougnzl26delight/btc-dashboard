"""Top 1% Predictor Engine — live coordinator.

Pulls everything together for the dashboard:
  - Standardizes current signal values to z-scores
  - Computes IC table (cached on disk, refreshed weekly)
  - Computes theme composites (LIQUIDITY, CREDIT, GROWTH, VALUATION, SENTIMENT, BTC_ONCHAIN)
  - Derives decision composites (top, early, bottom)
  - Computes BTC state (DEEP/SHALLOW/UNCONFIRMED)
  - Computes target allocation (Kelly + vol-target + DD brake + regime mult)
  - Runs failure checks
  - Returns one big dict the dashboard renders
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
IC_REFRESH_DAYS = 7


def _yf_close(ticker: str, period: str = "10y") -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty: return None
        s = df["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def _safe_z(values: list[float]) -> Optional[float]:
    """Fast z-score helper for sparse historical data."""
    arr = np.array([v for v in values if v is not None and not pd.isna(v)])
    if len(arr) < 30: return None
    return float((arr[-1] - arr.mean()) / arr.std()) if arr.std() > 0 else 0.0


# ============================================================
# Build current signal snapshot from existing modules
# ============================================================

def _current_signal_snapshot() -> dict[str, float]:
    """Collect current standardized z-scores via the signal registry.

    Returns flat {canonical_name: current_z_score}.
    Names match the canonical theme members in composites.py exactly,
    so theme composites compute properly.
    """
    from core.signal_registry import fetch_all_current
    snap = {}
    sigs = fetch_all_current()
    for name, info in sigs.items():
        z = info.get("z")
        if z is not None and not pd.isna(z):
            snap[name] = float(z)
    return snap


# ============================================================
# IC table — refreshed weekly
# ============================================================

def _ic_table_stale() -> bool:
    """Check if cached IC table is older than IC_REFRESH_DAYS."""
    from core.research.ic_table import IC_CACHE
    if not IC_CACHE.exists(): return True
    age_days = (time.time() - IC_CACHE.stat().st_mtime) / 86400
    return age_days > IC_REFRESH_DAYS


def refresh_ic_table() -> dict:
    """Compute IC table using the full signal registry.

    Returns summary {n_signals, top_5_by_ic_spy, top_5_by_ic_btc}.
    """
    from core.research.ic_table import build_ic_table, save_ic_table
    from core.signal_registry import fetch_all_historical

    spy = _yf_close("SPY", period="20y")
    btc = _yf_close("BTC-USD", period="10y")
    if spy is None: return {"error": "SPY history unavailable"}

    spy_ret = spy.pct_change().dropna()
    btc_ret = btc.pct_change().dropna() if btc is not None else None

    # Use the full registry — all 30 canonical signals
    signals = fetch_all_historical()
    # Also add raw price tickers as anchors
    signals["SPY_close"] = spy
    if btc is not None: signals["BTC_close"] = btc

    df_ic = build_ic_table(signals, spy_ret, btc_ret)
    save_ic_table(df_ic)

    # Summary
    top_spy = df_ic.head(5)[["signal", "best_ic_spy", "best_h_spy"]].to_dict("records")
    if "best_ic_btc" in df_ic.columns:
        df_btc = df_ic.sort_values("best_ic_btc", key=lambda c: c.abs(),
                                     ascending=False).head(5)
        top_btc = df_btc[["signal", "best_ic_btc", "best_h_btc"]].to_dict("records")
    else:
        top_btc = []

    return {
        "n_signals": len(df_ic),
        "n_passing_spy_oos_gate": int(df_ic["oos_pass_spy"].fillna(False).sum()),
        "top_5_signals_by_ic_spy": top_spy,
        "top_5_signals_by_ic_btc": top_btc,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# Main coordinator
# ============================================================

def predictor_engine_state(total_stake_nzd: float = 130_000,
                              current_equity_pct: float = 30.0,
                              current_btc_pct: float = 0.0) -> dict:
    """The big one. Single call → full engine state for the dashboard."""
    # 1. IC table refresh if stale
    ic_summary = None
    if _ic_table_stale():
        try: ic_summary = refresh_ic_table()
        except Exception as e: ic_summary = {"error": str(e)[:80]}
    else:
        from core.research.ic_table import load_ic_table
        tbl = load_ic_table()
        ic_summary = {
            "n_signals": len(tbl),
            "cached": True,
        }

    # 2. Load IC weights
    from core.research.ic_table import load_ic_weights
    ic_weights = load_ic_weights()

    # 3. Build current standardized signal snapshot
    raw_snap = _current_signal_snapshot()
    # Approximate z-scores: treat raw value as z if already z-like,
    # else use 0. Live system would call standardize.py with full history.
    standardized = {k: {"z": v, "raw": v} for k, v in raw_snap.items()
                    if isinstance(v, (int, float)) and not pd.isna(v)}

    # 4. Theme composites
    from core.composites import (compute_all_themes,
                                    composite_scores_for_decisions,
                                    THEME_DEFINITIONS)
    themes = compute_all_themes(standardized, ic_weights)
    decisions = composite_scores_for_decisions(themes)

    # 5. Regime (use existing rule-based)
    from core.btc_unified_decision import unified_decision
    ud = unified_decision(current_equity_pct=current_equity_pct,
                            current_btc_pct=current_btc_pct,
                            total_stake_nzd=total_stake_nzd)
    regime = ud["regime"]
    vetoes = ud["vetoes_active"]

    # 6. BTC state — use canonical signal_registry names so binding actually
    # passes the data through (was silently dropping signals due to name
    # mismatch: registry uses 'rcap_drawdown' + 'etf_flow_60d' not the older
    # 'realized_cap_drawdown' / 'etf_flow_60d_z' names).
    from core.btc_state import btc_bottom_composite, btc_entry_plan
    btc_input = {
        "mvrv_z":                raw_snap.get("mvrv_z"),
        "asopr":                 raw_snap.get("asopr"),
        "realized_cap_drawdown": raw_snap.get("rcap_drawdown"),  # registry name
        "reserve_risk":          raw_snap.get("reserve_risk"),
        "sth_mvrv":              raw_snap.get("sth_mvrv"),
        "puell":                 raw_snap.get("puell"),
        "hashrate_drawdown":     raw_snap.get("hashrate_drawdown"),
        "etf_flow_60d_z":        raw_snap.get("etf_flow_60d"),  # registry name
        "etf_flow_30d_z":        raw_snap.get("etf_flow_60d"),  # use same proxy
    }
    btc_state = btc_bottom_composite(btc_input)
    btc_plan = btc_entry_plan(btc_state["state"],
                                available_capital=total_stake_nzd * 0.3,
                                realized_vol_60d=0.60)

    # 7. Realized vols
    from core.position_size import realized_vol, compute_target_allocation
    spy_series = _yf_close("SPY", period="6mo")
    btc_series = _yf_close("BTC-USD", period="6mo")
    realized_vols = {
        "SPY": realized_vol(spy_series.pct_change(), window=60) if spy_series is not None else 0.18,
        "BTC": realized_vol(btc_series.pct_change(), window=60) if btc_series is not None else 0.60,
    }

    # 8. Position-size allocation (this is the calibrated one)
    sized = compute_target_allocation(
        composite_scores=decisions,
        regime=regime,
        realized_vols=realized_vols,
        current_drawdown=0.0,  # no live equity curve yet
        vetoes=vetoes,
        kelly_fraction=0.25,
        portfolio_vol_target=0.12,
        total_stake=total_stake_nzd,
    )

    # 9. Failure checks (use placeholders if no equity curve yet)
    from core.failure_detection import run_all_failure_checks
    failures = run_all_failure_checks(
        engine_returns=None, bench_returns=None,
        equity_curve=None,
        regime_history=None,
        hmm_agreement=None,
        rolling_24m_ic=None,
    )

    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "ic_table": ic_summary,
        "raw_signal_snapshot_n": len(raw_snap),
        "theme_composites": {
            theme: {"z": round(r.get("composite_z", 0.0), 2),
                    "n_signals": r.get("n_members_present", 0),
                    "n_defined":  r.get("n_members_defined", 0)}
            for theme, r in themes.items()
        },
        "decision_composites": {k: round(v, 2) for k, v in decisions.items()},
        "regime": regime,
        "regime_buckets": ud["regime_buckets"],
        "vetoes_active": vetoes,
        "btc_state": {
            "state": btc_state["state"],
            "composite_z": round(btc_state["composite_z"], 2),
            "bottom_probability": round(btc_state["bottom_probability"], 3),
            "n_components_used": btc_state.get("n_components", 0),
        },
        "btc_entry_plan": btc_plan,
        "realized_vols": {k: round(v, 3) for k, v in realized_vols.items()},
        "calibrated_allocation": sized,
        "rule_based_allocation": ud["target_allocation_pct"],  # the old engine
        "staging_basket": ud["staging_basket_pct"],
        "failure_checks": failures,
    }


def main():
    r = predictor_engine_state()
    print("=" * 70)
    print(f"PREDICTOR ENGINE — Regime: {r['regime']}")
    print("=" * 70)
    print(f"  IC table: {r['ic_table']}")
    print(f"  Signal snapshot: {r['raw_signal_snapshot_n']} signals")
    print(f"  Theme composites:")
    for theme, t in r["theme_composites"].items():
        print(f"    {theme:14s}  z={t['z']:+.2f}  ({t['n_signals']}/{t['n_defined']})")
    print(f"  Decision composites: {r['decision_composites']}")
    print(f"  BTC state: {r['btc_state']}")
    print(f"  Calibrated allocation: {r['calibrated_allocation']['weights_pct']}")
    print(f"  Rule-based allocation: {r['rule_based_allocation']}")
    print(f"  Failures: {r['failure_checks']['n_failures']}")


if __name__ == "__main__":
    main()
