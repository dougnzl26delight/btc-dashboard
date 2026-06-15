"""Macro-regime state machine — RISK_ON / LATE_CYCLE / RECESSIONARY_BEAR.

Combines:
  - Incremental macro layer (btc_macro_layer.all_macro_signals)
  - Existing leading indicators (btc_early_rotation)
  - Yield-curve un-inversion (computed locally for veto)

Outputs:
  - regime: one of three labels
  - bucket scores: growth (0-4), plumbing (0-4), credit (0-3)
  - active vetoes: list of override rule names
  - threshold table for scorecards in current regime
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


Regime = Literal["RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"]


# ============================================================
# Regime-dependent threshold table
# ============================================================
# Same scorecards, different trigger levels per regime.
# Logic: late-cycle = act earlier on weaker signals.

REGIME_THRESHOLDS = {
    "RISK_ON": {
        "top": {
            "trim":           4,
            "defensive":      6,
            "bear_confirmed": 8,
            "full_rotation":  9,
        },
        "early": {
            "watch":           2,
            "reduce":          4,
            "rotate_to_cash":  5,
        },
        "btc_partial": 5,
        "btc_full":    7,
        "max_btc_pct": 50.0,
        "baseline_equity_pct": 30.0,
    },
    "LATE_CYCLE": {
        "top": {
            "trim":           3,
            "defensive":      5,
            "bear_confirmed": 7,
            "full_rotation":  9,
        },
        "early": {
            "watch":           1,
            "reduce":          3,
            "rotate_to_cash":  3,
        },
        "btc_partial": 4,
        "btc_full":    6,
        "max_btc_pct": 70.0,
        "baseline_equity_pct": 30.0,
    },
    "RECESSIONARY_BEAR": {
        "top": {
            "trim":           2,
            "defensive":      3,
            "bear_confirmed": 5,
            "full_rotation":  7,
        },
        "early": {
            "watch":           0,
            "reduce":          2,
            "rotate_to_cash":  2,
        },
        "btc_partial": 3,
        "btc_full":    5,
        "max_btc_pct": 100.0,
        "baseline_equity_pct": 30.0,
    },
}


# ============================================================
# Yield-curve un-inversion (for veto check)
# ============================================================

def yield_curve_uninverting() -> dict:
    """True if T10Y2Y was inverted in last 180d AND now back > 0.

    This is THE recession-start signal — un-inversion typically happens
    0-6 months before recession officially begins.
    """
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("T10Y2Y", days=400)
        if df is None or df.empty:
            return {"firing": False, "status": "data unavailable"}
        df = df.sort_values("date").reset_index(drop=True)
        current = float(df["value"].iloc[-1])
        min_180d = float(df["value"].tail(180).min())
        was_inverted = min_180d < -0.2
        now_positive = current > 0.0
        firing = was_inverted and now_positive
        return {
            "firing": bool(firing),
            "current": current,
            "min_180d": min_180d,
            "status": f"T10Y2Y now {current:+.2f}, 180d min {min_180d:+.2f}",
        }
    except Exception as e:
        return {"firing": False, "status": f"error: {e!r}"[:80]}


# ============================================================
# Regime bucket scoring
# ============================================================

def regime_buckets(macro: dict, curve_uninvert: bool, early_rot: Optional[dict] = None) -> dict:
    """Score the three regime buckets from raw signals.

    Args:
      macro: output of btc_macro_layer.all_macro_signals()
      curve_uninvert: yield_curve_uninverting()["firing"]
      early_rot: optional early_rotation_signal() output (for HY widening)

    Returns:
      {"growth": int 0-4, "plumbing": int 0-4, "credit": int 0-3}
    """
    # GROWTH bucket
    growth = 0
    if macro.get("oecd_cli_6m", {}).get("firing"): growth += 1
    if macro.get("cb_lei_yoy", {}).get("firing"):  growth += 1
    if macro.get("claims_cross", {}).get("firing"): growth += 1
    if macro.get("sahm_rule", {}).get("firing"):    growth += 1

    # PLUMBING bucket
    plumbing = 0
    move_d = macro.get("move_elevated", {})
    if move_d.get("firing") or move_d.get("extreme"): plumbing += 1
    liq = macro.get("dollar_liq_stress", {})
    if liq.get("rrp_collapsing"): plumbing += 1
    if liq.get("sofr_above_iorb"): plumbing += 1
    # HY widening — pull from early rotation if available
    if early_rot:
        hy = early_rot.get("indicators", {}).get("hy_spread_widening", {})
        if hy.get("firing"): plumbing += 1

    # CREDIT bucket
    credit = 0
    if macro.get("credit_impulse", {}).get("firing"):     credit += 1
    if macro.get("sloos_tightening", {}).get("firing"):   credit += 1
    if curve_uninvert: credit += 1

    return {"growth": growth, "plumbing": plumbing, "credit": credit}


# ============================================================
# Regime classification
# ============================================================

def classify_regime(buckets: dict, macro: dict, curve_uninvert: bool) -> Regime:
    growth = buckets["growth"]
    plumbing = buckets["plumbing"]
    credit = buckets["credit"]

    sahm_firing = macro.get("sahm_rule", {}).get("firing", False)

    # RECESSIONARY_BEAR: hard signals confirmed
    if sahm_firing or curve_uninvert:
        return "RECESSIONARY_BEAR"
    if growth >= 3 or (growth >= 2 and plumbing >= 2):
        return "RECESSIONARY_BEAR"

    # LATE_CYCLE: warning signs accumulating
    if growth >= 1 or plumbing >= 2 or credit >= 1:
        return "LATE_CYCLE"

    # RISK_ON: clean bill of health
    return "RISK_ON"


# ============================================================
# Veto/override rules
# ============================================================

def compute_vetoes(regime: Regime, macro: dict, curve_uninvert: bool,
                    liquidity_z: float, credit_impulse_value: float) -> list[str]:
    """List of active veto rule names.

    These OVERRIDE the scorecard math:
      - force_cash_move_spike: MOVE > 150
      - no_btc_during_collapse: regime BEAR + liq z<-1 + credit impulse<-3
      - no_equity_add_recession_start: curve un-inverted + claims rising
    """
    active = []

    move_v = macro.get("move_elevated", {}).get("value")
    if move_v is not None and move_v > 150:
        active.append("force_cash_move_spike")

    if (regime == "RECESSIONARY_BEAR"
        and liquidity_z < -1.0
        and credit_impulse_value < -3.0):
        active.append("no_btc_during_collapse")

    claims = macro.get("claims_cross", {})
    if curve_uninvert and claims.get("firing"):
        active.append("no_equity_add_recession_start")

    return active


# ============================================================
# Top-level regime function
# ============================================================

def full_regime_analysis(macro: dict, liquidity_z: float = 0.0,
                          credit_impulse_value: Optional[float] = None,
                          early_rot: Optional[dict] = None) -> dict:
    """One-call wrapper.

    Returns:
      {regime, buckets, vetoes, thresholds, curve_uninvert}
    """
    curve_d = yield_curve_uninverting()
    curve_uninvert = curve_d.get("firing", False)

    if credit_impulse_value is None:
        credit_impulse_value = macro.get("credit_impulse", {}).get("value") or 0.0

    buckets = regime_buckets(macro, curve_uninvert, early_rot)
    regime = classify_regime(buckets, macro, curve_uninvert)
    vetoes = compute_vetoes(regime, macro, curve_uninvert, liquidity_z,
                              credit_impulse_value)

    return {
        "regime": regime,
        "buckets": buckets,
        "vetoes": vetoes,
        "thresholds": REGIME_THRESHOLDS[regime],
        "curve_uninvert": curve_uninvert,
        "curve_status": curve_d.get("status", ""),
    }


def main():
    from core.btc_macro_layer import all_macro_signals
    macro = all_macro_signals()
    result = full_regime_analysis(macro, liquidity_z=0.0)
    print("=" * 70)
    print(f"REGIME: {result['regime']}")
    print("=" * 70)
    b = result["buckets"]
    print(f"  Growth:    {b['growth']}/4")
    print(f"  Plumbing:  {b['plumbing']}/4")
    print(f"  Credit:    {b['credit']}/3")
    print(f"  Curve un-invert: {result['curve_uninvert']}")
    print(f"  Vetoes active:   {result['vetoes'] or 'none'}")
    print(f"  Thresholds for this regime:")
    for k, v in result["thresholds"].items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
