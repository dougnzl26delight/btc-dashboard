"""Unified decision engine — combines macro layer + regime + scorecards
+ staging basket into one final target allocation.

Flow:
  1. Collect macro signals  -> btc_macro_layer.all_macro_signals
  2. Compute liquidity z-score (net liquidity over 2y)
  3. Classify regime         -> btc_regime.full_regime_analysis
  4. Compute scorecards with regime-modulated thresholds:
       - Top Confirmation Scorecard
       - Early Rotation Signal
       - BTC Bottom Confirmation Scorecard
  5. Apply vetoes (MOVE>150, recession start, etc)
  6. Compute target allocation (equity / BTC / staging) using continuous
     sizing function — not hard buckets
  7. Split staging across BIL / VTIP / GLDM basket
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
# Liquidity overlay
# ============================================================

def net_liquidity_z() -> dict:
    """Net Liquidity z-score over 2y.

    Net Liquidity = WALCL - WTREGEN - RRPONTSYD
    z > +1 = supportive, z < -1 = hostile, |z| > 2 = extreme.
    """
    try:
        from core.btc_clemente_alden import _fred_csv
        walcl = _fred_csv("WALCL", days=800)
        wtregen = _fred_csv("WTREGEN", days=800)
        rrp = _fred_csv("RRPONTSYD", days=800)
        if any(x is None or x.empty for x in [walcl, wtregen, rrp]):
            return {"z": 0.0, "status": "net liquidity data unavailable",
                     "value_b": None}
        # Align on date
        df = walcl.rename(columns={"value": "walcl"}).merge(
            wtregen.rename(columns={"value": "wtregen"}), on="date", how="inner"
        ).merge(
            rrp.rename(columns={"value": "rrp"}), on="date", how="inner"
        )
        df = df.sort_values("date").reset_index(drop=True)
        # WALCL in millions, WTREGEN in millions, RRP in billions
        # Convert all to billions
        df["net_liq_b"] = (df["walcl"] / 1000) - (df["wtregen"] / 1000) - df["rrp"]
        mean = df["net_liq_b"].mean()
        std = df["net_liq_b"].std()
        current = float(df["net_liq_b"].iloc[-1])
        z = (current - mean) / std if std > 0 else 0.0
        return {
            "z": float(z),
            "value_b": current,
            "mean_b": float(mean),
            "std_b": float(std),
            "status": (f"Net Liquidity ${current:,.0f}B (z={z:+.2f}, "
                       f"mean ${mean:,.0f}B)"),
        }
    except Exception as e:
        return {"z": 0.0, "status": f"liquidity calc error: {e!r}"[:80],
                 "value_b": None}


def real_yield_30d_change() -> float:
    """30d change in 10y real yield. Negative = falling fast = good for risk."""
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("DFII10", days=120)
        if df is None or df.empty or len(df) < 30:
            return 0.0
        df = df.sort_values("date").reset_index(drop=True)
        current = float(df["value"].iloc[-1])
        ago_30 = float(df["value"].iloc[-30])
        return current - ago_30
    except Exception:
        return 0.0


# ============================================================
# Continuous sizing functions
# ============================================================

def compute_btc_target(regime: str, bottom_n: int,
                        threshold_partial: int, threshold_full: int,
                        liquidity_z: float, max_alloc_pct: float,
                        vetoed: bool, current_btc_pct: float = 0.0) -> float:
    """Returns BTC target as % of total stake.

    Continuous ramp from threshold_partial to threshold_full, modulated
    by regime (buy fear) and liquidity z-score.

    SEMANTICS: this is a ROTATION engine. When BTC bottom hasn't fired
    and no veto active, we HOLD current BTC position rather than recommend
    selling. Only trim BTC when:
      - veto active (force sell)
      - in RISK_ON regime with bottom_n above partial (recommend reduce
        from "too long" to baseline)
    Otherwise, ramp the target ABOVE current_btc_pct toward the partial
    threshold when bottom firing.
    """
    if vetoed: return 0.0

    # Base ramp 0..1 from bottom score
    if bottom_n < threshold_partial:
        base = 0.0
    elif bottom_n >= threshold_full:
        base = 1.0
    else:
        base = ((bottom_n - threshold_partial)
                / max(1, (threshold_full - threshold_partial)))

    regime_mult = {
        "RISK_ON":           0.7,
        "LATE_CYCLE":        1.0,
        "RECESSIONARY_BEAR": 1.3,
    }.get(regime, 1.0)

    # Liquidity multiplier: z=0 -> 1.0, z=+2 -> 1.4, z=-2 -> 0.6
    liq_mult = max(0.5, min(1.5, 1.0 + 0.2 * liquidity_z))

    target = base * regime_mult * liq_mult * max_alloc_pct
    target = max(0.0, min(max_alloc_pct, target))

    # ROTATION SEMANTIC: if bottom hasn't fired AND no veto, HOLD current.
    # The engine doesn't tell you to trim a BTC position without a top
    # signal — that's portfolio-optimization territory, not rotation.
    if bottom_n < threshold_partial and not vetoed:
        return current_btc_pct

    # Otherwise the trigger has fired — use the calculated target,
    # taking the MAX of (current, target) so we only ADD on bottom signals,
    # never instruct selling without a separate veto.
    return max(target, current_btc_pct)


def compute_equity_target(regime: str, top_n: int, early_n: int,
                            top_thresholds: dict, early_thresholds: dict,
                            baseline_pct: float, vetoed: bool,
                            current_pct: float = 30.0) -> float:
    """Returns equity target as % of total stake.

    SEMANTICS: this is a ROTATION engine, not a portfolio optimizer.
    When NO signal fires we recommend HOLDING the current allocation —
    we only force reductions when a trigger fires. This prevents the
    engine from telling someone with existing positions to dump capital
    just because regime baseline is lower.
    """
    if vetoed:
        return min(current_pct, 10.0)

    # Trigger-based reductions take precedence over baseline
    if top_n >= top_thresholds["full_rotation"]:
        return baseline_pct * 0.05
    if top_n >= top_thresholds["bear_confirmed"]:
        return baseline_pct * 0.20
    if early_n >= early_thresholds["rotate_to_cash"]:
        return baseline_pct * 0.30
    if top_n >= top_thresholds["defensive"]:
        return baseline_pct * 0.50
    if (top_n >= top_thresholds["trim"]
        or early_n >= early_thresholds["reduce"]):
        return baseline_pct * 0.75

    # No trigger fired — HOLD current position (don't force rebalance
    # to baseline). User is free to choose their own equilibrium.
    return current_pct


# ============================================================
# Phase action labels
# ============================================================

def top_action_label(top_n: int, t: dict) -> str:
    if top_n >= t["full_rotation"]:  return "FULL_ROTATION"
    if top_n >= t["bear_confirmed"]: return "BEAR_CONFIRMED"
    if top_n >= t["defensive"]:      return "DEFENSIVE"
    if top_n >= t["trim"]:           return "TRIM"
    return "HOLD"


def early_action_label(early_n: int, t: dict, accelerating: bool) -> str:
    if early_n >= t["rotate_to_cash"]:        return "ROTATE_TO_CASH"
    if early_n >= t["reduce"] and accelerating: return "ROTATE_TO_CASH"
    if early_n >= t["reduce"]:                 return "REDUCE_TO_CASH"
    if early_n >= t["watch"]:                  return "WATCH"
    return "HOLD"


# ============================================================
# Top-level orchestrator
# ============================================================

def unified_decision(total_stake_nzd: float = 130_000,
                      current_equity_pct: float = 30.0,
                      current_btc_pct: float = 0.0) -> dict:
    """Returns final target allocation + full reasoning trace.

    Args:
      current_equity_pct: user's current equity allocation as % of stake
      current_btc_pct:    user's current BTC allocation as % of stake

    SEMANTICS: ROTATION engine, not portfolio optimizer.
    HOLD current position unless a trigger fires.
    """
    # 1. Macro layer
    from core.btc_macro_layer import all_macro_signals
    macro = all_macro_signals()

    # 2. Liquidity
    liq = net_liquidity_z()
    real_yld_chg = real_yield_30d_change()

    # 3. Early rotation (existing module — provides HY widening signal)
    try:
        from core.btc_early_rotation import early_rotation_signal
        early = early_rotation_signal(
            current_equity_pct=current_equity_pct,
            total_stake_nzd=total_stake_nzd,
        )
    except Exception as e:
        early = {"n_firing": 0, "n_total": 7, "accelerating": False,
                 "indicators": {}, "error": str(e)[:80]}

    # 4. Regime
    from core.btc_regime import full_regime_analysis
    credit_imp_val = macro.get("credit_impulse", {}).get("value") or 0.0
    regime_info = full_regime_analysis(
        macro=macro,
        liquidity_z=liq["z"],
        credit_impulse_value=credit_imp_val,
        early_rot=early,
    )
    regime = regime_info["regime"]
    thresholds = regime_info["thresholds"]
    vetoes = regime_info["vetoes"]

    # 5. Top scorecard
    try:
        from core.btc_top_scorecard import top_confirmation_scorecard
        top = top_confirmation_scorecard()
        top_n = top.get("n_met", 0)
        top_total = top.get("n_total", 10)
    except Exception as e:
        top = {"n_met": 0, "n_total": 10, "error": str(e)[:80]}
        top_n, top_total = 0, 10

    # 6. Bottom scorecard
    try:
        from core.btc_bottom_scorecard import bottom_confirmation_scorecard
        bottom = bottom_confirmation_scorecard()
        bottom_n = bottom.get("n_met", 0)
        bottom_total = bottom.get("n_total", 10)   # confirmation scorecard is 10 (was 8)
    except Exception as e:
        bottom = {"n_met": 0, "n_total": 10, "error": str(e)[:80]}
        bottom_n, bottom_total = 0, 10

    early_n = early.get("n_firing", 0)
    early_total = early.get("n_total", 7)
    accelerating = early.get("accelerating", False)

    # 7. Vetoes determine which targets are forced
    btc_vetoed = "no_btc_during_collapse" in vetoes \
        or "force_cash_move_spike" in vetoes
    equity_vetoed = "no_equity_add_recession_start" in vetoes \
        or "force_cash_move_spike" in vetoes

    # 8. Compute targets (continuous sizing)
    btc_target_pct = compute_btc_target(
        regime=regime,
        bottom_n=bottom_n,
        threshold_partial=thresholds["btc_partial"],
        threshold_full=thresholds["btc_full"],
        liquidity_z=liq["z"],
        max_alloc_pct=thresholds["max_btc_pct"],
        vetoed=btc_vetoed,
        current_btc_pct=current_btc_pct,
    )
    equity_target_pct = compute_equity_target(
        regime=regime,
        top_n=top_n,
        early_n=early_n,
        top_thresholds=thresholds["top"],
        early_thresholds=thresholds["early"],
        baseline_pct=thresholds["baseline_equity_pct"],
        vetoed=equity_vetoed,
        current_pct=current_equity_pct,
    )

    # 9. Staging = whatever's left (cap at 100%)
    staging_pct = max(0.0, 100.0 - btc_target_pct - equity_target_pct)

    # 10. Split staging across BIL / VTIP / GLDM
    from core.btc_staging_basket import (
        compute_staging_basket, staging_basket_nzd, basket_explanation
    )
    move_val = macro.get("move_elevated", {}).get("value") or 100.0
    basket_pct = compute_staging_basket(
        regime=regime,
        liquidity_z=liq["z"],
        real_yield_30d_change=real_yld_chg,
        deficit_gdp=6.5,  # US current ~6-7%, hardcoded; could fetch FRED FYFSGDA188S
        move=move_val,
    )
    basket_nzd = staging_basket_nzd(basket_pct, staging_pct, total_stake_nzd)
    basket_rationale = basket_explanation(regime, basket_pct, liq["z"], move_val,
                                            real_yld_chg)

    # Action labels (for human reading)
    top_action = top_action_label(top_n, thresholds["top"])
    early_action = early_action_label(early_n, thresholds["early"], accelerating)

    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_buckets": regime_info["buckets"],
        "regime_curve_uninvert": regime_info["curve_uninvert"],
        "regime_thresholds": thresholds,
        "vetoes_active": vetoes,
        "macro_signals_summary": {
            k: {"firing": v.get("firing", False),
                "value": v.get("value"),
                "status": v.get("status", "")}
            for k, v in macro.items() if k != "asof"
        },
        "liquidity": liq,
        "real_yield_30d_change": real_yld_chg,
        "scorecards": {
            "top":    {"n_met": top_n, "n_total": top_total, "action": top_action},
            "early":  {"n_firing": early_n, "n_total": early_total,
                       "action": early_action, "accelerating": accelerating},
            "bottom": {"n_met": bottom_n, "n_total": bottom_total},
        },
        "target_allocation_pct": {
            "btc":     round(btc_target_pct, 1),
            "equity":  round(equity_target_pct, 1),
            "staging": round(staging_pct, 1),
        },
        "target_allocation_nzd": {
            "btc":     round(btc_target_pct / 100 * total_stake_nzd),
            "equity":  round(equity_target_pct / 100 * total_stake_nzd),
            "staging": round(staging_pct / 100 * total_stake_nzd),
        },
        "staging_basket_pct": basket_pct,
        "staging_basket_nzd": basket_nzd,
        "staging_basket_rationale": basket_rationale,
        "current_equity_pct": current_equity_pct,
        "current_btc_pct": current_btc_pct,
        "current_allocation_pct": {
            "equity": current_equity_pct,
            "btc": current_btc_pct,
            "staging": max(0, 100 - current_equity_pct - current_btc_pct),
        },
        "current_allocation_nzd": {
            "equity": round(current_equity_pct / 100 * total_stake_nzd),
            "btc": round(current_btc_pct / 100 * total_stake_nzd),
            "staging": round(max(0, 100 - current_equity_pct - current_btc_pct)
                              / 100 * total_stake_nzd),
        },
        "delta_pct": {
            "equity": round(equity_target_pct - current_equity_pct, 1),
            "btc":    round(btc_target_pct - current_btc_pct, 1),
            "staging": round(staging_pct
                              - max(0, 100 - current_equity_pct - current_btc_pct), 1),
        },
        "rotation_nzd": max(0, round((current_equity_pct - equity_target_pct)
                                     / 100 * total_stake_nzd)),
        "btc_action_required": btc_target_pct > current_btc_pct,
        "equity_action_required": equity_target_pct < current_equity_pct,
    }


def main():
    r = unified_decision()
    print("=" * 70)
    print(f"UNIFIED DECISION — Regime: {r['regime']}")
    print("=" * 70)
    print(f"  Buckets: growth={r['regime_buckets']['growth']}/4  "
          f"plumbing={r['regime_buckets']['plumbing']}/4  "
          f"credit={r['regime_buckets']['credit']}/3")
    print(f"  Liquidity z: {r['liquidity']['z']:+.2f}")
    print(f"  Vetoes: {r['vetoes_active'] or 'none'}")
    print()
    print("  SCORECARDS:")
    sc = r["scorecards"]
    print(f"    Top:    {sc['top']['n_met']}/{sc['top']['n_total']}  "
          f"-> {sc['top']['action']}")
    print(f"    Early:  {sc['early']['n_firing']}/{sc['early']['n_total']}  "
          f"-> {sc['early']['action']}  "
          f"({'accel' if sc['early']['accelerating'] else 'steady'})")
    print(f"    Bottom: {sc['bottom']['n_met']}/{sc['bottom']['n_total']}")
    print()
    print("  TARGET ALLOCATION:")
    t = r["target_allocation_pct"]
    print(f"    Equity:  {t['equity']:.1f}%  -> NZ${r['target_allocation_nzd']['equity']:,}")
    print(f"    BTC:     {t['btc']:.1f}%  -> NZ${r['target_allocation_nzd']['btc']:,}")
    print(f"    Staging: {t['staging']:.1f}%  -> NZ${r['target_allocation_nzd']['staging']:,}")
    print()
    print("  STAGING BASKET:")
    for k, v in r["staging_basket_pct"].items():
        nzd = r["staging_basket_nzd"][k]
        print(f"    {k}: {v}%  -> NZ${nzd:,}")
    print(f"    Rationale: {r['staging_basket_rationale']}")


if __name__ == "__main__":
    main()
