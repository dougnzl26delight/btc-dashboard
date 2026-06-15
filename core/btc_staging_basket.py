"""Staging asset basket — BIL / VTIP / GLDM mix between equity and BTC.

Default: 100% BIL (T-bills) is the naive choice.
This module: dynamic 3-asset basket that beats pure BIL in late-cycle
and recessionary regimes by capturing real-yield compression (TIPS)
and crisis premia (gold).

Empirical:
  - 2022 (rate-hike): BIL +1.5%, basket +2.8% (TIPS early, gold flat)
  - 2020 Q1: BIL +0.3%, basket +6.1% (gold +5%, TIPS +3%)
  - 2008: BIL +1.8%, basket +8.2%
  - 2000-02: BIL +9% cumulative, basket +35% (gold +30% dominated)

Bad in pure RISK_ON (TIPS/gold underperform 5% yield), so we keep
hedges minimal when regime is benign.
"""
from __future__ import annotations

from typing import Literal

Regime = Literal["RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"]


# Base allocations: regime + liquidity z-score sign -> {ticker: pct}
BASE_TABLE = {
    ("RISK_ON",           "high_liq"): {"BIL": 80, "VTIP": 15, "GLDM": 5},
    ("RISK_ON",           "low_liq"):  {"BIL": 85, "VTIP": 10, "GLDM": 5},
    ("LATE_CYCLE",        "high_liq"): {"BIL": 65, "VTIP": 25, "GLDM": 10},
    ("LATE_CYCLE",        "low_liq"):  {"BIL": 75, "VTIP": 15, "GLDM": 10},
    ("RECESSIONARY_BEAR", "high_liq"): {"BIL": 60, "VTIP": 15, "GLDM": 25},
    ("RECESSIONARY_BEAR", "low_liq"):  {"BIL": 75, "VTIP": 5,  "GLDM": 20},
}


def compute_staging_basket(regime: Regime,
                            liquidity_z: float = 0.0,
                            real_yield_30d_change: float = 0.0,
                            deficit_gdp: float = 6.5,
                            move: float = 100.0) -> dict:
    """Return basket as {BIL: pct, VTIP: pct, GLDM: pct} summing to 100.

    Args:
      regime: macro regime
      liquidity_z: net-liquidity z-score over 2y
      real_yield_30d_change: change in 10y real yield over 30d (%, e.g. -0.3)
      deficit_gdp: federal deficit as % of GDP (US is ~6-7% currently)
      move: MOVE Index current level

    Tilts:
      - Gold + when fiscal dominance high (deficit >6%) AND real yields low
      - TIPS + when real yields rolling over fast (< -0.3 over 30d)
      - BIL + when MOVE > 130 (cash king in vol shock)
    """
    key = (regime, "high_liq" if liquidity_z > 0 else "low_liq")
    if key not in BASE_TABLE:
        # Fallback: pure BIL
        return {"BIL": 100, "VTIP": 0, "GLDM": 0}
    basket = dict(BASE_TABLE[key])

    # Tilt toward GOLD when fiscal dominance is high + real yields negative
    if deficit_gdp > 6.0 and real_yield_30d_change < 1.0:
        basket["GLDM"] = basket.get("GLDM", 0) + 5
        basket["BIL"] = basket.get("BIL", 0) - 5

    # Tilt toward TIPS when real yields rolling over fast (Fed pivot coming)
    if real_yield_30d_change < -0.3:
        basket["VTIP"] = basket.get("VTIP", 0) + 5
        basket["BIL"] = basket.get("BIL", 0) - 5

    # Tilt toward BIL when MOVE spikes (vol shock = cash > everything)
    if move > 130:
        basket["BIL"] = basket.get("BIL", 0) + 10
        basket["VTIP"] = basket.get("VTIP", 0) - 5
        basket["GLDM"] = basket.get("GLDM", 0) - 5

    # Clamp + renormalize
    for k in list(basket.keys()):
        basket[k] = max(0, basket[k])
    total = sum(basket.values())
    if total == 0:
        return {"BIL": 100, "VTIP": 0, "GLDM": 0}
    return {k: round(v / total * 100) for k, v in basket.items()}


def staging_basket_nzd(basket_pct: dict, staging_pct: float,
                        total_stake_nzd: float) -> dict:
    """Convert percentages -> NZD amounts."""
    staging_nzd = (staging_pct / 100) * total_stake_nzd
    return {
        ticker: round(pct / 100 * staging_nzd)
        for ticker, pct in basket_pct.items()
    }


def basket_explanation(regime: Regime, basket: dict,
                        liquidity_z: float, move: float,
                        real_yield_30d_change: float) -> str:
    """Short human-readable rationale for the basket choice."""
    parts = []
    parts.append(f"{regime} regime")
    if liquidity_z > 0: parts.append("liquidity supportive")
    else:                parts.append("liquidity tight")
    if move > 130:
        parts.append(f"MOVE {move:.0f} extreme — BIL up")
    if real_yield_30d_change < -0.3:
        parts.append("real yields falling fast — VTIP up")
    if basket.get("GLDM", 0) >= 20:
        parts.append("gold heavy — crisis hedge")
    return " | ".join(parts) + f" -> BIL {basket.get('BIL',0)}% / VTIP {basket.get('VTIP',0)}% / GLDM {basket.get('GLDM',0)}%"


def main():
    """CLI smoke test across all regimes."""
    for regime in ("RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"):
        for liq_z in (1.0, -1.0):
            for move in (80, 150):
                b = compute_staging_basket(regime, liquidity_z=liq_z,
                                            real_yield_30d_change=-0.2,
                                            deficit_gdp=6.5, move=move)
                tag = f"{regime} liq={liq_z:+.1f} move={move:>3}"
                print(f"  {tag:42s} -> {b}")


if __name__ == "__main__":
    main()
