"""P7: BTC bottom state — DEEP_GENERATIONAL / SHALLOW_ETF_DRIVEN / UNCONFIRMED.

Cycle 5 problem: ETF buying may prevent the deep capitulation past
bottoms produced. We need to distinguish:
  - DEEP_GENERATIONAL (2015, 2018, 2022): on-chain extreme + ETF flat/negative
  - SHALLOW_ETF_DRIVEN (cycle 5 candidate): mid on-chain + ETF support
  - UNCONFIRMED: no signal

Each state has a different entry style and max allocation cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def btc_bottom_composite(state: dict) -> dict:
    """Continuous bottom score from z-scored BTC signals.

    Each input is expected to ALREADY be a z-score (from signal_registry).
    We sign-correct so positive = bottom-like (low MVRV-Z = bottom).

    Aliased signals (sth_mvrv, reserve_risk both proxy MVRV currently)
    are deduplicated — only counted once — to avoid triple-counting the
    same upstream data.

    Returns:
      composite_z: equal-weighted z (each component capped at ±3)
      bottom_probability: sigmoid(composite_z) -> 0..1
      state: DEEP_GENERATIONAL / SHALLOW_ETF_DRIVEN / BTC_BOTTOM_UNCONFIRMED
    """
    # Sign per signal — negative means "low value = bottom"
    SIGN = {
        "mvrv_z":              -1,
        "asopr":               -1,
        "realized_cap_drawdown": -1,
        "reserve_risk":        -1,
        "sth_mvrv":            -1,
        "puell":               -1,
        "hashrate_drawdown":   -1,
        "etf_flow_60d_z":      +1,  # positive flows = bullish not bottom
    }

    # Dedup signals that alias to the same upstream data.
    # Currently sth_mvrv + reserve_risk both proxy CapMVRVCur — count
    # only mvrv_z to avoid 3x weighting of the same signal.
    ALIAS_KEEP_ONE = {"mvrv_z", "sth_mvrv", "reserve_risk"}
    seen_alias_value = None
    components = {}
    for key, sign in SIGN.items():
        v = state.get(key)
        if v is None: continue
        try: z = float(v)
        except Exception: continue
        if abs(z) > 3.5: z = 3.5 * (1 if z > 0 else -1)  # clip wild outliers
        # Dedup MVRV aliases
        if key in ALIAS_KEEP_ONE:
            if seen_alias_value is not None and abs(z - seen_alias_value) < 0.01:
                continue  # this is the same upstream data, skip
            seen_alias_value = z
        components[key] = z * sign

    if not components:
        return {
            "composite_z": 0.0,
            "bottom_probability": 0.5,
            "state": "BTC_BOTTOM_UNCONFIRMED",
            "components": {},
            "n_components": 0,
            "error": "no_components_available",
        }

    # Equal-weighted average normalised by sqrt(N) so composite has ~unit vol
    composite_z = sum(components.values()) / np.sqrt(len(components))
    bottom_prob = 1 / (1 + np.exp(-composite_z))

    # State classification — REQUIRE multiple components for deep state
    mvrv_z = state.get("mvrv_z")
    mvrv_deep = mvrv_z is not None and float(mvrv_z) < -1.5
    etf_flow_30d_z = state.get("etf_flow_30d_z")
    etf_positive = etf_flow_30d_z is not None and float(etf_flow_30d_z) > 0

    if composite_z > 1.5 and mvrv_deep and len(components) >= 4:
        bot_state = "DEEP_GENERATIONAL"
    elif 0.5 < composite_z <= 1.5 and etf_positive and len(components) >= 3:
        bot_state = "SHALLOW_ETF_DRIVEN"
    else:
        bot_state = "BTC_BOTTOM_UNCONFIRMED"

    return {
        "composite_z": float(composite_z),
        "bottom_probability": float(bottom_prob),
        "state": bot_state,
        "components": {k: float(v) for k, v in components.items()},
        "n_components": len(components),
    }


def btc_entry_plan(bottom_state: str, available_capital: float = 0.0,
                    realized_vol_60d: float = 0.60) -> dict:
    """Concrete entry schedule for the given bottom state."""
    plans = {
        "DEEP_GENERATIONAL": {
            "initial_tranche_pct": 50,
            "dca_remaining_pct":   50,
            "dca_days":            30,
            "vol_scale":           False,
            "max_alloc_pct":       100,    # max of regime cap
            "rationale":           ("Deep capitulation — historic precedent. "
                                    "Half upfront, DCA rest over 30d. "
                                    "Don't try to time the exact bottom."),
        },
        "SHALLOW_ETF_DRIVEN": {
            "initial_tranche_pct": 20,
            "dca_remaining_pct":   80,
            "dca_days":            90,
            "vol_scale":           True,
            "max_alloc_pct":       60,    # cap lower — shallower bottom
            "rationale":           ("ETF support without deep capitulation. "
                                    "Smaller initial size, longer DCA, "
                                    "scale tranches by inverse vol."),
        },
        "BTC_BOTTOM_UNCONFIRMED": {
            "initial_tranche_pct": 0,
            "dca_remaining_pct":   0,
            "dca_days":            0,
            "vol_scale":           False,
            "max_alloc_pct":       0,
            "rationale":           ("No bottom signal. Stay in staging basket "
                                    "until composite_z > 0.5 + ETF flows "
                                    "or composite_z > 1.5."),
        },
    }
    plan = dict(plans.get(bottom_state, plans["BTC_BOTTOM_UNCONFIRMED"]))

    # Vol-scaled tranche when applicable
    if plan["vol_scale"] and realized_vol_60d > 0:
        target_vol = 0.60
        plan["tranche_scale"] = float(min(1.0, target_vol / realized_vol_60d))
    else:
        plan["tranche_scale"] = 1.0

    # Concrete NZD breakdown if capital provided
    if available_capital > 0:
        initial_nzd = available_capital * plan["initial_tranche_pct"] / 100 * plan["tranche_scale"]
        dca_total_nzd = available_capital * plan["dca_remaining_pct"] / 100 * plan["tranche_scale"]
        per_day = dca_total_nzd / plan["dca_days"] if plan["dca_days"] > 0 else 0
        plan["concrete_amounts"] = {
            "initial_buy_nzd": round(initial_nzd),
            "dca_per_day_nzd": round(per_day),
            "dca_total_nzd": round(dca_total_nzd),
        }

    return plan


def main():
    # Smoke tests across 3 scenarios
    scenarios = {
        "DEEP_GENERATIONAL_test": {
            "mvrv_z": -1.5, "asopr": 0.85, "realized_cap_drawdown": -20,
            "reserve_risk": 0.0015, "sth_mvrv": 0.75, "puell": 0.35,
            "hashrate_drawdown": -28, "etf_flow_60d_z": -1.2,
            "etf_flow_30d_z": -0.5,
        },
        "SHALLOW_ETF_test": {
            "mvrv_z": -0.3, "asopr": 0.95, "realized_cap_drawdown": -8,
            "reserve_risk": 0.005, "sth_mvrv": 0.95, "puell": 0.6,
            "hashrate_drawdown": -10, "etf_flow_60d_z": 0.5,
            "etf_flow_30d_z": 0.8,
        },
        "UNCONFIRMED_test": {
            "mvrv_z": 1.2, "asopr": 1.05, "realized_cap_drawdown": -2,
            "reserve_risk": 0.015, "sth_mvrv": 1.15, "puell": 1.2,
            "hashrate_drawdown": 0, "etf_flow_60d_z": 0.1,
            "etf_flow_30d_z": 0.0,
        },
    }
    for name, state in scenarios.items():
        r = btc_bottom_composite(state)
        plan = btc_entry_plan(r["state"], available_capital=10000)
        print(f"\n{name}:")
        print(f"  composite_z: {r['composite_z']:+.2f}  "
              f"prob: {r['bottom_probability']:.2%}  state: {r['state']}")
        print(f"  plan: initial {plan['initial_tranche_pct']}% / "
              f"DCA {plan['dca_remaining_pct']}% over {plan['dca_days']}d  "
              f"max_alloc: {plan['max_alloc_pct']}%")


if __name__ == "__main__":
    main()
