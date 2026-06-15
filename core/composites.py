"""P4: Theme composite scores.

Instead of one giant scorecard, compute 6 theme-level z-scores.
Each theme = IC-weighted average of standardized member signals.

Themes (Bridgewater-style):
  LIQUIDITY    — global $ supply for risk assets (Howell axis)
  CREDIT       — bond market stress (leads equity)
  GROWTH       — real economy late-cycle (LEI, Sahm, claims)
  VALUATION    — mean-reversion expected return (P/E, ERP)
  SENTIMENT    — crowd positioning extremes
  BTC_ONCHAIN  — Bitcoin-native cycle markers
"""
from __future__ import annotations

import numpy as np
import pandas as pd


THEME_DEFINITIONS = {
    "LIQUIDITY": {
        "members": [
            "net_liquidity_b",        # WALCL - WTREGEN - RRP
            "tip_yield",                # 10y real yield (inverted = liquidity-supportive)
            "move_index",               # bond vol (inverted)
            "rrp_balance_b",            # RRP balance (inverted: collapsing = stress)
            "sofr_iorb_bps",            # SOFR-IORB spread (inverted)
        ],
        # Sign: +1 means high z = liquidity SUPPORTIVE (good for risk)
        # -1 means high z = stress / bad for risk
        "signs": {
            "net_liquidity_b":  +1,
            "tip_yield":        -1,  # higher real yields = liquidity tight
            "move_index":       -1,
            "rrp_balance_b":    +1,  # higher RRP = more dry powder
            "sofr_iorb_bps":    -1,
        },
    },
    "CREDIT": {
        "members": [
            "hy_spread_bps",         # higher = stress
            "credit_impulse",         # higher = good
            "sloos_tightening",       # higher = stress
            "yield_curve_t10y2y",     # higher = ok; inverted = stress
        ],
        "signs": {
            "hy_spread_bps":     -1,
            "credit_impulse":    +1,
            "sloos_tightening":  -1,
            "yield_curve_t10y2y": +1,
        },
    },
    "GROWTH": {
        "members": [
            "oecd_cli",
            "lei_yoy",
            "sahm",                  # higher = recession
            "claims_4w_ma",          # higher = recession
            "ism_manufacturing",     # higher = good
        ],
        "signs": {
            "oecd_cli":           +1,
            "lei_yoy":            +1,
            "sahm":               -1,
            "claims_4w_ma":       -1,
            "ism_manufacturing":  +1,
        },
    },
    "VALUATION": {
        "members": [
            "spy_pe",
            "erp",
            "cape_proxy",
        ],
        "signs": {
            "spy_pe":      -1,   # high P/E = expensive = bearish forward
            "erp":         +1,   # high ERP = stocks attractive
            "cape_proxy":  -1,
        },
    },
    "SENTIMENT": {
        "members": [
            "aaii_bullish",
            "naaim_exposure",
            "breadth_200d_pct",
            "fear_greed",
            "put_call_ratio",
        ],
        "signs": {
            "aaii_bullish":       -1,  # extreme bull = contrarian sell
            "naaim_exposure":     -1,
            "breadth_200d_pct":   +1,
            "fear_greed":         -1,  # greed = contrarian sell
            "put_call_ratio":     +1,  # high P/C = fear = bullish
        },
    },
    "BTC_ONCHAIN": {
        "members": [
            "mvrv_z",
            "asopr",
            "rcap_drawdown",
            "reserve_risk",
            "sth_mvrv",
            "puell",
            "hashrate_drawdown",
            "etf_flow_60d",
        ],
        "signs": {
            "mvrv_z":              -1,  # low MVRV-Z = bottom = bullish forward
            "asopr":               -1,  # low aSOPR = bottom
            "rcap_drawdown":       -1,  # deep drawdown = bottom
            "reserve_risk":        -1,  # low RR = bottom
            "sth_mvrv":            -1,  # low STH MVRV = pain = bottom
            "puell":               -1,  # low Puell = miner cap = bottom
            "hashrate_drawdown":   -1,  # deep DD = bottom
            "etf_flow_60d":        +1,  # positive flows = bullish
        },
    },
}


def compute_theme_score(member_zs: dict, member_ics: dict, theme: str,
                         method: str = "ic_weighted") -> dict:
    """Compute a theme composite z-score from member z-scores.

    Args:
      member_zs: {signal_name: z_score (float)}
      member_ics: {signal_name: ic (float)} — used as weights
      theme: theme name from THEME_DEFINITIONS

    Returns dict with: composite_z, n_members_present, weights_used
    """
    if theme not in THEME_DEFINITIONS:
        return {"composite_z": 0.0, "n_members_present": 0,
                "weights_used": {}, "error": "unknown_theme"}

    spec = THEME_DEFINITIONS[theme]
    signs = spec["signs"]

    # Build weights — IC-weighted with sign-corrected member z's.
    # NOTE: use |IC| not max(0, IC). The SIGNS dict already encodes
    # expected direction; negative IC just means the signal moves
    # opposite to what we expected, which still has predictive power.
    weights = {}
    signed_zs = {}
    for name in spec["members"]:
        z = member_zs.get(name)
        if z is None or pd.isna(z): continue
        ic = member_ics.get(name, 0.05)  # fallback weight if no IC available
        try: ic = float(ic) if ic is not None else 0.05
        except Exception: ic = 0.05
        if method == "ic_weighted":
            weights[name] = abs(ic) if ic != 0 else 0.05
        else:  # equal weighted
            weights[name] = 1.0
        signed_zs[name] = float(z) * signs.get(name, +1)

    total_w = sum(weights.values())
    if total_w == 0 or not signed_zs:
        return {"composite_z": 0.0, "n_members_present": 0,
                "weights_used": weights, "members_used": {}}

    # Weighted sum scaled by sqrt of weight sum so composite has ~unit vol
    weighted = sum(signed_zs[n] * weights[n] for n in signed_zs)
    composite = weighted / np.sqrt(sum(w**2 for w in weights.values()))

    return {
        "composite_z": float(composite),
        "n_members_present": len(signed_zs),
        "n_members_defined": len(spec["members"]),
        "weights_used": weights,
        "members_used": signed_zs,
    }


def compute_all_themes(standardized_signals: dict[str, dict],
                        ic_weights: dict[str, float]) -> dict[str, dict]:
    """Compute all theme composites in one call.

    Args:
      standardized_signals: output of standardize.standardize_batch()
                            {signal_name: {z, percentile, ...}}
      ic_weights: {signal_name: ic_weight}

    Returns: {theme_name: {composite_z, n_members_present, ...}}
    """
    member_zs = {name: r.get("z") for name, r in standardized_signals.items()
                 if r.get("z") is not None}
    themes = {}
    for theme in THEME_DEFINITIONS:
        themes[theme] = compute_theme_score(member_zs, ic_weights, theme)
    return themes


def composite_scores_for_decisions(theme_zs: dict[str, dict]) -> dict[str, float]:
    """Derive the 3 decision composites from theme z's.

    These are what feeds the position sizer:
      - 'top' composite (equity top): high = exit equity
      - 'early' composite (early rotation): high = rotate to cash
      - 'bottom' composite (BTC bottom): high = deploy BTC
    """
    def _z(theme):
        return theme_zs.get(theme, {}).get("composite_z", 0.0)

    return {
        # Equity top score: high valuation + high sentiment = top
        "top":     -_z("VALUATION") - _z("SENTIMENT") * 0.5,
        # Early rotation: liquidity / credit / growth all stressing
        "early":   -_z("LIQUIDITY") - _z("CREDIT") - _z("GROWTH") * 0.5,
        # BTC bottom: on-chain depth + liquidity support
        "bottom":   _z("BTC_ONCHAIN") + _z("LIQUIDITY") * 0.3,
    }


def main():
    # Synthetic smoke test
    member_zs = {
        # LIQUIDITY (mixed)
        "net_liquidity_b": 0.5, "tip_yield": -0.2,
        "move_index": -0.3, "rrp_balance_b": 0.4,
        # CREDIT
        "hy_spread_bps": 0.8, "credit_impulse": -1.2,
        # GROWTH
        "oecd_cli": -1.0, "lei_yoy": -0.8,
        # VALUATION
        "spy_pe": 1.5, "erp": -1.3,
        # SENTIMENT
        "aaii_bullish": 1.2, "naaim_exposure": 0.9,
        # BTC
        "mvrv_z": 1.5, "asopr": 0.8,
    }
    ic_weights = {name: 0.08 for name in member_zs}

    # Wrap as standardize-output shape
    standardized = {name: {"z": z} for name, z in member_zs.items()}
    themes = compute_all_themes(standardized, ic_weights)
    print("Theme composites:")
    for theme, r in themes.items():
        print(f"  {theme:14s} z={r['composite_z']:+.2f}  "
              f"({r['n_members_present']}/{r['n_members_defined']} signals)")

    decisions = composite_scores_for_decisions(themes)
    print("\nDecision composites:")
    for name, z in decisions.items():
        print(f"  {name:8s} composite_z = {z:+.2f}")


if __name__ == "__main__":
    main()
