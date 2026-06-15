"""P6: Position sizing engine.

Combines:
  - composite scores (from composites.py)
  - regime
  - realized volatility per asset
  - drawdown brake
  - Kelly fraction (with cap)
  - turnover constraint

Outputs target weights for: equity / btc / staging.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Core sizing function
# ============================================================

def position_size(
    composite_score_z: float,
    asset_vol_annual: float,
    portfolio_vol_target: float = 0.12,
    kelly_fraction: float = 0.25,
    regime_multiplier: float = 1.0,
    max_position: float = 1.0,
) -> float:
    """Returns target weight as fraction of portfolio (0.0..max_position).

    Logic:
      1. signal_strength = composite_z / 3  (normalized -1..+1)
      2. Vol-target scaling: target_vol / asset_vol
      3. Kelly safety fraction (don't over-bet)
      4. Regime multiplier
      5. Cap at max_position
    """
    if asset_vol_annual <= 0: return 0.0

    signal_strength = float(np.clip(composite_score_z / 3.0, -1.0, 1.0))
    if signal_strength <= 0: return 0.0

    vol_ratio = portfolio_vol_target / asset_vol_annual
    raw_weight = signal_strength * vol_ratio * kelly_fraction
    adjusted = raw_weight * regime_multiplier
    return float(np.clip(adjusted, 0.0, max_position))


# ============================================================
# Drawdown brake
# ============================================================

def drawdown_brake(equity_curve: pd.Series,
                    lookback_days: int = 252,
                    threshold: float = -0.10,
                    floor: float = 0.5) -> float:
    """Returns multiplier 0.5..1.0 to scale ALL positions.

    Activates when 1y rolling DD exceeds threshold.
    """
    if equity_curve is None or len(equity_curve) < 30:
        return 1.0
    recent = equity_curve.tail(lookback_days)
    peak = recent.cummax()
    dd = float((recent / peak - 1).iloc[-1])
    if dd > threshold: return 1.0
    severity = (dd - threshold) / (-0.10)
    return float(max(floor, 1.0 - 0.5 * severity))


def drawdown_brake_value(current_dd: float, threshold: float = -0.10,
                           floor: float = 0.5) -> float:
    """Direct version when you already have DD as a scalar."""
    if current_dd > threshold: return 1.0
    severity = (current_dd - threshold) / (-0.10)
    return float(max(floor, 1.0 - 0.5 * severity))


# ============================================================
# Turnover constraint
# ============================================================

def apply_turnover_constraint(current_weights: dict,
                                 target_weights: dict,
                                 min_change: float = 0.05) -> dict:
    """If |target - current| < min_change, hold current."""
    rebalanced = {}
    for asset, target in target_weights.items():
        current = current_weights.get(asset, 0)
        if abs(target - current) < min_change:
            rebalanced[asset] = current
        else:
            rebalanced[asset] = target
    return rebalanced


# ============================================================
# Realized volatility helper
# ============================================================

def realized_vol(returns: pd.Series, window: int = 60,
                  annualize: bool = True) -> float:
    """Return annualized realized vol of daily returns over `window` days."""
    if returns is None or len(returns) < window:
        return 0.0
    s = pd.Series(returns).dropna()
    if len(s) < 10: return 0.0
    vol = float(s.tail(window).std())
    if annualize: vol *= np.sqrt(252)
    return vol


def compute_current_drawdown(equity_curve: pd.Series,
                                lookback: int = 252) -> float:
    """Current drawdown vs rolling peak."""
    if equity_curve is None or len(equity_curve) < 2: return 0.0
    s = pd.Series(equity_curve).dropna().tail(lookback)
    if s.empty: return 0.0
    return float(s.iloc[-1] / s.cummax().iloc[-1] - 1)


# ============================================================
# Final allocation orchestrator
# ============================================================

REGIME_ASSET_MULTIPLIERS = {
    "RISK_ON":           {"equity": 1.2, "btc": 0.7, "staging": 0.6},
    "LATE_CYCLE":        {"equity": 0.7, "btc": 1.0, "staging": 1.2},
    "RECESSIONARY_BEAR": {"equity": 0.3, "btc": 1.3, "staging": 1.5},
}

REGIME_MAX_BTC = {
    "RISK_ON":           0.30,
    "LATE_CYCLE":        0.50,
    "RECESSIONARY_BEAR": 1.00,
}


def compute_target_allocation(
    composite_scores: dict,         # {top, early, bottom} composite z's
    regime: str,
    realized_vols: dict,             # {SPY, BTC} annualized
    current_drawdown: float = 0.0,
    vetoes: Optional[list] = None,
    kelly_fraction: float = 0.25,
    portfolio_vol_target: float = 0.12,
    total_stake: float = 130_000,
) -> dict:
    """Final allocation combining composites + regime + vol-target + brakes."""
    vetoes = vetoes or []
    dd_mult = drawdown_brake_value(current_drawdown)
    regime_mults = REGIME_ASSET_MULTIPLIERS.get(regime,
                       REGIME_ASSET_MULTIPLIERS["LATE_CYCLE"])

    # Equity: invert top (high top score = SELL equity)
    raw_equity = position_size(
        composite_score_z=-composite_scores.get("top", 0.0),
        asset_vol_annual=realized_vols.get("SPY", 0.20),
        portfolio_vol_target=portfolio_vol_target,
        kelly_fraction=kelly_fraction,
        regime_multiplier=regime_mults["equity"] * dd_mult,
        max_position=0.50,
    )
    # The "wanting to maintain equity" path: in RISK_ON regimes with
    # weak top composite, ensure baseline equity allocation.
    if regime == "RISK_ON" and composite_scores.get("top", 0.0) < 0.5:
        raw_equity = max(raw_equity, 0.30)

    # BTC: high bottom composite = BUY BTC
    raw_btc = position_size(
        composite_score_z=composite_scores.get("bottom", 0.0),
        asset_vol_annual=realized_vols.get("BTC", 0.60),
        portfolio_vol_target=portfolio_vol_target,
        kelly_fraction=kelly_fraction,
        regime_multiplier=regime_mults["btc"] * dd_mult,
        max_position=REGIME_MAX_BTC.get(regime, 0.50),
    )

    # Vetoes
    if "force_cash_move_spike" in vetoes:
        raw_equity = min(raw_equity, 0.05)
        raw_btc = 0.0
    if "no_btc_during_collapse" in vetoes:
        raw_btc = 0.0
    if "no_equity_add_recession_start" in vetoes:
        raw_equity = min(raw_equity, 0.10)

    # Staging = whatever's left
    raw_staging = max(0.0, 1.0 - raw_equity - raw_btc)
    raw = {"equity": raw_equity, "btc": raw_btc, "staging": raw_staging}

    # Normalize to sum=1
    total = sum(raw.values())
    if total > 0:
        weights = {k: v / total for k, v in raw.items()}
    else:
        weights = {"equity": 0, "btc": 0, "staging": 1.0}

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "weights_pct": {k: round(v * 100, 1) for k, v in weights.items()},
        "nzd": {k: round(v * total_stake) for k, v in weights.items()},
        "regime": regime,
        "drawdown_multiplier": dd_mult,
        "current_drawdown": current_drawdown,
        "kelly_fraction": kelly_fraction,
        "portfolio_vol_target": portfolio_vol_target,
        "vetoes_applied": vetoes,
        "composite_inputs": composite_scores,
    }


def main():
    test_composites = {"top": 1.5, "early": 0.8, "bottom": 0.6}
    vols = {"SPY": 0.18, "BTC": 0.65}
    for regime in ("RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"):
        r = compute_target_allocation(test_composites, regime, vols,
                                       current_drawdown=0.0)
        print(f"\nRegime: {regime}")
        for k, v in r["weights_pct"].items():
            print(f"  {k}: {v:.1f}% (NZ${r['nzd'][k]:,})")


if __name__ == "__main__":
    main()
