"""Realistic cost model for backtests + post-cost Sharpe re-evaluation.

Costs modelled:
    Slippage    : 30 bps per side (60 bps round-trip) — typical for $500-2000
                  market orders on Binance majors. Higher for low-cap alts.
    Trading fee : 10 bps per side (20 bps round-trip) — Binance VIP-0 spot.
                  Perp futures: 5 bps per side (10 bps RT) at VIP-0.
    Funding     : 0.01% per 8h average for perps (-30%/+30% extremes).
                  Applied per position-hour-held.

These are FLOORS — actual costs on $35k account with 12 active pairs are likely
worse. Top-1% rule: when in doubt, double the cost assumption.

Usage:
    from core.cost_model import apply_costs_to_returns
    net_returns = apply_costs_to_returns(gross_returns, n_trades, holding_days, is_perp=True)
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# Default cost assumptions (bps = basis points = 1/100 of a percent)
SLIPPAGE_BPS_PER_SIDE = 30          # 30 bps each entry + exit = 60 RT
SPOT_FEE_BPS_PER_SIDE = 10          # Binance VIP-0 spot
PERP_FEE_BPS_PER_SIDE = 5           # Binance VIP-0 perp futures
FUNDING_BPS_PER_8H_AVG = 1.0        # ~0.01% per 8h funding average

# Multipliers for less-liquid pairs (top-50 OK, alts get worse)
LOW_CAP_PAIRS = {"TAO/USDT", "ONDO/USDT", "TIBBIR/USDT", "NPC/USDT"}
LOW_CAP_SLIPPAGE_MULT = 2.5          # 30bps -> 75bps per side on micro-caps


def round_trip_cost_bps(pair: str = "", is_perp: bool = False) -> float:
    """Total round-trip cost (slippage + fees) in basis points."""
    slip_mult = LOW_CAP_SLIPPAGE_MULT if pair in LOW_CAP_PAIRS else 1.0
    slip = SLIPPAGE_BPS_PER_SIDE * slip_mult * 2
    fee = (PERP_FEE_BPS_PER_SIDE if is_perp else SPOT_FEE_BPS_PER_SIDE) * 2
    return slip + fee


def funding_cost_bps(holding_days: float) -> float:
    """Funding accrual for a perp position over N days (positive = cost to long)."""
    n_funding_events = holding_days * 3  # 3 funding events per day (8h cadence)
    return FUNDING_BPS_PER_8H_AVG * n_funding_events


def apply_costs_to_returns(returns: Sequence[float], n_trades: int = 0,
                           total_holding_days: float = 0.0,
                           is_perp: bool = False, pair: str = "") -> list[float]:
    """Subtract realistic costs from a return series.

    Costs are amortized across the series (one transaction = 1 entry + 1 exit
    pair, distributed across the holding period).
    """
    rets = list(returns)
    if not rets or n_trades <= 0:
        return rets

    rt_cost_bps = round_trip_cost_bps(pair, is_perp)
    rt_cost = rt_cost_bps / 10000.0  # bps -> decimal
    total_tx_cost = rt_cost * n_trades

    if is_perp and total_holding_days > 0:
        # Estimate average position size = abs(mean of returns) — proxy
        avg_funding_bps = funding_cost_bps(total_holding_days / max(n_trades, 1))
        total_tx_cost += (avg_funding_bps / 10000.0) * n_trades

    # Distribute cost uniformly across the return series
    cost_per_bar = total_tx_cost / len(rets)
    return [r - cost_per_bar for r in rets]


def annualized_sharpe(returns: Sequence[float], periods_per_year: int = 365) -> float:
    """Annualized Sharpe from daily returns (no risk-free)."""
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 10 or arr.std() == 0:
        return 0.0
    return float((arr.mean() / arr.std()) * np.sqrt(periods_per_year))


def post_cost_sharpe(gross_returns: Sequence[float], n_trades: int,
                     holding_days: float, is_perp: bool = False,
                     pair: str = "", periods_per_year: int = 365) -> dict:
    """Sharpe before/after costs, plus cost contribution."""
    gross = annualized_sharpe(gross_returns, periods_per_year)
    net_rets = apply_costs_to_returns(gross_returns, n_trades, holding_days,
                                       is_perp=is_perp, pair=pair)
    net = annualized_sharpe(net_rets, periods_per_year)
    return {
        "gross_sharpe": gross,
        "net_sharpe": net,
        "delta": gross - net,
        "round_trip_cost_bps": round_trip_cost_bps(pair, is_perp),
        "n_trades": n_trades,
    }


# ===== One-liner re-evaluation helpers for existing backtests =====
def reevaluate_sleeve(name: str, gross_sharpe: float, n_trades_per_year: int = 20,
                      avg_holding_days: float = 5.0, is_perp: bool = False) -> dict:
    """Approximate post-cost Sharpe given a known gross Sharpe + trade frequency.

    Heuristic: each round-trip costs X bps of equity. Annual cost drag =
    n_trades * cost_per_trade. Sharpe impact = drag / annual_vol.
    """
    annual_drag = (n_trades_per_year * round_trip_cost_bps("", is_perp)) / 10000.0
    if is_perp:
        annual_drag += (n_trades_per_year * funding_cost_bps(avg_holding_days)) / 10000.0
    # Crude: subtract annualized drag from gross-return-equivalent and recompute
    # Assuming Sharpe = mean / std, drag ~ direct subtraction from mean
    # Assume 30% annualized vol typical for crypto strategies
    assumed_vol = 0.30
    sharpe_drag = annual_drag / assumed_vol
    return {
        "sleeve": name,
        "gross_sharpe": gross_sharpe,
        "approx_net_sharpe": gross_sharpe - sharpe_drag,
        "annual_cost_drag_pct": annual_drag * 100,
        "n_trades_per_year": n_trades_per_year,
        "is_perp": is_perp,
    }


def main():
    """Re-evaluate the known sleeve backtest Sharpes after realistic costs."""
    print("=" * 80)
    print("POST-COST SHARPE RE-EVALUATION (30bps slip + fee + funding)")
    print("=" * 80)
    print()
    print(f"{'Sleeve':<22s} {'Gross':>7s} {'Net':>7s} {'Drag%':>7s} {'N tx/yr':>8s}")
    print("-" * 80)
    sleeves = [
        # (name, gross_sharpe, n_trades_per_year, holding_days, is_perp)
        ("pro_trend_v5", 1.45, 24, 30, False),       # 2 entries/month, monthly hold
        ("bah_btc", 0.89, 12, 30, False),            # monthly rebal
        ("xsmom", 0.31, 26, 14, True),               # weekly rebal, perp
        ("oversold_bounce", 0.80, 60, 14, False),    # ~5/month, 2-week holds (est)
        ("overbought_fade", 0.60, 36, 7, True),      # ~3/month, 1-week, perp
        ("basis_arb", 0.50, 100, 30, True),          # frequent rebal, perp short
        ("vol_breakout", 0.40, 50, 7, False),
        ("short_term_momentum", 0.35, 80, 5, False),
        ("funding_basis", 0.25, 100, 30, True),
        ("diverse_mom_ethbtc", 0.50, 26, 14, False),
    ]
    threshold = 0.70
    survivors = []
    for s, gross, n_tx, hold, is_perp in sleeves:
        r = reevaluate_sleeve(s, gross, n_tx, hold, is_perp)
        net = r["approx_net_sharpe"]
        flag = "[KEEP]" if net >= threshold else "[CUT ]"
        print(f"  {flag}  {s:<22s} {gross:>+6.2f}  {net:>+6.2f}  "
              f"{r['annual_cost_drag_pct']:>5.2f}%  {n_tx:>5d}")
        if net >= threshold:
            survivors.append(s)
    print()
    print(f"Threshold: post-cost Sharpe >= {threshold:.2f}")
    print(f"Survivors: {len(survivors)} of {len(sleeves)} sleeves")
    for s in survivors:
        print(f"  - {s}")
    print()
    print("Recommendation: PRUNE the sleeves below threshold from the orchestrator.")
    print("They are running with no provable edge after realistic costs.")


if __name__ == "__main__":
    main()
