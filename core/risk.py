"""Risk management — position caps, drawdown gates, sizing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCaps:
    max_position_frac: float = 0.10   # CLAUDE.md: 5-10% per name
    max_loss_frac: float = 0.20       # CLAUDE.md: 15-20% per-position stop
    max_portfolio_dd: float = 0.30    # kill all when portfolio dd > this
    max_strategy_alloc: float = 0.30  # one strategy <= this fraction of book


DEFAULT = RiskCaps()


def position_size(
    equity: float,
    target_frac: float,
    price: float,
    caps: RiskCaps = DEFAULT,
) -> float:
    """Convert a target portfolio fraction in [-1, 1] to a base-asset quantity, clamped."""
    if price <= 0 or equity <= 0:
        return 0.0
    capped = max(-caps.max_position_frac, min(caps.max_position_frac, target_frac))
    return capped * equity / price


def drawdown_breach(
    equity: float,
    peak_equity: float,
    caps: RiskCaps = DEFAULT,
) -> tuple[bool, float]:
    """Return (kill_switch_triggered, drawdown_fraction)."""
    if peak_equity <= 0:
        return False, 0.0
    dd = 1.0 - equity / peak_equity
    return dd > caps.max_portfolio_dd, dd


def equity_value(
    cash: float, positions: dict[str, float], prices: dict[str, float]
) -> float:
    """Mark-to-market portfolio value."""
    return cash + sum(qty * prices.get(asset, 0.0) for asset, qty in positions.items())
