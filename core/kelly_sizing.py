"""Kelly criterion + ATR-adjusted position sizing.

Kelly fraction is the mathematically optimal bet size given known edge.
Full Kelly is theoretically optimal but practically suicidal due to:
    1. Edge uncertainty (estimated win-rate has wide CIs)
    2. Fat tails (single losses can exceed Kelly's assumption)
    3. Path dependence (drawdowns of 50%+ even at optimal Kelly)

Half-Kelly is the practitioner standard. Captures 75% of compound growth
with 50% of full-Kelly's drawdown profile.

ATR adjustment: position size scales inversely with current volatility.
Equivalent risk-per-unit across pairs regardless of vol regime.

References:
    Kelly Jr., J.L. (1956) "A New Interpretation of Information Rate"
    Thorp, E.O. (1962) "Beat the Dealer" — practical Kelly applications
    Ernie Chan (2013) — half-Kelly recommendation for finance
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def kelly_fraction(win_rate: float, avg_win_pct: float, avg_loss_pct: float,
                    kelly_multiplier: float = 0.5) -> float:
    """Kelly fraction with safety multiplier.

    Args:
        win_rate: P(winning trade), e.g., 0.55
        avg_win_pct: average winning trade return, e.g., 0.04 (4%)
        avg_loss_pct: average losing trade return, e.g., -0.02 (-2%; pass as POSITIVE)
        kelly_multiplier: 0.5 = half-Kelly (recommended), 1.0 = full Kelly

    Returns: fraction of equity to risk per trade [0, 1]
    """
    if avg_loss_pct <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    # Standard Kelly: f = p/L - q/W, where W = avg_win/avg_loss
    W = avg_win_pct / avg_loss_pct  # win/loss ratio
    p = win_rate
    q = 1 - win_rate
    # f = p - q/W  (Kelly per unit bet)
    raw_kelly = p - q / W
    if raw_kelly <= 0:
        return 0.0
    # Apply safety multiplier
    return max(0.0, min(1.0, raw_kelly * kelly_multiplier))


def atr_position_size(equity: float, risk_pct: float, entry_price: float,
                       stop_price: float, atr: Optional[float] = None) -> dict:
    """ATR-adjusted position size for fixed-risk-per-trade.

    Logic: risk_per_trade = equity * risk_pct
           stop_distance = |entry - stop|
           position_size_units = risk_per_trade / stop_distance

    Optionally pass ATR for volatility-cap (don't oversize a tight-stop trade).

    Returns: {qty, notional, risk_usd, distance_to_stop_pct}
    """
    if entry_price <= 0 or stop_price <= 0:
        return {"qty": 0, "notional": 0, "risk_usd": 0, "error": "invalid_prices"}
    risk_usd = equity * risk_pct
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return {"qty": 0, "notional": 0, "risk_usd": risk_usd, "error": "zero_stop_distance"}

    qty = risk_usd / stop_distance
    notional = qty * entry_price

    # ATR vol cap: if stop is < 1 ATR, cap qty to risk_usd/ATR instead
    if atr is not None and atr > 0:
        min_safe_stop = atr  # 1 ATR minimum vol-aware stop
        if stop_distance < min_safe_stop:
            qty = risk_usd / min_safe_stop
            notional = qty * entry_price

    dist_pct = stop_distance / entry_price
    return {
        "qty": qty,
        "notional": notional,
        "risk_usd": risk_usd,
        "stop_distance_pct": dist_pct,
        "stop_distance_usd": stop_distance,
    }


def composite_size(sleeve_equity: float,
                    win_rate: float, avg_win_pct: float, avg_loss_pct: float,
                    entry_price: float, stop_price: float,
                    atr: Optional[float] = None,
                    risk_pct_floor: float = 0.005,
                    risk_pct_cap: float = 0.02) -> dict:
    """Compose Kelly + ATR sizing for one trade.

    Final size:
        1. Compute Kelly fraction
        2. Clamp to [floor, cap] for sanity
        3. Apply ATR-adjusted unit sizing using clamped fraction as risk_pct

    Returns full sizing dict.
    """
    kelly = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_multiplier=0.5)
    risk_pct = max(risk_pct_floor, min(risk_pct_cap, kelly))
    sizing = atr_position_size(sleeve_equity, risk_pct, entry_price, stop_price, atr)
    sizing["kelly_fraction"] = kelly
    sizing["risk_pct_used"] = risk_pct
    sizing["clamped"] = kelly != risk_pct
    return sizing


# === Wiring into unified gates pipeline ===========================

# Kelly multiplier range — bounded like meta-confidence so it cannot
# dominate the other gates.
KELLY_SCALE_MIN = 0.5
KELLY_SCALE_MAX = 1.5

# Baseline Kelly that maps to 1.0× scale. A sleeve generating exactly this
# half-Kelly fraction keeps its natural sizing; lower than this scales DOWN,
# higher scales UP, both clamped to [MIN, MAX].
KELLY_BASELINE = 0.02   # 2% of equity — matches the per-trade sanity cap


def sleeve_kelly_stats(sleeve: str, days: int = 60) -> Optional[dict]:
    """Derive win-rate / avg-win / avg-loss from sleeve's daily returns.

    Trade-level breakdown isn't always available, so we use DAILY returns as
    a proxy: positive day = win, negative day = loss. This is robust to
    incomplete trade-level data and aligns with the rest of the gates pipeline
    (which already uses daily returns for Sharpe).

    Returns None if fewer than 14 daily observations.
    """
    try:
        from core.pnl_db import get_sleeve_returns
    except Exception:
        return None
    rets = get_sleeve_returns(sleeve, days=days)
    if rets is None or len(rets) < 14:
        return None
    wins = [r for r in rets if r > 0]
    losses = [-r for r in rets if r < 0]  # convert to positive magnitude
    if not wins or not losses:
        return None
    win_rate = len(wins) / len(rets)
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(losses))
    return {
        "n_obs": len(rets),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def kelly_multiplier(sleeve: str, days: int = 60) -> float:
    """Return a Kelly-derived scale in [KELLY_SCALE_MIN, KELLY_SCALE_MAX].

    Maps half-Kelly fraction to a position-size multiplier:
        kelly << baseline  -> 0.5x   (edge weak/uncertain)
        kelly ≈ baseline   -> 1.0x
        kelly >> baseline  -> 1.5x   (clear edge from live data)

    Returns 1.0 (no effect) when sleeve has <14 days of live returns, so the
    gate stays neutral during warm-up.
    """
    stats = sleeve_kelly_stats(sleeve, days=days)
    if stats is None:
        return 1.0
    k = kelly_fraction(stats["win_rate"], stats["avg_win"], stats["avg_loss"],
                       kelly_multiplier=0.5)
    if k <= 0:
        # Negative Kelly = no edge in recent window — minimum sizing.
        return KELLY_SCALE_MIN
    ratio = k / KELLY_BASELINE
    # Linear mapping: ratio of 1.0 → 1.0x scale.
    scale = 1.0 * ratio if ratio <= 1.0 else 1.0 + 0.5 * min(ratio - 1.0, 1.0)
    return max(KELLY_SCALE_MIN, min(KELLY_SCALE_MAX, scale))


def main():
    """CLI demo: Kelly + ATR for a sample trade."""
    print("=" * 70)
    print("KELLY + ATR POSITION SIZING")
    print("=" * 70)
    # Sample: pro_trend-like stats
    print("\nExample sleeve (pro_trend-like): WR 40%, avg win 8%, avg loss 3%")
    print(f"Equity: $10,000  Entry: $73,500  Stop: $71,400  ATR: $1,200")
    print()
    s = composite_size(
        sleeve_equity=10_000,
        win_rate=0.40, avg_win_pct=0.08, avg_loss_pct=0.03,
        entry_price=73_500, stop_price=71_400,
        atr=1_200,
    )
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:<24s}  {v:.4f}" if abs(v) < 1 else f"  {k:<24s}  ${v:,.2f}" if "usd" in k or "notional" in k else f"  {k:<24s}  {v:.4f}")
        else:
            print(f"  {k:<24s}  {v}")


if __name__ == "__main__":
    main()
