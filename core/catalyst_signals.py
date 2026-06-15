"""Catalyst signal overlays — halving cycle and ETF flow surges.

These don't generate entry signals on their own. Instead they SCALE the
position size of existing trend strategies when the macro setup is
particularly favorable.

Multiplier convention:
  1.0 = neutral (no boost, no reduction)
  1.5 = strong bullish setup (e.g., post-halving + heavy ETF inflows)
  0.5 = bearish overlay (e.g., late-cycle + outflows)
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.etf_flows import etf_flow_signal, fetch_etf_flows


# BTC halvings — historical and projected
HALVINGS = [
    date(2012, 11, 28),
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 19),
    date(2028, 4, 1),  # estimated
]


def days_since_last_halving(today: date | None = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    past = [h for h in HALVINGS if h <= today]
    if not past:
        return -1
    return (today - max(past)).days


def halving_cycle_multiplier(today: date | None = None) -> float:
    """Position-size multiplier based on BTC halving cycle.

    Historical pattern:
      - Months 0-6 post-halving: building, neutral (1.0)
      - Months 6-18 post-halving: peak bull cycle (1.5)
      - Months 18-30: distribution, late cycle (1.0)
      - Months 30-48: bear / accumulation (0.5)
    """
    days = days_since_last_halving(today)
    if days < 0:
        return 1.0
    months = days / 30.4
    if months < 6:
        return 1.0
    if months < 18:
        return 1.5
    if months < 30:
        return 1.0
    return 0.5


def etf_flow_multiplier(z_threshold_strong: float = 1.5) -> float:
    """Position-size multiplier from ETF flow signal.

    Returns:
      1.5 if recent ETF flows are >1.5σ above mean (strong inflow surge)
      1.2 if 0.5–1.5σ above
      1.0 in neutral range
      0.8 if 0.5–1.5σ below
      0.5 if <-1.5σ (heavy outflows)
    """
    try:
        flows = fetch_etf_flows()
        if flows.empty:
            return 1.0
        sig = etf_flow_signal(flows, ema_window=7)
        if sig.empty:
            return 1.0
        z = float(sig.iloc[-1]) * 2  # signal was clipped at ±1, undo
    except Exception:
        return 1.0

    if z >= z_threshold_strong:
        return 1.5
    if z >= 0.5:
        return 1.2
    if z <= -z_threshold_strong:
        return 0.5
    if z <= -0.5:
        return 0.8
    return 1.0


def combined_catalyst_multiplier(today: date | None = None) -> dict:
    """Combine halving + ETF multipliers (capped to avoid runaway).

    Cap at 2.0 (single-position) to keep risk bounded.
    """
    h = halving_cycle_multiplier(today)
    e = etf_flow_multiplier()
    combined = min(2.0, h * e)
    return {
        "halving_mult": h,
        "etf_mult": e,
        "combined_mult": combined,
        "days_since_halving": days_since_last_halving(today),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(combined_catalyst_multiplier(), indent=2, default=str))
