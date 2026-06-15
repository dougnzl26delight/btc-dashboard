"""Loss-streak sizing buffer — automated psychological discipline.

After N consecutive losing days, halve all NEW entry sizes for K days.

Why: after a losing streak, every trader's worst instinct is "revenge sizing"
— double down to "win back" losses. Statistically this destroys CTAs more
than any single market move. Hardcoding the buffer prevents the operator
(you) from making this mistake when feelings are loud.

Tier:
    0-2 losing days   -> scale 1.0 (full size)
    3-4               -> scale 0.50
    5-6               -> scale 0.25
    7+                -> scale 0.10 (almost-stopped; force a re-evaluation)

Reads from pnl_db.daily_pnl. If insufficient data, returns 1.0 (no buffer).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pnl_db import get_sleeve_returns


# Scale tiers — (max_consecutive_losing_days, scale_multiplier)
TIERS = [
    (2, 1.00),    # 0-2 losing days: full size
    (4, 0.50),    # 3-4: half
    (6, 0.25),    # 5-6: quarter
    (999, 0.10),  # 7+: nearly-stopped
]


def consecutive_losing_days(sleeve: str, lookback: int = 14) -> int:
    """Count the streak of consecutive losing days ENDING today.

    Returns 0 if today is positive (or no data).
    """
    returns = get_sleeve_returns(sleeve, days=lookback)
    if not returns:
        return 0
    # get_sleeve_returns returns most-recent-first
    streak = 0
    for r in returns:
        if r < 0:
            streak += 1
        else:
            break
    return streak


def loss_streak_scale(sleeve: str) -> float:
    """Multiplier in [0.1, 1.0] based on current loss streak."""
    streak = consecutive_losing_days(sleeve)
    for max_streak, scale in TIERS:
        if streak <= max_streak:
            return scale
    return 0.1


def status(sleeves: list[str]) -> list[dict]:
    """Per-sleeve loss-streak status for dashboard/reports."""
    out = []
    for s in sleeves:
        streak = consecutive_losing_days(s)
        scale = loss_streak_scale(s)
        out.append({
            "sleeve": s,
            "consecutive_losing_days": streak,
            "scale": scale,
            "buffer_active": scale < 1.0,
        })
    return out


def main():
    """CLI status."""
    sleeves = ["bah_btc", "xsmom", "pro_trend", "oversold_bounce", "overbought_fade",
               "spot_orchestrator", "perp_orchestrator"]
    print(f"{'Sleeve':<22s} {'Loss days':>10s} {'Scale':>7s} {'Status':<20s}")
    print("-" * 65)
    for r in status(sleeves):
        flag = "BUFFER ACTIVE" if r["buffer_active"] else "OK"
        print(f"{r['sleeve']:<22s} {r['consecutive_losing_days']:>10d} "
              f"{r['scale']:>6.2f}x {flag:<20s}")


if __name__ == "__main__":
    main()
