"""Drawdown-based position scaling.

Carver, Systematic Trading (2015), Ch. 9 (Risk Targeting).

As portfolio drawdown grows, scale exposure down continuously. Returns to
full size as drawdown recovers. Complements (does not replace) the binary
kill-switch in core/risk.py.

Default kink at 10%, kill at 30% — same scale as MAX_LOSS=20% and the
30% kill-switch already in RiskCaps.
"""

from __future__ import annotations


def drawdown_scale(
    equity: float, peak_equity: float, kink_dd: float = 0.10, kill_dd: float = 0.30
) -> float:
    """Continuous position scalar in [0, 1] as a function of current drawdown.

    - drawdown <= kink_dd: scale = 1.0 (full size)
    - kink_dd < drawdown < kill_dd: linear ramp from 1.0 to 0.0
    - drawdown >= kill_dd: scale = 0.0 (kill all positions)
    """
    if peak_equity <= 0:
        return 1.0
    dd = max(0.0, 1.0 - equity / peak_equity)
    if dd <= kink_dd:
        return 1.0
    if dd >= kill_dd:
        return 0.0
    return float(1.0 - (dd - kink_dd) / (kill_dd - kink_dd))


def update_peak(current_equity: float, recorded_peak: float) -> float:
    """Maintain a running peak for drawdown calculations."""
    return max(current_equity, recorded_peak)
