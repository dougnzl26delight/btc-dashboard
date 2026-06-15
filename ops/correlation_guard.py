"""Cross-sleeve correlation guard — detect collapsed diversification.

The premise of multi-sleeve allocation is DIVERSIFICATION: different signals
reduce single-strategy risk. If 3+ sleeves end up making the SAME trade
(e.g., all long alts during a relief rally), you don't have "3 strategies" —
you have "1 strategy x 3 implementations" with no actual diversification.

This module computes pairwise correlation of sleeve daily returns. When the
average pairwise correlation across active sleeves > THRESHOLD (default 0.7),
it returns a system-wide halving signal that all sleeves apply.

Use case:
    A bear-relief rally where oversold_bounce, BAH BTC, AND orchestrator all
    go long majors at the same time. Without this guard, the rig is 3x
    over-exposed. With this guard, all sleeves auto-scale down 50%.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.pnl_db import get_sleeve_returns


# Correlation collapse threshold — when avg pairwise corr exceeds this, halve sizes
CORR_THRESHOLD = 0.70
MIN_OBSERVATIONS = 14  # need at least 2 weeks of data to compute meaningful corr
MIN_ACTIVE_SLEEVES = 3  # need 3+ sleeves with data; otherwise no guard


def compute_pairwise_correlation(sleeves: list[str], lookback_days: int = 30) -> dict:
    """Pairwise correlation matrix from pnl_db daily returns.

    Returns: {
      "sleeves": list of sleeves with sufficient data,
      "matrix": 2D dict of corr values,
      "avg_pairwise": mean of all off-diagonal pairs,
      "max_pairwise": max off-diagonal,
    }
    """
    return_series = {}
    for s in sleeves:
        r = get_sleeve_returns(s, days=lookback_days)
        if len(r) >= MIN_OBSERVATIONS:
            # Reverse to chronological order
            return_series[s] = list(reversed(r))

    if len(return_series) < 2:
        return {"sleeves": list(return_series.keys()), "matrix": {}, "avg_pairwise": None,
                "max_pairwise": None, "n_sleeves_with_data": len(return_series)}

    # Align lengths: use min length across sleeves
    min_len = min(len(s) for s in return_series.values())
    aligned = {k: np.array(v[-min_len:]) for k, v in return_series.items()}

    sleeves_with_data = list(aligned.keys())
    matrix = {}
    pairwise_vals = []
    for i, s_i in enumerate(sleeves_with_data):
        matrix[s_i] = {}
        for j, s_j in enumerate(sleeves_with_data):
            if i == j:
                matrix[s_i][s_j] = 1.0
            else:
                xs = aligned[s_i]
                ys = aligned[s_j]
                if xs.std() == 0 or ys.std() == 0:
                    matrix[s_i][s_j] = 0.0
                else:
                    matrix[s_i][s_j] = float(np.corrcoef(xs, ys)[0, 1])
                if i < j:
                    pairwise_vals.append(matrix[s_i][s_j])

    avg = float(np.mean(pairwise_vals)) if pairwise_vals else None
    mx = float(np.max(np.abs(pairwise_vals))) if pairwise_vals else None
    return {
        "sleeves": sleeves_with_data,
        "matrix": matrix,
        "avg_pairwise": avg,
        "max_pairwise": mx,
        "n_sleeves_with_data": len(sleeves_with_data),
    }


def correlation_guard_scale(sleeves: list[str] | None = None,
                             threshold: float = CORR_THRESHOLD) -> float:
    """Return scale multiplier [0.5, 1.0].

    1.0 if diversification healthy. 0.5 if 3+ sleeves at avg corr > threshold.
    """
    if sleeves is None:
        sleeves = ["bah_btc", "xsmom", "pro_trend", "oversold_bounce",
                   "overbought_fade", "spot_orchestrator", "perp_orchestrator"]
    d = compute_pairwise_correlation(sleeves)
    avg = d.get("avg_pairwise")
    n_with_data = d.get("n_sleeves_with_data", 0)
    if avg is None or n_with_data < MIN_ACTIVE_SLEEVES:
        return 1.0  # insufficient data — no guard
    if avg > threshold:
        return 0.5  # correlation collapse — halve all
    return 1.0


def status() -> dict:
    """Detailed status for dashboard/reports."""
    sleeves = ["bah_btc", "xsmom", "pro_trend", "oversold_bounce",
               "overbought_fade", "spot_orchestrator", "perp_orchestrator"]
    d = compute_pairwise_correlation(sleeves)
    scale = correlation_guard_scale(sleeves)
    avg = d.get("avg_pairwise")
    return {
        "sleeves_with_data": d.get("sleeves", []),
        "n_sleeves": d.get("n_sleeves_with_data", 0),
        "avg_pairwise_correlation": avg,
        "max_pairwise_correlation": d.get("max_pairwise"),
        "scale_applied": scale,
        "guard_active": scale < 1.0,
        "threshold": CORR_THRESHOLD,
    }


def main():
    """CLI status."""
    s = status()
    print("=" * 60)
    print("CROSS-SLEEVE CORRELATION GUARD")
    print("=" * 60)
    print(f"Sleeves with data: {s['n_sleeves']}  ({', '.join(s['sleeves_with_data'])})")
    if s["avg_pairwise_correlation"] is None:
        print("Insufficient data for correlation analysis. Guard inactive.")
    else:
        print(f"Avg pairwise correlation: {s['avg_pairwise_correlation']:+.3f}")
        print(f"Max pairwise correlation: {s['max_pairwise_correlation']:+.3f}")
        print(f"Threshold: {s['threshold']:+.2f}")
        print(f"Scale applied: {s['scale_applied']:.2f}x")
        if s["guard_active"]:
            print("\n!! CORRELATION COLLAPSE: all sleeves should halve sizing")
        else:
            print("\nDiversification healthy.")


if __name__ == "__main__":
    main()
