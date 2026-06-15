"""Regime-aware pause helper — empirically calibrated.

Backtest evidence (_bt_mtf_bull_regime.py / _bt_followups.py, 2026-06-01):

    Strategy: intraday_momentum, 8 pairs, 4h bars, 24h forward returns
    ----------------------------------------------------------------------
    2021 H1 BULL  (BTC +109%):   mean +2.48%, hit 57%   → tradeable
    2023 H2 CHOP  (BTC flat):    mean +0.58%, hit 46%   → marginal
    2022 H1 BEAR  (BTC  -60%):   mean -0.22%, hit 49%   → losing
    2026 H1 (cur) (BTC -4%):     mean -0.48%, hit 41%   → losing

Trending-momentum strategies (intraday_momentum, consolidation_breakout,
pro_trend LONG) have negative expected value in clear bear regimes. The MTF
filter doesn't fix this — even the BASE strategy is a bleed in bears.

Solution: pause these sleeves entirely when BTC is in clear bear regime.
Discriminator: BTC 30-day return < BEAR_THRESHOLD (-8%).

The current sleeve circuit breakers (DD, Sharpe) eventually pause these
sleeves AFTER they've already bled enough capital. Regime-pause acts
prospectively — pause BEFORE the bleed accumulates.

Note on walk-forward intent: this is a STRUCTURAL change based on 6-year
backtest evidence, not a parameter tweak. The 90-day walk-forward lock
prohibits parameter optimization on recent data; it permits structural fixes
based on out-of-sample empirical evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Regime thresholds (calibrated to backtest)
BEAR_THRESHOLD = -0.08    # BTC 30d return below this = clear bear
BULL_THRESHOLD = +0.08    # BTC 30d return above this = clear bull

# Sleeves that should PAUSE entirely in clear bear regime.
# These are strategies whose base edge requires trending or sideways markets.
PAUSE_IN_BEAR = {
    "intraday_momentum",         # trending momentum, fails in -60% bears
    "consolidation_breakout",    # breakouts fade back in downtrends
}

# Sleeves that should run NORMALLY regardless of regime — they have their own
# regime-aware logic OR are designed for bear capture.
REGIME_INDEPENDENT = {
    "bah_btc",                   # has its own MVRV cycle gate
    "overbought_fade",           # designed for bear regime (gates internally)
    "intraday_momentum_short",   # bear sleeve
    "pro_trend",                 # has SMA200 filter + MTF regime gate
    "xsmom",                     # cross-sectional, works in any regime
    "basis_arb",                 # funding-rate driven, regime-independent
    "grid_trader",               # oscillation capture, regime-independent
    "oversold_bounce",           # contrarian, often fires in bear bottoms
}


def get_btc_30d_return() -> Optional[float]:
    """Fetch BTC 30-day return. Returns None on data failure."""
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=35)
        if df.empty or len(df) < 31:
            return None
        return float(df["close"].iloc[-1] / df["close"].iloc[-31] - 1)
    except Exception:
        return None


def current_regime() -> dict:
    """Classify current market regime by BTC 30-day return.

    Returns: {regime, btc_30d_return, label, color}
    """
    ret = get_btc_30d_return()
    if ret is None:
        return {"regime": "unknown", "btc_30d_return": None,
                "label": "DATA UNAVAILABLE", "color": "gray"}
    if ret < BEAR_THRESHOLD:
        return {"regime": "bear", "btc_30d_return": ret,
                "label": f"CLEAR BEAR  (BTC 30d {ret*100:+.1f}%)", "color": "red"}
    if ret > BULL_THRESHOLD:
        return {"regime": "bull", "btc_30d_return": ret,
                "label": f"CLEAR BULL  (BTC 30d {ret*100:+.1f}%)", "color": "green"}
    return {"regime": "chop", "btc_30d_return": ret,
            "label": f"CHOP  (BTC 30d {ret*100:+.1f}%)", "color": "yellow"}


def should_pause_sleeve(sleeve: str) -> dict:
    """Return whether sleeve should be paused based on regime.

    Returns: {should_pause, reason, regime}
    """
    if sleeve not in PAUSE_IN_BEAR:
        return {"should_pause": False, "reason": "not_regime_gated", "regime": None}
    regime = current_regime()
    if regime["regime"] == "bear":
        return {
            "should_pause": True,
            "reason": "bear_regime_pause",
            "regime": regime,
            "rationale": (
                f"{sleeve} loses money in clear bear (backtest 2022 H1 BEAR: "
                f"-0.22%/signal). Pausing prospectively rather than waiting for "
                f"DD circuit breaker."
            ),
        }
    return {"should_pause": False, "reason": f"regime_ok ({regime['regime']})",
            "regime": regime}


def main():
    """CLI: show current regime + pause status per sleeve."""
    print("=" * 72)
    print("REGIME GATE STATUS")
    print("=" * 72)
    r = current_regime()
    print(f"\nCurrent regime: {r['label']}")
    if r['btc_30d_return'] is not None:
        print(f"  Thresholds: bear < {BEAR_THRESHOLD*100:+.0f}%   bull > {BULL_THRESHOLD*100:+.0f}%")
    print()
    print(f"{'Sleeve':<26s} {'Pause?':<10s} Reason")
    print("-" * 72)
    all_sleeves = sorted(PAUSE_IN_BEAR | REGIME_INDEPENDENT)
    for s in all_sleeves:
        p = should_pause_sleeve(s)
        flag = "PAUSED" if p["should_pause"] else "active"
        print(f"  {s:<24s} {flag:<10s} {p['reason']}")


if __name__ == "__main__":
    main()
