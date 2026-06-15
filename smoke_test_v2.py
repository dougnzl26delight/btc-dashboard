"""Smoke test for the four practitioner-technique modules.

Tests:
  1. core/sizing.py        — fractional Kelly
  2. core/exits.py         — triple-barrier
  3. core/drawdown_scale.py — Carver drawdown scaling
  4. core/tail_overlay.py   — Daniel/Moskowitz crash overlay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from core import data, sizing, exits, drawdown_scale, tail_overlay
from research import signals as res_sig


def main():
    print("=" * 60)
    print("FETCHING DATA")
    print("=" * 60)
    df = data.ohlcv_extended("BTC/USDT", days_back=730)
    prices = df["close"]
    print(f"  {len(prices)} BTC daily bars, {prices.index[0].date()} -> {prices.index[-1].date()}")

    print("\n" + "=" * 60)
    print("1. FRACTIONAL KELLY SIZING")
    print("=" * 60)
    sig_series = res_sig.tsmom_multi(prices, horizons=(30, 90))
    fwd_ret = prices.pct_change().shift(-1)
    fk_size = sizing.kelly_from_signal_history(sig_series, fwd_ret, fraction=0.25)
    print(f"  TSMOM(30,90) fractional Kelly @ 0.25x: {fk_size:+.4f} of capital")
    full_kelly = sizing.kelly_fraction((sig_series * fwd_ret).dropna())
    print(f"  Full Kelly would be: {full_kelly:+.3f} (clipped to safer fractional)")

    print("\n" + "=" * 60)
    print("2. TRIPLE-BARRIER EXITS")
    print("=" * 60)
    # Sparse events: trigger entry whenever |signal| crosses 0.5
    events = pd.Series(0.0, index=sig_series.index)
    events[sig_series > 0.5] = 1.0
    events[sig_series < -0.5] = -1.0
    # Take only first event in each cluster (don't trigger every day)
    events = events.where(events != events.shift(1), 0)
    print(f"  generated {(events != 0).sum()} trade events from TSMOM signal")

    barriers = exits.triple_barrier(prices, events, horizon_days=30, pt_sigma=2.0, sl_sigma=1.5)
    summary = exits.barrier_summary(barriers)
    print(f"  trades closed: {summary['n_trades']}")
    print(f"  win rate: {summary['win_rate']:.1%}")
    print(f"  avg return per trade: {summary['avg_return']:+.2%}")
    print(f"  avg holding period: {summary['avg_duration']:.1f} days")
    print(f"  exit reasons: {summary['exit_reason_counts']}")

    print("\n" + "=" * 60)
    print("3. DRAWDOWN SCALING")
    print("=" * 60)
    scenarios = [
        ("no drawdown",      100_000, 100_000),
        ("5% drawdown",       95_000, 100_000),
        ("10% drawdown (kink)", 90_000, 100_000),
        ("20% drawdown (mid)", 80_000, 100_000),
        ("30% drawdown (kill)", 70_000, 100_000),
        ("40% drawdown (past kill)", 60_000, 100_000),
    ]
    for label, eq, peak in scenarios:
        scale = drawdown_scale.drawdown_scale(eq, peak)
        print(f"  {label:30s} -> scale={scale:.3f}")

    print("\n" + "=" * 60)
    print("4. TAIL-RISK OVERLAY")
    print("=" * 60)
    # Build a strategy P&L series from TSMOM signal
    weights = sig_series.shift(1).fillna(0).clip(-1, 1)
    strat_ret = (weights * fwd_ret).dropna()
    overlay_scale = tail_overlay.crash_adjusted_size(strat_ret, target_vol=0.15, span=30)
    print(f"  TSMOM strategy realized vol -> overlay scale: {overlay_scale:.3f}")
    print(f"    (1.0 = full size, < 1.0 = scaled down due to elevated strategy vol)")
    series = tail_overlay.crash_adjusted_series(strat_ret, target_vol=0.15, span=30)
    print(f"  series stats: mean={series.mean():.3f}, min={series.min():.3f}, max={series.max():.3f}")

    print("\nALL FOUR MODULES OK")


if __name__ == "__main__":
    main()
