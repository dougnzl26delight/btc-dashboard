"""Cross-sectional momentum (XSMOM) backtest — 2nd uncorrelated strategy.

Different signal class from pro_trend's TSMOM (time-series momentum):

  TSMOM: own-asset return > threshold → long; below → short. Each pair
         independent. Pro_trend is this.

  XSMOM: rank pairs by recent return; long top tercile, short bottom
         tercile. RELATIVE strength, not absolute. Each rebalance
         period, capital rotates to current winners away from current
         losers.

Why uncorrelated to TSMOM?
  - TSMOM goes long when ALL pairs trend up; XSMOM goes long top regardless
    of overall direction
  - In sideways markets where TSMOM flatlines, XSMOM still rotates
  - Documented Sharpe 1.0-1.5 in crypto literature
  - Daniel/Moskowitz (2016) show XSMOM uncorrelated to TSMOM in equities

Implementation:
  - Universe: top-7 by liquidity (BTC, ETH, SOL, BNB, AVAX, LINK, MATIC)
  - Rebalance: weekly
  - Signal: 30d cumulative return rank
  - Long: top 2 (≈top 30%)
  - Short: bottom 2 (≈bottom 30%)
  - Equal-weight within each leg, dollar-neutral
  - 1.5% risk per leg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


ANNUALIZATION = 365

# Wider universe than pro_trend — XSMOM benefits from more cross-section
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ATOM/USDT",
]


def fetch_close_panel(days_back: int = 1500, pairs: list[str] | None = None) -> pd.DataFrame:
    """Returns DataFrame with one column per pair (close prices)."""
    pairs = pairs or UNIVERSE
    out = {}
    for p in pairs:
        df = data.ohlcv_extended(p, days_back=days_back)
        if df.empty or len(df) < 100:
            continue
        out[p] = df["close"]
    if not out:
        return pd.DataFrame()
    panel = pd.concat(out, axis=1)
    return panel


def xsmom_backtest(
    days_back: int = 1500,
    starting_equity: float = 100_000.0,
    momentum_window: int = 30,    # days
    rebalance_freq: int = 7,       # rebalance every N days (1 = daily, 7 = weekly)
    long_n: int = 2,
    short_n: int = 2,
    risk_per_leg: float = 0.015,
    round_trip_bps: float = 30.0,
    pairs: list[str] | None = None,
) -> dict:
    """Equal-weight long/short XSMOM, dollar-neutral.

    Returns: equity path + summary stats. Uses spot data (no perp funding cost).
    """
    panel = fetch_close_panel(days_back, pairs)
    if panel.empty or len(panel.columns) < (long_n + short_n):
        return {"error": "insufficient pairs"}

    panel = panel.dropna(how="all")

    # Compute momentum: cumulative return over window
    momentum = panel.pct_change(momentum_window)
    daily_rets = panel.pct_change()

    # Build target weights — rebalance every N days
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    n_days = len(panel)
    last_weights = pd.Series(0.0, index=panel.columns)
    n_rebals = 0
    for i in range(momentum_window + 1, n_days):
        date = panel.index[i]
        if (i - momentum_window - 1) % rebalance_freq != 0:
            weights.iloc[i] = last_weights
            continue
        # Rank by momentum on this date
        m = momentum.iloc[i].dropna()
        if len(m) < (long_n + short_n):
            weights.iloc[i] = last_weights
            continue
        ranked = m.sort_values(ascending=False)
        new_w = pd.Series(0.0, index=panel.columns)
        if long_n > 0:
            for p in ranked.index[:long_n]:
                new_w[p] = risk_per_leg / long_n
        if short_n > 0:
            for p in ranked.index[-short_n:]:
                new_w[p] = -risk_per_leg / short_n
        weights.iloc[i] = new_w
        last_weights = new_w
        n_rebals += 1

    # Compute portfolio returns
    # PnL_t = sum(weights_{t-1} * daily_ret_t) on a per-dollar-of-equity basis
    portfolio_rets = (weights.shift(1) * daily_rets).sum(axis=1)
    portfolio_rets = portfolio_rets.fillna(0)

    # Apply turnover cost — bps per dollar of weight change
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0)
    cost_per_day = turnover * round_trip_bps / 2 / 10_000
    portfolio_rets = portfolio_rets - cost_per_day

    # Equity path
    eq = starting_equity * (1 + portfolio_rets).cumprod()
    eq.iloc[0] = starting_equity
    daily_eq_rets = eq.pct_change().dropna()

    sharpe = float(daily_eq_rets.mean() / daily_eq_rets.std() * np.sqrt(ANNUALIZATION)) if daily_eq_rets.std() > 0 else 0
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    final_eq = float(eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days_actual = (eq.index[-1] - eq.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days_actual, 1)) - 1

    return {
        "n_days": n_days_actual,
        "starting_equity": starting_equity,
        "final_equity": final_eq,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_rebalances": n_rebals,
        "equity_path": eq,
        "daily_returns": daily_eq_rets,
        "n_pairs_in_universe": len(panel.columns),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("XSMOM (cross-sectional momentum) BACKTEST — meaningful size")
    print("=" * 70)
    print()

    # Risk-per-leg sweep at the best parameter combo (14d window, 14d rebal, 2/2)
    print("Risk-per-leg sweep at window=14d, rebal=14d, L/S=2/2:")
    print(f"{'Risk/leg':>8s}  {'Annlzd':>8s}  {'Sharpe':>6s}  {'MaxDD':>6s}")
    best_run = None
    best_sharpe = -10
    for risk in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        r = xsmom_backtest(
            days_back=1500,
            momentum_window=14,
            rebalance_freq=14,
            long_n=2, short_n=2,
            risk_per_leg=risk,
        )
        if "error" in r:
            continue
        marker = ""
        if r["sharpe"] > best_sharpe:
            best_sharpe = r["sharpe"]
            best_run = (risk, r)
            marker = "  <-- best"
        print(f"{risk:>7.2f}   {r['annualized_return']:>+7.2%}  "
              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}{marker}")

    print()
    # Long-only variant (drop short leg)
    print("Long-only variants (no short leg):")
    print(f"{'Risk':>5s}  {'Window':>6s}  {'Rebal':>5s}  {'Long N':>6s}  "
          f"{'Annlzd':>8s}  {'Sharpe':>6s}  {'MaxDD':>6s}")
    for risk in [0.10, 0.20, 0.30]:
        for mw in [14, 30, 60]:
            for rebal in [7, 14, 30]:
                for lng in [2, 3]:
                    r = xsmom_backtest(
                        days_back=1500,
                        momentum_window=mw,
                        rebalance_freq=rebal,
                        long_n=lng, short_n=0,
                        risk_per_leg=risk,
                    )
                    if "error" in r:
                        continue
                    if r["sharpe"] > best_sharpe:
                        best_sharpe = r["sharpe"]
                        best_run = (risk, r)
                        marker = "  <-- best"
                    else:
                        marker = ""
                    if r["sharpe"] > 0.5 or marker:
                        print(f"{risk:>5.2f}   {mw:>5d}d   {rebal:>4d}d   "
                              f"{lng:>5d}    {r['annualized_return']:>+7.2%}  "
                              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}{marker}")

    if best_run:
        risk, best = best_run
        print()
        print(f"BEST overall: risk={risk}")
        print(f"  Annualized: {best['annualized_return']:+.2%}")
        print(f"  Sharpe:     {best['sharpe']:+.2f}")
        print(f"  MaxDD:      {best['max_drawdown']:.1%}")
