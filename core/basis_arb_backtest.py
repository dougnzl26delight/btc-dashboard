"""Funding-rate basis arbitrage backtest.

The trade:
  1. When BTC perp funding > entry threshold, open long-spot + short-perp
  2. Collect funding payment every 8h (paid TO short perp side)
  3. Close when funding < exit threshold (or held too long)

Per ScienceDirect 2024 paper, retail-realistic Sharpe 1-3, APR 10-30% in
favorable conditions. Documented Sharpe 5-15 in extreme conditions/leverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


# Defaults — tuned for retail-friendly thresholds
FUNDING_ENTRY_BPS_8H = 1.0       # 0.01% per 8h = ~11% annualized
FUNDING_EXIT_BPS_8H = 0.3        # 0.003% per 8h = ~3.3% annualized
MAX_ALLOCATION = 0.30            # max 30% of bankroll into basis trade
ROUND_TRIP_COST_BPS = 60         # 4 trades × 15 bps = 60 bps per round trip
ANNUALIZATION = 365


def backtest_basis_arb(
    perp_pair: str = "BTC/USDT:USDT",
    spot_pair: str = "BTC/USDT",
    days_back: int = 1000,
    starting_equity: float = 100_000.0,
    funding_entry_bps_8h: float = FUNDING_ENTRY_BPS_8H,
    funding_exit_bps_8h: float = FUNDING_EXIT_BPS_8H,
    max_allocation: float = MAX_ALLOCATION,
    round_trip_cost_bps: float = ROUND_TRIP_COST_BPS,
) -> dict:
    funding_df = data.funding_history_extended(perp_pair, days_back=days_back)
    spot_df = data.ohlcv_extended(spot_pair, days_back=days_back)

    if funding_df.empty or spot_df.empty:
        return {"error": "no data"}

    # Align: funding 8-hourly, spot daily — forward-fill spot to funding timestamps
    funding_df = funding_df.sort_index()
    spot_df = spot_df.sort_index()
    funding_df["spot_price"] = spot_df["close"].reindex(funding_df.index, method="ffill")
    funding_df = funding_df.dropna()
    if funding_df.empty:
        return {"error": "no aligned data"}

    funding_df["funding_bps"] = funding_df["funding_rate"] * 10_000.0

    n_events = len(funding_df)
    equity = starting_equity
    in_trade = False
    notional = 0.0
    cycle_start = None
    cycle_funding = 0.0
    trade_open_idx = -1

    equity_path: list[dict] = []
    trades: list[dict] = []

    for i in range(n_events):
        row = funding_df.iloc[i]
        f_bps = float(row["funding_bps"])

        if in_trade:
            # Funding payment received this period (positive funding × notional)
            funding_payment = notional * float(row["funding_rate"])
            equity += funding_payment
            cycle_funding += funding_payment

            # Exit?
            if f_bps < funding_exit_bps_8h:
                close_cost = notional * (round_trip_cost_bps / 2.0) / 10_000.0
                equity -= close_cost
                trades.append({
                    "open_ts": cycle_start,
                    "close_ts": row.name,
                    "n_periods": i - trade_open_idx,
                    "notional": notional,
                    "funding_collected": cycle_funding,
                    "close_cost": close_cost,
                    "net_pnl": cycle_funding - close_cost,
                    "annualized_return_in_trade": (
                        (cycle_funding - close_cost) / notional * ANNUALIZATION
                        / max(i - trade_open_idx, 1) * 3  # 3 periods per day (8h)
                    ),
                })
                in_trade = False
                notional = 0.0
                cycle_funding = 0.0
        else:
            if f_bps > funding_entry_bps_8h:
                notional = equity * max_allocation
                open_cost = notional * (round_trip_cost_bps / 2.0) / 10_000.0
                equity -= open_cost
                cycle_funding = -open_cost
                in_trade = True
                cycle_start = row.name
                trade_open_idx = i

        equity_path.append({
            "ts": row.name,
            "equity": equity,
            "in_trade": in_trade,
            "funding_bps": f_bps,
        })

    # Force-close any open trade
    if in_trade:
        close_cost = notional * (round_trip_cost_bps / 2.0) / 10_000.0
        equity -= close_cost
        trades.append({
            "open_ts": cycle_start,
            "close_ts": funding_df.index[-1],
            "n_periods": n_events - trade_open_idx,
            "notional": notional,
            "funding_collected": cycle_funding,
            "close_cost": close_cost,
            "net_pnl": cycle_funding - close_cost,
            "forced_close": True,
        })

    eq_df = pd.DataFrame(equity_path)
    if eq_df.empty:
        return {"error": "no equity path"}
    eq_df = eq_df.set_index("ts")

    daily_eq = eq_df["equity"].resample("1D").last().ffill()
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION))
        if daily_rets.std() > 0 else 0.0
    )
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    total_return = equity / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = ((1 + total_return) ** (ANNUALIZATION / max(n_days, 1))) - 1

    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["net_pnl"] > 0) / max(n_trades, 1)
    avg_pnl = sum(t["net_pnl"] for t in trades) / max(n_trades, 1)
    avg_duration = sum(t["n_periods"] for t in trades) / max(n_trades, 1) / 3  # in days

    pct_time_in_trade = sum(1 for e in equity_path if e["in_trade"]) / max(len(equity_path), 1)

    return {
        "n_funding_events": n_events,
        "n_days": n_days,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_pnl_per_trade_usdt": float(avg_pnl),
        "avg_duration_days": float(avg_duration),
        "pct_time_in_trade": float(pct_time_in_trade),
        "starting_equity": starting_equity,
        "ending_equity": float(equity),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "params": {
            "entry_bps_8h": funding_entry_bps_8h,
            "exit_bps_8h": funding_exit_bps_8h,
            "max_allocation": max_allocation,
            "round_trip_cost_bps": round_trip_cost_bps,
        },
    }


if __name__ == "__main__":
    import json

    print("=== BTC funding basis arb (default thresholds) ===")
    r = backtest_basis_arb(days_back=1000)
    if "error" in r:
        print(r["error"])
    else:
        for k in ["n_funding_events", "n_days", "n_trades", "win_rate",
                  "pct_time_in_trade", "total_return", "annualized_return",
                  "sharpe", "max_drawdown", "avg_duration_days"]:
            v = r[k]
            print(f"  {k:30s} {v:+.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    print()
    print("=== BTC basis arb sensitivity to entry threshold ===")
    print(f"{'Entry bps/8h':>12s}  {'N trades':>10s}  {'Total ret':>10s}  {'Sharpe':>8s}  {'MaxDD':>8s}")
    print("-" * 60)
    for entry in [0.5, 1.0, 2.0, 3.0, 5.0]:
        r = backtest_basis_arb(days_back=1000, funding_entry_bps_8h=entry, funding_exit_bps_8h=0.3)
        if "error" not in r:
            print(f"{entry:>10.1f}     {r['n_trades']:>8d}     {r['total_return']:>+8.2%}    "
                  f"{r['sharpe']:>+6.2f}    {r['max_drawdown']:>6.2%}")
