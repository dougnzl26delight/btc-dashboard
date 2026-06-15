"""Swing trading backtest — Donchian breakout + ATR stops + multi-timeframe filter.

Setup (canonical Turtle / Carver swing):
  - Entry: long when price breaks above 20-day high AND price > 50-day SMA (uptrend filter)
  - Stop: 2 × ATR(14) below entry
  - Trailing stop: as price moves up, trail at 2 × ATR
  - Time stop: max 30-day hold
  - Position size: risk-based (1.5% of equity per trade at stop distance)
  - Long-only (crypto has structural bull bias)
  - One position per pair at a time

Holding period: 3-30 days typical (real swing). Trade frequency: low.
Costs realistic: 30 bps round-trip per trade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


ANNUALIZATION = 365


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def swing_backtest_pair(
    pair: str,
    days_back: int = 1500,
    starting_equity: float = 100_000.0,
    donchian_window: int = 20,
    sma_filter: int = 50,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
    atr_trail_mult: float = 2.0,
    risk_pct_per_trade: float = 0.015,    # 1.5% of equity at stop
    max_hold_days: int = 30,
    round_trip_bps: float = 30.0,
) -> dict:
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty or len(df) < max(donchian_window, sma_filter, atr_period) * 2:
        return {"error": "insufficient data"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_window).max().shift(1)
    df["sma_filter"] = df["close"].rolling(sma_filter).mean()
    df["atr"] = compute_atr(df, atr_period)
    df = df.dropna()

    n = len(df)
    equity = starting_equity
    in_trade = False
    qty = 0.0
    entry_price = 0.0
    entry_idx = -1
    initial_stop = 0.0
    trail_stop = 0.0
    high_water = 0.0

    trades: list[dict] = []
    equity_path = []

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])

        if in_trade:
            # Update trailing stop on new highs
            if high > high_water:
                high_water = high
                new_trail = high - atr_trail_mult * atr
                if new_trail > trail_stop:
                    trail_stop = new_trail

            # Check exits in priority order: stop, trail, time
            stop_hit = low <= trail_stop
            time_exit = (i - entry_idx) >= max_hold_days

            if stop_hit or time_exit:
                exit_price = trail_stop if stop_hit else price
                exit_reason = "trail_stop" if stop_hit else "time_stop"
                gross = qty * (exit_price - entry_price)
                exit_cost = qty * exit_price * round_trip_bps / 2 / 10_000  # half RT (close side)
                net_pnl = gross - exit_cost
                equity += net_pnl
                trades.append({
                    "open_ts": df.index[entry_idx],
                    "close_ts": row.name,
                    "n_days": i - entry_idx,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "gross_pnl": gross,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "return_pct": (exit_price / entry_price - 1),
                })
                in_trade = False
                qty = 0.0
        else:
            # Entry signal: price breaks Donchian high AND above SMA filter (uptrend)
            in_uptrend = price > float(row["sma_filter"])
            breakout = high >= float(row["donchian_high"])
            if breakout and in_uptrend and atr > 0:
                # Position sized by risk: equity × risk_pct / stop_distance
                stop_distance = atr_stop_mult * atr
                qty = (equity * risk_pct_per_trade) / stop_distance
                # Cap to 30% of equity notional
                max_qty = equity * 0.30 / price
                qty = min(qty, max_qty)
                entry_price = price
                entry_idx = i
                initial_stop = price - stop_distance
                trail_stop = initial_stop
                high_water = high
                entry_cost = qty * entry_price * round_trip_bps / 2 / 10_000
                equity -= entry_cost
                in_trade = True

        equity_path.append({"ts": row.name, "equity": equity, "in_trade": in_trade})

    # Force-close at end
    if in_trade:
        exit_price = float(df["close"].iloc[-1])
        gross = qty * (exit_price - entry_price)
        exit_cost = qty * exit_price * round_trip_bps / 2 / 10_000
        equity += gross - exit_cost
        trades.append({
            "open_ts": df.index[entry_idx],
            "close_ts": df.index[-1],
            "n_days": n - 1 - entry_idx,
            "entry": entry_price,
            "exit": exit_price,
            "qty": qty,
            "gross_pnl": gross,
            "net_pnl": gross - exit_cost,
            "exit_reason": "forced",
            "return_pct": (exit_price / entry_price - 1),
        })

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"]
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION))
        if daily_rets.std() > 0 else 0.0
    )
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    total_return = equity / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1

    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["net_pnl"] > 0) / max(n_trades, 1)
    avg_win = (
        np.mean([t["return_pct"] for t in trades if t["net_pnl"] > 0])
        if any(t["net_pnl"] > 0 for t in trades) else 0
    )
    avg_loss = (
        np.mean([t["return_pct"] for t in trades if t["net_pnl"] < 0])
        if any(t["net_pnl"] < 0 for t in trades) else 0
    )
    avg_hold = np.mean([t["n_days"] for t in trades]) if trades else 0

    # Benchmark
    bench_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    return {
        "pair": pair,
        "n_days": n_days,
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "avg_hold_days": float(avg_hold),
        "starting_equity": starting_equity,
        "ending_equity": float(equity),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "buy_and_hold_return": bench_return,
        "alpha_vs_bah": float(total_return) - bench_return,
        "trades_per_year": n_trades / max(n_days / 365, 1),
    }


if __name__ == "__main__":
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
             "AVAX/USDT", "LINK/USDT", "DOT/USDT"]

    print(f"{'Pair':<12s}  {'N tr':>5s}  {'Win%':>6s}  {'AvgHold':>8s}  "
          f"{'Total':>9s}  {'Annlzd':>9s}  {'Sharpe':>8s}  {'MaxDD':>8s}  "
          f"{'BAH':>9s}  {'Alpha':>9s}")
    print("-" * 110)

    portfolio_total = 0.0
    portfolio_dd = 0.0
    n_pairs = 0
    bah_total = 0.0
    n_trades_all = 0
    win_count = 0
    total_count = 0

    for p in pairs:
        r = swing_backtest_pair(p, days_back=1500)
        if "error" in r:
            print(f"{p}  ERROR: {r['error']}")
            continue
        print(f"{p:<12s}  {r['n_trades']:>5d}  {r['win_rate']:>5.0%}   "
              f"{r['avg_hold_days']:>6.1f}d   {r['total_return']:>+8.1%}   "
              f"{r['annualized_return']:>+8.1%}   {r['sharpe']:>+6.2f}    "
              f"{r['max_drawdown']:>6.2%}    {r['buy_and_hold_return']:>+8.1%}   "
              f"{r['alpha_vs_bah']:>+8.1%}")
        portfolio_total += r["total_return"]
        portfolio_dd = max(portfolio_dd, r["max_drawdown"])
        bah_total += r["buy_and_hold_return"]
        n_pairs += 1
        n_trades_all += r["n_trades"]

    if n_pairs:
        print("-" * 110)
        print(f"{'AVG':<12s}                                  "
              f"{portfolio_total/n_pairs:>+8.1%}                              "
              f"{portfolio_dd:>6.2%}    {bah_total/n_pairs:>+8.1%}   "
              f"{(portfolio_total - bah_total)/n_pairs:>+8.1%}")
        print(f"\nTotal trades across {n_pairs} pairs: {n_trades_all}")
