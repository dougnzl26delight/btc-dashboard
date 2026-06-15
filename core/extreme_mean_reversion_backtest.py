"""Extreme mean-reversion strategy — buy 3+σ down moves, exit on reversion.

Different from naive z-score reversion (which we showed loses money) because
we ONLY trade extreme moves where statistical regression to mean is most
pronounced. Per Daniel/Moskowitz: tail moves have stronger mean-reversion
properties than middle-of-distribution z-scores.

Setup:
  - Compute 5-day return z-score
  - Long when z < -2.5 (price has dropped 2.5+σ in last 5 days)
  - Exit when z > -0.5 (most of the move has reverted) OR after 10 days
  - 1.5% risk per trade
  - Long-only (catching crashes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


ANNUALIZATION = 365


def extreme_revert_backtest(
    pair: str = "BTC/USDT",
    days_back: int = 1500,
    starting_equity: float = 100_000.0,
    return_window: int = 5,
    z_window: int = 60,
    z_entry: float = -2.5,
    z_exit: float = -0.5,
    max_hold_days: int = 10,
    risk_pct_per_trade: float = 0.015,
    stop_pct: float = 0.10,    # hard 10% stop
    round_trip_bps: float = 30.0,
) -> dict:
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty or len(df) < z_window * 2:
        return {"error": "insufficient data"}

    df = df.copy()
    df["ret_window"] = df["close"].pct_change(return_window)
    df["ret_mean"] = df["ret_window"].rolling(z_window).mean()
    df["ret_std"] = df["ret_window"].rolling(z_window).std()
    df["z"] = (df["ret_window"] - df["ret_mean"]) / df["ret_std"]
    df = df.dropna()
    n = len(df)

    cash = starting_equity
    in_trade = False
    qty = 0.0
    entry_price = 0.0
    entry_idx = -1
    stop_price = 0.0
    trades, equity_path = [], []

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        z = float(row["z"])

        if in_trade:
            # Hard stop
            if price <= stop_price:
                exit_p = stop_price
                pnl = qty * (exit_p - entry_price)
                cash += pnl - qty * exit_p * round_trip_bps / 2 / 10_000
                trades.append({
                    "n_days": i - entry_idx, "entry": entry_price, "exit": exit_p,
                    "pnl": pnl, "z_entry": float(df["z"].iloc[entry_idx]),
                    "reason": "hard_stop",
                })
                in_trade = False
                qty = 0
            elif z > z_exit or (i - entry_idx) >= max_hold_days:
                exit_p = price
                pnl = qty * (exit_p - entry_price)
                cash += pnl - qty * exit_p * round_trip_bps / 2 / 10_000
                trades.append({
                    "n_days": i - entry_idx, "entry": entry_price, "exit": exit_p,
                    "pnl": pnl, "z_entry": float(df["z"].iloc[entry_idx]),
                    "z_exit": z, "reason": "z_exit" if z > z_exit else "time_stop",
                })
                in_trade = False
                qty = 0
        else:
            if z < z_entry:
                qty = (cash * risk_pct_per_trade) / (price * stop_pct)
                qty = min(qty, cash * 0.30 / price)
                cash -= qty * price * round_trip_bps / 2 / 10_000
                entry_price = price
                entry_idx = i
                stop_price = price * (1 - stop_pct)
                in_trade = True

        unrealized = qty * (price - entry_price) if in_trade else 0
        equity_path.append({"ts": row.name, "equity": cash + unrealized + (qty * price if in_trade else 0)})

    if in_trade:
        exit_p = float(df["close"].iloc[-1])
        cash += qty * (exit_p - entry_price) - qty * exit_p * round_trip_bps / 2 / 10_000

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"]
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION)) if daily_rets.std() > 0 else 0
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1
    bah_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["pnl"] > 0) / max(n_trades, 1)
    avg_hold = np.mean([t["n_days"] for t in trades]) if trades else 0
    trades_per_year = n_trades / max(n_days / 365, 1)

    return {
        "pair": pair, "n_days": n_days,
        "n_trades": n_trades, "trades_per_year": float(trades_per_year),
        "win_rate": float(win_rate), "avg_hold_days": float(avg_hold),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe), "max_drawdown": float(max_dd),
        "bah_return": bah_return, "alpha_vs_bah": float(total_return) - bah_return,
    }


if __name__ == "__main__":
    print("=== Extreme mean-reversion backtest, 1500 days ===")
    print(f"{'Pair':<12s} {'N tr':>5s} {'TPY':>5s} {'Win%':>6s} {'Hold':>7s} "
          f"{'Total':>9s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s}")
    print("-" * 80)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]:
        r = extreme_revert_backtest(pair=pair, days_back=1500)
        if "error" in r:
            print(f"{pair}: {r['error']}")
            continue
        print(f"{pair:<12s} {r['n_trades']:>4d}  {r['trades_per_year']:>4.0f}  "
              f"{r['win_rate']:>5.0%}  {r['avg_hold_days']:>5.1f}d  "
              f"{r['total_return']:>+8.1%}  {r['annualized_return']:>+8.1%}  "
              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}")
