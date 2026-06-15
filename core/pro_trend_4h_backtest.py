"""Pro trend follower on 4h bars — higher frequency variant.

Same mechanic as daily pro trend but parameters scaled for 4h:
  SMA(1200) = 200 days
  Donchian(120) = 20 days
  ATR(84) = 14 days × 6 bars/day

Tests whether 3-5x more trades/year keeps the edge or whether costs eat it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr


ANNUALIZATION = 365


def backtest_4h(
    pair: str = "BTC/USDT",
    days_back: int = 1500,
    starting_equity: float = 100_000.0,
    sma_filter_bars: int = 1200,        # ~200 days
    donchian_window_bars: int = 120,    # ~20 days
    atr_period_bars: int = 84,          # ~14 days
    atr_stop_mult: float = 4.0,
    risk_pct_per_unit: float = 0.02,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.25,
    round_trip_bps: float = 30.0,
) -> dict:
    df = data.ohlcv_extended(pair, days_back=days_back, timeframe="4h")
    if df.empty or len(df) < sma_filter_bars * 2:
        return {"error": f"insufficient bars: {len(df)} (need >= {sma_filter_bars * 2})"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_window_bars).max().shift(1)
    df["sma_filter"] = df["close"].rolling(sma_filter_bars).mean()
    df["atr"] = compute_atr(df, atr_period_bars)
    df = df.dropna()
    n = len(df)

    cash = starting_equity
    units, trades, equity_path = [], [], []
    high_water = trail_stop = 0
    peak_equity = starting_equity

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])
        sma = float(row["sma_filter"])
        donchian = float(row["donchian_high"])

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        if equity_dd > drawdown_kill_pct and units:
            for u in units:
                pnl = u["qty"] * (price - u["entry_price"])
                cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                trades.append({"pnl": pnl, "n_bars": i - u["entry_idx"], "reason": "dd_kill"})
            units, high_water, trail_stop = [], 0, 0

        if units:
            if high > high_water:
                high_water = high
                new_trail = high - atr_stop_mult * atr
                if new_trail > trail_stop:
                    trail_stop = new_trail
            stop_hit = low <= trail_stop
            sma_break = price < sma
            if stop_hit or sma_break:
                exit_p = trail_stop if stop_hit else price
                for u in units:
                    pnl = u["qty"] * (exit_p - u["entry_price"])
                    cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                    trades.append({
                        "pnl": pnl, "n_bars": i - u["entry_idx"],
                        "reason": "trail" if stop_hit else "sma",
                    })
                units, high_water, trail_stop = [], 0, 0
            elif len(units) < max_pyramid_units:
                last = units[-1]
                if high >= last["entry_price"] + pyramid_atr_step * last["entry_atr"]:
                    stop_dist = atr_stop_mult * atr
                    if stop_dist > 0:
                        qty = min((mtm_eq * risk_pct_per_unit) / stop_dist, mtm_eq * 0.25 / price)
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        units.append({"qty": qty, "entry_price": price, "entry_atr": atr, "entry_idx": i})
        else:
            if price > sma and high >= donchian and atr > 0:
                stop_dist = atr_stop_mult * atr
                qty = min((mtm_eq * risk_pct_per_unit) / stop_dist, mtm_eq * 0.25 / price)
                cash -= qty * price * round_trip_bps / 2 / 10_000
                units = [{"qty": qty, "entry_price": price, "entry_atr": atr, "entry_idx": i}]
                high_water = high
                trail_stop = price - stop_dist

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        equity_path.append({"ts": row.name, "equity": cash + unrealized})

    if units:
        exit_p = float(df["close"].iloc[-1])
        for u in units:
            pnl = u["qty"] * (exit_p - u["entry_price"])
            cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"].resample("1D").last().ffill()
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
    avg_hold_bars = np.mean([t["n_bars"] for t in trades]) if trades else 0
    trades_per_year = n_trades / max(n_days / 365, 1)

    return {
        "pair": pair, "n_days": n_days, "n_bars": n,
        "n_trades": n_trades, "trades_per_year": float(trades_per_year),
        "win_rate": float(win_rate),
        "avg_hold_bars": float(avg_hold_bars),
        "avg_hold_days": float(avg_hold_bars / 6),
        "starting_equity": starting_equity, "ending_equity": final_eq,
        "total_return": float(total_return), "annualized_return": float(annualized),
        "sharpe": float(sharpe), "max_drawdown": float(max_dd),
        "bah_return": bah_return, "alpha_vs_bah": float(total_return) - bah_return,
    }


if __name__ == "__main__":
    print("=== 4h pro trend backtest, 1500 days ===")
    print(f"{'Pair':<12s} {'N tr':>6s} {'TPY':>6s} {'Win%':>6s} {'Hold':>7s} "
          f"{'Total':>9s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'vs BAH':>9s}")
    print("-" * 90)

    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        r = backtest_4h(pair=pair, days_back=1500)
        if "error" in r:
            print(f"{pair}  ERROR: {r['error']}")
            continue
        print(f"{pair:<12s} {r['n_trades']:>5d}  {r['trades_per_year']:>4.0f}  "
              f"{r['win_rate']:>5.0%}  {r['avg_hold_days']:>5.1f}d  "
              f"{r['total_return']:>+8.1%}  {r['annualized_return']:>+8.1%}  "
              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}   "
              f"{r['alpha_vs_bah']:>+8.1%}")

    print()
    print("=== Compare to daily pro trend on same window ===")
    from core.pro_trend_backtest import pro_trend_backtest
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        r = pro_trend_backtest(pair=pair, days_back=1500,
                                atr_stop_mult=4.0, max_pyramid_units=2,
                                risk_pct_per_unit=0.02)
        if "error" in r:
            continue
        tpy = r["n_trades"] / max(r["n_days"] / 365, 1)
        print(f"{pair:<12s} {r['n_trades']:>5d}  {tpy:>4.0f}  "
              f"{r['win_rate']:>5.0%}  {r['avg_hold_days']:>5.1f}d  "
              f"{r['total_return']:>+8.1%}  {r['annualized_return']:>+8.1%}  "
              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}   "
              f"{r['alpha_vs_bah']:>+8.1%}")
