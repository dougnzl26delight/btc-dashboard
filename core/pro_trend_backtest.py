"""Pro trend follower — wide stops, pyramiding, BTC focus.

The retail-trader killer: stops too tight, profit targets too narrow,
selling winners early. This system inverts all of that:

  - WIDE stops: 4-5 × ATR (survive normal noise to catch the big move)
  - PYRAMID: add a unit on every +2 ATR continuation (max 4 units)
  - LET WINNERS RUN: no max hold, no profit target, only exit on trail stop or trend break
  - LONG-ONLY on BTC: most liquid + structural bull bias + 4-year halving cycle
  - Asymmetric R/R: target 5R wins, 1R losses, 30%+ win rate is profitable

Math: 30% win × 5R win - 70% loss × 1R loss = +0.8R per trade × 1%/trade = 0.8% × N trades/yr.
At 30 trades/yr that's 24%/yr expected. The trick is whether reality matches.
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


def pro_trend_backtest(
    pair: str = "BTC/USDT",
    days_back: int = 2500,
    starting_equity: float = 100_000.0,
    sma_filter: int = 200,
    donchian_window: int = 20,
    atr_period: int = 14,
    atr_stop_mult: float = 4.0,
    risk_pct_per_unit: float = 0.01,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 4,
    drawdown_kill_pct: float = 0.25,
    round_trip_bps: float = 30.0,
) -> dict:
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty or len(df) < sma_filter * 2:
        return {"error": "insufficient data"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_window).max().shift(1)
    df["sma_filter"] = df["close"].rolling(sma_filter).mean()
    df["atr"] = compute_atr(df, atr_period)
    df = df.dropna()
    n = len(df)

    cash = starting_equity
    units: list[dict] = []
    high_water = 0.0
    trail_stop = 0.0
    peak_equity = starting_equity
    trades: list[dict] = []
    equity_path = []

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])
        sma = float(row["sma_filter"])
        donchian = float(row["donchian_high"])

        # Mark-to-market equity
        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0.0

        # Catastrophic kill switch
        if equity_dd > drawdown_kill_pct and units:
            for u in units:
                exit_p = price
                pnl = u["qty"] * (exit_p - u["entry_price"])
                exit_cost = u["qty"] * exit_p * round_trip_bps / 2 / 10_000.0
                cash += pnl - exit_cost
                trades.append({
                    "entry": u["entry_price"], "exit": exit_p, "qty": u["qty"],
                    "pnl": pnl - exit_cost, "n_days": i - u["entry_idx"],
                    "reason": "dd_kill",
                })
            units = []
            high_water = 0
            trail_stop = 0

        if units:
            # Update trailing stop on new highs (apply to whole position)
            if high > high_water:
                high_water = high
                new_trail = high - atr_stop_mult * atr
                if new_trail > trail_stop:
                    trail_stop = new_trail

            stop_hit = low <= trail_stop
            sma_break = price < sma

            if stop_hit or sma_break:
                exit_p = trail_stop if stop_hit else price
                reason = "trail_stop" if stop_hit else "sma_break"
                for u in units:
                    pnl = u["qty"] * (exit_p - u["entry_price"])
                    exit_cost = u["qty"] * exit_p * round_trip_bps / 2 / 10_000.0
                    cash += pnl - exit_cost
                    trades.append({
                        "entry": u["entry_price"], "exit": exit_p, "qty": u["qty"],
                        "pnl": pnl - exit_cost, "n_days": i - u["entry_idx"],
                        "reason": reason,
                    })
                units = []
                high_water = 0
                trail_stop = 0

            # Pyramid in if winning
            elif len(units) < max_pyramid_units:
                last_entry = units[-1]["entry_price"]
                last_atr = units[-1]["entry_atr"]
                if high >= last_entry + pyramid_atr_step * last_atr:
                    stop_dist = atr_stop_mult * atr
                    if stop_dist > 0:
                        qty = (mtm_eq * risk_pct_per_unit) / stop_dist
                        qty = min(qty, mtm_eq * 0.25 / price)  # 25% notional cap per unit
                        entry_cost = qty * price * round_trip_bps / 2 / 10_000.0
                        cash -= entry_cost
                        units.append({
                            "qty": qty, "entry_price": price, "entry_atr": atr,
                            "entry_idx": i,
                        })
        else:
            # Initial entry
            if price > sma and high >= donchian and atr > 0:
                stop_dist = atr_stop_mult * atr
                qty = (mtm_eq * risk_pct_per_unit) / stop_dist
                qty = min(qty, mtm_eq * 0.25 / price)
                entry_cost = qty * price * round_trip_bps / 2 / 10_000.0
                cash -= entry_cost
                units = [{"qty": qty, "entry_price": price, "entry_atr": atr, "entry_idx": i}]
                high_water = high
                trail_stop = price - stop_dist

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        equity_path.append({"ts": row.name, "equity": cash + unrealized,
                            "n_units": len(units)})

    # Close any open at end
    if units:
        exit_p = float(df["close"].iloc[-1])
        for u in units:
            pnl = u["qty"] * (exit_p - u["entry_price"])
            exit_cost = u["qty"] * exit_p * round_trip_bps / 2 / 10_000.0
            cash += pnl - exit_cost

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"]
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION))
        if daily_rets.std() > 0 else 0.0
    )
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1
    bah_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    n_trades = len(trades)
    if n_trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss else 0
        avg_hold = np.mean([t["n_days"] for t in trades])
    else:
        win_rate = avg_win = avg_loss = rr_ratio = avg_hold = 0

    return {
        "pair": pair, "n_days": n_days, "n_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_win_usdt": float(avg_win), "avg_loss_usdt": float(avg_loss),
        "rr_ratio": float(rr_ratio),
        "avg_hold_days": float(avg_hold),
        "starting_equity": starting_equity,
        "ending_equity": final_eq,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "bah_return": bah_return,
        "alpha_vs_bah": float(total_return) - bah_return,
        "params": {
            "atr_stop_mult": atr_stop_mult,
            "max_pyramid_units": max_pyramid_units,
            "risk_pct_per_unit": risk_pct_per_unit,
        },
    }


def walk_forward_pro_trend(
    pair: str,
    days_back: int = 2500,
    n_folds: int = 5,
    **kwargs,
) -> dict:
    """Walk-forward: split full window into N consecutive folds, run strategy on each.

    Each fold gets enough warmup (200-day SMA needs it) by adding 250 days
    of pre-fold data for indicator initialization.
    """
    df_full = data.ohlcv_extended(pair, days_back=days_back)
    if df_full.empty:
        return {"error": "no data"}
    if len(df_full) < 250 * n_folds + 250:
        return {"error": f"need >= {250 * n_folds + 250} bars"}

    warmup = 250
    fold_len = (len(df_full) - warmup) // n_folds
    fold_stats = []
    for k in range(n_folds):
        start = k * fold_len
        end = start + warmup + fold_len
        end = min(end, len(df_full))
        sub_df = df_full.iloc[start:end].copy()
        sub_days_back = (sub_df.index[-1] - sub_df.index[0]).days
        # Run backtest on this slice (need a way to inject the slice)
        r = _backtest_with_data(sub_df, **kwargs)
        if "error" in r:
            continue
        fold_stats.append({
            "fold": k,
            "start": str(sub_df.index[0].date()),
            "end": str(sub_df.index[-1].date()),
            "n_trades": r["n_trades"],
            "total_return": r["total_return"],
            "annualized": r["annualized_return"],
            "sharpe": r["sharpe"],
            "max_drawdown": r["max_drawdown"],
            "bah_return": r["bah_return"],
        })
    if not fold_stats:
        return {"error": "no fold results"}
    sharpes = np.array([f["sharpe"] for f in fold_stats])
    returns = np.array([f["annualized"] for f in fold_stats])
    return {
        "n_folds": len(fold_stats),
        "mean_sharpe": float(sharpes.mean()),
        "std_sharpe": float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0,
        "min_sharpe": float(sharpes.min()),
        "max_sharpe": float(sharpes.max()),
        "mean_annualized": float(returns.mean()),
        "n_positive_folds": int((sharpes > 0).sum()),
        "fold_stats": fold_stats,
    }


def _backtest_with_data(df, **kwargs) -> dict:
    """Internal: run pro trend logic on a pre-fetched DataFrame.
    Mirrors pro_trend_backtest but skips the data fetch (for walk-forward)."""
    sma_filter = kwargs.get("sma_filter", 200)
    donchian_window = kwargs.get("donchian_window", 20)
    atr_period = kwargs.get("atr_period", 14)
    atr_stop_mult = kwargs.get("atr_stop_mult", 4.0)
    risk_pct_per_unit = kwargs.get("risk_pct_per_unit", 0.01)
    pyramid_atr_step = kwargs.get("pyramid_atr_step", 2.0)
    max_pyramid_units = kwargs.get("max_pyramid_units", 4)
    drawdown_kill_pct = kwargs.get("drawdown_kill_pct", 0.25)
    round_trip_bps = kwargs.get("round_trip_bps", 30.0)
    starting_equity = kwargs.get("starting_equity", 100_000.0)

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_window).max().shift(1)
    df["sma_filter"] = df["close"].rolling(sma_filter).mean()
    df["atr"] = compute_atr(df, atr_period)
    df = df.dropna()
    if len(df) < 30:
        return {"error": "insufficient bars after warmup"}

    cash = starting_equity
    units, trades, equity_path = [], [], []
    high_water = trail_stop = peak_equity = 0
    peak_equity = starting_equity
    n = len(df)

    for i in range(n):
        row = df.iloc[i]
        price, high, low, atr = float(row["close"]), float(row["high"]), float(row["low"]), float(row["atr"])
        sma, donchian = float(row["sma_filter"]), float(row["donchian_high"])

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        if equity_dd > drawdown_kill_pct and units:
            for u in units:
                pnl = u["qty"] * (price - u["entry_price"])
                cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                trades.append({"pnl": pnl, "n_days": i - u["entry_idx"], "reason": "dd_kill"})
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
                    trades.append({"pnl": pnl, "n_days": i - u["entry_idx"], "reason": "trail" if stop_hit else "sma"})
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
    return {
        "n_trades": len(trades), "total_return": total_return,
        "annualized_return": annualized, "sharpe": sharpe,
        "max_drawdown": max_dd, "bah_return": bah_return,
    }


if __name__ == "__main__":
    print("=== Pro trend follower on BTC, 2500 days, parameter sweep ===")
    print(f"{'ATR mult':>8s}  {'Pyramid':>8s}  {'Risk%':>6s}  {'N tr':>5s}  "
          f"{'Win%':>6s}  {'R/R':>5s}  {'Total':>10s}  {'Annlzd':>8s}  "
          f"{'Sharpe':>7s}  {'MaxDD':>8s}  {'vs BAH':>10s}")
    print("-" * 110)

    BTC_BAH = None
    for stop in [3.0, 4.0, 5.0, 6.0]:
        for pyramid in [1, 2, 4]:
            for risk in [0.005, 0.01, 0.02]:
                r = pro_trend_backtest(
                    pair="BTC/USDT", days_back=2500,
                    atr_stop_mult=stop, max_pyramid_units=pyramid,
                    risk_pct_per_unit=risk,
                )
                if "error" in r:
                    continue
                if BTC_BAH is None:
                    BTC_BAH = r["bah_return"]
                print(f"{stop:>6.1f}    {pyramid:>6d}    {risk:>5.1%}   "
                      f"{r['n_trades']:>4d}   {r['win_rate']:>5.0%}   "
                      f"{r['rr_ratio']:>4.1f}   {r['total_return']:>+9.1%}   "
                      f"{r['annualized_return']:>+7.1%}   {r['sharpe']:>+5.2f}    "
                      f"{r['max_drawdown']:>6.1%}   {r['alpha_vs_bah']:>+8.1%}")

    if BTC_BAH is not None:
        print()
        print(f"BTC buy-and-hold over same period: {BTC_BAH:+.1%}")

    print()
    print("=" * 90)
    print("=== MULTI-ASSET TEST: same params (4.0 ATR / 2 pyramid / 2% risk) ===")
    print("=" * 90)
    BEST_KW = dict(atr_stop_mult=4.0, max_pyramid_units=2, risk_pct_per_unit=0.02)
    for asset in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        r = pro_trend_backtest(pair=asset, days_back=2000, **BEST_KW)
        if "error" in r:
            print(f"{asset}: error - {r['error']}")
            continue
        print(f"{asset}: {r['n_trades']} tr, {r['win_rate']:.0%} win, "
              f"R/R {r['rr_ratio']:.1f}, return {r['total_return']:+.1%} "
              f"(ann {r['annualized_return']:+.1%}, sharpe {r['sharpe']:+.2f}, "
              f"DD {r['max_drawdown']:.1%}); BAH {r['bah_return']:+.1%}, "
              f"alpha {r['alpha_vs_bah']:+.1%}")

    print()
    print("=" * 90)
    print("=== WALK-FORWARD on BTC: 5 consecutive non-overlapping folds ===")
    print("=" * 90)
    wf = walk_forward_pro_trend("BTC/USDT", days_back=2500, n_folds=5, **BEST_KW)
    if "error" in wf:
        print(f"error: {wf['error']}")
    else:
        print(f"Mean OOS Sharpe across folds: {wf['mean_sharpe']:+.2f}")
        print(f"Std Sharpe: {wf['std_sharpe']:.2f}")
        print(f"Range: [{wf['min_sharpe']:+.2f}, {wf['max_sharpe']:+.2f}]")
        print(f"Positive folds: {wf['n_positive_folds']}/{wf['n_folds']}")
        print()
        print("Per-fold breakdown:")
        for f in wf["fold_stats"]:
            print(f"  Fold {f['fold']}: {f['start']} -> {f['end']}, "
                  f"{f['n_trades']} tr, ann {f['annualized']:+.1%}, "
                  f"Sharpe {f['sharpe']:+.2f}, DD {f['max_drawdown']:.1%}, "
                  f"BAH {f['bah_return']:+.1%}")
