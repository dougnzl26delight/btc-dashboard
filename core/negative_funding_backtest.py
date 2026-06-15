"""Negative-funding short-basis backtest.

Symmetric to existing positive-funding basis arb. When perp funding is very
NEGATIVE (shorts pay longs), the inverse trade collects funding:

  - SHORT spot (need to borrow/margin spot — operational complication)
  - LONG perp (collect funding from shorts on the perp side)

Delta-neutral by construction. The directional exposure cancels; what
remains is funding income minus basis spread risk minus costs.

Backtest mirrors basis_arb_backtest_v2.py but with inverse entry:
  - Enter when funding_bps_8h < -1.0 (~-11% ann)
  - Exit when funding > -0.3 bps (or any positive)
  - Same cost assumptions (spot+perp round-trip + slippage + spread)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


ANNUALIZATION = 365


def neg_funding_basis_backtest(
    perp_pair: str = "BTC/USDT:USDT",
    spot_pair: str = "BTC/USDT",
    days_back: int = 1000,
    starting_equity: float = 100_000.0,
    funding_entry_bps_8h: float = -1.0,
    funding_exit_bps_8h: float = -0.3,
    max_allocation: float = 0.30,
    spot_round_trip_bps: float = 20.0,
    perp_round_trip_bps: float = 10.0,
    slippage_bps_per_side: float = 5.0,
    basis_daily_vol_bps: float = 10.0,
    basis_blowout_exit_bps: float = 100.0,
    rng_seed: int = 42,
) -> dict:
    funding_df = data.funding_history_extended(perp_pair, days_back=days_back)
    spot_df = data.ohlcv_extended(spot_pair, days_back=days_back)
    if funding_df.empty or spot_df.empty:
        return {"error": "no data"}

    funding_df = funding_df.sort_index()
    spot_df = spot_df.sort_index()
    funding_df["spot_price"] = spot_df["close"].reindex(funding_df.index, method="ffill")
    funding_df = funding_df.dropna()
    funding_df["funding_bps"] = funding_df["funding_rate"] * 10_000.0

    rng = np.random.default_rng(rng_seed)
    rt_cost_bps = (spot_round_trip_bps + perp_round_trip_bps
                    + 4 * slippage_bps_per_side)

    n_events = len(funding_df)
    equity = starting_equity
    in_trade = False
    notional = 0.0
    cycle_basis_bps = 0.0
    cycle_funding = 0.0
    trade_open_idx = -1
    cycle_start = None

    trades = []
    equity_path = []
    blowout_closes = 0

    for i in range(n_events):
        row = funding_df.iloc[i]
        f_bps = float(row["funding_bps"])

        if in_trade:
            # Inverse trade: long perp earns funding when funding < 0
            # Funding payment on long perp = -funding_rate * notional
            #   (if funding < 0, shorts pay longs, we get +)
            funding_payment = -notional * float(row["funding_rate"])
            equity += funding_payment
            cycle_funding += funding_payment

            basis_step_bps = rng.normal(0, basis_daily_vol_bps / np.sqrt(3))
            cycle_basis_bps += basis_step_bps

            # Basis blowout exit (inverse: spread MOVES AGAINST short-spot long-perp)
            if abs(cycle_basis_bps) > basis_blowout_exit_bps:
                # Realize the basis P&L; for inverse trade, gain when basis falls
                basis_pnl = -notional * cycle_basis_bps / 10_000.0
                exit_cost = notional * rt_cost_bps / 10_000.0 / 2
                equity += basis_pnl - exit_cost
                trades.append({
                    "entry_idx": trade_open_idx, "exit_idx": i,
                    "cycle_funding": cycle_funding,
                    "basis_pnl": basis_pnl, "blowout": True,
                    "n_events": i - trade_open_idx,
                })
                blowout_closes += 1
                in_trade = False
                notional = cycle_basis_bps = cycle_funding = 0
                trade_open_idx = -1

            # Normal exit on funding reversion
            elif f_bps > funding_exit_bps_8h:
                basis_pnl = -notional * cycle_basis_bps / 10_000.0
                exit_cost = notional * rt_cost_bps / 10_000.0 / 2
                equity += basis_pnl - exit_cost
                trades.append({
                    "entry_idx": trade_open_idx, "exit_idx": i,
                    "cycle_funding": cycle_funding,
                    "basis_pnl": basis_pnl, "blowout": False,
                    "n_events": i - trade_open_idx,
                })
                in_trade = False
                notional = cycle_basis_bps = cycle_funding = 0
                trade_open_idx = -1
        else:
            # Entry condition: funding very negative
            if f_bps < funding_entry_bps_8h:
                notional = equity * max_allocation
                entry_cost = notional * rt_cost_bps / 10_000.0 / 2
                equity -= entry_cost
                in_trade = True
                cycle_basis_bps = 0.0
                cycle_funding = 0.0
                trade_open_idx = i
                cycle_start = row.name

        equity_path.append({"ts": row.name, "equity": equity})

    # Close any open at end
    if in_trade:
        last = funding_df.iloc[-1]
        basis_pnl = -notional * cycle_basis_bps / 10_000.0
        exit_cost = notional * rt_cost_bps / 10_000.0 / 2
        equity += basis_pnl - exit_cost
        trades.append({"cycle_funding": cycle_funding, "basis_pnl": basis_pnl,
                       "blowout": False, "n_events": n_events - trade_open_idx})

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    if eq_df.empty or len(eq_df) < 2:
        return {"error": "no equity path"}
    # Funding events are every 8h = 3/day; resample to daily for Sharpe
    daily_eq = eq_df["equity"].resample("D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION)) if daily_rets.std() > 0 else 0
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (daily_eq.index[-1] - daily_eq.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1

    return {
        "perp_pair": perp_pair,
        "n_trades": len(trades),
        "n_blowouts": blowout_closes,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "final_equity": final_eq,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("NEGATIVE-FUNDING SHORT-BASIS BACKTEST")
    print("=" * 72)
    print()
    print(f"{'Pair':<12s}  {'Trades':>6s}  {'Total':>9s}  {'Annlzd':>9s}  "
          f"{'Sharpe':>7s}  {'MaxDD':>6s}  {'Blowouts':>9s}")

    pairs = [
        ("BTC/USDT", "BTC/USDT:USDT"),
        ("ETH/USDT", "ETH/USDT:USDT"),
        ("SOL/USDT", "SOL/USDT:USDT"),
        ("ATOM/USDT", "ATOM/USDT:USDT"),
        ("DOGE/USDT", "DOGE/USDT:USDT"),
        ("SUI/USDT", "SUI/USDT:USDT"),
    ]

    # Threshold sweep — see if tighter entry helps
    print("Threshold sweep — sweep entry/exit to find sweet spot:")
    print(f"{'Entry':>5s}  {'Exit':>5s}  {'Pair':<10s}  {'Trades':>6s}  "
          f"{'Annlzd':>9s}  {'Sharpe':>7s}  {'MaxDD':>6s}")
    for entry, exit_thr in [(-1.0, -0.3), (-1.5, -0.5), (-2.0, -0.7), (-3.0, -1.0)]:
        for spot, perp in pairs[:3]:  # Just top 3 for sweep
            sub_returns, sub_sharpes, sub_trades, sub_dds = [], [], [], []
            for seed in range(3):
                r = neg_funding_basis_backtest(
                    spot_pair=spot, perp_pair=perp,
                    days_back=1000, rng_seed=42 + seed,
                    funding_entry_bps_8h=entry,
                    funding_exit_bps_8h=exit_thr,
                )
                if "error" in r:
                    continue
                sub_returns.append(r["total_return"])
                sub_sharpes.append(r["sharpe"])
                sub_trades.append(r["n_trades"])
                sub_dds.append(r["max_drawdown"])
            if not sub_returns:
                continue
            avg_ret = np.mean(sub_returns)
            avg_sh = np.mean(sub_sharpes)
            avg_tr = int(np.mean(sub_trades))
            avg_dd = np.mean(sub_dds)
            ann = (1 + avg_ret) ** (365 / 1000) - 1
            print(f"{entry:>5.2f}  {exit_thr:>5.2f}  {spot:<10s}  {avg_tr:>5d}   "
                  f"{ann:>+8.2%}  {avg_sh:>+6.2f}   {avg_dd:>5.1%}")
        print()

    print()
    print("Standard -1.0/-0.3 (baseline):")
    all_results = []
    for spot, perp in pairs:
        # Average over 5 seeds to denoise basis spread walk
        sub_returns, sub_sharpes, sub_trades, sub_dds = [], [], [], []
        for seed in range(5):
            r = neg_funding_basis_backtest(
                spot_pair=spot, perp_pair=perp,
                days_back=1000, rng_seed=42 + seed,
            )
            if "error" in r:
                continue
            sub_returns.append(r["total_return"])
            sub_sharpes.append(r["sharpe"])
            sub_trades.append(r["n_trades"])
            sub_dds.append(r["max_drawdown"])
        if not sub_returns:
            print(f"{spot:<12s}  no data")
            continue
        avg_ret = np.mean(sub_returns)
        avg_sh = np.mean(sub_sharpes)
        avg_tr = int(np.mean(sub_trades))
        avg_dd = np.mean(sub_dds)
        ann = (1 + avg_ret) ** (365 / 1000) - 1
        avg_blow = np.mean([1 if t > 0 else 0 for t in sub_trades])  # crude
        print(f"{spot:<12s}  {avg_tr:>5d}   "
              f"{avg_ret:>+8.2%}  {ann:>+8.2%}  {avg_sh:>+6.2f}   "
              f"{avg_dd:>5.1%}")
        all_results.append({"pair": spot, "ann": ann, "sharpe": avg_sh,
                             "dd": avg_dd, "trades": avg_tr})

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if all_results:
        mean_sharpe = np.mean([r["sharpe"] for r in all_results])
        mean_ann = np.mean([r["ann"] for r in all_results])
        print(f"Mean Sharpe across pairs:  {mean_sharpe:+.2f}")
        print(f"Mean annualized:           {mean_ann:+.2%}")
        if mean_sharpe > 0.5:
            print("PASSES — wire negative-funding leg into basis_run.py")
        elif mean_sharpe > 0:
            print("MARGINAL — positive but weak edge; consider as opportunistic only")
        else:
            print("FAILS — negative funding doesn't profitably reverse in these pairs")
