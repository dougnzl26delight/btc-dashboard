"""Basis arbitrage backtest v2 — REALISTIC frictions.

Adds the things v1 was missing:
  - Basis spread random walk (perp diverges from spot daily)
  - Realistic costs: 30bps round trip (10bps spot + 5bps perp, both sides)
  - Entry/exit slippage on top of fees
  - Funding-based EXIT also when basis blows out

Run side-by-side with v1 to see how much of the Sharpe was idealized.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


ANNUALIZATION = 365


def backtest_basis_arb_realistic(
    perp_pair: str = "BTC/USDT:USDT",
    spot_pair: str = "BTC/USDT",
    days_back: int = 1000,
    starting_equity: float = 100_000.0,
    funding_entry_bps_8h: float = 1.0,
    funding_exit_bps_8h: float = 0.3,
    max_allocation: float = 0.30,
    spot_round_trip_bps: float = 20.0,    # 10 bps × 2 sides
    perp_round_trip_bps: float = 10.0,    # 5 bps × 2 sides
    slippage_bps_per_side: float = 5.0,   # per side
    basis_daily_vol_bps: float = 10.0,    # bp std dev of daily basis change
    basis_blowout_exit_bps: float = 100.0, # close if basis moves > 100bps adverse
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

    # Total round-trip cost (entry + exit, spot + perp + slippage)
    rt_cost_bps = (
        spot_round_trip_bps + perp_round_trip_bps + 4 * slippage_bps_per_side
    )

    n_events = len(funding_df)
    equity = starting_equity
    in_trade = False
    notional = 0.0
    cycle_basis_bps = 0.0          # cumulative basis movement since entry
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
            # Funding income for short perp side (positive funding → income)
            funding_payment = notional * float(row["funding_rate"])
            equity += funding_payment
            cycle_funding += funding_payment

            # Basis random walk (each 8h period has σ basis_daily_vol_bps/sqrt(3))
            basis_step_bps = rng.normal(0, basis_daily_vol_bps / np.sqrt(3))
            cycle_basis_bps += basis_step_bps

            # Mark-to-market basis P&L (delta-neutral, so only basis movement matters)
            # Negative basis movement (perp drops vs spot) is BAD for short perp + long spot
            basis_pnl = -notional * basis_step_bps / 10_000.0
            equity += basis_pnl

            # Force close on basis blowout
            forced = abs(cycle_basis_bps) > basis_blowout_exit_bps
            normal_exit = f_bps < funding_exit_bps_8h

            if forced or normal_exit:
                close_cost = notional * (rt_cost_bps / 2) / 10_000.0
                equity -= close_cost
                if forced:
                    blowout_closes += 1
                trades.append({
                    "open_ts": cycle_start,
                    "close_ts": row.name,
                    "n_periods": i - trade_open_idx,
                    "notional": notional,
                    "funding_collected": cycle_funding,
                    "cumulative_basis_bps": cycle_basis_bps,
                    "close_cost": close_cost,
                    "net_pnl": cycle_funding - close_cost - notional * cycle_basis_bps / 10_000.0,
                    "forced_close": forced,
                })
                in_trade = False
                notional = 0.0
                cycle_funding = 0.0
                cycle_basis_bps = 0.0
        else:
            if f_bps > funding_entry_bps_8h:
                notional = equity * max_allocation
                open_cost = notional * (rt_cost_bps / 2) / 10_000.0
                equity -= open_cost
                cycle_funding = -open_cost
                cycle_basis_bps = 0.0
                in_trade = True
                cycle_start = row.name
                trade_open_idx = i

        equity_path.append({"ts": row.name, "equity": equity, "in_trade": in_trade})

    eq_df = pd.DataFrame(equity_path).set_index("ts")
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
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1
    win_rate = sum(1 for t in trades if t["net_pnl"] > 0) / max(len(trades), 1)

    return {
        "n_funding_events": n_events,
        "n_days": n_days,
        "n_trades": len(trades),
        "n_blowout_closes": blowout_closes,
        "win_rate": float(win_rate),
        "starting_equity": starting_equity,
        "ending_equity": float(equity),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "params": {
            "rt_cost_bps": rt_cost_bps,
            "basis_daily_vol_bps": basis_daily_vol_bps,
            "basis_blowout_exit_bps": basis_blowout_exit_bps,
        },
    }


def monte_carlo_audit(
    perp_pair: str = "BTC/USDT:USDT",
    n_simulations: int = 100,
    days_back: int = 1000,
    **kwargs,
) -> dict:
    """Run N simulations with different random basis paths to get distribution."""
    results = []
    for seed in range(n_simulations):
        r = backtest_basis_arb_realistic(
            perp_pair=perp_pair, days_back=days_back, rng_seed=seed, **kwargs,
        )
        if "error" not in r:
            results.append(r)
    if not results:
        return {"error": "all sims failed"}

    sharpes = np.array([r["sharpe"] for r in results])
    returns = np.array([r["total_return"] for r in results])
    dds = np.array([r["max_drawdown"] for r in results])
    n_blowouts = np.array([r["n_blowout_closes"] for r in results])

    return {
        "n_simulations": len(results),
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_std": float(sharpes.std()),
        "sharpe_p5": float(np.percentile(sharpes, 5)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "return_mean": float(returns.mean()),
        "return_p5": float(np.percentile(returns, 5)),
        "return_p95": float(np.percentile(returns, 95)),
        "maxdd_mean": float(dds.mean()),
        "maxdd_p95": float(np.percentile(dds, 95)),
        "avg_blowout_closes": float(n_blowouts.mean()),
        "prob_negative_return": float((returns < 0).mean()),
    }


if __name__ == "__main__":
    print("=== v1 (no basis risk modeled) — for comparison ===")
    from core import basis_arb_backtest as v1
    r1 = v1.backtest_basis_arb(days_back=1000)
    print(f"  Sharpe: {r1['sharpe']:+.2f}, Total: {r1['total_return']:+.2%}, "
          f"MaxDD: {r1['max_drawdown']:.2%}, N trades: {r1['n_trades']}")

    print()
    print("=== v2 (REALISTIC frictions, single seed) ===")
    r2 = backtest_basis_arb_realistic(days_back=1000)
    print(f"  Sharpe: {r2['sharpe']:+.2f}, Total: {r2['total_return']:+.2%}, "
          f"MaxDD: {r2['max_drawdown']:.2%}, N trades: {r2['n_trades']}, "
          f"Blowouts: {r2['n_blowout_closes']}")
    print(f"  RT cost: {r2['params']['rt_cost_bps']} bps, basis vol: "
          f"{r2['params']['basis_daily_vol_bps']} bps/day")

    print()
    print("=== v2 Monte Carlo audit (100 simulations with random basis paths) ===")
    print("This is the HONEST distribution of outcomes:")
    mc = monte_carlo_audit(n_simulations=100, days_back=1000)
    for k, v in mc.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:+.4f}")
        else:
            print(f"  {k:30s} {v}")

    print()
    print("=== Sensitivity: basis vol assumption ===")
    print(f"{'basis vol bps/day':>20s}  {'mean Sharpe':>12s}  {'mean return':>12s}  "
          f"{'P(loss)':>10s}  {'avg blowouts':>14s}")
    print("-" * 75)
    for bvol in [5.0, 10.0, 20.0, 50.0]:
        mc = monte_carlo_audit(n_simulations=50, days_back=1000, basis_daily_vol_bps=bvol)
        if "error" not in mc:
            print(f"{bvol:>18.1f}     {mc['sharpe_mean']:>+10.2f}    "
                  f"{mc['return_mean']:>+10.2%}    {mc['prob_negative_return']:>8.0%}    "
                  f"{mc['avg_blowout_closes']:>12.1f}")
