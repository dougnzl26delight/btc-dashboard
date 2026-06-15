"""Test: does lowering basis arb entry threshold from 1.0 to 0.7 bps/8h help?

Re-runs the realistic basis arb v2 backtest at multiple entry/exit thresholds.
Current production threshold is 1.0 bps/8h (~11% annualized funding).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.basis_arb_backtest_v2 import backtest_basis_arb_realistic


PAIRS = [
    ("BTC/USDT", "BTC/USDT:USDT"),
    ("ETH/USDT", "ETH/USDT:USDT"),
    ("SOL/USDT", "SOL/USDT:USDT"),
]


if __name__ == "__main__":
    print("=" * 80)
    print("BASIS ARB THRESHOLD SWEEP (realistic v2 with basis spread + costs)")
    print("=" * 80)
    print()
    print(f"{'Entry':>6s} {'Exit':>5s}  {'Pair':<10s}  {'Trades':>6s}  "
          f"{'Total':>9s}  {'Annlzd':>9s}  {'Sharpe':>6s}  {'MaxDD':>6s}")
    print("-" * 80)

    sweeps = [
        (1.5, 0.5, "conservative"),
        (1.0, 0.3, "current"),
        (0.7, 0.2, "moderate"),
        (0.5, 0.15, "aggressive"),
        (0.3, 0.1, "very aggressive"),
    ]

    aggregates = {}
    for entry, exit_thr, label in sweeps:
        agg_returns = []
        agg_sharpes = []
        agg_trades = 0
        for spot, perp in PAIRS:
            try:
                # Average over 5 simulations to denoise the basis-spread random walk
                sub_returns = []
                sub_sharpes = []
                sub_trades = []
                sub_dds = []
                for seed in range(5):
                    r = backtest_basis_arb_realistic(
                        spot_pair=spot, perp_pair=perp,
                        days_back=1000,
                        funding_entry_bps_8h=entry,
                        funding_exit_bps_8h=exit_thr,
                        rng_seed=42 + seed,
                    )
                    if "error" in r:
                        continue
                    sub_returns.append(r.get("total_return", 0))
                    sub_sharpes.append(r.get("sharpe", 0))
                    sub_trades.append(r.get("n_trades", 0))
                    sub_dds.append(r.get("max_drawdown", 0))
                if not sub_returns:
                    continue
                avg_ret = np.mean(sub_returns)
                avg_sh = np.mean(sub_sharpes)
                avg_tr = int(np.mean(sub_trades))
                avg_dd = np.mean(sub_dds)
                # Approx annualization (1000 days)
                ann = (1 + avg_ret) ** (365 / 1000) - 1
                marker = "  *" if entry == 1.0 else ""
                print(f"{entry:>5.2f}  {exit_thr:>4.2f}   {spot:<10s}  {avg_tr:>5d}   "
                      f"{avg_ret:>+8.2%}  {ann:>+8.2%}  {avg_sh:>+5.2f}   "
                      f"{avg_dd:>5.1%}{marker}")
                agg_returns.append(avg_ret)
                agg_sharpes.append(avg_sh)
                agg_trades += avg_tr
            except Exception as e:
                print(f"  {spot}: {type(e).__name__}: {e}")
                continue
        if agg_returns:
            mean_ret = np.mean(agg_returns)
            mean_sh = np.mean(agg_sharpes)
            ann = (1 + mean_ret) ** (365 / 1000) - 1
            print(f"  {label:<26s}  {agg_trades:>5d}   "
                  f"{mean_ret:>+8.2%}  {ann:>+8.2%}  {mean_sh:>+5.2f}")
            aggregates[label] = {
                "entry": entry, "exit": exit_thr,
                "mean_return": mean_ret, "annualized": ann, "mean_sharpe": mean_sh,
                "total_trades": agg_trades,
            }
        print()

    print("=" * 80)
    print("RANKED BY MEAN PORTFOLIO SHARPE:")
    print("=" * 80)
    sorted_aggs = sorted(aggregates.items(), key=lambda x: -x[1]["mean_sharpe"])
    for label, a in sorted_aggs:
        print(f"  {label:<20s}  entry={a['entry']:.2f} exit={a['exit']:.2f}  "
              f"Sharpe {a['mean_sharpe']:+.2f}  ann {a['annualized']:+.2%}  "
              f"trades {a['total_trades']}")

    print()
    print("(* = current production threshold)")
