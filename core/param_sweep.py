"""Combined parameter sweep: DD kill, ATR=4.5, top-K on full history.

Three independent investigations rolled into one run for efficiency:
  A. DD kill threshold (0.30/0.35/0.40/0.45/0.50)
  B. ATR_STOP_MULT fine-grained (3.5/4.0/4.5/5.0)
  C. Top-K universe selection on FULL 2500-day history
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run
from core.pro_trend_backtest import pro_trend_backtest


BASE_KW = dict(
    starting_equity=100_000.0, base_risk=0.04,
    portfolio_risk_cap=0.15, atr_stop_mult=4.0, drawdown_kill_pct=0.35,
)


def run_dd_sweep(pair_data):
    print("=" * 78)
    print("[A] DD KILL THRESHOLD SWEEP")
    print("=" * 78)
    print(f"{'DD kill':>8s}  {'Annlzd':>8s}  {'Sharpe':>6s}  {'MaxDD':>6s}  "
          f"{'DDkills':>7s}  {'Trades':>6s}")
    rows = []
    for dd in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        kw = {**BASE_KW, "drawdown_kill_pct": dd}
        r = portfolio_run(pair_data=pair_data, **kw)
        marker = "  *" if dd == 0.35 else ""
        print(f"{dd:>7.2f}   {r['annualized_return']:>+7.2%}  "
              f"{r['sharpe']:>+5.2f}   {r['max_drawdown']:>5.1%}   "
              f"{r['n_dd_kills']:>5d}    {r['n_trades']:>4d}{marker}")
        rows.append({"dd_kill": dd, **r})
    print("(* = current production)")
    print()
    rows.sort(key=lambda x: -x["sharpe"])
    print("Top by Sharpe:")
    for r in rows[:3]:
        print(f"  dd={r['dd_kill']:.2f}: Sharpe {r['sharpe']:.2f}, "
              f"ann {r['annualized_return']:+.1%}, DD {r['max_drawdown']:.1%}")
    print()
    return rows


def run_atr_sweep(pair_data):
    print("=" * 78)
    print("[B] ATR_STOP_MULT FINE-GRAINED")
    print("=" * 78)
    print(f"{'ATR':>4s}  {'Annlzd':>8s}  {'Sharpe':>6s}  {'Sortino':>7s}  "
          f"{'MaxDD':>6s}  {'Trades':>6s}")
    rows = []
    for atr in [3.5, 4.0, 4.5, 5.0, 5.5]:
        kw = {**BASE_KW, "atr_stop_mult": atr}
        r = portfolio_run(pair_data=pair_data, **kw)
        marker = "  *" if atr == 4.0 else ""
        print(f"{atr:>4.1f}   {r['annualized_return']:>+7.2%}  "
              f"{r['sharpe']:>+5.2f}   {r['sortino']:>+6.2f}    "
              f"{r['max_drawdown']:>5.1%}   {r['n_trades']:>4d}{marker}")
        rows.append({"atr": atr, **r})
    print("(* = current production)")
    print()
    return rows


def run_topk_full_history(pair_data):
    print("=" * 78)
    print("[C] TOP-K UNIVERSE — FULL 2500-DAY HISTORY")
    print("=" * 78)

    # Per-pair backtest on full history
    all_pairs = list(pair_data.keys())
    print("Per-pair full-history backtest:")
    pair_perf = {}
    for p in all_pairs:
        r = pro_trend_backtest(
            pair=p, days_back=2500,
            atr_stop_mult=4.0, max_pyramid_units=2,
            risk_pct_per_unit=0.04, drawdown_kill_pct=0.35,
        )
        if "error" not in r:
            pair_perf[p] = r
            print(f"  {p:<12s} ann {r['annualized_return']:>+7.2%} "
                  f"Sharpe {r['sharpe']:>+5.2f} DD {r['max_drawdown']:>5.1%} "
                  f"trades {r['n_trades']}")
    print()

    ranked = sorted(pair_perf.items(), key=lambda x: -x[1]["sharpe"])
    print(f"Ranked by full-history Sharpe: {[p for p,_ in ranked]}")
    print()

    print("Portfolio backtest with top-K subsets:")
    print(f"{'K':>3s}  {'Universe':<60s}  {'Annlzd':>8s}  {'Sharpe':>6s}  {'MaxDD':>6s}")
    for k in [1, 2, 3, 4, 5]:
        if k > len(ranked):
            continue
        subset = [p for p, _ in ranked[:k]]
        sub_data = {p: pair_data[p] for p in subset if p in pair_data}
        r = portfolio_run(pair_data=sub_data, **BASE_KW)
        print(f"{k:>3d}  {str(subset)[:58]:<60s}  "
              f"{r['annualized_return']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']:>5.1%}")
    print()


if __name__ == "__main__":
    print("Fetching max-history data...")
    pair_data = fetch_all(days_back=2500)
    print(f"Got {len(pair_data)} pairs over up to {max(len(df) for df in pair_data.values())} bars")
    print()

    dd_rows = run_dd_sweep(pair_data)
    atr_rows = run_atr_sweep(pair_data)
    run_topk_full_history(pair_data)
