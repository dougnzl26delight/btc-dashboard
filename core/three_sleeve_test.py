"""Three-sleeve allocation test: pro_trend + XSMOM + BAH BTC.

Hypothesis: adding a BAH BTC sleeve captures chop-regime upside that
pro_trend's SMA200 filter prevents. Trade some XSMOM allocation for it.

Tests multiple allocation splits across 4 regimes + full 6.3y history.

Decision gate:
  - Full-history Sharpe must stay >= 1.40 (current 70/30 production)
  - Recent 18-month return must be materially better than 70/30 (>+5pp)
  - Max DD must not deteriorate beyond 50%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.strategy_bakeoff import (
    fetch_panel, PAIRS_5, PAIRS_XSMOM, perf_stats,
    pro_trend_long_only, bah_single,
)
from core.xsmom_backtest import xsmom_backtest


def combined_equity_path(pair_data, pair_data_xs, allocations, date_start, date_end):
    """Run pro_trend, xsmom, BAH BTC each with their allocation; return total equity.

    Daily rebalance to maintain target weights (simplified — in practice would
    rebalance monthly or quarterly to reduce costs, but for backtest accuracy
    we use daily reweight).
    """
    start_equity = 100_000.0
    pt_alloc = allocations.get("pro_trend", 0)
    xs_alloc = allocations.get("xsmom", 0)
    bah_alloc = allocations.get("bah_btc", 0)

    # Run each sleeve at its own starting capital
    pt_eq = pro_trend_long_only(pair_data, date_start, date_end, use_v5_filter=True) if pt_alloc > 0 else pd.Series(dtype=float)

    # XSMOM equity over the window
    xs_full = xsmom_backtest(
        days_back=2500, momentum_window=14, rebalance_freq=14,
        long_n=2, short_n=2, risk_per_leg=0.20,
    )
    if "error" not in xs_full and xs_alloc > 0:
        xs_eq_full = xs_full["equity_path"]
        xs_eq = xs_eq_full[(xs_eq_full.index >= date_start) & (xs_eq_full.index <= date_end)]
    else:
        xs_eq = pd.Series(dtype=float)

    # BAH BTC
    if bah_alloc > 0:
        bah_eq = bah_single(pair_data, "BTC/USDT", date_start, date_end)
    else:
        bah_eq = pd.Series(dtype=float)

    # Align dates and combine — daily-rebalance to target allocation
    sleeves = []
    if not pt_eq.empty and pt_alloc > 0:
        pt_rets = pt_eq.pct_change().fillna(0)
        sleeves.append(("pro_trend", pt_rets, pt_alloc))
    if not xs_eq.empty and xs_alloc > 0:
        xs_rets = xs_eq.pct_change().fillna(0)
        sleeves.append(("xsmom", xs_rets, xs_alloc))
    if not bah_eq.empty and bah_alloc > 0:
        bah_rets = bah_eq.pct_change().fillna(0)
        sleeves.append(("bah_btc", bah_rets, bah_alloc))

    if not sleeves:
        return pd.Series(dtype=float)

    # Get common index
    indices = [s[1].index for s in sleeves]
    common = indices[0]
    for idx in indices[1:]:
        common = common.intersection(idx)

    if len(common) == 0:
        return pd.Series(dtype=float)

    # Weighted daily returns (constant target allocation = monthly-rebal approximation)
    combined_rets = pd.Series(0.0, index=common)
    for name, rets, alloc in sleeves:
        combined_rets = combined_rets + rets.loc[common] * alloc

    # Build equity curve
    eq = start_equity * (1 + combined_rets).cumprod()
    return eq


if __name__ == "__main__":
    print("=" * 100)
    print("THREE-SLEEVE ALLOCATION TEST — pro_trend + XSMOM + BAH BTC")
    print("=" * 100)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    pair_data_xs = fetch_panel(PAIRS_XSMOM, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())

    windows = [
        ("A: 2020-21 mega-bull",
         pd.Timestamp("2020-01-21", tz="UTC"), pd.Timestamp("2021-04-24", tz="UTC")),
        ("B: 2021-22 top/LUNA",
         pd.Timestamp("2021-04-25", tz="UTC"), pd.Timestamp("2022-07-28", tz="UTC")),
        ("C: 2022-23 bear/recovery",
         pd.Timestamp("2022-07-29", tz="UTC"), pd.Timestamp("2023-10-31", tz="UTC")),
        ("D: 2024-26 recent chop",
         pd.Timestamp("2024-11-01", tz="UTC"), end_date),
        ("ALL: full 6.3y",
         pd.Timestamp("2020-01-21", tz="UTC"), end_date),
    ]

    allocations_to_test = {
        "current 70/30/0 (no BAH)":
            {"pro_trend": 0.70, "xsmom": 0.30, "bah_btc": 0.00},
        "proposed 60/20/20":
            {"pro_trend": 0.60, "xsmom": 0.20, "bah_btc": 0.20},
        "alt 50/20/30":
            {"pro_trend": 0.50, "xsmom": 0.20, "bah_btc": 0.30},
        "alt 70/10/20":
            {"pro_trend": 0.70, "xsmom": 0.10, "bah_btc": 0.20},
        "alt 70/20/10":
            {"pro_trend": 0.70, "xsmom": 0.20, "bah_btc": 0.10},
        "alt 80/0/20 (drop XSMOM)":
            {"pro_trend": 0.80, "xsmom": 0.00, "bah_btc": 0.20},
        "alt 50/30/20":
            {"pro_trend": 0.50, "xsmom": 0.30, "bah_btc": 0.20},
        "alt 40/30/30":
            {"pro_trend": 0.40, "xsmom": 0.30, "bah_btc": 0.30},
    }

    print(f"{'Allocation':<30s}", end="")
    for label, _, _ in windows:
        print(f"  {label[:15]:>17s}", end="")
    print()
    print("-" * 30 + ("-" * 19) * len(windows))

    results = {}
    for alloc_name, allocs in allocations_to_test.items():
        results[alloc_name] = {}
        print(f"{alloc_name:<30s}", end="")
        for label, ds, de in windows:
            try:
                eq = combined_equity_path(pair_data, pair_data_xs, allocs, ds, de)
                stats = perf_stats(eq)
                results[alloc_name][label] = stats
                cell = f"{stats['annualized']:+5.1%}/Sh{stats['sharpe']:+3.1f}"
                print(f"  {cell:>17s}", end="")
            except Exception as e:
                print(f"  {'ERR':>17s}", end="")
                results[alloc_name][label] = {"error": str(e)}
        print()

    print()
    print("=" * 100)
    print("RANKED BY FULL 6.3-YEAR SHARPE")
    print("=" * 100)
    ranked = sorted([(a, r.get("ALL: full 6.3y", {}))
                     for a, r in results.items()],
                    key=lambda x: -x[1].get("sharpe", 0))
    print(f"{'Allocation':<30s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for a, r in ranked:
        if "error" in r:
            continue
        marker = "  <-- CURRENT" if a == "current 70/30/0 (no BAH)" else ""
        print(f"{a:<30s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}{marker}")

    print()
    print("=" * 100)
    print("RANKED BY RECENT 18-MONTH WINDOW (where production is suffering)")
    print("=" * 100)
    ranked_d = sorted([(a, r.get("D: 2024-26 recent chop", {}))
                       for a, r in results.items()],
                      key=lambda x: -x[1].get("sharpe", 0))
    print(f"{'Allocation':<30s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for a, r in ranked_d:
        if "error" in r:
            continue
        marker = "  <-- CURRENT" if a == "current 70/30/0 (no BAH)" else ""
        print(f"{a:<30s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}{marker}")

    print()
    print("=" * 100)
    print("DECISION GATE")
    print("=" * 100)
    current = results.get("current 70/30/0 (no BAH)", {})
    current_full_sharpe = current.get("ALL: full 6.3y", {}).get("sharpe", 0)
    current_recent = current.get("D: 2024-26 recent chop", {}).get("return", 0)

    print(f"Current 70/30/0 baseline:  Full Sharpe {current_full_sharpe:+.2f}, "
          f"Recent return {current_recent:+.2%}")
    print()
    print(f"{'Allocation':<30s} {'FullSh':>7s} {'Vs Cur':>8s} {'Recent':>9s} {'Vs Cur':>8s}  Verdict")
    for alloc_name, r in results.items():
        if alloc_name == "current 70/30/0 (no BAH)":
            continue
        full = r.get("ALL: full 6.3y", {})
        recent = r.get("D: 2024-26 recent chop", {})
        if "error" in full or "error" in recent:
            continue
        sharpe_delta = full["sharpe"] - current_full_sharpe
        recent_delta = recent["return"] - current_recent
        max_dd = full.get("max_dd", 0)

        verdict = []
        if full["sharpe"] >= current_full_sharpe - 0.05:
            verdict.append("Full-Sh OK")
        else:
            verdict.append("Full-Sh DOWN")
        if recent_delta > 0.05:
            verdict.append("Recent BETTER")
        elif recent_delta < -0.02:
            verdict.append("Recent WORSE")
        else:
            verdict.append("Recent flat")
        if max_dd > 0.50:
            verdict.append("DD>50% bad")

        passes = ("Full-Sh OK" in verdict and "Recent BETTER" in verdict
                  and "DD>50% bad" not in verdict)
        verdict_str = " | ".join(verdict)
        marker = "  <-- PASSES" if passes else ""

        print(f"{alloc_name:<30s} {full['sharpe']:>+6.2f}  {sharpe_delta:>+7.2f}  "
              f"{recent['return']:>+8.2%}  {recent_delta:>+7.2%}  {verdict_str}{marker}")
