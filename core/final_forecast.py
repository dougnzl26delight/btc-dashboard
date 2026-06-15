"""Final realistic forecast — what return can you actually expect?

Uses the 50/30/20 allocation backtest + 8M-path Monte Carlo to give the
honest expected distribution across multiple horizons.

Applies a 50% Sharpe haircut (standard for backtest -> live degradation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.three_sleeve_test import combined_equity_path
from core.strategy_bakeoff import fetch_panel, PAIRS_5
from core.monte_carlo_live_sim import simulate_paths


if __name__ == "__main__":
    print("=" * 80)
    print("FINAL REALISTIC FORECAST — 50/30/20 allocation")
    print("=" * 80)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())
    start_date = pd.Timestamp("2020-01-21", tz="UTC")

    print("Building 6.3y daily equity curve of 50/30/20 production allocation...")
    eq = combined_equity_path(
        pair_data, None,
        {"pro_trend": 0.50, "xsmom": 0.30, "bah_btc": 0.20},
        start_date, end_date,
    )
    daily_rets = eq.pct_change().dropna()
    obs_sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365))
    obs_ann = (1 + daily_rets.mean()) ** 365 - 1
    print(f"Observed backtest Sharpe: {obs_sharpe:+.2f}")
    print(f"Observed annualized:      {obs_ann:+.2%}")
    print()

    horizons = [
        ("3 months", 90),
        ("6 months", 180),
        ("1 year", 365),
        ("2 years", 730),
        ("3 years", 1095),
        ("5 years", 1825),
    ]

    for label, haircut in [
        ("optimistic (no haircut, 100% of backtest)", 1.0),
        ("realistic (50% haircut, typical live deg)", 0.5),
        ("pessimistic (30% haircut, severe degradation)", 0.3),
    ]:
        print("=" * 80)
        print(f"SCENARIO: {label}")
        print("=" * 80)
        print(f"{'Horizon':<12s} {'P5':>11s} {'P25':>11s} {'P50':>12s} {'P75':>11s} {'P95':>11s} {'P(profit)':>9s}")
        for h_label, h in horizons:
            r = simulate_paths(daily_rets, n_paths=10000, horizon_days=h,
                                sharpe_haircut=haircut)
            if "error" in r:
                continue
            start = 100_000
            p5 = start * (1 + r["ann_return_p5"])
            p25 = start * (1 + r["ann_return_p25"])
            p50 = start * (1 + r["ann_return_p50"])
            p75 = start * (1 + r["ann_return_p75"])
            p95 = start * (1 + r["ann_return_p95"])
            print(f"{h_label:<12s} ${p5:>10,.0f} ${p25:>10,.0f} ${p50:>11,.0f} "
                  f"${p75:>10,.0f} ${p95:>10,.0f}  {r['prob_profit']:>7.1%}")
        print()

    print("=" * 80)
    print("HEADLINE: median path on $100k under each scenario")
    print("=" * 80)
    print(f"{'Scenario':<22s}", end="")
    for h_label, _ in horizons:
        print(f"  {h_label[:8]:>10s}", end="")
    print()
    print("-" * 22 + ("-" * 12) * len(horizons))

    for label, haircut in [
        ("Optimistic", 1.0),
        ("Realistic (50% haircut)", 0.5),
        ("Pessimistic", 0.3),
    ]:
        print(f"{label:<22s}", end="")
        for h_label, h in horizons:
            r = simulate_paths(daily_rets, n_paths=10000, horizon_days=h,
                                sharpe_haircut=haircut)
            if "error" in r:
                continue
            p50 = 100_000 * (1 + r["ann_return_p50"])
            print(f"  ${p50:>9,.0f}", end="")
        print()
    print()

    print("=" * 80)
    print("RISK CHECK — probability of catastrophic outcomes (50% haircut)")
    print("=" * 80)
    for h_label, h in horizons:
        r = simulate_paths(daily_rets, n_paths=10000, horizon_days=h,
                            sharpe_haircut=0.5)
        if "error" in r:
            continue
        print(f"{h_label:<12s} P(DD>30%) = {r['prob_dd_above_30pct']:>5.1%}   "
              f"P(DD>50%) = {r['prob_dd_above_50pct']:>5.1%}   "
              f"P(loss>20%) = {r['prob_loss_20pct']:>5.1%}")
