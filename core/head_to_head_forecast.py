"""Head-to-head 5-year forecast: 50/30/20 system vs BAH BTC vs DCA BTC.

Apples-to-apples Monte Carlo on each strategy's 6.3y historic daily returns.
Same 50% Sharpe haircut applied to all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.three_sleeve_test import combined_equity_path
from core.strategy_bakeoff import (
    fetch_panel, PAIRS_5, bah_single, dca_btc,
)
from core.monte_carlo_live_sim import simulate_paths


def perf_summary(daily_rets, label):
    if len(daily_rets) < 30:
        return None
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 0 else 0
    eq = (1 + daily_rets).cumprod()
    total = float(eq.iloc[-1] - 1)
    n_days = (daily_rets.index[-1] - daily_rets.index[0]).days
    ann = (1 + total) ** (365 / max(n_days, 1)) - 1
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    return {"label": label, "sharpe": sharpe, "ann": ann,
            "max_dd": max_dd, "total": total, "n_days": n_days,
            "rets": daily_rets}


if __name__ == "__main__":
    print("=" * 90)
    print("HEAD-TO-HEAD 5-YEAR FORECAST — apples-to-apples Monte Carlo")
    print("=" * 90)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())
    start_date = pd.Timestamp("2020-01-21", tz="UTC")

    print("Building 6.3y daily return series for each strategy...")
    # 50/30/20 system
    sys_eq = combined_equity_path(
        pair_data, None,
        {"pro_trend": 0.50, "xsmom": 0.30, "bah_btc": 0.20},
        start_date, end_date,
    )
    sys_rets = sys_eq.pct_change().dropna()

    # BAH BTC
    bah_eq = bah_single(pair_data, "BTC/USDT", start_date, end_date)
    bah_rets = bah_eq.pct_change().dropna()

    # DCA BTC
    dca_eq = dca_btc(pair_data, start_date, end_date)
    dca_rets = dca_eq.pct_change().dropna()

    strategies = [
        perf_summary(sys_rets, "50/30/20 system (production)"),
        perf_summary(bah_rets, "BAH BTC only"),
        perf_summary(dca_rets, "DCA $1k/wk BTC"),
    ]

    print()
    print("HISTORIC BACKTEST (6.3 years)")
    print("-" * 90)
    print(f"{'Strategy':<32s} {'Sharpe':>7s} {'Annlzd':>9s} {'Total':>11s} {'MaxDD':>7s} {'$100k->':>12s}")
    for s in strategies:
        end_val = 100_000 * (1 + s["total"])
        print(f"{s['label']:<32s} {s['sharpe']:>+6.2f}  {s['ann']:>+8.1%}  "
              f"{s['total']:>+10.1%}  {s['max_dd']:>5.1%}   ${end_val:>9,.0f}")
    print()

    print("=" * 90)
    print("MONTE CARLO 5-YEAR FORECAST (50% Sharpe haircut, 10000 paths)")
    print("=" * 90)
    print(f"{'Strategy':<32s} {'P5':>11s} {'P25':>11s} {'P50':>12s} {'P75':>11s} {'P95':>11s} {'P(profit)':>9s}")
    forecasts = {}
    for s in strategies:
        r = simulate_paths(s["rets"], n_paths=10000, horizon_days=1825,
                            sharpe_haircut=0.5)
        forecasts[s["label"]] = r
        p5 = 100_000 * (1 + r["ann_return_p5"])
        p25 = 100_000 * (1 + r["ann_return_p25"])
        p50 = 100_000 * (1 + r["ann_return_p50"])
        p75 = 100_000 * (1 + r["ann_return_p75"])
        p95 = 100_000 * (1 + r["ann_return_p95"])
        print(f"{s['label']:<32s} ${p5:>10,.0f} ${p25:>10,.0f} ${p50:>11,.0f} "
              f"${p75:>10,.0f} ${p95:>10,.0f}  {r['prob_profit']:>7.1%}")
    print()

    print("=" * 90)
    print("DRAWDOWN RISK (5-year horizon, 50% haircut)")
    print("=" * 90)
    print(f"{'Strategy':<32s} {'P(DD>30%)':>10s} {'P(DD>50%)':>10s} {'P(loss>20%)':>12s} {'P(loss>50%)':>12s}")
    for s in strategies:
        r = forecasts[s["label"]]
        loss_50 = float(((r.get("ann_return_p5", 0) < -0.50)))  # crude proxy
        # Build a better one: recompute
        sub = simulate_paths(s["rets"], n_paths=10000, horizon_days=1825,
                              sharpe_haircut=0.5)
        print(f"{s['label']:<32s} {sub['prob_dd_above_30pct']:>9.1%}  "
              f"{sub['prob_dd_above_50pct']:>9.1%}  "
              f"{sub['prob_loss_20pct']:>11.1%}  "
              f"{'<5%' if sub['prob_loss_20pct'] < 0.05 else '~':>11s}")
    print()

    print("=" * 90)
    print("HEAD-TO-HEAD AT MEDIAN — which strategy wins for the typical investor?")
    print("=" * 90)
    sys_p50 = 100_000 * (1 + forecasts["50/30/20 system (production)"]["ann_return_p50"])
    bah_p50 = 100_000 * (1 + forecasts["BAH BTC only"]["ann_return_p50"])
    dca_p50 = 100_000 * (1 + forecasts["DCA $1k/wk BTC"]["ann_return_p50"])
    print(f"  50/30/20 system: ${sys_p50:>10,.0f}")
    print(f"  BAH BTC only:    ${bah_p50:>10,.0f}")
    print(f"  DCA $1k/wk BTC:  ${dca_p50:>10,.0f}")
    print()

    print("=" * 90)
    print("HEAD-TO-HEAD AT P25 (worst-quarter typical outcome)")
    print("=" * 90)
    sys_p25 = 100_000 * (1 + forecasts["50/30/20 system (production)"]["ann_return_p25"])
    bah_p25 = 100_000 * (1 + forecasts["BAH BTC only"]["ann_return_p25"])
    dca_p25 = 100_000 * (1 + forecasts["DCA $1k/wk BTC"]["ann_return_p25"])
    print(f"  50/30/20 system: ${sys_p25:>10,.0f}")
    print(f"  BAH BTC only:    ${bah_p25:>10,.0f}")
    print(f"  DCA $1k/wk BTC:  ${dca_p25:>10,.0f}")
    print()

    print("=" * 90)
    print("HEAD-TO-HEAD AT P5 (worst-case downside)")
    print("=" * 90)
    sys_p5 = 100_000 * (1 + forecasts["50/30/20 system (production)"]["ann_return_p5"])
    bah_p5 = 100_000 * (1 + forecasts["BAH BTC only"]["ann_return_p5"])
    dca_p5 = 100_000 * (1 + forecasts["DCA $1k/wk BTC"]["ann_return_p5"])
    print(f"  50/30/20 system: ${sys_p5:>10,.0f}")
    print(f"  BAH BTC only:    ${bah_p5:>10,.0f}")
    print(f"  DCA $1k/wk BTC:  ${dca_p5:>10,.0f}")
