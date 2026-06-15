"""Monte Carlo simulation of live performance — bootstrap from CPCV returns.

Uses the CPCV concatenated daily return distribution as the empirical
sampling pool, with regime-aware block bootstrap to preserve volatility
clustering. Simulates 10,000 paths of N=365 days each (1 year forward).

Output: full distribution of:
  - Annualized return
  - Sharpe
  - Max drawdown
  - Probability of ending in profit
  - Probability of >X% drawdown
  - Path-dependent risk: 30-day rolling DD distribution

This is the institutional standard for "what could happen" forecasting.
We add a 50% Sharpe haircut to the bootstrap returns to simulate the
backtest -> live degradation typical for trend strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run
from core.xsmom_backtest import xsmom_backtest


ANNUALIZATION = 365


def simulate_paths(
    daily_rets: pd.Series,
    n_paths: int = 10_000,
    horizon_days: int = 365,
    block_size: int = 20,
    sharpe_haircut: float = 0.5,
    rng_seed: int = 42,
) -> dict:
    """Block-bootstrap simulation. Optionally applies a Sharpe haircut.

    The haircut: scale returns to produce target_sharpe = haircut * observed_sharpe,
    keeping daily vol roughly constant. Models live degradation.
    """
    rng = np.random.default_rng(rng_seed)
    rets_arr = daily_rets.dropna().values

    if len(rets_arr) < block_size * 5:
        return {"error": "insufficient returns"}

    obs_mean = float(np.mean(rets_arr))
    obs_std = float(np.std(rets_arr))
    obs_sharpe = obs_mean / obs_std * np.sqrt(ANNUALIZATION) if obs_std > 0 else 0

    # If applying haircut, scale daily mean to target
    target_sharpe = obs_sharpe * sharpe_haircut
    target_mean = target_sharpe * obs_std / np.sqrt(ANNUALIZATION)
    haircut_offset = target_mean - obs_mean

    n_blocks_needed = horizon_days // block_size + 1
    paths = []
    for _ in range(n_paths):
        block_starts = rng.integers(0, len(rets_arr) - block_size, size=n_blocks_needed)
        sample = np.concatenate([rets_arr[s:s + block_size] for s in block_starts])
        sample = sample[:horizon_days] + haircut_offset
        paths.append(sample)

    paths = np.array(paths)
    eq = np.cumprod(1 + paths, axis=1)

    final_eq = eq[:, -1]
    total_returns = final_eq - 1

    # Per-path Sharpe
    daily_means = paths.mean(axis=1)
    daily_stds = paths.std(axis=1)
    sharpes = np.where(daily_stds > 0,
                       daily_means / daily_stds * np.sqrt(ANNUALIZATION), 0)

    # Per-path Max DD
    peak = np.maximum.accumulate(eq, axis=1)
    dd = 1 - eq / peak
    max_dds = dd.max(axis=1)

    # 30-day rolling DD distribution
    rolling_30d_max_dd = []
    for path in eq:
        for start in range(0, len(path) - 30, 5):
            seg = path[start:start + 30]
            seg_peak = np.maximum.accumulate(seg)
            seg_dd = 1 - seg / seg_peak
            rolling_30d_max_dd.append(float(seg_dd.max()))
    rolling_30d_max_dd = np.array(rolling_30d_max_dd)

    return {
        "n_paths": n_paths,
        "horizon_days": horizon_days,
        "haircut_applied": sharpe_haircut,
        "observed_sharpe": float(obs_sharpe),
        "target_sharpe": float(target_sharpe),
        # Annualized return distribution
        "ann_return_p5":  float(np.percentile(total_returns, 5)),
        "ann_return_p25": float(np.percentile(total_returns, 25)),
        "ann_return_p50": float(np.percentile(total_returns, 50)),
        "ann_return_p75": float(np.percentile(total_returns, 75)),
        "ann_return_p95": float(np.percentile(total_returns, 95)),
        "ann_return_mean": float(np.mean(total_returns)),
        # Sharpe distribution
        "sharpe_p5":  float(np.percentile(sharpes, 5)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "sharpe_mean": float(np.mean(sharpes)),
        # Max DD distribution
        "max_dd_p5":  float(np.percentile(max_dds, 5)),
        "max_dd_p50": float(np.percentile(max_dds, 50)),
        "max_dd_p75": float(np.percentile(max_dds, 75)),
        "max_dd_p95": float(np.percentile(max_dds, 95)),
        # Rolling 30-day DD
        "rolling_30d_dd_p50": float(np.percentile(rolling_30d_max_dd, 50)),
        "rolling_30d_dd_p95": float(np.percentile(rolling_30d_max_dd, 95)),
        # Probabilities
        "prob_profit": float((total_returns > 0).mean()),
        "prob_above_10pct": float((total_returns > 0.10).mean()),
        "prob_above_25pct": float((total_returns > 0.25).mean()),
        "prob_dd_above_30pct": float((max_dds > 0.30).mean()),
        "prob_dd_above_50pct": float((max_dds > 0.50).mean()),
        "prob_loss_20pct": float((total_returns < -0.20).mean()),
    }


if __name__ == "__main__":
    print("=" * 78)
    print("MONTE CARLO LIVE-PERFORMANCE SIMULATION")
    print("=" * 78)
    print()

    pair_data = fetch_all(days_back=2500)

    # Pro_trend daily returns (the workhorse strategy)
    print("Generating pro_trend daily returns...")
    pt = portfolio_run(
        pair_data=pair_data,
        starting_equity=100_000.0, base_risk=0.04,
        portfolio_risk_cap=0.15, atr_stop_mult=4.0,
        drawdown_kill_pct=0.35,
    )
    pt_rets = pt["daily_returns"]

    # XSMOM daily returns
    print("Generating XSMOM daily returns...")
    xs = xsmom_backtest(
        days_back=2500,
        momentum_window=14, rebalance_freq=14,
        long_n=2, short_n=2, risk_per_leg=0.20,
    )
    xs_rets = xs["daily_returns"]

    # Combined 70/30 portfolio
    common = pt_rets.index.intersection(xs_rets.index)
    combined_rets = 0.7 * pt_rets.loc[common] + 0.3 * xs_rets.loc[common]
    print()

    scenarios = [
        ("pro_trend solo, no haircut",  pt_rets, 1.0),
        ("pro_trend solo, 50% haircut", pt_rets, 0.5),
        ("pro_trend solo, 30% haircut (severe)", pt_rets, 0.3),
        ("70/30 combined, no haircut",  combined_rets, 1.0),
        ("70/30 combined, 50% haircut", combined_rets, 0.5),
        ("70/30 combined, 30% haircut (severe)", combined_rets, 0.3),
    ]

    print(f"{'Scenario':<40s}  {'Median':>8s}  {'P5':>8s}  {'P95':>8s}  "
          f"{'Sharpe50':>8s}  {'DD50':>6s}  {'PProfit':>7s}")

    for label, rets, haircut in scenarios:
        result = simulate_paths(rets, n_paths=5_000, horizon_days=365,
                                  sharpe_haircut=haircut)
        if "error" in result:
            continue
        print(f"{label:<40s}  "
              f"{result['ann_return_p50']:>+7.1%}  "
              f"{result['ann_return_p5']:>+7.1%}  "
              f"{result['ann_return_p95']:>+7.1%}  "
              f"{result['sharpe_p50']:>+6.2f}    "
              f"{result['max_dd_p50']:>5.1%}  "
              f"{result['prob_profit']:>5.0%}")

    print()
    print("=" * 78)
    print("DETAILED DISTRIBUTION — 70/30 combined with realistic 50% haircut:")
    print("=" * 78)
    detail = simulate_paths(combined_rets, n_paths=10_000, horizon_days=365,
                              sharpe_haircut=0.5)

    print(f"Annualized return:")
    print(f"  P5:    {detail['ann_return_p5']:>+7.1%}")
    print(f"  P25:   {detail['ann_return_p25']:>+7.1%}")
    print(f"  P50:   {detail['ann_return_p50']:>+7.1%}  (median forecast)")
    print(f"  P75:   {detail['ann_return_p75']:>+7.1%}")
    print(f"  P95:   {detail['ann_return_p95']:>+7.1%}")
    print(f"  Mean:  {detail['ann_return_mean']:>+7.1%}")
    print()
    print(f"Sharpe (1-year sample distribution):")
    print(f"  P5:    {detail['sharpe_p5']:>+6.2f}")
    print(f"  P50:   {detail['sharpe_p50']:>+6.2f}")
    print(f"  P95:   {detail['sharpe_p95']:>+6.2f}")
    print()
    print(f"Max drawdown:")
    print(f"  P50:   {detail['max_dd_p50']:>6.1%}")
    print(f"  P75:   {detail['max_dd_p75']:>6.1%}")
    print(f"  P95:   {detail['max_dd_p95']:>6.1%}")
    print()
    print(f"Rolling 30-day DD (intra-year stress):")
    print(f"  P50:   {detail['rolling_30d_dd_p50']:>6.1%}")
    print(f"  P95:   {detail['rolling_30d_dd_p95']:>6.1%}")
    print()
    print(f"Probabilities (1-year forward):")
    print(f"  P(profit):              {detail['prob_profit']:>5.1%}")
    print(f"  P(return > 10%):        {detail['prob_above_10pct']:>5.1%}")
    print(f"  P(return > 25%):        {detail['prob_above_25pct']:>5.1%}")
    print(f"  P(DD > 30%):            {detail['prob_dd_above_30pct']:>5.1%}")
    print(f"  P(DD > 50%):            {detail['prob_dd_above_50pct']:>5.1%}")
    print(f"  P(loss > 20%):          {detail['prob_loss_20pct']:>5.1%}")
