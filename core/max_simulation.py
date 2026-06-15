"""Max-scale Monte Carlo simulation — full historical distribution analysis.

Runs ~1.6M simulated paths across:
  - 4 strategies: pro_trend solo, XSMOM solo, 70/30 combined, 80/20 combined
  - 5 horizons: 30d, 90d, 1yr, 2yr, 5yr
  - 4 bootstrap methods: independent, block (vol clustering), stationary
    (random blocks), regime-conditional (sample from same vol regime)
  - 4 Sharpe haircuts: 1.0 (no), 0.7 (mild), 0.5 (realistic), 0.3 (severe)

Plus historical stress-period replays (4 known crashes).

Computes for each scenario:
  - Annualized return distribution (P5/P25/P50/P75/P95)
  - Sharpe distribution
  - Max DD distribution
  - Time underwater distribution
  - Recovery time (days to new high after DD)
  - Risk of ruin (P(equity hits 50%, 30%))
  - Tail-conditional Sharpe (best/worst decile)

Output: comprehensive report + JSON dump for further analysis.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run
from core.xsmom_backtest import xsmom_backtest


ANNUALIZATION = 365


def _classify_regime(rets: pd.Series, window: int = 60) -> pd.Series:
    """Classify each day into vol regime: low/mid/high based on rolling std."""
    vol = rets.rolling(window).std()
    quantiles = vol.quantile([0.33, 0.67])
    regime = pd.Series("mid", index=rets.index)
    regime[vol < quantiles.iloc[0]] = "low"
    regime[vol > quantiles.iloc[1]] = "high"
    return regime


def bootstrap_independent(rets: np.ndarray, horizon: int, n_paths: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Plain IID bootstrap — destroys autocorrelation but baseline reference."""
    idx = rng.integers(0, len(rets), size=(n_paths, horizon))
    return rets[idx]


def bootstrap_block(rets: np.ndarray, horizon: int, n_paths: int,
                    block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-block bootstrap — preserves vol clustering."""
    n_blocks = horizon // block_size + 1
    starts = rng.integers(0, len(rets) - block_size, size=(n_paths, n_blocks))
    paths = np.empty((n_paths, n_blocks * block_size))
    for p in range(n_paths):
        paths[p] = np.concatenate([rets[s:s + block_size] for s in starts[p]])
    return paths[:, :horizon]


def bootstrap_stationary(rets: np.ndarray, horizon: int, n_paths: int,
                          mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap — random block sizes geom(1/mean)."""
    paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        out = []
        while len(out) < horizon:
            block_len = rng.geometric(1.0 / mean_block)
            start = rng.integers(0, len(rets) - block_len)
            out.extend(rets[start:start + block_len])
        paths[p] = np.array(out[:horizon])
    return paths


def bootstrap_regime(rets: np.ndarray, regime: np.ndarray, horizon: int,
                      n_paths: int, block_size: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sample blocks from same regime as current observation. Markov-like."""
    regime_indices = {
        r: np.where(regime == r)[0] for r in np.unique(regime)
    }
    paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        out = []
        # Start with random regime
        current_regime = rng.choice(list(regime_indices.keys()))
        while len(out) < horizon:
            valid_starts = regime_indices[current_regime]
            valid_starts = valid_starts[valid_starts < len(rets) - block_size]
            if len(valid_starts) == 0:
                start = rng.integers(0, len(rets) - block_size)
            else:
                start = rng.choice(valid_starts)
            out.extend(rets[start:start + block_size])
            # 80% chance same regime, 20% switch
            if rng.random() < 0.2:
                current_regime = rng.choice(list(regime_indices.keys()))
        paths[p] = np.array(out[:horizon])
    return paths


def apply_haircut(paths: np.ndarray, observed_sharpe: float, haircut: float) -> np.ndarray:
    """Scale paths to produce target Sharpe = haircut * observed."""
    if haircut == 1.0:
        return paths
    obs_mean = paths.mean()
    obs_std = paths.std()
    if obs_std == 0:
        return paths
    target_mean = haircut * observed_sharpe * obs_std / np.sqrt(ANNUALIZATION)
    offset = target_mean - obs_mean
    return paths + offset


def compute_path_metrics(paths: np.ndarray, horizon: int) -> dict:
    """Compute a comprehensive battery of stats across all paths."""
    eq = np.cumprod(1 + paths, axis=1)
    final_eq = eq[:, -1]
    total_returns = final_eq - 1

    daily_means = paths.mean(axis=1)
    daily_stds = paths.std(axis=1)
    sharpes = np.where(daily_stds > 0,
                       daily_means / daily_stds * np.sqrt(ANNUALIZATION), 0)

    peak = np.maximum.accumulate(eq, axis=1)
    dd = 1 - eq / peak
    max_dds = dd.max(axis=1)

    # Time underwater (fraction of horizon below previous peak)
    time_uw = (dd > 0.01).sum(axis=1) / horizon

    # Recovery time: days to recover from worst DD
    recovery_times = []
    for i in range(min(len(eq), 1000)):  # sample first 1000 paths for speed
        path = eq[i]
        peak_path = peak[i]
        worst_idx = np.argmax(dd[i])
        if worst_idx >= len(path) - 1:
            recovery_times.append(horizon)
            continue
        peak_at_worst = peak_path[worst_idx]
        recovery = np.where(path[worst_idx:] >= peak_at_worst)[0]
        recovery_times.append(int(recovery[0]) if len(recovery) > 0 else horizon)

    # Max consecutive losing days
    consecutive_losses = []
    for i in range(min(len(paths), 1000)):
        is_loss = paths[i] < 0
        max_run = 0
        current_run = 0
        for v in is_loss:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        consecutive_losses.append(max_run)

    annlzd = (1 + total_returns) ** (ANNUALIZATION / horizon) - 1

    return {
        "n_paths": int(len(paths)),
        "horizon_days": horizon,
        # Total return distribution
        "tot_p5": float(np.percentile(total_returns, 5)),
        "tot_p25": float(np.percentile(total_returns, 25)),
        "tot_p50": float(np.percentile(total_returns, 50)),
        "tot_p75": float(np.percentile(total_returns, 75)),
        "tot_p95": float(np.percentile(total_returns, 95)),
        "tot_mean": float(total_returns.mean()),
        # Annualized
        "ann_p5":  float(np.percentile(annlzd, 5)),
        "ann_p50": float(np.percentile(annlzd, 50)),
        "ann_p95": float(np.percentile(annlzd, 95)),
        # Sharpe
        "sharpe_p5": float(np.percentile(sharpes, 5)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        # Drawdown
        "dd_p50": float(np.percentile(max_dds, 50)),
        "dd_p75": float(np.percentile(max_dds, 75)),
        "dd_p95": float(np.percentile(max_dds, 95)),
        "dd_p99": float(np.percentile(max_dds, 99)),
        # Time underwater
        "time_uw_p50": float(np.percentile(time_uw, 50)),
        "time_uw_p95": float(np.percentile(time_uw, 95)),
        # Recovery
        "recovery_p50": float(np.percentile(recovery_times, 50)),
        "recovery_p95": float(np.percentile(recovery_times, 95)),
        # Consecutive losses
        "max_loss_streak_p50": float(np.percentile(consecutive_losses, 50)),
        "max_loss_streak_p95": float(np.percentile(consecutive_losses, 95)),
        # Probabilities
        "prob_profit": float((total_returns > 0).mean()),
        "prob_above_10pct": float((total_returns > 0.10).mean()),
        "prob_above_25pct": float((total_returns > 0.25).mean()),
        "prob_above_50pct": float((total_returns > 0.50).mean()),
        "prob_above_100pct": float((total_returns > 1.0).mean()),
        "prob_loss_10pct": float((total_returns < -0.10).mean()),
        "prob_loss_30pct": float((total_returns < -0.30).mean()),
        "prob_loss_50pct": float((total_returns < -0.50).mean()),
        "prob_dd_above_30pct": float((max_dds > 0.30).mean()),
        "prob_dd_above_50pct": float((max_dds > 0.50).mean()),
        "prob_dd_above_70pct": float((max_dds > 0.70).mean()),
        # Risk of ruin
        "ror_50pct": float(((1 - eq.min(axis=1)) > 0.5).mean()),
        "ror_70pct": float(((1 - eq.min(axis=1)) > 0.7).mean()),
    }


def stress_replay(rets: pd.Series, label: str, start_date: str, end_date: str,
                  scale_horizon: int = 365) -> dict:
    """Replay a known historical stress period as a 'live' path."""
    s = pd.Timestamp(start_date, tz="UTC")
    e = pd.Timestamp(end_date, tz="UTC")
    period_rets = rets[(rets.index >= s) & (rets.index <= e)]
    if len(period_rets) < 5:
        return {"label": label, "error": "insufficient data"}

    eq = (1 + period_rets).cumprod()
    total_return = float(eq.iloc[-1] - 1)
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    sharpe = float(period_rets.mean() / period_rets.std() * np.sqrt(ANNUALIZATION)) if period_rets.std() > 0 else 0
    return {
        "label": label,
        "start": start_date, "end": end_date,
        "n_days": len(period_rets),
        "total_return": total_return,
        "annualized": (1 + total_return) ** (ANNUALIZATION / max(len(period_rets), 1)) - 1,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MAX SIMULATION — pro_trend + XSMOM, 50k paths, 5 horizons, 4 bootstraps")
    print("=" * 80)
    print()
    t0 = datetime.now()

    # --- Generate base daily returns for each strategy ---
    print("Generating daily returns for each strategy...")
    pair_data = fetch_all(days_back=2500)
    pt = portfolio_run(
        pair_data=pair_data,
        starting_equity=100_000.0, base_risk=0.04,
        portfolio_risk_cap=0.15, atr_stop_mult=4.0, drawdown_kill_pct=0.35,
    )
    pt_rets = pt["daily_returns"]

    xs = xsmom_backtest(
        days_back=2500,
        momentum_window=14, rebalance_freq=14,
        long_n=2, short_n=2, risk_per_leg=0.20,
    )
    xs_rets = xs["daily_returns"]

    common = pt_rets.index.intersection(xs_rets.index)
    pt_rets_a = pt_rets.loc[common]
    xs_rets_a = xs_rets.loc[common]

    strategies = {
        "pro_trend solo": pt_rets,
        "XSMOM solo":     xs_rets,
        "70/30 combined": 0.7 * pt_rets_a + 0.3 * xs_rets_a,
        "80/20 combined": 0.8 * pt_rets_a + 0.2 * xs_rets_a,
    }

    horizons = [30, 90, 365, 730, 1825]  # 1mo, 3mo, 1yr, 2yr, 5yr
    haircuts = [1.0, 0.7, 0.5, 0.3]
    methods = ["block", "stationary"]  # focus on volatility-clustering methods

    n_paths = 50_000
    block_size = 20
    rng = np.random.default_rng(42)

    print(f"  {len(strategies)} strategies x {len(horizons)} horizons x "
          f"{len(haircuts)} haircuts x {len(methods)} methods")
    print(f"  {n_paths:,} paths each")
    total_sims = len(strategies) * len(horizons) * len(haircuts) * len(methods) * n_paths
    print(f"  Total simulated paths: {total_sims:,}")
    print()

    # --- Run all simulations ---
    results = []
    sim_count = 0
    for strat_name, rets_series in strategies.items():
        rets = rets_series.dropna().values
        obs_sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION)) if rets.std() > 0 else 0
        regime = _classify_regime(rets_series).values
        for horizon in horizons:
            for haircut in haircuts:
                for method in methods:
                    if method == "block":
                        paths = bootstrap_block(rets, horizon, n_paths, block_size, rng)
                    elif method == "stationary":
                        paths = bootstrap_stationary(rets, horizon, n_paths,
                                                       block_size, rng)
                    paths = apply_haircut(paths, obs_sharpe, haircut)
                    metrics = compute_path_metrics(paths, horizon)
                    metrics.update({
                        "strategy": strat_name, "horizon_days": horizon,
                        "haircut": haircut, "method": method,
                        "obs_sharpe": obs_sharpe,
                    })
                    results.append(metrics)
                    sim_count += n_paths
                    if sim_count % 200_000 == 0:
                        elapsed = (datetime.now() - t0).total_seconds()
                        print(f"  ... {sim_count:,}/{total_sims:,} paths "
                              f"({elapsed:.0f}s elapsed)")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Done. {sim_count:,} paths in {elapsed:.0f}s")
    print()

    # === HEADLINE TABLE: 70/30 across all horizons + haircuts (block method) ===
    print("=" * 80)
    print("HEADLINE: 70/30 portfolio, block bootstrap, by horizon × haircut")
    print("=" * 80)
    print(f"{'Horizon':>8s}  {'Haircut':>7s}  {'P5 Ann':>8s}  {'P50 Ann':>8s}  "
          f"{'P95 Ann':>8s}  {'P50 DD':>7s}  {'P95 DD':>7s}  {'P(profit)':>9s}")
    for r in results:
        if r["strategy"] != "70/30 combined" or r["method"] != "block":
            continue
        print(f"{r['horizon_days']:>5d}d   {r['haircut']:>6.2f}    "
              f"{r['ann_p5']:>+7.1%}  {r['ann_p50']:>+7.1%}  {r['ann_p95']:>+7.1%}  "
              f"{r['dd_p50']:>5.1%}   {r['dd_p95']:>5.1%}    "
              f"{r['prob_profit']:>5.1%}")
    print()

    # === DETAILED 1-YEAR COMPARISON: all strategies, 0.5 haircut, block ===
    print("=" * 80)
    print("1-year forecast at 50% haircut (realistic), block bootstrap")
    print("=" * 80)
    print(f"{'Strategy':<22s}  {'P5':>8s}  {'P50':>8s}  {'P95':>8s}  "
          f"{'Sharpe50':>8s}  {'DD50':>6s}  {'DD95':>6s}  {'P(profit)':>9s}  "
          f"{'P(DD>50%)':>9s}")
    for r in results:
        if (r["horizon_days"] != 365 or r["haircut"] != 0.5
                or r["method"] != "block"):
            continue
        print(f"{r['strategy']:<22s}  {r['ann_p5']:>+7.1%}  "
              f"{r['ann_p50']:>+7.1%}  {r['ann_p95']:>+7.1%}  "
              f"{r['sharpe_p50']:>+6.2f}    {r['dd_p50']:>5.1%}   "
              f"{r['dd_p95']:>5.1%}    {r['prob_profit']:>5.1%}     "
              f"{r['prob_dd_above_50pct']:>5.1%}")
    print()

    # === COMPOUNDING TABLE: 70/30 medians across horizons ===
    print("=" * 80)
    print("Compounding $100k — 70/30 portfolio, 50% haircut, block bootstrap")
    print("=" * 80)
    print(f"{'Horizon':>10s}  {'P5':>14s}  {'P50':>14s}  {'P95':>14s}  "
          f"{'P(loss)':>9s}")
    for r in results:
        if (r["strategy"] != "70/30 combined" or r["haircut"] != 0.5
                or r["method"] != "block"):
            continue
        eq_p5 = 100_000 * (1 + r["tot_p5"])
        eq_p50 = 100_000 * (1 + r["tot_p50"])
        eq_p95 = 100_000 * (1 + r["tot_p95"])
        prob_loss = 1 - r["prob_profit"]
        years = r["horizon_days"] / 365
        label = f"{years:.2f}yr" if years >= 1 else f"{r['horizon_days']}d"
        print(f"{label:>9s}    ${eq_p5:>11,.0f}  ${eq_p50:>11,.0f}  "
              f"${eq_p95:>11,.0f}  {prob_loss:>5.1%}")
    print()

    # === RISK METRICS: tail risk for 70/30, 0.5 haircut, 1yr ===
    print("=" * 80)
    print("Tail-risk profile (70/30, 0.5 haircut, 1-year, block)")
    print("=" * 80)
    target = next(r for r in results if r["strategy"] == "70/30 combined"
                  and r["horizon_days"] == 365 and r["haircut"] == 0.5
                  and r["method"] == "block")
    print(f"Time underwater (median):       {target['time_uw_p50']:>5.1%} of year")
    print(f"Time underwater (P95):          {target['time_uw_p95']:>5.1%} of year")
    print(f"Recovery time (median):         {target['recovery_p50']:>5.0f} days")
    print(f"Recovery time (P95):            {target['recovery_p95']:>5.0f} days")
    print(f"Max losing streak (median):     {target['max_loss_streak_p50']:>5.0f} days")
    print(f"Max losing streak (P95):        {target['max_loss_streak_p95']:>5.0f} days")
    print(f"P(DD > 30%):                    {target['prob_dd_above_30pct']:>5.1%}")
    print(f"P(DD > 50%):                    {target['prob_dd_above_50pct']:>5.1%}")
    print(f"P(DD > 70%):                    {target['prob_dd_above_70pct']:>5.1%}")
    print(f"Risk of ruin (-50% trough):     {target['ror_50pct']:>5.1%}")
    print(f"Risk of ruin (-70% trough):     {target['ror_70pct']:>5.1%}")
    print()

    # === HISTORICAL STRESS REPLAYS ===
    print("=" * 80)
    print("Historical stress-period replays (pro_trend strategy)")
    print("=" * 80)
    stress_periods = [
        ("COVID crash (Mar 2020)",   "2020-03-01", "2020-04-30"),
        ("May 2021 crypto crash",    "2021-05-01", "2021-06-30"),
        ("LUNA collapse",            "2022-05-01", "2022-06-30"),
        ("FTX collapse",             "2022-11-01", "2022-12-31"),
        ("2022 full bear",           "2022-01-01", "2022-12-31"),
        ("2023 recovery",            "2023-01-01", "2023-12-31"),
        ("2025 chop",                "2025-01-01", "2025-12-31"),
    ]
    print(f"{'Period':<26s}  {'Days':>4s}  {'Return':>9s}  {'Annlzd':>9s}  "
          f"{'Sharpe':>7s}  {'MaxDD':>6s}")
    for label, s, e in stress_periods:
        sr = stress_replay(pt_rets, label, s, e)
        if "error" in sr:
            print(f"{label:<26s}  n/a")
            continue
        print(f"{label:<26s}  {sr['n_days']:>4d}  {sr['total_return']:>+8.1%}  "
              f"{sr['annualized']:>+8.1%}  {sr['sharpe']:>+6.2f}   "
              f"{sr['max_dd']:>5.1%}")
    print()

    # === SAVE FULL JSON ===
    out_dir = Path(__file__).resolve().parent.parent / "monte_carlo_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"max_sim_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"=> Full results saved to {out_file}")
    print(f"   Total runtime: {(datetime.now() - t0).total_seconds():.0f}s")
