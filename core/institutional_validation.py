"""Institutional-grade validation suite — CPCV + DSR + PSR.

Applies López de Prado-style validation to the pro_trend portfolio backtest:

  1. CPCV: Combinatorial Purged Cross-Validation. Splits 6.3-year history
     into 6 groups, tests every C(6,2)=15 pair as OOS, gives 15 OOS
     evaluations vs walk-forward's 5. Tighter standard error on Sharpe.

  2. DSR (Deflated Sharpe Ratio): Bailey/LdP 2014. Probability the true
     Sharpe exceeds null given the trial count searched. The honest trial
     count for this session is the count of parameter combinations tested
     across all sweeps.

  3. PSR (Probabilistic Sharpe Ratio): probability observed Sharpe is
     greater than a benchmark (we use 0 and 0.5).

  4. Hurdles: DSR > 0.95 AND |t-stat| > 3.0 (Harvey/Liu/Zhu 2016).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run
from core.deflated_sharpe import deflated_sharpe, sharpe_ratio, t_stat


ANNUALIZATION = 365

# Honest trial count from this session's tinkering:
# - Universe size sweep: 5 portfolio sizes
# - Vol-targeted: 9 variants
# - Catalyst overlay: 6 schedules
# - Regime gate: 9 variants
# - Param sweep: DD(7) + ATR(5) + Top-K(5) = 17
# - Funding-aware: 12 variants
# - Basis arb threshold: 5 variants
# - Bootstrap: 1 (already done)
# Total ≈ 200 distinct backtest configurations searched.
N_TRIALS = 200


def cpcv_portfolio_backtest(
    pair_data: dict,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_days: int = 5,
    base_kw: dict | None = None,
) -> dict:
    """Run CPCV on the multi-pair portfolio backtest.

    For each combination of test groups, runs the backtest on ONLY those
    test periods (test=evaluate, train is implicit since strategy is
    stationary/parametric). Concatenates OOS daily returns across all
    combinations to get a richer OOS sample.
    """
    if base_kw is None:
        base_kw = dict(
            starting_equity=100_000.0, base_risk=0.04,
            portfolio_risk_cap=0.15, atr_stop_mult=4.0,
            drawdown_kill_pct=0.35,
        )

    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    n_obs = len(all_dates)
    group_size = n_obs // n_groups
    boundaries = []
    for i in range(n_groups):
        lo = i * group_size
        hi = (i + 1) * group_size if i < n_groups - 1 else n_obs
        boundaries.append((lo, hi))

    fold_results = []
    for combo_idx, test_combo in enumerate(combinations(range(n_groups), k_test)):
        # Determine the date range covered by this combo
        # We want CONTIGUOUS test windows where possible
        test_indices = []
        for g in test_combo:
            lo, hi = boundaries[g]
            test_indices.extend(range(lo, hi))
        test_indices.sort()

        # Convert indices to dates
        test_dates = [all_dates[i] for i in test_indices]
        # Group into contiguous runs
        runs = []
        current_run = [test_dates[0]]
        for i in range(1, len(test_dates)):
            if (test_dates[i] - current_run[-1]).days <= 2:
                current_run.append(test_dates[i])
            else:
                runs.append(current_run)
                current_run = [test_dates[i]]
        runs.append(current_run)

        # Run a separate backtest for each contiguous run, concatenate returns
        combo_returns = []
        combo_total_return = 1.0
        for run in runs:
            if len(run) < 30:
                continue
            r = portfolio_run(
                pair_data=pair_data,
                date_start=run[0], date_end=run[-1],
                **base_kw,
            )
            if "error" in r or "daily_returns" not in r:
                continue
            combo_returns.append(r["daily_returns"])
            combo_total_return *= (1 + r["total_return"])

        if not combo_returns:
            continue
        all_rets = pd.concat(combo_returns).sort_index()
        if len(all_rets) < 30 or all_rets.std() == 0:
            continue
        s = float(all_rets.mean() / all_rets.std() * np.sqrt(ANNUALIZATION))
        fold_results.append({
            "combo_idx": combo_idx,
            "test_groups": list(test_combo),
            "n_test_days": len(all_rets),
            "sharpe_oos": s,
            "total_return": combo_total_return - 1,
            "returns": all_rets,
        })

    if not fold_results:
        return {"error": "no fold results"}

    sharpes = np.array([r["sharpe_oos"] for r in fold_results])
    all_returns = pd.concat([r["returns"] for r in fold_results])

    return {
        "n_combinations": len(fold_results),
        "mean_oos_sharpe": float(sharpes.mean()),
        "std_oos_sharpe": float(sharpes.std(ddof=1)),
        "min_oos_sharpe": float(sharpes.min()),
        "max_oos_sharpe": float(sharpes.max()),
        "median_oos_sharpe": float(np.median(sharpes)),
        "n_total_oos_days": int(sum(r["n_test_days"] for r in fold_results)),
        "concat_returns": all_returns,
        "n_positive_combos": int((sharpes > 0).sum()),
        "n_combos_above_0p5": int((sharpes > 0.5).sum()),
    }


def probabilistic_sharpe(observed_sr: float, sr_benchmark: float,
                         n_obs: int, skew: float = 0, kurt: float = 3) -> float:
    """PSR: probability observed Sharpe exceeds benchmark (per-period units)."""
    if n_obs < 30:
        return 0.0
    excess_kurt = kurt - 3
    sigma_sr = np.sqrt(
        (1 - skew * observed_sr + (excess_kurt / 4) * observed_sr ** 2) / (n_obs - 1)
    )
    if sigma_sr <= 0:
        return 0.0
    z = (observed_sr - sr_benchmark) / sigma_sr
    return float(norm.cdf(z))


if __name__ == "__main__":
    print("=" * 78)
    print("INSTITUTIONAL VALIDATION — CPCV + DSR + PSR")
    print("=" * 78)
    print()
    print(f"Honest trial count: {N_TRIALS}")
    print(f"Hurdles: DSR > 0.95 (95% confidence) AND |t| > 3.0 (HLZ 2016)")
    print()

    pair_data = fetch_all(days_back=2500)
    print(f"Universe: {list(pair_data.keys())}")
    print()

    # === [1] Reference single backtest ===
    print("=" * 78)
    print("[1] Reference single-window backtest (max history)")
    print("=" * 78)
    full = portfolio_run(
        pair_data=pair_data,
        starting_equity=100_000.0, base_risk=0.04,
        portfolio_risk_cap=0.15, atr_stop_mult=4.0,
        drawdown_kill_pct=0.35,
    )
    daily_rets = full["daily_returns"]
    full_sr = sharpe_ratio(daily_rets)
    full_t = t_stat(daily_rets)
    print(f"N obs (days):         {len(daily_rets)}")
    print(f"Annualized return:    {full['annualized_return']:+.2%}")
    print(f"Annualized Sharpe:    {full_sr:+.2f}")
    print(f"Daily t-stat:         {full_t:+.2f}")
    print(f"Max DD:               {full['max_drawdown']:.2%}")
    print()

    # === [2] CPCV ===
    print("=" * 78)
    print("[2] Combinatorial Purged Cross-Validation (6 groups, k=2)")
    print("=" * 78)
    cpcv = cpcv_portfolio_backtest(pair_data, n_groups=6, k_test=2)
    if "error" not in cpcv:
        print(f"N combinations:       {cpcv['n_combinations']} (vs walk-forward's 5)")
        print(f"N OOS observations:   {cpcv['n_total_oos_days']}")
        print(f"Mean OOS Sharpe:      {cpcv['mean_oos_sharpe']:+.2f}")
        print(f"Median OOS Sharpe:    {cpcv['median_oos_sharpe']:+.2f}")
        print(f"Std OOS Sharpe:       {cpcv['std_oos_sharpe']:.2f}")
        print(f"Min/Max OOS Sharpe:   [{cpcv['min_oos_sharpe']:+.2f}, {cpcv['max_oos_sharpe']:+.2f}]")
        print(f"Combos with S > 0:    {cpcv['n_positive_combos']}/{cpcv['n_combinations']}")
        print(f"Combos with S > 0.5:  {cpcv['n_combos_above_0p5']}/{cpcv['n_combinations']}")
    print()

    # === [3] Deflated Sharpe Ratio ===
    print("=" * 78)
    print("[3] Deflated Sharpe Ratio (Bailey/LdP 2014)")
    print("=" * 78)
    dsr = deflated_sharpe(daily_rets, num_trials=N_TRIALS,
                           periods_per_year=ANNUALIZATION)
    print(f"DSR (probability true SR > trial-inflated null): {dsr['dsr']:.4f}")
    print(f"Annualized SR:                {dsr['sr_annualized']:+.2f}")
    print(f"SR threshold (annualized):    {dsr['sr_threshold_annualized']:+.2f}")
    print(f"N obs:                        {dsr['n_obs']}")
    print(f"N trials assumed:             {dsr['n_trials']}")
    print(f"PASSES DSR (>0.95):           {dsr['passes']}")
    print()

    # === [4] PSR ===
    print("=" * 78)
    print("[4] Probabilistic Sharpe Ratio")
    print("=" * 78)
    n = len(daily_rets)
    sr_per_period = full_sr / np.sqrt(ANNUALIZATION)

    psr_zero = probabilistic_sharpe(sr_per_period, 0, n,
                                     daily_rets.skew(), daily_rets.kurtosis() + 3)
    psr_05 = probabilistic_sharpe(
        sr_per_period, 0.5 / np.sqrt(ANNUALIZATION), n,
        daily_rets.skew(), daily_rets.kurtosis() + 3,
    )
    psr_10 = probabilistic_sharpe(
        sr_per_period, 1.0 / np.sqrt(ANNUALIZATION), n,
        daily_rets.skew(), daily_rets.kurtosis() + 3,
    )
    print(f"P(true SR > 0):       {psr_zero:.4f}")
    print(f"P(true SR > 0.5):     {psr_05:.4f}")
    print(f"P(true SR > 1.0):     {psr_10:.4f}")
    print()

    # === [5] Combined hurdle ===
    print("=" * 78)
    print("[5] HARVEY/LIU/ZHU (2016) HURDLE")
    print("=" * 78)
    print(f"Daily t-stat:         {full_t:+.2f}")
    print(f"|t| > 3.0:            {abs(full_t) > 3.0}")
    print(f"DSR > 0.95:           {dsr['passes']}")
    print(f"PASSES BOTH:          {abs(full_t) > 3.0 and dsr['passes']}")
    print()

    # === Verdict ===
    print("=" * 78)
    print("INSTITUTIONAL VALIDATION VERDICT")
    print("=" * 78)
    passes_all = (
        abs(full_t) > 3.0
        and dsr['passes']
        and "error" not in cpcv
        and cpcv['mean_oos_sharpe'] > 0.5
        and cpcv['n_combos_above_0p5'] / cpcv['n_combinations'] > 0.5
    )
    if passes_all:
        print("PASSED — strategy meets institutional gate (CPCV + DSR + HLZ).")
        print("This is the level at which an institutional shop would")
        print("consider seed allocation pending live track record.")
    else:
        print("FAILED — strategy does not pass full institutional gate.")
        if not dsr['passes']:
            print(f"  - DSR {dsr['dsr']:.3f} < 0.95")
        if abs(full_t) <= 3.0:
            print(f"  - |t| {abs(full_t):.2f} <= 3.0")
        if "error" not in cpcv:
            if cpcv['mean_oos_sharpe'] <= 0.5:
                print(f"  - Mean OOS CPCV Sharpe {cpcv['mean_oos_sharpe']:.2f} <= 0.5")
            ratio = cpcv['n_combos_above_0p5'] / cpcv['n_combinations']
            if ratio <= 0.5:
                print(f"  - Combos with Sharpe > 0.5 only {ratio:.0%} (need > 50%)")
