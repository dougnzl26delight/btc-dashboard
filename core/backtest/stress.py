"""P9: Synthetic stress tests via Monte Carlo.

Bootstrap returns from each regime to construct novel scenarios.
Tests robustness to UNSEEN combinations of conditions.

Pass criterion: 5th percentile Sharpe across N simulations > 0.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.backtest.replay import (
    sharpe, max_drawdown, total_return,
    fetch_scenario_data, compute_returns, simple_engine_replay,
)


# Per-regime daily return parameters (empirical approximations)
REGIME_RETURN_PARAMS = {
    "RISK_ON": {
        "SPY":     {"mean": 0.0008,  "std": 0.0095},  # ~20% annual / 15% vol
        "BTC-USD": {"mean": 0.0020,  "std": 0.0400},  # ~50% annual / 60% vol
        "BIL":     {"mean": 0.00015, "std": 0.0002},
        "GLDM":    {"mean": 0.0003,  "std": 0.0085},
        "VTIP":    {"mean": 0.00020, "std": 0.0035},
    },
    "LATE_CYCLE": {
        "SPY":     {"mean": 0.0003,  "std": 0.0120},
        "BTC-USD": {"mean": 0.0000,  "std": 0.0450},
        "BIL":     {"mean": 0.00015, "std": 0.0002},
        "GLDM":    {"mean": 0.0006,  "std": 0.0095},
        "VTIP":    {"mean": 0.00025, "std": 0.0040},
    },
    "RECESSIONARY_BEAR": {
        "SPY":     {"mean": -0.0010, "std": 0.0180},
        "BTC-USD": {"mean": -0.0005, "std": 0.0550},
        "BIL":     {"mean": 0.00018, "std": 0.0002},
        "GLDM":    {"mean": 0.0012,  "std": 0.0110},
        "VTIP":    {"mean": 0.0005,  "std": 0.0050},
    },
}

# Empirical transition probabilities (per day)
REGIME_TRANSITION_MATRIX = {
    "RISK_ON":           {"RISK_ON": 0.992, "LATE_CYCLE": 0.007, "RECESSIONARY_BEAR": 0.001},
    "LATE_CYCLE":        {"RISK_ON": 0.005, "LATE_CYCLE": 0.985, "RECESSIONARY_BEAR": 0.010},
    "RECESSIONARY_BEAR": {"RISK_ON": 0.002, "LATE_CYCLE": 0.013, "RECESSIONARY_BEAR": 0.985},
}


def simulate_regime_chain(length: int,
                           initial_regime: str = "RISK_ON",
                           rng: Optional[np.random.Generator] = None) -> list[str]:
    """Generate a Markov chain of regimes with realistic persistence."""
    rng = rng or np.random.default_rng()
    regimes = [initial_regime]
    for _ in range(length - 1):
        current = regimes[-1]
        probs = REGIME_TRANSITION_MATRIX[current]
        next_regime = rng.choice(list(probs.keys()), p=list(probs.values()))
        regimes.append(next_regime)
    return regimes


def simulate_returns(regime_seq: list[str],
                       assets: list[str] = None,
                       rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Sample daily returns for each asset from the regime-conditional distribution."""
    rng = rng or np.random.default_rng()
    assets = assets or ["SPY", "BTC-USD", "BIL", "GLDM", "VTIP"]
    n = len(regime_seq)
    data = {a: np.zeros(n) for a in assets}
    for i, regime in enumerate(regime_seq):
        params = REGIME_RETURN_PARAMS.get(regime, REGIME_RETURN_PARAMS["LATE_CYCLE"])
        for a in assets:
            p = params.get(a, {"mean": 0, "std": 0.01})
            data[a][i] = rng.normal(p["mean"], p["std"])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx)


def run_single_simulation(length_days: int = 756,  # 3 years
                            initial_regime: str = "RISK_ON",
                            seed: Optional[int] = None) -> dict:
    """One Monte Carlo run. Returns engine + benchmark metrics."""
    rng = np.random.default_rng(seed)
    regime_seq = simulate_regime_chain(length_days, initial_regime, rng)
    returns = simulate_returns(regime_seq, rng=rng)
    regime_series = pd.Series(regime_seq, index=returns.index)

    engine_rets = simple_engine_replay(returns, regime_series=regime_series)
    benchmark_6040 = 0.6 * returns["SPY"] + 0.4 * returns["BIL"]

    return {
        "engine_sharpe": sharpe(engine_rets),
        "engine_dd": max_drawdown(engine_rets),
        "engine_total": total_return(engine_rets),
        "bench_sharpe": sharpe(benchmark_6040),
        "bench_dd": max_drawdown(benchmark_6040),
        "bench_total": total_return(benchmark_6040),
        "n_days_by_regime": {r: regime_seq.count(r) for r in set(regime_seq)},
    }


def monte_carlo_stress(n_simulations: int = 100,
                         length_days: int = 756,
                         seed: int = 42) -> dict:
    """Run N simulations. Report distribution of Sharpes + DDs.

    Pass criterion: 5th percentile Sharpe > 0 (engine never blows up
    even on adversarial regime sequences).
    """
    base_rng = np.random.default_rng(seed)
    results = []
    for sim in range(n_simulations):
        sim_seed = int(base_rng.integers(0, 1_000_000))
        # Random initial regime
        init = base_rng.choice(["RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"])
        r = run_single_simulation(length_days, init, seed=sim_seed)
        results.append(r)

    engine_sharpes = [r["engine_sharpe"] for r in results]
    bench_sharpes = [r["bench_sharpe"] for r in results]
    engine_dds = [r["engine_dd"] for r in results]

    return {
        "n_simulations": n_simulations,
        "length_days": length_days,
        "engine_sharpe_median": float(np.median(engine_sharpes)),
        "engine_sharpe_p5": float(np.percentile(engine_sharpes, 5)),
        "engine_sharpe_p95": float(np.percentile(engine_sharpes, 95)),
        "bench_sharpe_median": float(np.median(bench_sharpes)),
        "engine_dd_median": float(np.median(engine_dds)),
        "engine_dd_worst": float(np.min(engine_dds)),
        "p_negative_sharpe": float(np.mean([s < 0 for s in engine_sharpes])),
        "p_worse_than_bench": float(np.mean(
            [e < b for e, b in zip(engine_sharpes, bench_sharpes)])),
        "pass": float(np.percentile(engine_sharpes, 5)) > 0.0,
    }


def synthetic_scenario_stress() -> dict:
    """Specific named scenarios: high-inflation+negative-growth, policy-error, etc."""
    scenarios = {
        "stagflation": {  # Sticky inflation, growth contracting
            "regime_sequence": ["LATE_CYCLE"] * 100 + ["RECESSIONARY_BEAR"] * 200 +
                              ["LATE_CYCLE"] * 100,
            "inflation_adjust": 1.5,  # vols 1.5x
        },
        "policy_error": {  # Hike-too-far → liquidity event
            "regime_sequence": ["RISK_ON"] * 80 + ["LATE_CYCLE"] * 40 +
                              ["RECESSIONARY_BEAR"] * 180,
            "inflation_adjust": 1.0,
        },
        "etf_mania_no_recession": {  # Bull continuation without macro support
            "regime_sequence": ["RISK_ON"] * 500,
            "inflation_adjust": 1.0,
        },
    }

    results = {}
    rng = np.random.default_rng(99)
    for name, sc in scenarios.items():
        regime_seq = sc["regime_sequence"]
        returns = simulate_returns(regime_seq, rng=rng)
        # Scale vol by inflation adjustment
        for col in returns.columns:
            returns[col] *= sc["inflation_adjust"]
        regime_series = pd.Series(regime_seq, index=returns.index)
        engine = simple_engine_replay(returns, regime_series=regime_series)
        bench = 0.6 * returns["SPY"] + 0.4 * returns["BIL"]

        results[name] = {
            "engine_sharpe": sharpe(engine),
            "engine_dd": max_drawdown(engine),
            "engine_total": total_return(engine),
            "bench_sharpe": sharpe(bench),
            "bench_dd": max_drawdown(bench),
            "bench_total": total_return(bench),
            "pass": sharpe(engine) >= sharpe(bench) - 0.10,  # within 0.1 of bench
        }
    return results


def main():
    print("Monte Carlo stress test (50 sims, 3-year horizon)")
    r = monte_carlo_stress(n_simulations=50, length_days=756)
    print(f"  Median engine Sharpe: {r['engine_sharpe_median']:+.2f}")
    print(f"  P5 engine Sharpe:     {r['engine_sharpe_p5']:+.2f}")
    print(f"  P95 engine Sharpe:    {r['engine_sharpe_p95']:+.2f}")
    print(f"  Median bench Sharpe:  {r['bench_sharpe_median']:+.2f}")
    print(f"  Worst engine DD:      {r['engine_dd_worst']:.2%}")
    print(f"  P(neg Sharpe):        {r['p_negative_sharpe']:.0%}")
    print(f"  Pass (P5>0): {r['pass']}")

    print("\nSynthetic named scenarios:")
    sc = synthetic_scenario_stress()
    for name, res in sc.items():
        print(f"  {name:30s} engine_sharpe {res['engine_sharpe']:+.2f}  "
              f"bench_sharpe {res['bench_sharpe']:+.2f}  pass {res['pass']}")


if __name__ == "__main__":
    main()
