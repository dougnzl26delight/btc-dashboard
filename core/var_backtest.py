"""VaR backtest validation — Kupiec POF + Christoffersen test.

VaR models are usually presented as "1% 1-day VaR = $X". But how do you know
the model is correctly CALIBRATED? You backtest it:

    1. Count actual losses that exceeded VaR (breaches)
    2. Compare to expected count (1% × N days)
    3. Run statistical test for whether observed breaches match expected

Kupiec POF (Proportion of Failures) test:
    H0: model is well-calibrated; breach rate = stated confidence level
    H1: model is mis-calibrated

Christoffersen test extends Kupiec to check INDEPENDENCE of breaches
(clustered breaches = bad model even if total count is correct).

If your VaR model fails Kupiec, you're either:
    - Over-estimating risk (over-conservative, missing trades)
    - Under-estimating risk (too aggressive, blow-up risk)

References:
    Kupiec (1995) "Techniques for Verifying the Accuracy of Risk Measurement Models"
    Christoffersen (1998) "Evaluating Interval Forecasts"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof_test(n_breaches: int, n_observations: int,
                     confidence_level: float = 0.99) -> dict:
    """Kupiec Proportion of Failures test.

    Args:
        n_breaches: count of days where loss exceeded VaR
        n_observations: total days in backtest
        confidence_level: VaR confidence (e.g., 0.99 for 1% VaR)

    Returns: test result with p-value and verdict.
    """
    alpha = 1 - confidence_level  # expected breach rate
    if n_observations < 30:
        return {"error": "insufficient_observations", "n": n_observations}

    pi_hat = n_breaches / n_observations
    expected_breaches = alpha * n_observations

    # Likelihood ratio test
    # L_alpha = alpha^x * (1-alpha)^(N-x)
    # L_pi_hat = pi_hat^x * (1-pi_hat)^(N-x)
    # LR = -2 * ln(L_alpha / L_pi_hat) ~ chi-squared(1)
    if pi_hat == 0 or pi_hat == 1:
        # Avoid log(0)
        ll_alpha = n_breaches * np.log(alpha) + (n_observations - n_breaches) * np.log(1 - alpha)
        ll_pi = -np.inf if pi_hat in (0, 1) else (
            n_breaches * np.log(pi_hat) + (n_observations - n_breaches) * np.log(1 - pi_hat)
        )
    else:
        ll_alpha = n_breaches * np.log(alpha) + (n_observations - n_breaches) * np.log(1 - alpha)
        ll_pi = n_breaches * np.log(pi_hat) + (n_observations - n_breaches) * np.log(1 - pi_hat)

    lr_pof = -2 * (ll_alpha - ll_pi)
    p_value = 1 - stats.chi2.cdf(lr_pof, df=1)

    # Verdict
    if p_value < 0.05:
        if pi_hat > alpha:
            verdict = "FAIL — model UNDER-estimates risk (too aggressive)"
        else:
            verdict = "FAIL — model OVER-estimates risk (too conservative)"
    else:
        verdict = "PASS — model calibration acceptable"

    return {
        "n_breaches_observed": n_breaches,
        "n_breaches_expected": expected_breaches,
        "n_observations": n_observations,
        "observed_breach_rate_pct": pi_hat * 100,
        "expected_breach_rate_pct": alpha * 100,
        "LR_POF": lr_pof,
        "p_value": p_value,
        "verdict": verdict,
        "confidence_level": confidence_level,
    }


def christoffersen_independence_test(breach_sequence: list[int]) -> dict:
    """Christoffersen independence test.

    Tests whether breaches are CLUSTERED (model is mis-calibrated for clusters
    even if total breach count is right).

    breach_sequence: list of 0/1 over N days (1 = breach)
    """
    n = len(breach_sequence)
    if n < 30:
        return {"error": "insufficient_observations"}

    # Count transition pairs
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev = breach_sequence[i - 1]
        curr = breach_sequence[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n - 1) if n > 1 else 0

    # LR_IND = -2 * ln(L_independent / L_dependent)
    def _safe_log(x):
        return np.log(x) if x > 0 else -1e10

    ll_dep = (n00 * _safe_log(1 - pi_01) + n01 * _safe_log(pi_01)
              + n10 * _safe_log(1 - pi_11) + n11 * _safe_log(pi_11))
    ll_ind = ((n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi))
    lr_ind = -2 * (ll_ind - ll_dep)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)

    verdict = "FAIL — breaches CLUSTERED" if p_value < 0.05 else "PASS — breaches independent"

    return {
        "transitions": {"00": n00, "01": n01, "10": n10, "11": n11},
        "LR_IND": lr_ind,
        "p_value": p_value,
        "verdict": verdict,
    }


def backtest_var(returns: list[float], var_pct: float = 0.01,
                  confidence_level: float = 0.99) -> dict:
    """Backtest a fixed-percentage VaR model on a return series.

    For a real model that VARIES daily, you'd pass per-day VaR estimates instead.
    """
    arr = np.array(returns)
    threshold = -var_pct  # losses below this = breach
    breach_seq = [1 if r < threshold else 0 for r in arr]
    n_breaches = sum(breach_seq)

    kupiec = kupiec_pof_test(n_breaches, len(arr), confidence_level=confidence_level)
    christoffersen = christoffersen_independence_test(breach_seq)

    # Joint verdict
    if "PASS" in kupiec.get("verdict", "") and "PASS" in christoffersen.get("verdict", ""):
        joint = "MODEL VALID"
    else:
        joint = "MODEL FAILS — recalibrate"

    return {
        "var_pct": var_pct,
        "confidence_level": confidence_level,
        "kupiec": kupiec,
        "christoffersen": christoffersen,
        "joint_verdict": joint,
    }


def main():
    """Demo: synthetic returns with mild fat tails."""
    print("=" * 70)
    print("VaR BACKTEST VALIDATION — Kupiec + Christoffersen")
    print("=" * 70)
    np.random.seed(42)
    # Returns with student-t (fat tail) distribution
    returns = stats.t.rvs(df=4, size=500) * 0.015  # 1.5% scale, df=4
    print(f"\nSynthetic returns: 500 days, std {returns.std()*100:.2f}%, "
          f"skew {stats.skew(returns):+.2f}, kurt {stats.kurtosis(returns)+3:.2f}")
    print()
    for var_pct in [0.01, 0.02, 0.03]:
        print(f"\n--- {var_pct*100:.0f}% VaR ---")
        r = backtest_var(list(returns), var_pct=var_pct, confidence_level=0.99)
        print(f"  Observed breaches: {r['kupiec']['n_breaches_observed']}  "
              f"Expected: {r['kupiec']['n_breaches_expected']:.1f}")
        print(f"  Kupiec p-value: {r['kupiec']['p_value']:.3f}  -> {r['kupiec']['verdict']}")
        print(f"  Christoffersen p-value: {r['christoffersen']['p_value']:.3f}  -> {r['christoffersen']['verdict']}")
        print(f"  JOINT: {r['joint_verdict']}")


if __name__ == "__main__":
    main()
