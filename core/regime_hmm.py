"""P5: HMM validator for the rule-based regime classifier.

Fit a 3-state Gaussian HMM on theme z-scores. Compare its regime
labels to the rule-based classifier. If they disagree more than 20%
of the time, the rule-based rules are mis-calibrated.

Mapping HMM states -> regime labels: sort states by historical SPY
return; lowest = RECESSIONARY_BEAR, highest = RISK_ON.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _try_hmm():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError:
        return None


def fit_hmm_regimes(theme_zs_df: pd.DataFrame, n_states: int = 3,
                      n_iter: int = 200, random_state: int = 42) -> dict:
    """Fit a Gaussian HMM on theme z-score timeseries.

    Args:
      theme_zs_df: DataFrame of theme z-scores (cols = theme names,
                   rows = dates)

    Returns dict with:
      states: array of regime label per date
      transition_matrix: 3x3 transition probs
      means: per-state means of features
      n_iter_actual, converged
    """
    GaussianHMM = _try_hmm()
    if GaussianHMM is None:
        return {"available": False,
                 "reason": "hmmlearn not installed (pip install hmmlearn)"}

    X = theme_zs_df.dropna().values
    if len(X) < 200:
        return {"available": False, "reason": "insufficient_history"}

    try:
        model = GaussianHMM(n_components=n_states,
                              covariance_type="full",
                              n_iter=n_iter,
                              random_state=random_state)
        model.fit(X)
        states = model.predict(X)
        return {
            "available": True,
            "states": states,
            "transition_matrix": model.transmat_.tolist(),
            "means": model.means_.tolist(),
            "covariances": model.covars_.tolist(),
            "n_states": n_states,
            "dates": theme_zs_df.dropna().index.tolist(),
        }
    except Exception as e:
        return {"available": False, "reason": f"hmm_fit_error: {e!r}"[:80]}


def map_states_to_regimes(states: np.ndarray, dates: list,
                            returns: pd.Series) -> dict[int, str]:
    """Map HMM state IDs to RISK_ON / LATE_CYCLE / RECESSIONARY_BEAR
    based on mean SPY return when in each state.

    Lowest mean return = RECESSIONARY_BEAR; highest = RISK_ON.
    """
    if returns is None or len(returns) == 0:
        # Fallback: just label by state index
        return {s: ["RECESSIONARY_BEAR", "LATE_CYCLE", "RISK_ON"][s]
                for s in range(3)}

    states_series = pd.Series(states, index=pd.DatetimeIndex(dates))
    aligned = pd.concat([states_series, returns], axis=1,
                         join="inner").dropna()
    aligned.columns = ["state", "ret"]
    means = aligned.groupby("state")["ret"].mean().sort_values()
    sorted_ids = means.index.tolist()

    labels = ["RECESSIONARY_BEAR", "LATE_CYCLE", "RISK_ON"]
    mapping = {}
    for i, sid in enumerate(sorted_ids):
        mapping[int(sid)] = labels[min(i, 2)]
    return mapping


def validate_against_hmm(rule_regimes: pd.Series,
                            theme_zs_df: pd.DataFrame,
                            returns: pd.Series) -> dict:
    """Compute agreement rate between rule-based and HMM regimes.

    Returns dict with agreement_rate + pass flag (>0.80 = pass).
    """
    hmm_result = fit_hmm_regimes(theme_zs_df)
    if not hmm_result.get("available"):
        return {"agreement_rate": None, "pass": None,
                "reason": hmm_result.get("reason")}

    mapping = map_states_to_regimes(hmm_result["states"],
                                       hmm_result["dates"], returns)
    hmm_labels = pd.Series(
        [mapping.get(int(s), "UNKNOWN") for s in hmm_result["states"]],
        index=pd.DatetimeIndex(hmm_result["dates"])
    )
    aligned = pd.concat([rule_regimes, hmm_labels], axis=1,
                         join="inner").dropna()
    aligned.columns = ["rule", "hmm"]
    if len(aligned) < 30:
        return {"agreement_rate": None, "pass": None,
                "reason": "insufficient_overlap"}

    agreement = float((aligned["rule"] == aligned["hmm"]).mean())
    confusion = pd.crosstab(aligned["rule"], aligned["hmm"])

    return {
        "agreement_rate": agreement,
        "pass": agreement > 0.80,
        "n_observations": len(aligned),
        "confusion_matrix": confusion.to_dict(),
        "hmm_state_mapping": mapping,
        "hmm_transition_matrix": hmm_result["transition_matrix"],
    }


def regime_return_segmentation(regimes: pd.Series,
                                  returns: dict[str, pd.Series]) -> dict:
    """Show that regimes meaningfully separate asset returns.

    PASS CRITERION: Sharpe differential > 0.5 between RISK_ON and
    RECESSIONARY_BEAR for SPY.
    """
    rows = []
    for asset, ret in returns.items():
        if ret is None: continue
        for regime in ("RISK_ON", "LATE_CYCLE", "RECESSIONARY_BEAR"):
            mask = regimes == regime
            aligned = pd.concat([mask, ret], axis=1, join="inner").dropna()
            if aligned.iloc[:, 0].sum() < 30: continue
            r = aligned[aligned.iloc[:, 0]].iloc[:, 1]
            if r.std() == 0 or len(r) < 30: continue
            rows.append({
                "asset": asset, "regime": regime,
                "n_days": int(aligned.iloc[:, 0].sum()),
                "mean_annual_pct": float(r.mean() * 252 * 100),
                "vol_annual_pct": float(r.std() * np.sqrt(252) * 100),
                "sharpe": float((r.mean() / r.std()) * np.sqrt(252)),
                "min_return": float(r.min()),
                "max_return": float(r.max()),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"segmentation": df, "pass": False, "reason": "no_data"}

    # Pass: SPY Sharpe(RISK_ON) - SPY Sharpe(RECESSIONARY_BEAR) > 0.5
    spy = df[df["asset"] == "SPY"].set_index("regime")["sharpe"]
    if "RISK_ON" in spy.index and "RECESSIONARY_BEAR" in spy.index:
        sharpe_diff = float(spy["RISK_ON"] - spy["RECESSIONARY_BEAR"])
        passes = sharpe_diff > 0.5
    else:
        sharpe_diff = None
        passes = False

    return {
        "segmentation": df.to_dict(orient="records"),
        "spy_sharpe_diff": sharpe_diff,
        "pass": passes,
        "reason": "ok" if passes else ("insufficient_regimes"
                                        if sharpe_diff is None else "weak_segmentation"),
    }


def main():
    GaussianHMM = _try_hmm()
    print(f"HMM available: {GaussianHMM is not None}")
    if GaussianHMM is None:
        print("Install: pip install hmmlearn")
        return

    rng = np.random.default_rng(13)
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n)
    # 3 regimes with different distributions
    regimes_true = np.repeat([0, 1, 2, 1, 0], n // 5)[:n]
    features = np.array([
        [rng.normal(2 if r == 2 else (-1 if r == 0 else 0.5), 0.5),
         rng.normal(-1 if r == 0 else (0.5 if r == 2 else 0), 0.5),
         rng.normal(-1 if r == 0 else (0.5 if r == 2 else 0), 0.5)]
        for r in regimes_true
    ])
    df = pd.DataFrame(features, index=idx,
                       columns=["LIQUIDITY", "GROWTH", "CREDIT"])
    rets = pd.Series(
        [rng.normal(0.001 if r == 2 else (-0.001 if r == 0 else 0.0), 0.01)
         for r in regimes_true], index=idx)
    rule_labels = pd.Series(
        ["RISK_ON" if r == 2 else ("RECESSIONARY_BEAR" if r == 0 else "LATE_CYCLE")
         for r in regimes_true], index=idx)

    v = validate_against_hmm(rule_labels, df, rets)
    print("HMM validation smoke test:")
    for k, v_ in v.items():
        if k in ("confusion_matrix", "hmm_transition_matrix"): continue
        print(f"  {k}: {v_}")


if __name__ == "__main__":
    main()
