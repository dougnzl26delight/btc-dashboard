"""P3: Walk-forward threshold optimization.

Never optimize threshold on the full sample (curve-fitting).
Use expanding-window walk-forward.

Pass criteria:
  - threshold_stability_cov < 0.20 (stable across folds)
  - oos_consistency > 0.65 (works in most OOS periods)

A signal that fails these gates is overfit — drop it.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def _backtest_threshold(signal: pd.Series, returns: pd.Series,
                          threshold: float, horizon: int,
                          direction: str = "above") -> dict:
    """Simulate: when signal crosses threshold, hold position for `horizon` days.
    Returns hit_rate, mean_return, sharpe."""
    aligned = pd.concat([signal, returns], axis=1, join="inner").dropna()
    if len(aligned) < horizon + 50:
        return {"hit_rate": 0.0, "mean_return": 0.0, "sharpe": 0.0, "n_events": 0}
    aligned.columns = ["sig", "ret"]

    if direction == "above":
        events = aligned["sig"] > threshold
    else:
        events = aligned["sig"] < threshold

    if events.sum() < 5:
        return {"hit_rate": 0.0, "mean_return": 0.0, "sharpe": 0.0, "n_events": int(events.sum())}

    fwd_returns = aligned["ret"].rolling(horizon).sum().shift(-horizon)
    triggered_returns = fwd_returns[events].dropna()
    if len(triggered_returns) == 0:
        return {"hit_rate": 0.0, "mean_return": 0.0, "sharpe": 0.0, "n_events": int(events.sum())}

    mean = float(triggered_returns.mean())
    std = float(triggered_returns.std()) if len(triggered_returns) > 1 else 1.0
    return {
        "hit_rate": float((triggered_returns > 0).mean()),
        "mean_return": mean,
        "sharpe": (mean / std) * np.sqrt(252 / horizon) if std > 0 else 0.0,
        "n_events": int(events.sum()),
    }


def _optimize_threshold(signal: pd.Series, returns: pd.Series,
                          horizon: int, direction: str = "above",
                          n_candidates: int = 20) -> float:
    """Find threshold maximizing OOS Sharpe on training data.
    Search across percentiles 10..90 of the signal's distribution."""
    s = signal.dropna()
    if len(s) < 100: return float(s.median()) if len(s) else 0.0

    if direction == "above":
        pcts = np.linspace(50, 95, n_candidates)
    else:
        pcts = np.linspace(5, 50, n_candidates)
    candidates = [float(np.percentile(s, p)) for p in pcts]

    best = None
    best_score = -np.inf
    for thr in candidates:
        r = _backtest_threshold(signal, returns, thr, horizon, direction)
        if r["n_events"] < 5: continue
        # Prefer high sharpe AND enough events
        score = r["sharpe"] - 0.1 * max(0, 20 - r["n_events"])
        if score > best_score:
            best_score = score
            best = thr
    return best if best is not None else float(s.median())


def walk_forward_threshold(signal: pd.Series, returns: pd.Series,
                              horizon: int,
                              direction: str = "above",
                              n_folds: int = 8,
                              min_train_size: int = 252) -> dict:
    """Walk-forward fit + OOS evaluation.

    Returns recommended threshold + pass/fail gate metrics.
    """
    aligned = pd.concat([signal, returns], axis=1, join="inner").dropna()
    if len(aligned) < (min_train_size + horizon) * 2:
        return {"pass": False, "reason": "insufficient_history",
                "threshold_recommended": None,
                "threshold_stability_cov": None,
                "oos_consistency": None,
                "n_folds": 0}

    s = aligned.iloc[:, 0]
    r = aligned.iloc[:, 1]

    fold_size = (len(s) - min_train_size) // n_folds
    if fold_size < horizon * 2:
        n_folds = max(2, (len(s) - min_train_size) // (horizon * 2))
        fold_size = (len(s) - min_train_size) // n_folds

    threshold_history = []
    oos_returns = []
    oos_sharpes = []

    for fold in range(n_folds):
        train_end = min_train_size + fold_size * fold
        test_end = min_train_size + fold_size * (fold + 1)
        if test_end > len(s): break

        train_sig = s.iloc[:train_end]
        train_ret = r.iloc[:train_end]
        test_sig = s.iloc[train_end:test_end]
        test_ret = r.iloc[train_end:test_end]

        thr = _optimize_threshold(train_sig, train_ret, horizon, direction)
        threshold_history.append(thr)

        oos_perf = _backtest_threshold(test_sig, test_ret, thr, horizon, direction)
        oos_returns.append(oos_perf["mean_return"])
        oos_sharpes.append(oos_perf["sharpe"])

    if not threshold_history:
        return {"pass": False, "reason": "no_folds",
                "threshold_recommended": None,
                "threshold_stability_cov": None,
                "oos_consistency": None,
                "n_folds": 0}

    thresh_mean = float(np.mean(threshold_history))
    thresh_std = float(np.std(threshold_history))
    cov = abs(thresh_std / thresh_mean) if thresh_mean != 0 else float("inf")
    consistency = float(np.mean([rr > 0 for rr in oos_returns]))
    median_thresh = float(np.median(threshold_history))

    passes = (cov < 0.20) and (consistency > 0.65)

    return {
        "threshold_recommended": median_thresh,
        "threshold_stability_cov": cov,
        "oos_consistency": consistency,
        "mean_oos_sharpe": float(np.mean(oos_sharpes)),
        "n_folds": len(threshold_history),
        "threshold_history": threshold_history,
        "oos_returns_by_fold": oos_returns,
        "pass": bool(passes),
        "reason": ("ok" if passes
                   else "unstable_threshold" if cov >= 0.20
                   else "inconsistent_oos"),
    }


def validate_all_signals(signals_dict: dict[str, pd.Series],
                            returns: pd.Series,
                            horizon: int = 126,
                            direction_map: Optional[dict] = None) -> pd.DataFrame:
    """Run walk-forward validation on a batch of signals."""
    direction_map = direction_map or {}
    rows = []
    for name, sig in signals_dict.items():
        direction = direction_map.get(name, "above")
        r = walk_forward_threshold(sig, returns, horizon=horizon, direction=direction)
        rows.append({
            "signal": name,
            "pass": r.get("pass"),
            "threshold": r.get("threshold_recommended"),
            "cov_stability": r.get("threshold_stability_cov"),
            "oos_consistency": r.get("oos_consistency"),
            "mean_oos_sharpe": r.get("mean_oos_sharpe"),
            "n_folds": r.get("n_folds"),
            "reason": r.get("reason"),
        })
    return pd.DataFrame(rows).sort_values("mean_oos_sharpe", ascending=False)


def main():
    rng = np.random.default_rng(11)
    n = 2000
    idx = pd.date_range("2018-01-01", periods=n)
    # Mean-reverting signal that predicts positive returns when low
    sig = pd.Series(np.cumsum(rng.normal(0, 1, n)) +
                    np.sin(np.arange(n) / 100) * 10, index=idx)
    # Returns negatively correlated with signal at h=60
    rets = -sig.shift(60).pct_change(5).fillna(0) / 10 + rng.normal(0, 0.005, n)
    rets = pd.Series(rets, index=idx)

    r = walk_forward_threshold(sig, rets, horizon=60, direction="below")
    print("Walk-forward smoke test:")
    for k, v in r.items():
        if k in ("threshold_history", "oos_returns_by_fold"):
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
