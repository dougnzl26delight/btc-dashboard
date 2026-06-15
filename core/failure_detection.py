"""P10: Engine failure detection.

The engine knows when it's wrong. If any failure criterion fires, the
appropriate action is taken: halt rebalancing, reduce positions, demote
low-IC signals, or trigger full recalibration.

The user is alerted via the regime alert channel.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
FAILURE_LOG = REPO_ROOT / ".engine_failures.json"


# ============================================================
# Failure criteria definitions
# ============================================================

FAILURE_CRITERIA = {
    "underperform_benchmark_12m": {
        "description": "Engine 12m Sharpe vs 60/40 benchmark",
        "threshold_diff": -0.30,  # if engine - bench < -0.30
        "action":         "halt_rebalancing_pending_recalibration",
        "severity":       "critical",
    },
    "missed_btc_upcycle": {
        "description": "Capture rate of BTC 100%+ rally",
        "threshold":     0.50,  # captured less than 50%
        "action":         "review_bottom_composite_thresholds",
        "severity":       "high",
    },
    "drawdown_exceeded": {
        "description": "Realized 1y drawdown",
        "threshold":    -0.15,
        "action":         "reduce_all_positions_50pct",
        "severity":       "critical",
    },
    "regime_misclassification": {
        "description": "HMM vs rule-based agreement",
        "threshold":    0.70,
        "action":         "rerun_regime_calibration",
        "severity":       "high",
    },
    "signal_ic_decay": {
        "description": "Average composite IC over rolling 24m",
        "threshold":    0.02,
        "action":         "demote_low_ic_signals_zero_weight",
        "severity":       "medium",
    },
    "regime_thrashing": {
        "description": "Regime transitions per 90d window",
        "threshold":    5,  # more than 5 transitions = thrashing
        "action":         "increase_regime_persistence_window",
        "severity":       "medium",
    },
}


# ============================================================
# Individual checks
# ============================================================

def check_underperform_benchmark(engine_returns: pd.Series,
                                    bench_returns: pd.Series,
                                    lookback: int = 252) -> dict:
    """Engine 12m Sharpe vs 60/40 benchmark."""
    if engine_returns is None or bench_returns is None:
        return {"fires": False, "reason": "no_data"}
    e = engine_returns.tail(lookback).dropna()
    b = bench_returns.tail(lookback).dropna()
    if len(e) < 100 or len(b) < 100:
        return {"fires": False, "reason": "insufficient_history"}
    e_sharpe = float((e.mean() / e.std()) * np.sqrt(252)) if e.std() > 0 else 0
    b_sharpe = float((b.mean() / b.std()) * np.sqrt(252)) if b.std() > 0 else 0
    diff = e_sharpe - b_sharpe
    fires = diff < FAILURE_CRITERIA["underperform_benchmark_12m"]["threshold_diff"]
    return {
        "fires": bool(fires),
        "engine_sharpe": e_sharpe,
        "bench_sharpe": b_sharpe,
        "diff": diff,
        "action": FAILURE_CRITERIA["underperform_benchmark_12m"]["action"]
                   if fires else None,
    }


def check_drawdown_exceeded(equity_curve: pd.Series,
                              lookback: int = 252) -> dict:
    if equity_curve is None or len(equity_curve) < 30:
        return {"fires": False, "reason": "no_data"}
    s = pd.Series(equity_curve).tail(lookback).dropna()
    dd = float(s.iloc[-1] / s.cummax().iloc[-1] - 1)
    fires = dd < FAILURE_CRITERIA["drawdown_exceeded"]["threshold"]
    return {
        "fires": bool(fires),
        "current_dd": dd,
        "threshold": FAILURE_CRITERIA["drawdown_exceeded"]["threshold"],
        "action": FAILURE_CRITERIA["drawdown_exceeded"]["action"] if fires else None,
    }


def check_regime_misclassification(agreement_rate: Optional[float]) -> dict:
    if agreement_rate is None:
        return {"fires": False, "reason": "hmm_unavailable"}
    fires = agreement_rate < FAILURE_CRITERIA["regime_misclassification"]["threshold"]
    return {
        "fires": bool(fires),
        "agreement_rate": agreement_rate,
        "threshold": FAILURE_CRITERIA["regime_misclassification"]["threshold"],
        "action": FAILURE_CRITERIA["regime_misclassification"]["action"]
                   if fires else None,
    }


def check_signal_ic_decay(rolling_24m_ic: Optional[float]) -> dict:
    if rolling_24m_ic is None:
        return {"fires": False, "reason": "no_ic_data"}
    fires = rolling_24m_ic < FAILURE_CRITERIA["signal_ic_decay"]["threshold"]
    return {
        "fires": bool(fires),
        "rolling_24m_ic": rolling_24m_ic,
        "threshold": FAILURE_CRITERIA["signal_ic_decay"]["threshold"],
        "action": FAILURE_CRITERIA["signal_ic_decay"]["action"] if fires else None,
    }


def check_regime_thrashing(regime_history: pd.Series,
                              window_days: int = 90) -> dict:
    if regime_history is None or len(regime_history) < window_days:
        return {"fires": False, "reason": "insufficient_history"}
    recent = regime_history.tail(window_days)
    transitions = int((recent != recent.shift()).sum())
    fires = transitions > FAILURE_CRITERIA["regime_thrashing"]["threshold"]
    return {
        "fires": bool(fires),
        "transitions_90d": transitions,
        "threshold": FAILURE_CRITERIA["regime_thrashing"]["threshold"],
        "action": FAILURE_CRITERIA["regime_thrashing"]["action"] if fires else None,
    }


def check_missed_btc_upcycle(engine_btc_weights: pd.Series,
                                btc_returns: pd.Series,
                                rally_threshold: float = 1.0) -> dict:
    """Find rallies of >100% in BTC and check if engine held BTC during them."""
    if engine_btc_weights is None or btc_returns is None:
        return {"fires": False, "reason": "no_data"}
    if len(btc_returns) < 252: return {"fires": False, "reason": "insufficient"}

    # Detect rallies: 252-day forward return > threshold
    fwd_ret = (1 + btc_returns).rolling(252).apply(lambda x: x.prod() - 1).shift(-252)
    rally_starts = fwd_ret[fwd_ret > rally_threshold].index
    if rally_starts.empty: return {"fires": False, "reason": "no_rallies_found"}

    captures = []
    for start in rally_starts:
        try:
            w = engine_btc_weights.loc[start:start + pd.Timedelta(days=252)]
            captures.append(float(w.mean()))
        except Exception:
            continue
    if not captures: return {"fires": False, "reason": "no_captures"}

    avg_capture = float(np.mean(captures))
    fires = avg_capture < FAILURE_CRITERIA["missed_btc_upcycle"]["threshold"]
    return {
        "fires": bool(fires),
        "avg_btc_weight_during_rallies": avg_capture,
        "n_rallies": len(rally_starts),
        "threshold": FAILURE_CRITERIA["missed_btc_upcycle"]["threshold"],
        "action": FAILURE_CRITERIA["missed_btc_upcycle"]["action"] if fires else None,
    }


# ============================================================
# Top-level check
# ============================================================

def run_all_failure_checks(engine_returns: Optional[pd.Series] = None,
                             bench_returns: Optional[pd.Series] = None,
                             equity_curve: Optional[pd.Series] = None,
                             regime_history: Optional[pd.Series] = None,
                             hmm_agreement: Optional[float] = None,
                             rolling_24m_ic: Optional[float] = None,
                             engine_btc_weights: Optional[pd.Series] = None,
                             btc_returns: Optional[pd.Series] = None) -> dict:
    """Run all failure checks, return list of active failures."""
    checks = {
        "underperform_benchmark_12m":
            check_underperform_benchmark(engine_returns, bench_returns),
        "drawdown_exceeded":
            check_drawdown_exceeded(equity_curve),
        "regime_misclassification":
            check_regime_misclassification(hmm_agreement),
        "signal_ic_decay":
            check_signal_ic_decay(rolling_24m_ic),
        "regime_thrashing":
            check_regime_thrashing(regime_history),
        "missed_btc_upcycle":
            check_missed_btc_upcycle(engine_btc_weights, btc_returns),
    }
    active_failures = [
        {"name": k, **v, "severity": FAILURE_CRITERIA[k]["severity"]}
        for k, v in checks.items() if v.get("fires")
    ]

    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "active_failures": active_failures,
        "n_failures": len(active_failures),
        "highest_severity": (max((f["severity"] for f in active_failures),
                                  key=lambda s: ["medium", "high", "critical"].index(s))
                              if active_failures else None),
        "checks_run": {k: v.get("fires") for k, v in checks.items()},
        "halt_required": any(f.get("action") == "halt_rebalancing_pending_recalibration"
                              for f in active_failures),
    }


def log_failure(failure_result: dict) -> None:
    """Persist failure detection result to disk."""
    try:
        history = []
        if FAILURE_LOG.exists():
            history = json.loads(FAILURE_LOG.read_text())
        history.append(failure_result)
        history = history[-100:]  # keep last 100
        FAILURE_LOG.write_text(json.dumps(history, indent=2, default=str))
    except Exception:
        pass


def main():
    print("Failure detection smoke test (synthetic):")
    rng = np.random.default_rng(7)
    n = 300
    eng = pd.Series(rng.normal(0.0002, 0.012, n))   # mildly profitable
    bench = pd.Series(rng.normal(0.0008, 0.010, n)) # benchmark better
    equity = (1 + eng).cumprod() * 100

    r = run_all_failure_checks(
        engine_returns=eng, bench_returns=bench,
        equity_curve=equity,
        hmm_agreement=0.75,
        rolling_24m_ic=0.025,
    )
    print(f"  Active failures: {r['n_failures']}")
    for f in r["active_failures"]:
        print(f"    [{f['severity']:8s}] {f['name']}: action={f.get('action')}")
    print(f"  Halt required: {r['halt_required']}")


if __name__ == "__main__":
    main()
