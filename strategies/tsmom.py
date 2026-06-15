"""TSMOM strategy: vol-targeted long/flat on trailing 60-day momentum.

VALIDATED = False. Will not run in live mode until evaluate() returns
passes_combined=True (DSR > 0.95 AND |t-stat| > 3.0).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, cv, data, deflated_sharpe, evidence, factor_decomp
from signals.tsmom import tsmom_signal, tsmom_signal_multi


VALIDATED = False
NAME = "tsmom"


def evaluate(pair: str = "BTC/USDT", lookback_days: int = 60, num_trials: int = 20) -> dict:
    """Single-horizon in-sample evaluation. Useful baseline; not validation-grade."""
    df = data.ohlcv(pair, timeframe="1d", limit=730)
    sig = tsmom_signal(df["close"], lookback_days=lookback_days)
    bt = backtest.run(df["close"], sig)
    summary = backtest.summarize(bt)
    hurdle = deflated_sharpe.passes_quant_hurdle(bt["ret"].dropna(), num_trials=num_trials)
    evidence.record(
        NAME,
        f"in-sample {pair} lookback={lookback_days}",
        {"summary": summary, "hurdle": hurdle},
    )
    return {"summary": summary, "hurdle": hurdle, "validated": hurdle["passes_combined"]}


def evaluate_strict(pair: str = "BTC/USDT") -> dict:
    """Validation-grade evaluation: multi-horizon signal, walk-forward CV,
    factor decomposition vs benchmark, plus the deflated-Sharpe hurdle on
    OOS-concatenated returns. This is what `VALIDATED = True` requires.
    """
    df = data.ohlcv(pair, timeframe="1d", limit=1000)
    bench_returns = df["close"].pct_change().fillna(0.0)

    wf = cv.walk_forward(
        df["close"],
        signal_fn=lambda p: tsmom_signal_multi(p),
        n_folds=5,
        min_train=365,
    )

    full_sig = tsmom_signal_multi(df["close"])
    full_bt = backtest.run(df["close"], full_sig)
    decomp = factor_decomp.decompose(full_bt["ret"], bench_returns)

    hurdle = (
        deflated_sharpe.passes_quant_hurdle(wf["concatenated_returns"], num_trials=20)
        if wf.get("n_folds", 0) > 0 and len(wf.get("concatenated_returns", [])) >= 30
        else {"passes_combined": False, "reason": "insufficient OOS data"}
    )

    validated = (
        wf.get("passes", False)
        and decomp.get("passes_alpha_t", False)
        and hurdle.get("passes_combined", False)
    )

    result = {
        "walk_forward": {k: v for k, v in wf.items() if k != "concatenated_returns"},
        "factor_decomp": decomp,
        "hurdle_oos": hurdle,
        "validated": validated,
    }
    evidence.record(NAME, f"strict eval {pair}", result)
    return result


def latest_signal(pair: str = "BTC/USDT") -> float:
    """Latest multi-horizon signal as a target weight in [-1, 1]."""
    df = data.ohlcv(pair, timeframe="1d", limit=540)
    sig = tsmom_signal_multi(df["close"])
    return float(sig.iloc[-1]) if not sig.empty else 0.0


if __name__ == "__main__":
    import json
    print("=== STRICT EVALUATION (walk-forward + factor decomp) ===")
    print(json.dumps(evaluate_strict(), indent=2, default=str))
