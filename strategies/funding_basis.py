"""Funding-rate contrarian-sentiment overlay strategy.

VALIDATED = False. Spot-only sentiment tilt; full basis arb (perp + spot)
deferred until broker abstraction supports both legs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, cv, data, deflated_sharpe, evidence, factor_decomp
from signals.funding_basis import funding_signal, funding_signal_series


VALIDATED = False
NAME = "funding_basis"


def latest_signal(pair: str = "BTC/USDT", perp_pair: str = "BTC/USDT:USDT") -> float:
    return funding_signal(perp_pair)


def evaluate_strict(
    pair: str = "BTC/USDT", perp_pair: str = "BTC/USDT:USDT"
) -> dict:
    """Validation-grade evaluation: walk-forward CV, factor decomp vs spot,
    plus the deflated-Sharpe hurdle on OOS-concatenated returns.
    """
    df = data.ohlcv(pair, timeframe="1d", limit=1000)
    bench_returns = df["close"].pct_change().fillna(0.0)

    fund_series = funding_signal_series(perp_pair=perp_pair, days_back=1000)
    if fund_series.empty:
        result = {
            "validated": False,
            "reason": "no funding history available",
        }
        evidence.record(NAME, f"strict eval {pair} (skipped)", result)
        return result

    # Reindex funding signal to align with daily price bars
    aligned = fund_series.reindex(df.index, method="ffill").fillna(0)
    n_real = (aligned != 0).sum()
    if n_real < 365:
        result = {
            "validated": False,
            "reason": f"only {n_real} days of aligned funding data; need >= 365",
        }
        evidence.record(NAME, f"strict eval {pair} (insufficient data)", result)
        return result

    def signal_fn(_prices):
        # Funding signal is independent of prices; reuse the precomputed alignment
        return aligned

    wf = cv.walk_forward(df["close"], signal_fn=signal_fn, n_folds=5, min_train=365)

    full_bt = backtest.run(df["close"], aligned)
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
        "n_aligned_days": int(n_real),
    }
    evidence.record(NAME, f"strict eval {pair}", result)
    return result


if __name__ == "__main__":
    import json
    print("=== STRICT EVALUATION (walk-forward + factor decomp) ===")
    print(json.dumps(evaluate_strict(), indent=2, default=str))
