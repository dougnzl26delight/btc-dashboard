"""Diverse-signal ensemble: TSMOM(30,90) + ETH/BTC z-score reversion(20).

Selected as production candidate from a 45-trial research sweep:
  - Best risk-adjusted ensemble found: OOS Sharpe 0.86, alpha_t 1.65, beta 0.012
  - Component correlation: -0.10 (genuinely orthogonal axes)
  - Pure alpha (beta to BTC near zero) — not levered market exposure

VALIDATED = False until evaluate_strict() returns passes_combined=True.
The orchestrator will NOT trade live with this strategy until that flag flips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, cv, data, deflated_sharpe, evidence, factor_decomp
from research import signals as res_sig


VALIDATED = False
NAME = "diverse_mom_ethbtc"

MOMENTUM_HORIZONS = (30, 90)
ETHBTC_ZSCORE_WINDOW = 20

# Total candidates considered across the full research search. Used for the
# multiple-testing penalty in deflated Sharpe. Honest accounting matters.
TRIALS_CONSIDERED = 45


def _ensemble_signal(btc_prices: pd.Series, eth_btc_ratio: pd.Series) -> pd.Series:
    """Equal-weighted ensemble of TSMOM(30,90) and ETH/BTC zscore reversion(20)."""
    s_mom = res_sig.tsmom_multi(btc_prices, horizons=MOMENTUM_HORIZONS)
    s_revert = res_sig.zscore_revert(
        eth_btc_ratio.reindex(btc_prices.index).ffill().bfill(),
        window=ETHBTC_ZSCORE_WINDOW,
    )
    return ((s_mom + s_revert) / 2).fillna(0).clip(-1, 1)


def latest_signal(pair: str = "BTC/USDT") -> float:
    """BTC-specific (uses ETH/BTC ratio); returns 0 for other pairs to avoid
    misapplying the BTC signal across the universe."""
    if pair != "BTC/USDT":
        return 0.0
    btc = data.ohlcv_extended(pair, days_back=540)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=540)["close"]
    eth_btc = (eth / btc).dropna()
    sig = _ensemble_signal(btc, eth_btc)
    return float(sig.iloc[-1]) if not sig.empty else 0.0


def evaluate_strict(pair: str = "BTC/USDT", days_back: int = 2000) -> dict:
    """Walk-forward + factor decomp + DSR/t hurdle. Logs to evidence ledger."""
    btc = data.ohlcv_extended(pair, days_back=days_back)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=days_back)["close"]
    eth_btc = (eth / btc).dropna()
    bench = btc.pct_change().fillna(0)

    def signal_fn(prices: pd.Series) -> pd.Series:
        return _ensemble_signal(prices, eth_btc)

    wf = cv.walk_forward(btc, signal_fn=signal_fn, n_folds=5, min_train=365)
    bt = backtest.run(btc, signal_fn(btc))
    decomp = factor_decomp.decompose(bt["ret"], bench)

    hurdle = (
        deflated_sharpe.passes_quant_hurdle(
            wf["concatenated_returns"], num_trials=TRIALS_CONSIDERED
        )
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
        "components": {
            "momentum_horizons": list(MOMENTUM_HORIZONS),
            "ethbtc_zscore_window": ETHBTC_ZSCORE_WINDOW,
        },
        "trials_considered": TRIALS_CONSIDERED,
    }
    evidence.record(NAME, f"strict eval {pair}", result)
    return result


if __name__ == "__main__":
    import json

    print("=== latest signal ===")
    print(f"  {NAME}: {latest_signal():+.4f}")
    print("\n=== strict evaluation ===")
    print(json.dumps(evaluate_strict(), indent=2, default=str))
