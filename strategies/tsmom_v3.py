"""TSMOM v3 — v1 ensemble signal + GARCH vol-targeting + drawdown scaling.

Built on the v2 finding: discrete entries lose to retail-grade costs.
v3 keeps continuous weights but layers in two SMOOTH practitioner adjustments
that don't add round-trip cost events:

  1. GARCH(1,1) conditional vol targeting (Engle 1982, Bollerslev 1986)
     Replaces naive realized vol in the magnitude scaling. Adapts faster
     to vol regime changes than a 60-day rolling window.

  2. Drawdown-based position scaling (Carver Systematic Trading 2015 Ch.9)
     Continuously scales exposure down as drawdown grows; full kill at 30%.

Both adjustments are continuous and don't fire extra trades, so cost drag
should match v1 (not v2's blowup).

VALIDATED = False. Counts as trial 47 in the multiple-testing budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, data, deflated_sharpe, evidence, factor_decomp, garch_vol
from research import signals as res_sig


VALIDATED = False
NAME = "tsmom_v3"
TRIALS_CONSIDERED = 47

TARGET_VOL_ANN = 0.20
KINK_DD = 0.10
KILL_DD = 0.30


def _ensemble_base_signal(btc: pd.Series, eth_btc: pd.Series) -> pd.Series:
    """Same composition as v1: TSMOM(30,90) + ETH/BTC reversion(20), equal weight."""
    s_mom = res_sig.tsmom_multi(btc, horizons=(30, 90))
    s_revert = res_sig.zscore_revert(
        eth_btc.reindex(btc.index).ffill().bfill(), window=20
    )
    return ((s_mom + s_revert) / 2).fillna(0).clip(-1, 1)


def _vol_scalar_series(btc: pd.Series, target_vol_ann: float = TARGET_VOL_ANN) -> pd.Series:
    """target_vol / GARCH-conditional-vol, clipped to [0, 1].

    Uses full-sample GARCH(1,1) fit; conditional vol at each point uses
    only past data (no information leakage in the vol estimate itself,
    minor parameter leakage). Acceptable for first-pass evaluation; for
    final validation, rolling refit is preferable.
    """
    log_ret = np.log(btc / btc.shift(1)).dropna()
    cond_vol = garch_vol.garch_conditional_vol(log_ret)
    cond_vol = cond_vol.reindex(btc.index).ffill().bfill()
    return (target_vol_ann / cond_vol).clip(0, 1)


def latest_signal(pair: str = "BTC/USDT") -> float:
    """BTC-specific (uses ETH/BTC ratio inside); returns 0 for other pairs."""
    if pair != "BTC/USDT":
        return 0.0
    btc = data.ohlcv_extended(pair, days_back=540)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=540)["close"]
    eth_btc = (eth / btc).dropna()
    base = _ensemble_base_signal(btc, eth_btc)
    try:
        vol_s = _vol_scalar_series(btc)
        return float((base * vol_s).iloc[-1])
    except Exception:
        return float(base.iloc[-1])


def evaluate_strict(pair: str = "BTC/USDT", days_back: int = 2000) -> dict:
    btc = data.ohlcv_extended(pair, days_back=days_back)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=days_back)["close"]
    eth_btc = (eth / btc).dropna()
    bench = btc.pct_change().fillna(0)

    base_signal = _ensemble_base_signal(btc, eth_btc)
    vol_scalar = _vol_scalar_series(btc)

    bt = backtest.run_path_dependent(
        btc,
        base_signal,
        starting_equity=100_000.0,
        use_dd_scaling=True,
        kink_dd=KINK_DD,
        kill_dd=KILL_DD,
        vol_target_series=vol_scalar,
    )
    summary = backtest.summarize(bt)
    decomp = factor_decomp.decompose(bt["ret"], bench)

    # Hurdle on the strategy returns (path-dependent, so use full-sample stats
    # rather than walk-forward for now — a proper rolling-refit evaluation is
    # backlog work)
    active = bt["ret"][bt["ret"] != 0]
    hurdle = (
        deflated_sharpe.passes_quant_hurdle(active, num_trials=TRIALS_CONSIDERED)
        if len(active) >= 30
        else {"passes_combined": False}
    )

    validated = (
        decomp.get("passes_alpha_t", False) and hurdle.get("passes_combined", False)
    )

    result = {
        "summary": summary,
        "factor_decomp": decomp,
        "hurdle_oos": hurdle,
        "validated": validated,
        "vol_scalar_stats": {
            "mean": float(vol_scalar.mean()),
            "min": float(vol_scalar.min()),
            "max": float(vol_scalar.max()),
            "frac_full_size": float((vol_scalar >= 0.99).mean()),
        },
    }
    evidence.record(NAME, f"strict eval {pair}", result)
    return result


if __name__ == "__main__":
    import json

    print("=== STRICT EVALUATION (GARCH vol-target + DD scaling) ===")
    print(json.dumps(evaluate_strict(), indent=2, default=str))
