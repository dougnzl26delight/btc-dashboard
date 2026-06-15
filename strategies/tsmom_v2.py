"""TSMOM v2 — event-driven entries with triple-barrier exits.

Built on the v1 finding: TSMOM signal under triple-barrier exits showed
62% win rate, 4-day average hold. The fixed-weight v1 backtest was diluting
real short-term predictive power by holding too long.

v2 differences from v1:
  - Entry on signal sign-change (not continuous weight)
  - Exit via triple-barrier (profit / stop / horizon)
  - Position sizing via fractional Kelly (Carver 2015 default)
  - Daily-MTM equivalent backtest for fair Sharpe comparison

VALIDATED = False. This is a NEW research candidate. Counts as one
additional trial against the multiple-testing budget — accumulated to 46.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, deflated_sharpe, evidence, exits, factor_decomp, sizing
from research import signals as res_sig


VALIDATED = False
NAME = "tsmom_v2"
TRIALS_CONSIDERED = 46

# Triple-barrier params (from López de Prado AFML 3.3 — symmetric profit/stop)
HORIZON_DAYS = 30
PT_SIGMA = 2.0
SL_SIGMA = 1.5  # asymmetric: tighter stop than target (positive expectancy bias)
VOL_WINDOW = 30


def _events_from_signal(signal: pd.Series) -> pd.Series:
    """Convert a continuous signal into sparse sign-change events."""
    sign = pd.Series(0.0, index=signal.index)
    sign[signal > 0] = 1.0
    sign[signal < 0] = -1.0
    prev = sign.shift(1).fillna(0.0)
    events = pd.Series(0.0, index=signal.index)
    events[(sign == 1.0) & (prev != 1.0)] = 1.0
    events[(sign == -1.0) & (prev != -1.0)] = -1.0
    return events


def latest_signal(pair: str = "BTC/USDT") -> float:
    """For orchestrator wiring: returns the underlying continuous signal.
    The orchestrator's portfolio combiner handles entry/exit logic.
    """
    btc = data.ohlcv_extended(pair, days_back=540)["close"]
    return float(res_sig.tsmom_multi(btc, horizons=(30, 90)).iloc[-1])


def evaluate_strict(pair: str = "BTC/USDT", days_back: int = 2000) -> dict:
    """Run v2 evaluation: triple-barrier exits + fractional Kelly sizing."""
    btc = data.ohlcv_extended(pair, days_back=days_back)["close"]
    primary = res_sig.tsmom_multi(btc, horizons=(30, 90))
    events = _events_from_signal(primary)

    barriers = exits.triple_barrier(
        btc,
        events,
        horizon_days=HORIZON_DAYS,
        pt_sigma=PT_SIGMA,
        sl_sigma=SL_SIGMA,
        vol_window=VOL_WINDOW,
    )
    if barriers.empty:
        return {"validated": False, "reason": "no events fired"}

    per_trade = exits.event_metrics(barriers, observation_days=len(btc))
    daily_pnl = exits.barriers_to_daily_returns(barriers, btc, cost_bps_per_side=15)

    # Daily-MTM Sharpe (comparable to v1's continuous-weight Sharpe)
    daily_clean = daily_pnl[daily_pnl != 0]
    if len(daily_clean) > 30 and daily_clean.std() > 0:
        sharpe_mtm = float(
            daily_pnl.mean() / daily_pnl[daily_pnl != 0].std() * np.sqrt(365)
        )
    else:
        sharpe_mtm = 0.0

    # Fractional Kelly size based on per-trade returns
    kelly_size = sizing.fractional_kelly_size(
        expected_return_ann=per_trade["annualized_return"],
        expected_variance_ann=(barriers["return"].std() ** 2) * per_trade["trades_per_year"],
        fraction=0.25,
        max_size=0.20,
    )

    # Factor decomp on daily P&L vs BTC benchmark
    bench = btc.pct_change().fillna(0)
    decomp = factor_decomp.decompose(daily_pnl, bench)

    # Hurdle on daily P&L (use only non-zero positions periods for fair stats)
    active_returns = daily_pnl[daily_pnl != 0]
    hurdle = (
        deflated_sharpe.passes_quant_hurdle(
            active_returns, num_trials=TRIALS_CONSIDERED
        )
        if len(active_returns) >= 30
        else {"passes_combined": False, "reason": "insufficient active observations"}
    )

    validated = decomp.get("passes_alpha_t", False) and hurdle.get("passes_combined", False)

    result = {
        "per_trade": per_trade,
        "daily_mtm_sharpe": sharpe_mtm,
        "fractional_kelly_size": kelly_size,
        "factor_decomp": decomp,
        "hurdle_oos": hurdle,
        "validated": validated,
        "params": {
            "horizon_days": HORIZON_DAYS,
            "pt_sigma": PT_SIGMA,
            "sl_sigma": SL_SIGMA,
            "vol_window": VOL_WINDOW,
        },
    }
    evidence.record(NAME, f"strict eval {pair}", result)
    return result


if __name__ == "__main__":
    import json

    print("=== STRICT EVALUATION (triple-barrier + fractional Kelly) ===")
    print(json.dumps(evaluate_strict(), indent=2, default=str))
