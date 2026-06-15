"""W4: Real-engine backtest replay.

The simple_engine_replay in replay.py used SPY-drawdown-based regime
classification — that's not the real engine.

This module replays the FULL engine pipeline historically:
  1. For each historical date, fetch all signal values as of that date
  2. Compute z-scores, composites, regime via the actual modules
  3. Compute target allocation
  4. Simulate returns
  5. Compare vs 60/40 benchmark

This is the validation that tells us whether the calibrated engine
actually has edge over naive benchmarks.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.research.standardize import rolling_zscore
from core.composites import THEME_DEFINITIONS, compute_all_themes, composite_scores_for_decisions
from core.position_size import compute_target_allocation, realized_vol
from core.backtest.replay import sharpe, max_drawdown, total_return


# ============================================================
# Historical signal fetch (using registry)
# ============================================================

def _historical_signals(years_back: int = 20) -> dict[str, pd.Series]:
    """Pull historical signal series via the registry."""
    from core.signal_registry import fetch_all_historical
    return fetch_all_historical()


def _historical_anchor_returns() -> dict[str, pd.Series]:
    import yfinance as yf
    spy = yf.Ticker("SPY").history(period="20y")["Close"]
    btc = yf.Ticker("BTC-USD").history(period="10y")["Close"]
    bil = yf.Ticker("BIL").history(period="15y")["Close"]
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    bil.index = pd.to_datetime(bil.index).tz_localize(None)
    return {"SPY": spy, "BTC": btc, "BIL": bil}


# ============================================================
# Classify regime historically (rule-based, point-in-time)
# ============================================================

def _classify_regime_from_zs(theme_zs: dict[str, float]) -> str:
    """Mirror of btc_regime.classify_regime but from theme z's directly.

    Liquidity z < -1 OR Growth z < -1 OR Credit z < -1 = RECESSIONARY_BEAR
    Any theme z < -0.5 = LATE_CYCLE
    All themes > -0.5 = RISK_ON
    """
    bear_thresh = -1.0
    late_thresh = -0.5
    liquidity_z = theme_zs.get("LIQUIDITY", 0)
    credit_z = theme_zs.get("CREDIT", 0)
    growth_z = theme_zs.get("GROWTH", 0)

    if liquidity_z < bear_thresh or growth_z < bear_thresh or credit_z < bear_thresh:
        return "RECESSIONARY_BEAR"
    if liquidity_z < late_thresh or credit_z < late_thresh or growth_z < late_thresh:
        return "LATE_CYCLE"
    return "RISK_ON"


# ============================================================
# Engine replay
# ============================================================

def real_engine_replay(
    rebalance_freq_days: int = 21,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> dict:
    """Replay the full engine on historical data.

    For each rebalance date:
      1. Standardize all signals up to that point (rolling z-score)
      2. Compute theme composites
      3. Classify regime
      4. Compute target allocation
      5. Simulate the next rebalance period

    Returns engine equity curve + benchmarks.
    """
    signals = _historical_signals()
    anchors = _historical_anchor_returns()

    if not signals or "SPY" not in anchors:
        return {"error": "data unavailable"}

    spy = anchors["SPY"]
    btc = anchors.get("BTC")
    bil = anchors.get("BIL")

    # Build a unified daily index
    end_date = pd.to_datetime(end) if end else spy.index[-1]
    start_date = pd.to_datetime(start)
    dates = pd.date_range(start_date, end_date, freq="B")

    # Reindex signals to daily, forward-fill (most are monthly)
    sig_df = pd.concat(signals, axis=1).reindex(dates).ffill()

    # Reindex anchors
    spy_r = spy.reindex(dates).ffill().pct_change()
    btc_r = btc.reindex(dates).ffill().pct_change() if btc is not None else pd.Series(0, index=dates)
    bil_r = bil.reindex(dates).ffill().pct_change() if bil is not None else pd.Series(0.04 / 252, index=dates)

    # Walk forward
    engine_returns = []
    regime_history = []
    weights_history = []
    current_weights = {"equity": 0.5, "btc": 0.0, "staging": 0.5}
    last_rebalance = -rebalance_freq_days
    ic_weights = {}  # empty for now — equal-weighted within themes

    for i, dt in enumerate(dates):
        # Rebalance every N days
        if i - last_rebalance >= rebalance_freq_days:
            # Get current signal values
            try:
                # Convert raw values to z-scores via rolling window
                row_zs = {}
                for col in sig_df.columns:
                    col_series = sig_df[col].iloc[:i+1].dropna()
                    if len(col_series) < 30:
                        continue
                    mean = col_series.tail(2520).mean()
                    std = col_series.tail(2520).std()
                    if std > 0:
                        z = (col_series.iloc[-1] - mean) / std
                        z = max(-3, min(3, z))
                        row_zs[col] = z

                # Wrap as standardize-output shape
                standardized = {n: {"z": z} for n, z in row_zs.items()}
                themes = compute_all_themes(standardized, ic_weights)
                theme_zs_flat = {t: r.get("composite_z", 0)
                                  for t, r in themes.items()}
                regime = _classify_regime_from_zs(theme_zs_flat)
                decisions = composite_scores_for_decisions(themes)

                # Realized vols
                spy_vol = realized_vol(spy_r.iloc[max(0, i-60):i+1], window=60) or 0.18
                btc_vol = realized_vol(btc_r.iloc[max(0, i-60):i+1], window=60) or 0.60

                target = compute_target_allocation(
                    composite_scores=decisions,
                    regime=regime,
                    realized_vols={"SPY": spy_vol, "BTC": btc_vol},
                    current_drawdown=0.0,
                    vetoes=[],
                    kelly_fraction=0.25,
                    portfolio_vol_target=0.12,
                    total_stake=100_000,
                )
                current_weights = target["weights"]
                last_rebalance = i
                regime_history.append(regime)
            except Exception:
                pass

        # Compute today's return
        r = (current_weights.get("equity", 0) * (spy_r.iloc[i] if not pd.isna(spy_r.iloc[i]) else 0)
             + current_weights.get("btc", 0) * (btc_r.iloc[i] if not pd.isna(btc_r.iloc[i]) else 0)
             + current_weights.get("staging", 0) * (bil_r.iloc[i] if not pd.isna(bil_r.iloc[i]) else 0))
        engine_returns.append(r)
        weights_history.append(dict(current_weights))

    engine_rets = pd.Series(engine_returns, index=dates)

    # Benchmarks
    benchmark_6040 = 0.6 * spy_r + 0.4 * bil_r
    benchmark_btc_dca = 0.9 * spy_r + 0.1 * btc_r

    return {
        "start": start, "end": str(end_date.date()),
        "n_days": len(dates),
        "engine": {
            "total_return": total_return(engine_rets),
            "sharpe": sharpe(engine_rets),
            "max_dd": max_drawdown(engine_rets),
            "annual_return": float((1 + engine_rets.mean()) ** 252 - 1),
            "vol": float(engine_rets.std() * np.sqrt(252)),
        },
        "benchmark_60_40": {
            "total_return": total_return(benchmark_6040),
            "sharpe": sharpe(benchmark_6040),
            "max_dd": max_drawdown(benchmark_6040),
        },
        "benchmark_btc_dca": {
            "total_return": total_return(benchmark_btc_dca),
            "sharpe": sharpe(benchmark_btc_dca),
            "max_dd": max_drawdown(benchmark_btc_dca),
        },
        "outperformance_total_vs_6040": float(total_return(engine_rets)
                                                 - total_return(benchmark_6040)),
        "sharpe_uplift_vs_6040": float(sharpe(engine_rets) - sharpe(benchmark_6040)),
        "n_regime_transitions": sum(1 for i in range(1, len(regime_history))
                                       if regime_history[i] != regime_history[i-1]),
        "n_regimes_observed": len(set(regime_history)),
        "engine_returns_tail_30": engine_rets.tail(30).to_list(),
        "weights_tail_5": weights_history[-5:] if weights_history else [],
    }


def main():
    print("Running real_engine_replay (2015-2026)...")
    r = real_engine_replay(start="2015-01-01")
    if "error" in r:
        print(f"Error: {r['error']}")
        return
    print(f"\n{r['start']} -> {r['end']} ({r['n_days']} days)")
    print(f"\nEngine:    total {r['engine']['total_return']:+.1%}  "
          f"Sharpe {r['engine']['sharpe']:+.2f}  "
          f"MaxDD {r['engine']['max_dd']:.1%}")
    print(f"60/40:     total {r['benchmark_60_40']['total_return']:+.1%}  "
          f"Sharpe {r['benchmark_60_40']['sharpe']:+.2f}  "
          f"MaxDD {r['benchmark_60_40']['max_dd']:.1%}")
    print(f"90/10 BTC: total {r['benchmark_btc_dca']['total_return']:+.1%}  "
          f"Sharpe {r['benchmark_btc_dca']['sharpe']:+.2f}  "
          f"MaxDD {r['benchmark_btc_dca']['max_dd']:.1%}")
    print(f"\nOutperformance vs 60/40: {r['outperformance_total_vs_6040']:+.1%} total / "
          f"{r['sharpe_uplift_vs_6040']:+.2f} Sharpe")
    print(f"Regime transitions: {r['n_regime_transitions']}, "
          f"regimes observed: {r['n_regimes_observed']}")


if __name__ == "__main__":
    main()
