"""Ensemble the top momentum candidates from the sweep.

Hypothesis: averaging correlated-but-not-identical momentum signals
diversifies away idiosyncratic noise and raises OOS Sharpe. Same OOS
pipeline (walk-forward + factor decomp + DSR/t hurdle) as the sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from research import signals as sig
from research.sweep import evaluate_candidate


def top5_ensemble(prices: pd.Series) -> pd.Series:
    """Equal-weighted average of the top 5 momentum variants from the sweep."""
    sigs = [
        sig.tsmom_multi(prices, horizons=(30, 90, 180)),
        sig.tsmom_single(prices, lookback=365),
        sig.tsmom_single(prices, lookback=90),
        sig.tsmom_multi(prices, horizons=(60, 180, 365)),
        sig.tsmom_multi(prices, horizons=(30, 60, 90, 180)),
    ]
    return sum(sigs) / len(sigs)


def top3_ensemble(prices: pd.Series) -> pd.Series:
    """Tighter ensemble — top 3 only, less internal redundancy."""
    sigs = [
        sig.tsmom_multi(prices, horizons=(30, 90, 180)),
        sig.tsmom_single(prices, lookback=365),
        sig.tsmom_single(prices, lookback=90),
    ]
    return sum(sigs) / len(sigs)


def diverse_momentum_ensemble(prices: pd.Series) -> pd.Series:
    """Span more sub-families: TSMOM + MA crossover + Donchian + vol breakout.
    Lower expected internal correlation than pure-TSMOM ensembles.
    """
    sigs = [
        sig.tsmom_multi(prices, horizons=(30, 90, 180)),
        sig.tsmom_single(prices, lookback=365),
        sig.ma_crossover(prices, fast=20, slow=50),
        sig.donchian_breakout(prices, window=50),
        sig.vol_breakout(prices, window=30),
    ]
    return sum(sigs) / len(sigs)


def evaluate_ensembles(pair: str = "BTC/USDT") -> pd.DataFrame:
    df = data.ohlcv(pair, timeframe="1d", limit=1000)
    bench = df["close"].pct_change().fillna(0)

    candidates = [
        {"name": "top5_momentum_ensemble", "fn": top5_ensemble},
        {"name": "top3_momentum_ensemble", "fn": top3_ensemble},
        {"name": "diverse_momentum_ensemble", "fn": diverse_momentum_ensemble},
    ]

    # Honest trial count: 27 from sweep + 3 ensembles (we considered them all)
    num_trials = 30

    rows: list[dict] = []
    for c in candidates:
        r = evaluate_candidate(c, df["close"], bench, num_trials=num_trials)
        rows.append(r)
        marker = "VALID" if r.get("validated") else ""
        print(
            f"{c['name']:30s} OOS={r.get('mean_sharpe_oos',0):+.2f} "
            f"(min={r.get('min_sharpe_oos',0):+.2f}) alpha_t={r.get('alpha_t',0):+.2f} "
            f"dsr={r.get('dsr',0):.2f} {marker}"
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    df = evaluate_ensembles()
    print()
    cols = ["name", "mean_sharpe_oos", "min_sharpe_oos", "alpha_t", "beta", "dsr", "hurdle_t", "validated"]
    print(df[cols].to_string(index=False))
