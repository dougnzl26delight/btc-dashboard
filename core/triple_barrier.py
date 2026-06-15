"""Triple-Barrier Method — López de Prado AFML Chapter 3.

Standard backtest labeling: "did price go up tomorrow?" — useless for systems
that exit on stops, targets, or time-caps.

Triple-barrier labels each signal with the ACTUAL realized outcome:
    +1  profit target hit FIRST
    -1  stop loss hit FIRST
     0  time-cap expired before either

This gives you a 3-class label that matches how your system actually trades.

Use case: feed historical signals into label_signals() to get a realistic
outcome distribution. Compute Sharpe / win rate / R-multiple on the labels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def label_signal(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: int,                # +1 long, -1 short
    profit_target_pct: float,
    stop_loss_pct: float,
    time_cap_bars: int,
) -> dict:
    """Label one signal using triple-barrier method.

    Args:
        df: OHLCV dataframe (must have 'high', 'low', 'close' columns)
        entry_idx: index in df where signal triggered
        entry_price: actual entry price
        direction: +1 for long, -1 for short
        profit_target_pct: e.g., 0.05 for +5%
        stop_loss_pct: e.g., 0.02 for 2% stop
        time_cap_bars: max bars to hold

    Returns: {label: +1/-1/0, exit_price, holding_bars, realized_pct}
    """
    if direction > 0:
        target_price = entry_price * (1 + profit_target_pct)
        stop_price = entry_price * (1 - stop_loss_pct)
    else:
        target_price = entry_price * (1 - profit_target_pct)
        stop_price = entry_price * (1 + stop_loss_pct)

    end_idx = min(entry_idx + time_cap_bars + 1, len(df))
    for i in range(entry_idx + 1, end_idx):
        high = float(df.iloc[i]["high"])
        low = float(df.iloc[i]["low"])
        if direction > 0:
            # Long position: target hit if high >= target, stop hit if low <= stop
            target_hit = high >= target_price
            stop_hit = low <= stop_price
        else:
            target_hit = low <= target_price
            stop_hit = high >= stop_price
        if target_hit and stop_hit:
            # Both touched in same bar — conservatively assume STOP hit first
            return {
                "label": -1,
                "exit_price": stop_price,
                "holding_bars": i - entry_idx,
                "realized_pct": -stop_loss_pct,
                "reason": "stop_first_same_bar",
            }
        if target_hit:
            return {
                "label": 1,
                "exit_price": target_price,
                "holding_bars": i - entry_idx,
                "realized_pct": profit_target_pct,
                "reason": "target",
            }
        if stop_hit:
            return {
                "label": -1,
                "exit_price": stop_price,
                "holding_bars": i - entry_idx,
                "realized_pct": -stop_loss_pct,
                "reason": "stop",
            }
    # Time cap reached
    exit_close = float(df.iloc[end_idx - 1]["close"])
    realized = (exit_close / entry_price - 1) * direction
    return {
        "label": 0,
        "exit_price": exit_close,
        "holding_bars": end_idx - entry_idx - 1,
        "realized_pct": realized,
        "reason": "time_cap",
    }


def label_signals(
    df: pd.DataFrame,
    signal_indices: Iterable[int],
    direction: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    time_cap_bars: int,
) -> list[dict]:
    """Label all signals at once. Adds entry_idx + entry_price to each output."""
    out = []
    for idx in signal_indices:
        if idx >= len(df):
            continue
        entry_price = float(df.iloc[idx]["close"])
        r = label_signal(df, idx, entry_price, direction,
                          profit_target_pct, stop_loss_pct, time_cap_bars)
        r["entry_idx"] = idx
        r["entry_price"] = entry_price
        out.append(r)
    return out


def evaluate_labels(labels: list[dict]) -> dict:
    """Compute backtest statistics from triple-barrier labels.

    Returns realistic metrics:
        win_rate (label = +1)
        loss_rate (label = -1)
        time_cap_rate (label = 0)
        avg_R (avg realized / |stop|)
        sharpe_per_trade  (mean return / std return)
        expectancy
    """
    if not labels:
        return {"n": 0}

    arr = np.array([l["realized_pct"] for l in labels])
    win_mask = np.array([l["label"] == 1 for l in labels])
    loss_mask = np.array([l["label"] == -1 for l in labels])
    timecap_mask = np.array([l["label"] == 0 for l in labels])
    n = len(arr)
    mean_ret = float(arr.mean())
    std_ret = float(arr.std()) if n > 1 else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    avg_holding = float(np.mean([l["holding_bars"] for l in labels]))

    return {
        "n_signals": n,
        "win_rate": float(win_mask.sum() / n),
        "loss_rate": float(loss_mask.sum() / n),
        "time_cap_rate": float(timecap_mask.sum() / n),
        "avg_realized_pct": mean_ret,
        "median_realized_pct": float(np.median(arr)),
        "std_realized_pct": std_ret,
        "annualized_sharpe": sharpe,
        "avg_holding_bars": avg_holding,
        "expectancy_per_trade": mean_ret,
    }


def main():
    """Demo: triple-barrier on synthetic price walk."""
    print("=" * 70)
    print("TRIPLE-BARRIER LABELING — Lopez de Prado AFML Ch 3 demo")
    print("=" * 70)
    np.random.seed(42)
    # Synthetic price walk
    n_bars = 1000
    returns = np.random.normal(0.0005, 0.015, n_bars)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": price,
        "high": price * (1 + np.abs(np.random.normal(0, 0.003, n_bars))),
        "low":  price * (1 - np.abs(np.random.normal(0, 0.003, n_bars))),
    })
    # Generate 50 random signal entries
    signals = list(np.random.choice(range(50, n_bars - 50), size=50, replace=False))

    labels = label_signals(df, signals, direction=1,
                            profit_target_pct=0.03, stop_loss_pct=0.015,
                            time_cap_bars=20)
    stats = evaluate_labels(labels)
    print()
    print("Triple-barrier labels (target=3%, stop=1.5%, time=20 bars):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<24s}  {v:+.4f}")
        else:
            print(f"  {k:<24s}  {v}")


if __name__ == "__main__":
    main()
