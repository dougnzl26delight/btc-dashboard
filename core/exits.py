"""Triple-barrier exit labels.

López de Prado, Advances in Financial Machine Learning (2018), Ch. 3.3-3.4.

Replaces signal-only entry-and-hold with three barriers:
  - Profit target  (upper barrier, +pt_sigma * sigma above entry)
  - Stop loss      (lower barrier, -sl_sigma * sigma below entry)
  - Time horizon   (vertical barrier, max holding period)

The position closes when ANY barrier is hit. First-touch wins. This forces
strategies to have an explicit exit rule rather than relying on signal flips.

Use the return labels for either:
  - Backtest with realistic holding behavior
  - Meta-labeling training set (LdP Ch. 3.6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(
    prices: pd.Series,
    events: pd.Series,
    horizon_days: int = 60,
    pt_sigma: float = 2.0,
    sl_sigma: float = 2.0,
    vol_window: int = 30,
) -> pd.DataFrame:
    """Apply triple-barrier exits to non-zero events in `events`.

    `events` should be a sparse signal series — non-zero values indicate trade
    entries with the sign giving direction. Zero values are ignored.

    Returns DataFrame indexed sequentially with columns:
      event_ts, direction, entry_price, exit_price, exit_reason
      ('profit'/'stop'/'horizon'), duration_days, return.
    """
    log_ret = np.log(prices / prices.shift(1))
    realized = log_ret.rolling(vol_window).std()

    out: list[dict] = []
    nonzero = events[events != 0]
    for event_ts in nonzero.index:
        if event_ts not in prices.index:
            continue
        direction = float(np.sign(nonzero.loc[event_ts]))
        entry_idx = prices.index.get_loc(event_ts)
        if entry_idx + 1 >= len(prices):
            continue
        entry_price = float(prices.iloc[entry_idx])
        sigma = realized.loc[event_ts] if event_ts in realized.index else None
        if sigma is None or pd.isna(sigma) or sigma == 0:
            continue

        pt_price = entry_price * np.exp(direction * pt_sigma * float(sigma))
        sl_price = entry_price * np.exp(-direction * sl_sigma * float(sigma))

        end_idx = min(entry_idx + horizon_days, len(prices) - 1)
        future = prices.iloc[entry_idx + 1: end_idx + 1]

        exit_reason = "horizon"
        exit_price = float(future.iloc[-1]) if len(future) else entry_price
        duration = len(future)

        for i, p in enumerate(future.values, start=1):
            p = float(p)
            if direction > 0:
                if p >= pt_price:
                    exit_reason, exit_price, duration = "profit", pt_price, i
                    break
                if p <= sl_price:
                    exit_reason, exit_price, duration = "stop", sl_price, i
                    break
            else:
                if p <= pt_price:
                    exit_reason, exit_price, duration = "profit", pt_price, i
                    break
                if p >= sl_price:
                    exit_reason, exit_price, duration = "stop", sl_price, i
                    break

        out.append(
            {
                "event_ts": event_ts,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "duration_days": duration,
                "return": direction * (exit_price / entry_price - 1.0),
            }
        )

    return pd.DataFrame(out)


def barrier_summary(events_df: pd.DataFrame) -> dict:
    """Aggregate stats from a triple-barrier event log."""
    if events_df.empty:
        return {"n": 0}
    return {
        "n_trades": int(len(events_df)),
        "win_rate": float((events_df["return"] > 0).mean()),
        "avg_return": float(events_df["return"].mean()),
        "avg_duration": float(events_df["duration_days"].mean()),
        "exit_reason_counts": events_df["exit_reason"].value_counts().to_dict(),
        "expectancy": float(events_df["return"].mean()),
    }


def barriers_to_daily_returns(
    barriers_df: pd.DataFrame,
    prices: pd.Series,
    cost_bps_per_side: int = 15,
) -> pd.Series:
    """Convert a triple-barrier trade list into a daily mark-to-market return series.

    Used to evaluate event-driven strategies on the same time-weighted Sharpe
    basis as continuous-weight backtests. Cost is applied at entry and exit.
    """
    import numpy as np

    daily = pd.Series(0.0, index=prices.index)
    for _, row in barriers_df.iterrows():
        entry_ts = row["event_ts"]
        if entry_ts not in prices.index:
            continue
        entry_idx = prices.index.get_loc(entry_ts)
        duration = int(row["duration_days"])
        exit_idx = min(entry_idx + duration, len(prices) - 1)
        direction = float(row["direction"])

        if exit_idx <= entry_idx:
            continue
        seg_prices = prices.iloc[entry_idx : exit_idx + 1].values
        seg_rets = np.diff(seg_prices) / seg_prices[:-1] * direction
        # Apply seg_rets to daily series at positions entry_idx+1 .. exit_idx
        for j, r in enumerate(seg_rets):
            daily.iloc[entry_idx + 1 + j] += float(r)

        # Costs at entry and exit
        if entry_idx + 1 < len(prices):
            daily.iloc[entry_idx + 1] -= cost_bps_per_side / 10_000.0
        if exit_idx < len(prices):
            daily.iloc[exit_idx] -= cost_bps_per_side / 10_000.0

    return daily


def event_metrics(barriers_df: pd.DataFrame, observation_days: int) -> dict:
    """Per-trade view of strategy stats. Pair with daily-MTM Sharpe for comparison."""
    import numpy as np

    if barriers_df.empty or len(barriers_df) < 2:
        return {"n_trades": int(len(barriers_df))}

    rets = barriers_df["return"].values
    durations = barriers_df["duration_days"].values
    avg_dur = float(durations.mean())
    trades_per_year = 365.0 / avg_dur if avg_dur > 0 else 0.0

    sharpe_per_trade = (
        float(rets.mean() / rets.std(ddof=1) * np.sqrt(trades_per_year))
        if rets.std(ddof=1) > 0
        else 0.0
    )
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = float((1 - equity / peak).max()) if len(equity) else 0.0
    total = float(equity[-1] - 1) if len(equity) else 0.0
    annualized = (1 + total) ** (365 / observation_days) - 1 if observation_days > 0 else 0.0

    return {
        "n_trades": int(len(rets)),
        "win_rate": float((rets > 0).mean()),
        "avg_return_per_trade": float(rets.mean()),
        "avg_duration_days": avg_dur,
        "trades_per_year": trades_per_year,
        "total_return": total,
        "annualized_return": float(annualized),
        "sharpe_per_trade_ann": sharpe_per_trade,
        "max_drawdown_per_trade": dd,
    }
