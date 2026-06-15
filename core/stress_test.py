"""Historical stress test framework — replay strategies through known crashes.

Crypto's biggest historical drawdowns (rolling 30d):
  - 2020-03 (COVID): BTC -50% in days
  - 2021-05 (China ban): BTC -40%
  - 2022-05 (LUNA): BTC -25%, alts wiped
  - 2022-06 (3AC/Celsius): BTC -25%
  - 2022-11 (FTX): BTC -25%, ETH -25%
  - 2024-08 (yen carry unwind): BTC -25% in 24h

For each event, replay the production candidate strategy's signals through
the historical window and report what would have happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, data


# (start, end, label) — windows around major crypto stress events
CRASH_WINDOWS = [
    ("2020-02-15", "2020-04-15", "covid_crash"),
    ("2021-04-15", "2021-06-15", "china_ban"),
    ("2022-05-01", "2022-06-30", "luna_collapse"),
    ("2022-10-15", "2022-12-15", "ftx_collapse"),
    ("2024-07-15", "2024-08-15", "yen_carry_unwind"),
]


def stress_test_strategy(
    signal_fn,
    pair: str = "BTC/USDT",
) -> pd.DataFrame:
    """Replay signal_fn through each crash window. Returns per-window stats."""
    df = data.ohlcv_extended(pair, days_back=2500)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for start, end, label in CRASH_WINDOWS:
        try:
            window = df.loc[start:end]
        except Exception:
            continue
        if window.empty or len(window) < 10:
            rows.append({
                "event": label, "n_obs": 0, "skipped": True,
            })
            continue

        # Signal computed using only history through end of window (no lookahead)
        full_history = df.loc[:end]
        sig = signal_fn(full_history["close"])
        sig_in_window = sig.reindex(window.index).fillna(0)

        bt = backtest.run(window["close"], sig_in_window, starting_equity=100_000)
        if bt.empty:
            continue
        equity = bt["equity"]
        eq_start = float(equity.iloc[0])
        eq_end = float(equity.iloc[-1])
        peak = equity.cummax()
        max_dd = float((1 - equity / peak).max())
        bench_start = float(window["close"].iloc[0])
        bench_end = float(window["close"].iloc[-1])

        rows.append({
            "event": label,
            "start": start,
            "end": end,
            "n_days": int(len(window)),
            "btc_return_pct": (bench_end / bench_start - 1) * 100,
            "strategy_return_pct": (eq_end / eq_start - 1) * 100,
            "strategy_max_dd_pct": max_dd * 100,
            "alpha_pct": ((eq_end / eq_start - 1) - (bench_end / bench_start - 1)) * 100,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from research import signals as res_sig

    sig_fn = lambda p: res_sig.tsmom_multi(p, horizons=(30, 90, 180))
    out = stress_test_strategy(sig_fn)
    if not out.empty:
        print(out.to_string(index=False))
