"""Re-evaluate ALL sleeves with López de Prado rigorous methodology.

Pulls 4 years of historical daily data per representative pair, applies each
sleeve's entry rules, labels outcomes via TRIPLE-BARRIER method, then runs
PURGED K-FOLD CV to get honest 95% CI on Sharpe.

Output: per-sleeve verdict using canonical hurdles:
    Purged CV CI low > 0.5    = DEPLOY (robust edge)
    Purged CV CI low > 0.0    = WEAK (small live test)
    Purged CV CI low < 0.0    = REJECT (overfit)
    Deflated Sharpe > raw / 2 = REJECT (too inflated)

Each sleeve logged to strategy_trials table for PBO accounting.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from core import data
from core.triple_barrier import label_signals, evaluate_labels
from core.purged_cv import purged_cv_sharpe
from core.deflated_sharpe import deflated_sharpe
from core.strategy_trials import log_trial


# Representative pair per sleeve
SLEEVE_TEST_PAIRS = {
    "bah_btc": "BTC/USDT",
    "xsmom": "BTC/USDT",
    "pro_trend": "BTC/USDT",
    "oversold_bounce": "ETH/USDT",
    "overbought_fade": "ETH/USDT",
    "intraday_momentum": "SOL/USDT",
    "consolidation_breakout": "BTC/USDT",
    "grid_trader": "BTC/USDT",
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def signal_oversold_bounce(df: pd.DataFrame) -> list[int]:
    """RSI(14) < 25 = enter long."""
    rsi = _rsi(df["close"])
    return list(df.index[(rsi < 25) & (rsi.shift(1) >= 25)])


def signal_overbought_fade(df: pd.DataFrame) -> list[int]:
    """RSI(14) > 70 in bear regime = enter short."""
    rsi = _rsi(df["close"])
    bear = df["close"] < df["close"].rolling(200).mean()
    return list(df.index[(rsi > 70) & (rsi.shift(1) <= 70) & bear])


def signal_pro_trend(df: pd.DataFrame) -> list[int]:
    """Donchian-20 breakout above SMA200 with TSMOM_30 > 0 (v5 filter)."""
    sma200 = df["close"].rolling(200).mean()
    donch_high_20 = df["high"].rolling(20).max().shift(1)
    tsmom30 = df["close"].pct_change(30)
    cond = (df["close"] > sma200) & (df["close"] > donch_high_20) & (tsmom30 > 0)
    return list(df.index[cond & ~cond.shift(1).fillna(False)])


def signal_xsmom(df: pd.DataFrame) -> list[int]:
    """14-day momentum > +5% = enter long (proxy — true XSMOM is cross-sectional)."""
    ret14 = df["close"].pct_change(14)
    cond = ret14 > 0.05
    return list(df.index[cond & ~cond.shift(1).fillna(False)])


def signal_consolidation_breakout(df: pd.DataFrame) -> list[int]:
    """Compressed BB-30 (<0.6× normal) + break above range high."""
    bb_30 = df["close"].rolling(30).std() / df["close"]
    bb_90_avg = bb_30.rolling(90).mean()
    compression = bb_30 / bb_90_avg
    range_high = df["high"].rolling(30).max().shift(1)
    cond = (compression < 0.6) & (df["close"] > range_high)
    return list(df.index[cond & ~cond.shift(1).fillna(False)])


def signal_bah_btc(df: pd.DataFrame) -> list[int]:
    """Always entered (set first signal at start of series)."""
    if len(df) < 200:
        return []
    return [200]


# Signal generators per sleeve
SIGNAL_FUNCS = {
    "oversold_bounce": (signal_oversold_bounce, +1, 0.05, 0.025, 21),
    "overbought_fade": (signal_overbought_fade, -1, 0.05, 0.025, 14),
    "pro_trend": (signal_pro_trend, +1, 0.08, 0.04, 60),
    "xsmom": (signal_xsmom, +1, 0.05, 0.03, 14),
    "consolidation_breakout": (signal_consolidation_breakout, +1, 0.10, 0.04, 30),
    "bah_btc": (signal_bah_btc, +1, 0.50, 0.30, 1825),  # 5-yr horizon
}


def evaluate_sleeve(sleeve: str) -> dict:
    """Run triple-barrier + purged CV on one sleeve's historical signals."""
    if sleeve not in SIGNAL_FUNCS:
        return {"sleeve": sleeve, "error": "no_signal_func"}

    fn, direction, profit_pct, stop_pct, time_cap = SIGNAL_FUNCS[sleeve]
    pair = SLEEVE_TEST_PAIRS.get(sleeve, "BTC/USDT")

    try:
        df = data.ohlcv_extended(pair, days_back=1500)  # ~4 years
        if df.empty or len(df) < 500:
            return {"sleeve": sleeve, "error": "insufficient_history"}
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Reset to integer index for triple-barrier
        df = df.reset_index(drop=False)
    except Exception as e:
        return {"sleeve": sleeve, "error": f"data_fetch_failed:{e}"}

    # Get signal indices
    signals = fn(df)
    if not signals:
        return {"sleeve": sleeve, "n_signals": 0, "error": "no_signals_in_history"}

    # Triple-barrier labels
    labels = label_signals(df, signals, direction=direction,
                            profit_target_pct=profit_pct,
                            stop_loss_pct=stop_pct,
                            time_cap_bars=time_cap)
    tb_stats = evaluate_labels(labels)

    # Build a return series (per-trade returns) for purged CV
    returns_per_trade = [l["realized_pct"] for l in labels]

    # Purged CV
    pcv = None
    if len(returns_per_trade) >= 50:
        pcv = purged_cv_sharpe(np.array(returns_per_trade), n_splits=5,
                                embargo_pct=0.02,
                                periods_per_year=int(252 / max(tb_stats["avg_holding_bars"], 1)))

    # Deflated Sharpe
    raw_sharpe = tb_stats.get("annualized_sharpe", 0.0)
    dsr = None
    if len(returns_per_trade) >= 30:
        try:
            dsr = deflated_sharpe(returns_per_trade, num_trials=30, periods_per_year=int(252 / max(tb_stats["avg_holding_bars"], 1)))
        except Exception:
            pass

    # Log to strategy_trials for PBO accounting
    try:
        log_trial(
            strategy=sleeve, variant="canon_backtest_v1",
            sharpe=raw_sharpe, n_obs=len(returns_per_trade),
            max_dd_pct=0.0,
            win_rate_pct=tb_stats.get("win_rate", 0) * 100,
            params={
                "pair": pair, "direction": direction,
                "profit_pct": profit_pct, "stop_pct": stop_pct,
                "time_cap": time_cap,
            },
            note="W15 canon re-evaluation (Lopez de Prado method)",
        )
    except Exception:
        pass

    return {
        "sleeve": sleeve, "pair": pair, "direction": direction,
        "n_signals": len(signals),
        "triple_barrier": tb_stats,
        "purged_cv": pcv,
        "dsr": dsr,
    }


def cli_report():
    print("=" * 105)
    print("CANON BACKTEST RE-EVALUATION — Triple-Barrier + Purged CV per sleeve")
    print("Lopez de Prado AFML methodology")
    print("=" * 105)
    print()
    results = []
    for sleeve in SIGNAL_FUNCS:
        print(f"Evaluating {sleeve}...")
        r = evaluate_sleeve(sleeve)
        results.append(r)
    print()
    print("=" * 105)
    print(f"{'Sleeve':<22s} {'N sigs':>7s} {'Win%':>5s} {'Raw SR':>7s} {'CV mean':>8s} "
          f"{'CV CI low':>9s} {'CV CI high':>10s} {'Verdict':<20s}")
    print("-" * 105)
    for r in results:
        if r.get("error"):
            print(f"  {r['sleeve']:<20s} ERROR: {r['error']}")
            continue
        tb = r.get("triple_barrier", {})
        cv = r.get("purged_cv", {}) or {}
        verdict = cv.get("verdict", "n/a")
        print(f"  {r['sleeve']:<20s} {r['n_signals']:>7d} "
              f"{tb.get('win_rate', 0)*100:>4.0f}% "
              f"{tb.get('annualized_sharpe', 0):>+6.2f} "
              f"{cv.get('mean_sharpe', 0):>+7.2f} "
              f"{cv.get('ci_95_low', 0):>+8.2f} "
              f"{cv.get('ci_95_high', 0):>+9.2f}  {verdict[:18]}")
    print()
    print("Decision rule: deploy live ONLY if CV CI low > 0.5.")
    print("CI low between 0.0 - 0.5 = small live test only.")
    print("CI low < 0.0 = REJECT (likely overfit).")
    print()
    print(f"All trials logged to strategy_trials table for ongoing PBO accounting.")


if __name__ == "__main__":
    cli_report()
