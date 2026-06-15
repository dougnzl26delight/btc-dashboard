"""Test: how does expanding pro_trend universe to top-N pairs affect returns?

Pulls top N USDT pairs by 24h volume from Binance, filters by data history,
runs pro_trend backtest on each, aggregates.

Compares: 11-pair (current) vs 25 vs 50 vs 100.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.pro_trend_backtest import pro_trend_backtest


def get_top_n_by_volume(n: int = 100, min_history_days: int = 1500) -> list[dict]:
    """Get top N USDT spot pairs by 24h volume that have enough history."""
    try:
        tickers = data._EX.fetch_tickers()
    except Exception as e:
        return []

    # Filter to USDT pairs and rank by quote volume
    usdt_pairs = []
    for symbol, t in tickers.items():
        if not symbol.endswith("/USDT") or symbol == "USDT/USDT":
            continue
        if symbol.endswith("UP/USDT") or symbol.endswith("DOWN/USDT"):
            continue  # leveraged tokens
        if "BULL" in symbol or "BEAR" in symbol:
            continue
        vol = t.get("quoteVolume") or 0
        last = t.get("last")
        if not last or vol < 1_000_000:  # need at least $1M daily volume
            continue
        usdt_pairs.append({"pair": symbol, "vol": float(vol), "last": float(last)})

    usdt_pairs.sort(key=lambda x: -x["vol"])
    top_n = usdt_pairs[: n * 2]  # take 2x to allow for history filter

    # Filter by data history
    qualified = []
    for p in top_n:
        try:
            df = data.ohlcv_extended(p["pair"], days_back=min_history_days, timeframe="1d")
            if len(df) >= min_history_days * 0.8:  # at least 80% of requested history
                p["bars"] = len(df)
                qualified.append(p)
                if len(qualified) >= n:
                    break
        except Exception:
            continue

    return qualified


def run_universe_test(pairs: list[str], days_back: int = 1500) -> dict:
    """Run pro_trend backtest on each pair, aggregate."""
    results = []
    for pair in pairs:
        try:
            r = pro_trend_backtest(
                pair=pair, days_back=days_back,
                atr_stop_mult=4.0, max_pyramid_units=2,
                risk_pct_per_unit=0.04, drawdown_kill_pct=0.35,
            )
            if "error" not in r:
                results.append(r)
        except Exception:
            continue

    if not results:
        return {"error": "no successful backtests"}

    annlzd = np.array([r["annualized_return"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    dds = np.array([r["max_drawdown"] for r in results])
    bah = np.array([r["bah_return"] for r in results])
    alpha = np.array([r["alpha_vs_bah"] for r in results])
    n_trades = np.array([r["n_trades"] for r in results])

    return {
        "n_pairs": len(results),
        "mean_annualized": float(annlzd.mean()),
        "median_annualized": float(np.median(annlzd)),
        "std_annualized": float(annlzd.std()),
        "n_positive_pairs": int((annlzd > 0).sum()),
        "n_negative_pairs": int((annlzd < 0).sum()),
        "best_pair": pairs[int(np.argmax(annlzd))] if len(annlzd) > 0 else None,
        "best_return": float(annlzd.max()) if len(annlzd) > 0 else 0,
        "worst_pair": pairs[int(np.argmin(annlzd))] if len(annlzd) > 0 else None,
        "worst_return": float(annlzd.min()) if len(annlzd) > 0 else 0,
        "mean_sharpe": float(sharpes.mean()),
        "mean_max_dd": float(dds.mean()),
        "max_max_dd": float(dds.max()),
        "n_pairs_beat_bah": int((alpha > 0).sum()),
        "mean_n_trades": float(n_trades.mean()),
        "total_n_trades": int(n_trades.sum()),
    }


if __name__ == "__main__":
    print("Fetching top pairs by 24h volume + history filter...")
    top_pairs = get_top_n_by_volume(n=100, min_history_days=1500)
    print(f"Found {len(top_pairs)} pairs with adequate history")
    print()

    sizes = [11, 25, 50, len(top_pairs)] if len(top_pairs) >= 50 else [11, 25, len(top_pairs)]
    sizes = sorted(set([s for s in sizes if s <= len(top_pairs)]))

    for size in sizes:
        subset_pairs = [p["pair"] for p in top_pairs[:size]]
        print(f"=== Top {size} pairs ===")
        print(f"  Sample: {subset_pairs[:5]}{'...' if size > 5 else ''}")
        result = run_universe_test(subset_pairs, days_back=1500)
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue
        print(f"  Backtested: {result['n_pairs']} pairs")
        print(f"  Mean annualized return: {result['mean_annualized']:+.2%}")
        print(f"  Median annualized: {result['median_annualized']:+.2%}")
        print(f"  Std annualized: {result['std_annualized']:.2%}")
        print(f"  Positive / Negative: {result['n_positive_pairs']} / {result['n_negative_pairs']}")
        print(f"  Best: {result['best_pair']} {result['best_return']:+.1%}")
        print(f"  Worst: {result['worst_pair']} {result['worst_return']:+.1%}")
        print(f"  Mean Sharpe: {result['mean_sharpe']:.2f}")
        print(f"  Mean max DD: {result['mean_max_dd']:.1%}")
        print(f"  Pairs that BEAT BAH: {result['n_pairs_beat_bah']} / {result['n_pairs']}")
        print(f"  Total trades: {result['total_n_trades']}, mean per pair: {result['mean_n_trades']:.0f}")
        print()
