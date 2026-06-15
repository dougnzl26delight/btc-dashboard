"""Regime-aware BAH BTC test — does scaling down in bear save the system?

Tests 3 BAH variants:
  v0: Unconditional 20% BAH BTC (current production)
  v1: Scale down by BTC distance from SMA200:
        BTC > SMA200:           100% of target (full 20%)
        0 to -10% below SMA200: 50% of target (10%)
        > -10% below SMA200:    0% (cash)
  v2: Binary cutoff:
        BTC > SMA200:           full 20%
        BTC < SMA200:           0%
  v3: Even tighter cutoff:
        BTC > SMA200 by >2%:    full 20%
        else:                   0%

Each variant tested across 4 regimes + full 6.3y.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.strategy_bakeoff import (
    fetch_panel, PAIRS_5, perf_stats, pro_trend_long_only,
)
from core.xsmom_backtest import xsmom_backtest


def regime_aware_bah_eq(pair_data, date_start, date_end, mode="unconditional"):
    """BAH BTC with optional regime gate. Returns daily equity series."""
    btc = pair_data["BTC/USDT"]
    btc = btc[(btc.index >= date_start - pd.Timedelta(days=210))
              & (btc.index <= date_end)].copy()
    btc["sma200"] = btc["close"].rolling(200).mean()
    btc = btc.dropna()
    btc = btc[(btc.index >= date_start) & (btc.index <= date_end)]
    if len(btc) < 2:
        return pd.Series(dtype=float)

    cash = 100_000.0
    btc_qty = 0.0
    daily_eqs = []
    target_alloc = 0.20  # 20% of capital target

    for i, (date, row) in enumerate(btc.iterrows()):
        price = float(row["close"])
        sma = float(row["sma200"])
        pct_below = price / sma - 1

        # Determine target allocation per mode
        if mode == "unconditional":
            allocation = target_alloc
        elif mode == "v1_scaled":
            if pct_below >= 0:
                allocation = target_alloc
            elif pct_below >= -0.10:
                allocation = target_alloc * 0.5
            else:
                allocation = 0.0
        elif mode == "v2_binary":
            allocation = target_alloc if pct_below >= 0 else 0.0
        elif mode == "v3_tight":
            allocation = target_alloc if pct_below >= 0.02 else 0.0
        else:
            allocation = target_alloc

        # Rebalance monthly OR on regime change
        rebalance = (i == 0) or (date.month != btc.index[i - 1].month)
        # Force rebalance if current allocation differs significantly from target
        current_value = btc_qty * price
        total_equity = cash + current_value
        current_alloc = current_value / total_equity if total_equity > 0 else 0
        target_value = total_equity * allocation
        if abs(current_alloc - allocation) > 0.05:
            rebalance = True

        if rebalance:
            new_qty = target_value / price if price > 0 else 0
            qty_delta = new_qty - btc_qty
            cash_delta = -qty_delta * price
            cash += cash_delta
            btc_qty = new_qty

        # MTM
        equity = cash + btc_qty * price
        daily_eqs.append({"ts": date, "equity": equity})

    return pd.DataFrame(daily_eqs).set_index("ts")["equity"]


def three_sleeve_with_bah_mode(pair_data, date_start, date_end, bah_mode):
    """Combined 50/30/20 portfolio with BAH using specified regime mode."""
    pt_eq = pro_trend_long_only(pair_data, date_start, date_end, use_v5_filter=True)
    xs_full = xsmom_backtest(days_back=2500, momentum_window=14,
                              rebalance_freq=14, long_n=2, short_n=2, risk_per_leg=0.20)
    xs_eq = (xs_full["equity_path"]
             [(xs_full["equity_path"].index >= date_start)
              & (xs_full["equity_path"].index <= date_end)]
             if "error" not in xs_full else pd.Series(dtype=float))
    bah_eq = regime_aware_bah_eq(pair_data, date_start, date_end, mode=bah_mode)

    sleeves = [
        ("pro_trend", pt_eq.pct_change().fillna(0), 0.50),
        ("xsmom", xs_eq.pct_change().fillna(0), 0.30),
        ("bah_btc", bah_eq.pct_change().fillna(0), 0.20),
    ]
    indices = [s[1].index for s in sleeves if not s[1].empty]
    if not indices:
        return pd.Series(dtype=float)
    common = indices[0]
    for idx in indices[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        return pd.Series(dtype=float)

    combined = pd.Series(0.0, index=common)
    for name, rets, alloc in sleeves:
        if not rets.empty:
            combined += rets.loc[common] * alloc
    eq = 100_000 * (1 + combined).cumprod()
    return eq


if __name__ == "__main__":
    print("=" * 100)
    print("REGIME-AWARE BAH BTC TEST")
    print("=" * 100)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())

    windows = [
        ("A: 2020-21 mega-bull",
         pd.Timestamp("2020-01-21", tz="UTC"), pd.Timestamp("2021-04-24", tz="UTC")),
        ("B: 2021-22 top/LUNA bear",
         pd.Timestamp("2021-04-25", tz="UTC"), pd.Timestamp("2022-07-28", tz="UTC")),
        ("C: 2022-23 bear/recovery",
         pd.Timestamp("2022-07-29", tz="UTC"), pd.Timestamp("2023-10-31", tz="UTC")),
        ("D: 2024-26 recent chop",
         pd.Timestamp("2024-11-01", tz="UTC"), end_date),
        ("ALL: full 6.3y",
         pd.Timestamp("2020-01-21", tz="UTC"), end_date),
    ]

    modes = {
        "v0 unconditional 20%": "unconditional",
        "v1 scaled (full/half/0)": "v1_scaled",
        "v2 binary cutoff (SMA200)": "v2_binary",
        "v3 tight cutoff (SMA200+2%)": "v3_tight",
    }

    print(f"{'BAH Mode':<28s}", end="")
    for label, _, _ in windows:
        print(f"  {label[:15]:>17s}", end="")
    print()
    print("-" * 28 + ("-" * 19) * len(windows))

    results = {}
    for mode_name, mode_key in modes.items():
        results[mode_name] = {}
        print(f"{mode_name:<28s}", end="")
        for label, ds, de in windows:
            try:
                eq = three_sleeve_with_bah_mode(pair_data, ds, de, mode_key)
                stats = perf_stats(eq)
                results[mode_name][label] = stats
                cell = f"{stats['annualized']:+5.1%}/Sh{stats['sharpe']:+3.1f}"
                print(f"  {cell:>17s}", end="")
            except Exception as e:
                print(f"  {'ERR':>17s}", end="")
                results[mode_name][label] = {"error": str(e)}
        print()

    print()
    print("=" * 100)
    print("DETAILED COMPARISON ACROSS REGIMES")
    print("=" * 100)
    for label, _, _ in windows:
        print(f"\n{label}")
        print(f"  {'Mode':<32s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>10s} {'MaxDD':>7s}")
        for mode_name, mode_results in results.items():
            r = mode_results.get(label, {})
            if "error" in r:
                continue
            print(f"  {mode_name:<32s} {r['sharpe']:>+6.2f}  "
                  f"{r['annualized']:>+8.1%}  {r['return']:>+9.1%}  "
                  f"{r['max_dd']:>5.1%}")

    print()
    print("=" * 100)
    print("KEY QUESTION: which mode protects in BEAR (window B: 2021-22 LUNA)?")
    print("=" * 100)
    print(f"{'Mode':<32s} {'Bear Return':>12s} {'Bear MaxDD':>11s} {'Full Sharpe':>12s}")
    for mode_name, mode_results in results.items():
        bear = mode_results.get("B: 2021-22 top/LUNA bear", {})
        full = mode_results.get("ALL: full 6.3y", {})
        if "error" in bear or "error" in full:
            continue
        print(f"{mode_name:<32s} {bear['return']:>+11.1%}  {bear['max_dd']:>10.1%}  "
              f"{full['sharpe']:>+11.2f}")
