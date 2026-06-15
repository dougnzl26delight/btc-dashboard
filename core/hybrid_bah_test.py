"""Hybrid cycle-aware BAH BTC sizing — Mayer-Multiple-based dynamic allocation.

Concept: instead of unconditional 20% BAH BTC, size up when cheap (Mayer < 0.8)
and size down when expensive (Mayer > 1.5). Keeps capital deployed but
opportunistically heavy/light based on extreme valuations.

Variants:
  V0  Unconditional 20% (current production)
  H1  Modest swing  (15% / 20% / 25%, Mayer breakpoints 1.5 / 0.8)
  H2  Moderate swing (10% / 20% / 30%, breakpoints 1.5 / 0.8)
  H3  Aggressive    (0% / 20% / 30%, breakpoints 1.5 / 0.8)
  H4  Wider thresh  (10% / 20% / 30%, breakpoints 1.6 / 0.7)
  H5  Very wide     (0% / 20% / 35%, breakpoints 1.8 / 0.6)

Compared against:
  Plain BAH BTC unconditional
  v3 Cycle timing (halving calendar)
  Current 50/30/20 system

Tests applied within full 50/30/20 portfolio (50% pro_trend + 30% xsmom + 20% BAH variant).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.strategy_bakeoff import fetch_panel, PAIRS_5, perf_stats, pro_trend_long_only
from core.xsmom_backtest import xsmom_backtest


# ============================================================================
# Hybrid BAH BTC — dynamic Mayer-Multiple sizing
# ============================================================================

def hybrid_bah_eq(
    pair_data, date_start, date_end,
    base_alloc=0.20,          # baseline allocation when Mayer in middle zone
    high_alloc=0.20,          # allocation when Mayer > sell_threshold
    low_alloc=0.20,           # allocation when Mayer < buy_threshold
    sell_threshold=1.5,       # Mayer above this -> reduce
    buy_threshold=0.8,        # Mayer below this -> increase
    rebalance_drift=0.02,     # rebalance if actual allocation drifts >2pp from target
):
    """BAH BTC with dynamic sizing based on Mayer Multiple."""
    btc = pair_data["BTC/USDT"]
    # Need enough lookback for SMA200
    btc = btc[(btc.index >= date_start - pd.Timedelta(days=210))
              & (btc.index <= date_end)].copy()
    btc["sma200"] = btc["close"].rolling(200).mean()
    btc["mayer"] = btc["close"] / btc["sma200"]
    btc = btc.dropna(subset=["mayer"])
    btc = btc[(btc.index >= date_start) & (btc.index <= date_end)]
    if len(btc) < 2:
        return pd.Series(dtype=float)

    cash = 100_000.0
    btc_qty = 0.0
    daily_eq = []

    for i, (date_idx, row) in enumerate(btc.iterrows()):
        price = float(row["close"])
        mayer = float(row["mayer"])

        # Determine target allocation per Mayer zone
        if mayer > sell_threshold:
            target_alloc = high_alloc
        elif mayer < buy_threshold:
            target_alloc = low_alloc
        else:
            target_alloc = base_alloc

        current_value = btc_qty * price
        total_equity = cash + current_value
        current_alloc = current_value / total_equity if total_equity > 0 else 0
        drift = abs(current_alloc - target_alloc)

        # Rebalance on first day OR drift exceeds tolerance OR month change
        first_day = (i == 0)
        month_change = (i > 0 and date_idx.month != btc.index[i - 1].month)
        zone_change = False
        if i > 0:
            prev_mayer = float(btc.iloc[i - 1]["mayer"])
            prev_zone = ("high" if prev_mayer > sell_threshold
                         else "low" if prev_mayer < buy_threshold else "mid")
            cur_zone = ("high" if mayer > sell_threshold
                        else "low" if mayer < buy_threshold else "mid")
            zone_change = (prev_zone != cur_zone)

        if first_day or drift > rebalance_drift or month_change or zone_change:
            target_value = total_equity * target_alloc
            new_qty = target_value / price if price > 0 else 0
            qty_delta = new_qty - btc_qty
            cash -= qty_delta * price  # buy if positive, sell if negative
            btc_qty = new_qty

        equity = cash + btc_qty * price
        daily_eq.append({"ts": date_idx, "equity": equity})

    return pd.DataFrame(daily_eq).set_index("ts")["equity"]


def portfolio_with_hybrid_bah(pair_data, date_start, date_end,
                                bah_params, allocation=(0.50, 0.30, 0.20)):
    """Combined 50/30/20 portfolio with the hybrid BAH variant."""
    pt_alloc, xs_alloc, bah_alloc = allocation
    pt_eq = pro_trend_long_only(pair_data, date_start, date_end, use_v5_filter=True)
    xs_full = xsmom_backtest(days_back=2500, momentum_window=14,
                              rebalance_freq=14, long_n=2, short_n=2, risk_per_leg=0.20)
    xs_eq = (xs_full["equity_path"]
             [(xs_full["equity_path"].index >= date_start)
              & (xs_full["equity_path"].index <= date_end)]
             if "error" not in xs_full else pd.Series(dtype=float))
    bah_eq = hybrid_bah_eq(pair_data, date_start, date_end, **bah_params)

    sleeves = [
        ("pro_trend", pt_eq.pct_change().fillna(0), pt_alloc),
        ("xsmom", xs_eq.pct_change().fillna(0), xs_alloc),
        ("bah", bah_eq.pct_change().fillna(0), bah_alloc),
    ]
    indices = [s[1].index for s in sleeves if not s[1].empty]
    if not indices:
        return pd.Series(dtype=float)
    common = indices[0]
    for idx in indices[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        return pd.Series(dtype=float)

    combined_rets = pd.Series(0.0, index=common)
    for name, rets, alloc in sleeves:
        if not rets.empty:
            combined_rets += rets.loc[common] * alloc
    eq = 100_000 * (1 + combined_rets).cumprod()
    return eq


if __name__ == "__main__":
    print("=" * 100)
    print("HYBRID CYCLE-AWARE BAH BTC TEST")
    print("=" * 100)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())

    windows = [
        ("A: 2020-21 mega-bull",
         pd.Timestamp("2020-01-21", tz="UTC"), pd.Timestamp("2021-04-24", tz="UTC")),
        ("B: 2021-22 top/LUNA",
         pd.Timestamp("2021-04-25", tz="UTC"), pd.Timestamp("2022-07-28", tz="UTC")),
        ("C: 2022-23 bear/recovery",
         pd.Timestamp("2022-07-29", tz="UTC"), pd.Timestamp("2023-10-31", tz="UTC")),
        ("D: 2024-26 recent chop",
         pd.Timestamp("2024-11-01", tz="UTC"), end_date),
        ("ALL: full 6.3y",
         pd.Timestamp("2020-01-21", tz="UTC"), end_date),
    ]

    variants = {
        "V0 unconditional 20% (current)":
            dict(base_alloc=0.20, high_alloc=0.20, low_alloc=0.20,
                 sell_threshold=99, buy_threshold=-1),
        "H1 modest (15/20/25, 1.5/0.8)":
            dict(base_alloc=0.20, high_alloc=0.15, low_alloc=0.25,
                 sell_threshold=1.5, buy_threshold=0.8),
        "H2 moderate (10/20/30, 1.5/0.8)":
            dict(base_alloc=0.20, high_alloc=0.10, low_alloc=0.30,
                 sell_threshold=1.5, buy_threshold=0.8),
        "H3 aggressive (0/20/30, 1.5/0.8)":
            dict(base_alloc=0.20, high_alloc=0.00, low_alloc=0.30,
                 sell_threshold=1.5, buy_threshold=0.8),
        "H4 wider (10/20/30, 1.6/0.7)":
            dict(base_alloc=0.20, high_alloc=0.10, low_alloc=0.30,
                 sell_threshold=1.6, buy_threshold=0.7),
        "H5 very wide (0/20/35, 1.8/0.6)":
            dict(base_alloc=0.20, high_alloc=0.00, low_alloc=0.35,
                 sell_threshold=1.8, buy_threshold=0.6),
    }

    print(f"{'Variant':<36s}", end="")
    for label, _, _ in windows:
        print(f"  {label[:14]:>16s}", end="")
    print()
    print("-" * 36 + ("-" * 18) * len(windows))

    results = {}
    for name, params in variants.items():
        results[name] = {}
        print(f"{name:<36s}", end="")
        for label, ds, de in windows:
            try:
                eq = portfolio_with_hybrid_bah(pair_data, ds, de, params)
                stats = perf_stats(eq)
                results[name][label] = stats
                cell = f"{stats['annualized']:+5.1%}/Sh{stats['sharpe']:+3.1f}"
                print(f"  {cell:>16s}", end="")
            except Exception as e:
                print(f"  {'ERR':>16s}", end="")
                results[name][label] = {"error": str(e)}
        print()

    print()
    print("=" * 100)
    print("RANKED BY FULL 6.3Y SHARPE")
    print("=" * 100)
    ranked = sorted([(a, r.get("ALL: full 6.3y", {})) for a, r in results.items()],
                    key=lambda x: -x[1].get("sharpe", 0))
    print(f"{'Variant':<36s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for a, r in ranked:
        if "error" in r:
            continue
        marker = "  <-- CURRENT" if "current" in a else ""
        print(f"{a:<36s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}{marker}")

    print()
    print("=" * 100)
    print("RANKED BY RECENT 18-MONTH")
    print("=" * 100)
    ranked_d = sorted([(a, r.get("D: 2024-26 recent chop", {})) for a, r in results.items()],
                      key=lambda x: -x[1].get("sharpe", 0))
    print(f"{'Variant':<36s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for a, r in ranked_d:
        if "error" in r:
            continue
        marker = "  <-- CURRENT" if "current" in a else ""
        print(f"{a:<36s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}{marker}")

    print()
    print("=" * 100)
    print("BEAR-WINDOW STRESS TEST (B: 2021-22 top/LUNA — where BAH HURT)")
    print("=" * 100)
    ranked_b = sorted([(a, r.get("B: 2021-22 top/LUNA", {})) for a, r in results.items()],
                      key=lambda x: -x[1].get("sharpe", 0))
    print(f"{'Variant':<36s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for a, r in ranked_b:
        if "error" in r:
            continue
        marker = "  <-- CURRENT" if "current" in a else ""
        print(f"{a:<36s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}{marker}")

    print()
    print("=" * 100)
    print("CURRENT Mayer + current allocation per each variant")
    print("=" * 100)
    btc = pair_data["BTC/USDT"]
    btc = btc.copy()
    btc["sma200"] = btc["close"].rolling(200).mean()
    btc["mayer"] = btc["close"] / btc["sma200"]
    current_mayer = float(btc["mayer"].iloc[-1])
    print(f"Today's Mayer Multiple: {current_mayer:.2f}")
    print(f"BTC price:              ${btc['close'].iloc[-1]:,.0f}")
    print(f"SMA200:                 ${btc['sma200'].iloc[-1]:,.0f}")
    print()
    for name, params in variants.items():
        if current_mayer > params["sell_threshold"]:
            alloc = params["high_alloc"]
            zone = "high (expensive)"
        elif current_mayer < params["buy_threshold"]:
            alloc = params["low_alloc"]
            zone = "low (cheap)"
        else:
            alloc = params["base_alloc"]
            zone = "mid (fair)"
        print(f"  {name:<36s} -> {alloc:.0%} BAH ({zone})")
