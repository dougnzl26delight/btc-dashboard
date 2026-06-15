"""Test: does extreme mean reversion work as a chop-regime hedge for pro_trend?

Two questions:
  1. What's the standalone Sharpe/return of extreme mean reversion across
     the same 5-pair universe?
  2. What's the correlation of its DAILY P&L to pro_trend's DAILY P&L?
     If <0.3, it's genuinely diversifying. If >0.5, it's redundant.

Approach: rebuild both strategies on the same date axis, capture daily
equity curves, compute return correlation. Test combined portfolio
(70% pro_trend + 30% mean rev) vs pro_trend alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.comprehensive_backtest import fetch_all, portfolio_run
from core.extreme_mean_reversion_backtest import extreme_revert_backtest


ANNUALIZATION = 365
UNIVERSE = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]


def mean_rev_portfolio(pairs: list, days_back: int = 2500,
                       starting_equity: float = 100_000.0) -> dict:
    """Run extreme mean reversion across pairs, capture combined equity curve.

    Capital splits equally per pair. Each pair has independent state but
    we aggregate to a daily portfolio equity series.
    """
    per_pair_eq = {}
    capital_per_pair = starting_equity / len(pairs)

    for p in pairs:
        try:
            r = extreme_revert_backtest(
                pair=p, days_back=days_back,
                starting_equity=capital_per_pair,
                risk_pct_per_trade=0.015,
            )
            if "error" in r:
                continue
            # extreme_revert_backtest doesn't expose equity_path directly; we
            # reconstruct via per-trade simulation. Easiest: re-run with a
            # mod that captures equity. For now, use the summary stats.
            per_pair_eq[p] = r
        except Exception as e:
            print(f"  {p}: {e}")
            continue

    if not per_pair_eq:
        return {"error": "no successful backtests"}

    annlzds = np.array([r["annualized_return"] for r in per_pair_eq.values()])
    sharpes = np.array([r["sharpe"] for r in per_pair_eq.values()])
    dds = np.array([r["max_drawdown"] for r in per_pair_eq.values()])
    n_trades = np.array([r["n_trades"] for r in per_pair_eq.values()])

    return {
        "n_pairs": len(per_pair_eq),
        "mean_annualized": float(annlzds.mean()),
        "median_annualized": float(np.median(annlzds)),
        "mean_sharpe": float(sharpes.mean()),
        "n_positive": int((annlzds > 0).sum()),
        "mean_max_dd": float(dds.mean()),
        "total_trades": int(n_trades.sum()),
        "per_pair": per_pair_eq,
    }


def mean_rev_with_equity_path(
    pair: str, days_back: int = 2500,
    starting_equity: float = 20_000.0,  # 1/5 of $100k
    return_window: int = 5,
    z_window: int = 60,
    z_entry: float = -2.5,
    z_exit: float = -0.5,
    max_hold_days: int = 10,
    risk_pct_per_trade: float = 0.015,
    stop_pct: float = 0.10,
    round_trip_bps: float = 30.0,
) -> pd.DataFrame:
    """Re-implementation that exposes equity_path for correlation analysis."""
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty or len(df) < z_window * 2:
        return pd.DataFrame()
    df = df.copy()
    df["ret_window"] = df["close"].pct_change(return_window)
    df["ret_mean"] = df["ret_window"].rolling(z_window).mean()
    df["ret_std"] = df["ret_window"].rolling(z_window).std()
    df["z"] = (df["ret_window"] - df["ret_mean"]) / df["ret_std"]
    df = df.dropna()

    cash = starting_equity
    in_trade = False
    qty = 0.0
    entry_price = 0.0
    entry_idx = -1
    stop_price = 0.0
    equity_path = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        z = float(row["z"])

        if in_trade:
            if price <= stop_price:
                exit_p = stop_price
                pnl = qty * (exit_p - entry_price)
                cash += pnl - qty * exit_p * round_trip_bps / 2 / 10_000
                in_trade = False
                qty = 0
            elif z > z_exit or (i - entry_idx) >= max_hold_days:
                exit_p = price
                pnl = qty * (exit_p - entry_price)
                cash += pnl - qty * exit_p * round_trip_bps / 2 / 10_000
                in_trade = False
                qty = 0
        else:
            if z < z_entry:
                qty = (cash * risk_pct_per_trade) / (price * stop_pct)
                qty = min(qty, cash * 0.30 / price)
                cash -= qty * price * round_trip_bps / 2 / 10_000
                entry_price = price
                entry_idx = i
                stop_price = price * (1 - stop_pct)
                in_trade = True

        unrealized = qty * (price - entry_price) if in_trade else 0
        equity_path.append({"ts": row.name, "equity": cash + unrealized + (qty * price if in_trade else 0)})

    if in_trade:
        exit_p = float(df["close"].iloc[-1])
        cash += qty * (exit_p - entry_price) - qty * exit_p * round_trip_bps / 2 / 10_000

    return pd.DataFrame(equity_path).set_index("ts")


def combine_equity_paths(paths: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum equity paths across pairs to get portfolio equity."""
    if not paths:
        return pd.DataFrame()
    combined = pd.concat([p["equity"] for p in paths], axis=1)
    combined.columns = [f"p{i}" for i in range(len(paths))]
    combined = combined.fillna(method="ffill").fillna(0)
    combined["total"] = combined.sum(axis=1)
    return combined


def equity_to_returns(eq: pd.Series) -> pd.Series:
    return eq.pct_change().dropna()


def stats(rets: pd.Series, label: str = "") -> dict:
    if len(rets) < 30:
        return {}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION)) if rets.std() > 0 else 0
    eq = (1 + rets).cumprod()
    total_ret = float(eq.iloc[-1] - 1)
    n_days = (rets.index[-1] - rets.index[0]).days
    ann = (1 + total_ret) ** (ANNUALIZATION / max(n_days, 1)) - 1
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    return {"label": label, "ann": float(ann), "sharpe": sharpe, "max_dd": max_dd}


if __name__ == "__main__":
    print("=" * 78)
    print("MEAN REVERSION AS CHOP HEDGE — TEST")
    print("=" * 78)
    print()

    # === [1] Standalone mean rev across 5 pairs ===
    print("[1] Standalone mean reversion on 5-pair universe")
    print("-" * 78)
    mr = mean_rev_portfolio(UNIVERSE, days_back=2500)
    if "error" in mr:
        print(f"FAIL: {mr['error']}")
        sys.exit(1)
    print(f"Pairs:           {mr['n_pairs']}")
    print(f"Mean annualized: {mr['mean_annualized']:+.2%}")
    print(f"Median:          {mr['median_annualized']:+.2%}")
    print(f"Mean Sharpe:     {mr['mean_sharpe']:+.2f}")
    print(f"Pairs positive:  {mr['n_positive']}/{mr['n_pairs']}")
    print(f"Mean max DD:     {mr['mean_max_dd']:.1%}")
    print(f"Total trades:    {mr['total_trades']}")
    print()
    print("Per-pair details:")
    for p, r in mr["per_pair"].items():
        print(f"  {p:<12s} ann {r['annualized_return']:>+7.2%}  "
              f"Sharpe {r['sharpe']:>+5.2f}  DD {r['max_drawdown']:>5.1%}  "
              f"trades {r['n_trades']}")
    print()

    # Quick gate: if standalone Sharpe is bad, kill the idea
    if mr["mean_sharpe"] < 0.3:
        print(f"GATE FAILED: mean Sharpe {mr['mean_sharpe']:.2f} < 0.3")
        print("Mean reversion alone is too weak to add as a sleeve.")
        print("Recommendation: skip this lever.")
        sys.exit(0)

    # === [2] Combine with pro_trend, measure correlation ===
    print("[2] Build daily equity paths for both strategies, measure correlation")
    print("-" * 78)

    # pro_trend equity path
    pair_data = fetch_all(days_back=2500)
    pt_result = portfolio_run(
        pair_data=pair_data,
        starting_equity=70_000.0,    # 70% allocation
        base_risk=0.04,
        portfolio_risk_cap=0.15,
    )
    pt_eq = pt_result["equity_path"]["equity"]

    # mean rev: aggregate equity paths across pairs
    mr_paths = []
    for p in UNIVERSE:
        eq = mean_rev_with_equity_path(p, days_back=2500,
                                       starting_equity=30_000.0 / len(UNIVERSE))
        if not eq.empty:
            mr_paths.append(eq)
    mr_combined = combine_equity_paths(mr_paths)
    mr_eq = mr_combined["total"] if "total" in mr_combined.columns else pd.Series()

    if mr_eq.empty:
        print("Mean reversion equity path empty — skipping combined test")
        sys.exit(0)

    # Align dates
    common = pt_eq.index.intersection(mr_eq.index)
    pt_eq = pt_eq.loc[common]
    mr_eq = mr_eq.loc[common]

    # Daily returns
    pt_rets = equity_to_returns(pt_eq)
    mr_rets = equity_to_returns(mr_eq)
    common = pt_rets.index.intersection(mr_rets.index)
    pt_rets = pt_rets.loc[common]
    mr_rets = mr_rets.loc[common]

    correlation = float(pt_rets.corr(mr_rets))
    print(f"Daily P&L correlation:  {correlation:+.3f}")
    print(f"  Interpretation:")
    print(f"  >  0.5: redundant — same regime exposure")
    print(f"  0.0-0.5: somewhat correlated (typical for crypto strategies)")
    print(f"  < 0.0: genuinely uncorrelated / hedging")
    print()

    # Combined 70/30 portfolio
    combined_rets = 0.7 * pt_rets + 0.3 * mr_rets
    pt_stats = stats(pt_rets, "pro_trend (70% allocation, equiv 100% capital)")
    mr_stats = stats(mr_rets, "mean_rev (30% allocation, equiv 100% capital)")
    combined_stats = stats(combined_rets, "combined 70/30")

    print("[3] Stats comparison")
    print("-" * 78)
    print(f"{'Strategy':<40s} {'Annlzd':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for s in [pt_stats, mr_stats, combined_stats]:
        if s:
            print(f"{s['label']:<40s} {s['ann']:>+7.2%} {s['sharpe']:>+6.2f} "
                  f"{s['max_dd']:>6.1%}")
    print()

    print("=" * 78)
    print("DECISION GATE")
    print("=" * 78)
    if correlation < 0.3 and mr_stats and mr_stats["sharpe"] > 0.3 and combined_stats:
        sharpe_improvement = combined_stats["sharpe"] - pt_stats["sharpe"]
        print(f"PASS: correlation {correlation:.2f} < 0.3, "
              f"mean rev Sharpe {mr_stats['sharpe']:.2f} > 0.3")
        print(f"Sharpe improvement: {sharpe_improvement:+.2f}")
        print(f"Recommendation: WIRE mean reversion sleeve at 30% allocation")
    else:
        print(f"FAIL gate criteria: correlation {correlation:.2f} (need <0.3), "
              f"mean rev Sharpe {mr_stats.get('sharpe', 0):.2f} (need >0.3)")
        print("Recommendation: skip mean reversion sleeve")
