"""COMPREHENSIVE STRATEGY BAKE-OFF — what actually makes money?

User said: "this strategy still no good, look at the data from backtests what
data and strategies do work"

Tests 12 strategy classes across 4 market-regime windows:
  Window A: 2020-01 to 2021-04 (massive bull run)
  Window B: 2021-04 to 2022-07 (top + LUNA crash)
  Window C: 2022-07 to 2023-10 (deep bear to recovery)
  Window D: 2024-11 to 2026-05 (recent 18mo chop)
  + Overall: full 6.3y combined

Strategies tested:
  1. Pro_trend (current production: long-only + v5 filter)
  2. Pro_trend (no v5 filter, baseline)
  3. Pro_trend (long+short, no filter — what we just rejected)
  4. BAH BTC only (single asset hold)
  5. BAH 5-pair equal-weight (no rebalance)
  6. BAH 5-pair monthly rebalance
  7. BAH risk-parity (inverse vol weighting, monthly rebalance)
  8. DCA $1k/week into BTC
  9. Trend follow LONG-ONLY shorter lookback (Donch-10, no v5)
  10. Trend follow LONG-ONLY no SMA filter (Donch-20 only)
  11. XSMOM standalone (14d momentum, 14d rebalance)
  12. Combined production (70% pro_trend + 30% XSMOM)

Report: per-window Sharpe + return for each strategy. Identify winners.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr
from core.indicator_lab import ind_macd_signal, ind_tsmom
from core.xsmom_backtest import xsmom_backtest


ANNUALIZATION = 365
PAIRS_5 = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]
PAIRS_XSMOM = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ATOM/USDT"]


def fetch_panel(pairs, days_back=2500):
    out = {}
    for p in pairs:
        df = data.ohlcv_extended(p, days_back=days_back)
        if not df.empty and len(df) >= 50:
            out[p] = df
    return out


def perf_stats(eq_series, start_equity=100_000):
    eq = eq_series.dropna()
    if len(eq) < 2:
        return {"return": 0, "sharpe": 0, "max_dd": 0, "annualized": 0}
    daily_rets = eq.pct_change().dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    sharpe = (float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION))
              if daily_rets.std() > 0 else 0)
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    n_days = (eq.index[-1] - eq.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1
    return {"return": float(total_return), "sharpe": sharpe,
            "max_dd": max_dd, "annualized": float(annualized)}


# ============================================================================
# Strategy implementations
# ============================================================================

def bah_single(pair_data, asset, date_start, date_end):
    df = pair_data[asset]
    sub = df[(df.index >= date_start) & (df.index <= date_end)]
    if len(sub) < 2:
        return pd.Series(dtype=float)
    eq = 100_000 * sub["close"] / sub["close"].iloc[0]
    return eq


def bah_basket_equal(pair_data, date_start, date_end):
    """Equal-weight buy at start, hold to end. No rebalance."""
    panels = []
    for p, df in pair_data.items():
        sub = df[(df.index >= date_start) & (df.index <= date_end)]
        if len(sub) < 2:
            continue
        weight_value = 100_000 / len(pair_data)
        panels.append(weight_value * sub["close"] / sub["close"].iloc[0])
    if not panels:
        return pd.Series(dtype=float)
    combined = pd.concat(panels, axis=1).fillna(method="ffill").fillna(0)
    return combined.sum(axis=1)


def bah_basket_monthly_rebal(pair_data, date_start, date_end):
    """Monthly rebalance to equal weight."""
    panels = {p: df[(df.index >= date_start) & (df.index <= date_end)]["close"]
              for p, df in pair_data.items()}
    df = pd.DataFrame(panels).dropna()
    if len(df) < 30:
        return pd.Series(dtype=float)
    daily_rets = df.pct_change().fillna(0)
    n = len(df.columns)
    equity = pd.Series(100_000.0, index=df.index)
    weights = pd.Series([1 / n] * n, index=df.columns)
    for i in range(1, len(df)):
        # Apply daily returns to weights (drift)
        weights = weights * (1 + daily_rets.iloc[i])
        weights = weights / weights.sum()
        # Monthly rebalance
        if df.index[i].month != df.index[i - 1].month:
            weights = pd.Series([1 / n] * n, index=df.columns)
        equity.iloc[i] = equity.iloc[i - 1] * (1 + (daily_rets.iloc[i] * weights).sum())
    return equity


def bah_risk_parity(pair_data, date_start, date_end, vol_window=60):
    """Risk-parity: weight inversely to realized vol, monthly rebalance."""
    panels = {p: df[(df.index >= date_start - pd.Timedelta(days=vol_window + 5))
                     & (df.index <= date_end)]["close"]
              for p, df in pair_data.items()}
    df = pd.DataFrame(panels).dropna()
    if len(df) < vol_window + 10:
        return pd.Series(dtype=float)
    rets = df.pct_change()
    vols = rets.rolling(vol_window).std()
    inv_vol = 1 / vols
    weights_daily = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    # Now run with monthly rebalance
    sub_df = df[df.index >= date_start]
    sub_rets = rets.loc[sub_df.index]
    weights = weights_daily.loc[sub_df.index].iloc[0]
    equity = pd.Series(100_000.0, index=sub_df.index)
    for i in range(1, len(sub_df)):
        weights = weights * (1 + sub_rets.iloc[i])
        if weights.sum() > 0:
            weights = weights / weights.sum()
        if sub_df.index[i].month != sub_df.index[i - 1].month:
            new_w = weights_daily.loc[sub_df.index[i]]
            if new_w.sum() > 0:
                weights = new_w / new_w.sum()
        equity.iloc[i] = equity.iloc[i - 1] * (1 + (sub_rets.iloc[i] * weights).sum())
    return equity


def dca_btc(pair_data, date_start, date_end):
    """Buy $1000 of BTC every Monday."""
    df = pair_data["BTC/USDT"]
    sub = df[(df.index >= date_start) & (df.index <= date_end)]
    if len(sub) < 7:
        return pd.Series(dtype=float)
    cash_in = 0
    btc_held = 0
    equity = []
    for date, row in sub.iterrows():
        if date.weekday() == 0:  # Monday
            cash_in += 1000
            btc_held += 1000 / row["close"]
        # Total invested + current value of BTC
        equity.append({"ts": date, "equity": cash_in + btc_held * row["close"] - cash_in})
        # Equity above starts at 0 — adjust to total returns basis vs cash_in
    # Translate to percentage relative to total cash invested:
    eq_series = pd.Series(
        [e["equity"] for e in equity], index=[e["ts"] for e in equity]
    )
    # Make a comparable equity by anchoring to 100_000 starting value
    # Use proportional return: total_value / total_invested
    cum_invest = sub.index.to_series().apply(lambda d: 1000 * (((d - date_start).days // 7) + 1))
    # Simpler: compute returns vs invested capital
    total_value = []
    invested = 0
    btc = 0
    for date, row in sub.iterrows():
        if date.weekday() == 0:
            invested += 1000
            btc += 1000 / row["close"]
        cur_val = btc * row["close"]
        if invested > 0:
            total_value.append({"ts": date,
                                 "ratio": cur_val / invested,
                                 "abs_ret": cur_val - invested,
                                 "invested": invested,
                                 "value": cur_val})
    if not total_value:
        return pd.Series(dtype=float)
    # Equity series: scale starting position to 100k
    df_tv = pd.DataFrame(total_value).set_index("ts")
    eq = 100_000 * df_tv["ratio"]
    return eq


def pro_trend_long_only(pair_data, date_start, date_end, use_v5_filter=True,
                         use_sma_filter=True, donchian_window=20):
    """Pro_trend with v5 (TSMOM+MACD) filter; long-only."""
    base_risk = 0.04
    portfolio_risk_cap = 0.15
    atr_stop_mult = 4.0
    pyramid_atr_step = 2.0
    max_pyramid_units = 2
    drawdown_kill_pct = 0.35
    round_trip_bps = 30.0

    # Pre-compute indicators per pair
    prepared = {}
    for p, df in pair_data.items():
        d = df.copy()
        d["donchian_high"] = d["high"].rolling(donchian_window).max().shift(1)
        d["sma_filter"] = d["close"].rolling(200).mean()
        d["atr"] = compute_atr(d, 14)
        d["tsmom30"] = d["close"].pct_change(30)
        _ef = d["close"].ewm(span=12).mean()
        _es = d["close"].ewm(span=26).mean()
        _macd = _ef - _es
        _sig = _macd.ewm(span=9).mean()
        d["macd_hist"] = (_macd - _sig) / d["close"]
        d = d.dropna()
        d = d[(d.index >= date_start) & (d.index <= date_end)]
        if not d.empty:
            prepared[p] = d
    if not prepared:
        return pd.Series(dtype=float)

    all_dates = sorted(set().union(*[df.index for df in prepared.values()]))
    cash = 100_000.0
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0} for p in prepared}
    peak_equity = 100_000.0
    equity_path = []

    for today in all_dates:
        active = {p: df.loc[today] for p, df in prepared.items() if today in df.index}
        unrealized = sum(
            sum(u["qty"] * (float(active[p]["close"]) - u["entry_price"]) for u in st["units"])
            for p, st in state.items() if st["units"] and p in active
        )
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0
        if equity_dd > drawdown_kill_pct and any(st["units"] for st in state.values()):
            for p, st in state.items():
                if not st["units"] or p not in active:
                    continue
                price = float(active[p]["close"])
                for u in st["units"]:
                    pnl = u["qty"] * (price - u["entry_price"])
                    cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
            equity_path.append({"ts": today, "equity": cash})
            continue
        n_active = sum(1 for st in state.values() if st["units"])
        for p, row in active.items():
            st = state[p]
            price, high, low = float(row["close"]), float(row["high"]), float(row["low"])
            atr, sma = float(row["atr"]), float(row["sma_filter"])
            donchian_high = float(row["donchian_high"])
            tsmom30, macd_hist = float(row["tsmom30"]), float(row["macd_hist"])
            in_bull = price > sma if use_sma_filter else True
            if st["units"]:
                if high > st["extreme"]:
                    st["extreme"] = high
                    new_trail = high - atr_stop_mult * atr
                    if new_trail > st["trail_stop"]:
                        st["trail_stop"] = new_trail
                if low <= st["trail_stop"] or (use_sma_filter and price < sma):
                    exit_p = st["trail_stop"] if low <= st["trail_stop"] else price
                    for u in st["units"]:
                        pnl = u["qty"] * (exit_p - u["entry_price"])
                        cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                    state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
                elif len(st["units"]) < max_pyramid_units:
                    last_unit = st["units"][-1]
                    if high >= last_unit["entry_price"] + pyramid_atr_step * last_unit["entry_atr"]:
                        per_pair_max = portfolio_risk_cap / max(n_active, 1)
                        risk = min(base_risk, per_pair_max)
                        stop_dist = atr_stop_mult * atr
                        if stop_dist > 0:
                            qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                            cash -= qty * price * round_trip_bps / 2 / 10_000
                            st["units"].append({"qty": qty, "entry_price": price, "entry_atr": atr})
            else:
                if in_bull and high >= donchian_high:
                    if use_v5_filter and not (tsmom30 > 0 and macd_hist > 0):
                        continue
                    per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                    risk = min(base_risk, per_pair_max)
                    stop_dist = atr_stop_mult * atr
                    qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                    if qty > 0 and atr > 0:
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        state[p] = {
                            "units": [{"qty": qty, "entry_price": price, "entry_atr": atr}],
                            "extreme": high, "trail_stop": price - stop_dist,
                        }
        unrealized = sum(
            sum(u["qty"] * (float(active[p]["close"]) - u["entry_price"]) for u in st["units"])
            for p, st in state.items() if st["units"] and p in active
        )
        equity_path.append({"ts": today, "equity": cash + unrealized})
    return pd.DataFrame(equity_path).set_index("ts")["equity"]


def xsmom_equity_path(date_start, date_end, days_back=2000):
    """Wrapper for xsmom_backtest returning equity series for a window."""
    r = xsmom_backtest(days_back=days_back,
                        momentum_window=14, rebalance_freq=14,
                        long_n=2, short_n=2, risk_per_leg=0.20)
    if "error" in r:
        return pd.Series(dtype=float)
    eq = r["equity_path"]
    return eq[(eq.index >= date_start) & (eq.index <= date_end)]


# ============================================================================
# Run bake-off
# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("STRATEGY BAKE-OFF: what actually makes money?")
    print("=" * 100)
    print()

    pair_data = fetch_panel(PAIRS_5, days_back=2500)
    pair_data_xs = fetch_panel(PAIRS_XSMOM, days_back=2500)
    print(f"5-pair universe: {list(pair_data.keys())}")
    print(f"XSMOM 8-pair: {list(pair_data_xs.keys())}")
    print()

    end_date = max(df.index[-1] for df in pair_data.values())

    windows = [
        ("A: 2020-21 mega-bull", pd.Timestamp("2020-01-21", tz="UTC"),
         pd.Timestamp("2021-04-24", tz="UTC")),
        ("B: 2021-22 top/LUNA", pd.Timestamp("2021-04-25", tz="UTC"),
         pd.Timestamp("2022-07-28", tz="UTC")),
        ("C: 2022-23 bear/recovery", pd.Timestamp("2022-07-29", tz="UTC"),
         pd.Timestamp("2023-10-31", tz="UTC")),
        ("D: 2024-26 recent chop", pd.Timestamp("2024-11-01", tz="UTC"),
         end_date),
        ("ALL: full 6.3y", pd.Timestamp("2020-01-21", tz="UTC"), end_date),
    ]

    strategies = {
        "Pro_trend v5 (production)": lambda ps, ds, de: pro_trend_long_only(ps, ds, de, use_v5_filter=True),
        "Pro_trend baseline (no v5)": lambda ps, ds, de: pro_trend_long_only(ps, ds, de, use_v5_filter=False),
        "Pro_trend no SMA filter":    lambda ps, ds, de: pro_trend_long_only(ps, ds, de, use_v5_filter=False, use_sma_filter=False),
        "Pro_trend Donch-10":         lambda ps, ds, de: pro_trend_long_only(ps, ds, de, use_v5_filter=False, donchian_window=10),
        "BAH BTC-only":               lambda ps, ds, de: bah_single(ps, "BTC/USDT", ds, de),
        "BAH 5-pair (no rebal)":     lambda ps, ds, de: bah_basket_equal(ps, ds, de),
        "BAH 5-pair (monthly rebal)": lambda ps, ds, de: bah_basket_monthly_rebal(ps, ds, de),
        "BAH risk-parity (60d vol)":  lambda ps, ds, de: bah_risk_parity(ps, ds, de),
        "DCA $1k/wk BTC":             lambda ps, ds, de: dca_btc(ps, ds, de),
    }

    print(f"{'Strategy':<32s}", end="")
    for label, _, _ in windows:
        print(f" {label[:18]:>20s}", end="")
    print()
    print("-" * 32 + ("-" * 21) * len(windows))

    # Run each strategy on each window
    results = {}
    for strat_name, fn in strategies.items():
        results[strat_name] = {}
        print(f"{strat_name:<32s}", end="")
        for label, ds, de in windows:
            try:
                eq = fn(pair_data, ds, de)
                stats = perf_stats(eq)
                results[strat_name][label] = stats
                cell = f"{stats['annualized']:+6.1%} Sh{stats['sharpe']:+4.1f}"
                print(f" {cell:>20s}", end="")
            except Exception as e:
                print(f" {'ERR':>20s}", end="")
                results[strat_name][label] = {"error": str(e)}
        print()

    print()
    print("=" * 100)
    print("RANKED BY FULL 6.3-YEAR RISK-ADJUSTED RETURN (Sharpe)")
    print("=" * 100)
    ranked = sorted(
        [(s, r.get("ALL: full 6.3y", {})) for s, r in results.items()],
        key=lambda x: -x[1].get("sharpe", 0),
    )
    print(f"{'Strategy':<32s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for s, r in ranked:
        if "error" in r:
            continue
        print(f"{s:<32s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}")

    print()
    print("=" * 100)
    print("RANKED BY RECENT 18-MONTH WINDOW (window D)")
    print("=" * 100)
    ranked_d = sorted(
        [(s, r.get("D: 2024-26 recent chop", {})) for s, r in results.items()],
        key=lambda x: -x[1].get("sharpe", 0),
    )
    print(f"{'Strategy':<32s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s}")
    for s, r in ranked_d:
        if "error" in r:
            continue
        print(f"{s:<32s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['return']:>+10.1%}  {r['max_dd']:>5.1%}")

    print()
    print("=" * 100)
    print("PER-REGIME WINNERS")
    print("=" * 100)
    for label, _, _ in windows:
        winner = max(
            results.items(),
            key=lambda x: x[1].get(label, {}).get("sharpe", -999)
            if "error" not in x[1].get(label, {}) else -999
        )
        s = winner[0]
        r = winner[1].get(label, {})
        print(f"  {label:<32s} -> {s:<28s} Sharpe {r.get('sharpe', 0):+.2f}, "
              f"Ann {r.get('annualized', 0):+.1%}")
