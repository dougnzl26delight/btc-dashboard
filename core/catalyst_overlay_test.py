"""Test: does the BTC halving-cycle multiplier add value, or is it curve-fit?

The current overlay scales position size:
  Months 0-6 post-halving:  1.0x
  Months 6-18:               1.5x  (peak bull)
  Months 18-30:              1.0x  (distribution)
  Months 30-48:              0.5x  (bear)

Backtests on top-5 universe with portcap-15 (lever 2 winner) under variants:
  none        — flat 1.0x always
  default     — 1.0/1.5/1.0/0.5 schedule (current)
  steep       — 0.5/1.5/1.0/0.5 (more aggressive bear scaling)
  flat-bull   — 1.0/1.0/1.0/1.0 (kill the overlay; sanity check vs none)
  inverse     — 1.5/1.0/1.0/1.5 (counter-intuitive: overweight in bear)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.catalyst_signals import HALVINGS
from core.swing_backtest import compute_atr


ANNUALIZATION = 365
UNIVERSE = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]


SCHEDULES = {
    "none":       lambda m: 1.0,
    "default":    lambda m: 1.0 if m < 6 else (1.5 if m < 18 else (1.0 if m < 30 else 0.5)),
    "steep":      lambda m: 0.5 if m < 6 else (1.5 if m < 18 else (1.0 if m < 30 else 0.5)),
    "flat-bull":  lambda m: 1.0,  # same as none — control
    "inverse":    lambda m: 1.5 if m < 6 else (1.0 if m < 18 else (1.0 if m < 30 else 1.5)),
    "two-stage":  lambda m: 1.5 if m < 18 else 0.7,  # simpler bull/bear
}


def months_since_halving(d: pd.Timestamp) -> float:
    today = d.date()
    past = [h for h in HALVINGS if h <= today]
    if not past:
        return 999
    days = (today - max(past)).days
    return days / 30.4


def fetch_all(days_back: int = 1500) -> dict[str, pd.DataFrame]:
    out = {}
    for p in UNIVERSE:
        df = data.ohlcv_extended(p, days_back=days_back)
        if df.empty or len(df) < 250:
            continue
        df = df.copy()
        df["donchian_high"] = df["high"].rolling(20).max().shift(1)
        df["sma_filter"] = df["close"].rolling(200).mean()
        df["atr"] = compute_atr(df, 14)
        df = df.dropna()
        out[p] = df
    return out


def portfolio_backtest_with_catalyst(
    pair_data: dict[str, pd.DataFrame],
    schedule_fn,
    starting_equity: float = 100_000.0,
    base_risk: float = 0.04,
    portfolio_risk_cap: float = 0.15,
    atr_stop_mult: float = 4.0,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.35,
    round_trip_bps: float = 30.0,
) -> dict:
    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0} for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    n_trades = 0

    for d_idx, today in enumerate(all_dates):
        m_since = months_since_halving(today)
        catalyst = schedule_fn(m_since)
        active_rows = {p: df.loc[today] for p, df in pair_data.items() if today in df.index}

        # MTM
        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            for u in st["units"]:
                unrealized += u["qty"] * (price - u["entry_price"])
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        if equity_dd > drawdown_kill_pct and any(st["units"] for st in state.values()):
            for p, st in state.items():
                if not st["units"] or p not in active_rows:
                    continue
                price = float(active_rows[p]["close"])
                for u in st["units"]:
                    pnl = u["qty"] * (price - u["entry_price"])
                    cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                    n_trades += 1
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
            continue

        n_active = sum(1 for st in state.values() if st["units"])

        for p, row in active_rows.items():
            st = state[p]
            price, high, low = float(row["close"]), float(row["high"]), float(row["low"])
            atr, sma, donchian = float(row["atr"]), float(row["sma_filter"]), float(row["donchian_high"])

            if st["units"]:
                if high > st["extreme"]:
                    st["extreme"] = high
                    new_trail = high - atr_stop_mult * atr
                    if new_trail > st["trail_stop"]:
                        st["trail_stop"] = new_trail
                if low <= st["trail_stop"] or price < sma:
                    exit_p = st["trail_stop"] if low <= st["trail_stop"] else price
                    for u in st["units"]:
                        pnl = u["qty"] * (exit_p - u["entry_price"])
                        cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                        n_trades += 1
                    state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
                elif len(st["units"]) < max_pyramid_units:
                    last = st["units"][-1]
                    if high >= last["entry_price"] + pyramid_atr_step * last["entry_atr"]:
                        per_pair_max = portfolio_risk_cap / max(n_active, 1)
                        risk = min(base_risk, per_pair_max) * catalyst
                        stop_dist = atr_stop_mult * atr
                        if stop_dist > 0:
                            qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                            cash -= qty * price * round_trip_bps / 2 / 10_000
                            st["units"].append({"qty": qty, "entry_price": price,
                                                 "entry_atr": atr})
            else:
                if price > sma and high >= donchian and atr > 0:
                    per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                    risk = min(base_risk, per_pair_max) * catalyst
                    stop_dist = atr_stop_mult * atr
                    qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                    if qty > 0:
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        st["units"] = [{"qty": qty, "entry_price": price, "entry_atr": atr}]
                        st["extreme"] = high
                        st["trail_stop"] = price - stop_dist

        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            for u in st["units"]:
                unrealized += u["qty"] * (price - u["entry_price"])
        equity_path.append({"ts": today, "equity": cash + unrealized})

    final_day = all_dates[-1]
    for p, st in state.items():
        if st["units"] and final_day in pair_data[p].index:
            price = float(pair_data[p].loc[final_day, "close"])
            for u in st["units"]:
                pnl = u["qty"] * (price - u["entry_price"])
                cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"]
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION)) if daily_rets.std() > 0 else 0
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": n_trades,
    }


if __name__ == "__main__":
    print("Fetching data...")
    pair_data = fetch_all(days_back=1500)
    print(f"  {len(pair_data)} pairs")
    print()

    rows = []
    for label, fn in SCHEDULES.items():
        r = portfolio_backtest_with_catalyst(pair_data=pair_data, schedule_fn=fn)
        print(f"{label:<14s}  ann {r['annualized_return']:>+7.2%}  "
              f"Sharpe {r['sharpe']:>+5.2f}  DD {r['max_drawdown']:>5.1%}  "
              f"trades {r['n_trades']:>3d}")
        rows.append({"label": label, **r})

    print()
    print("=" * 70)
    print("RANKED:")
    print("=" * 70)
    rows.sort(key=lambda x: -x["sharpe"])
    for r in rows:
        print(f"{r['label']:<14s}  Sharpe {r['sharpe']:>+5.2f}  "
              f"ann {r['annualized_return']:>+7.2%}  DD {r['max_drawdown']:>5.1%}")
