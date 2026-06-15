"""Direct comparison: long-only vs long+short with various filters.

Hypothesis from exhaustion_filter_test: shorts don't add value to pro_trend
on crypto majors. Long-only baseline should match or beat the best shorted
variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exhaustion_filter_test import (
    fetch_all_with_donch, portfolio_run_with_exhaustion, PAIRS,
)


def long_only_backtest(pair_data, **kw):
    """Wrapper: set exhaustion_pct very high so longs aren't filtered, but
    disable shorts by setting min_pct (not used; we hack via temp disabling).
    Easier: create a variant that simply skips the short branch."""
    return portfolio_run_with_exhaustion_long_only(pair_data, **kw)


def portfolio_run_with_exhaustion_long_only(
    pair_data, exhaustion_pct=0.0, starting_equity=100_000.0,
    base_risk=0.04, portfolio_risk_cap=0.15, atr_stop_mult=4.0,
    pyramid_atr_step=2.0, max_pyramid_units=2, drawdown_kill_pct=0.35,
    round_trip_bps=30.0, date_start=None, date_end=None,
):
    """Same as portfolio_run_with_exhaustion but SHORTS DISABLED."""
    if date_start or date_end:
        pair_data = {
            p: df.loc[(df.index >= (date_start or df.index[0]))
                       & (df.index <= (date_end or df.index[-1]))]
            for p, df in pair_data.items()
        }
        pair_data = {p: df for p, df in pair_data.items() if not df.empty}
        if not pair_data:
            return {"error": "no data"}

    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0, "side": None} for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    n_trades = 0
    n_dd_kills = 0

    for today in all_dates:
        active_rows = {p: df.loc[today] for p, df in pair_data.items() if today in df.index}
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
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
            n_dd_kills += 1
            equity_path.append({"ts": today, "equity": cash})
            continue

        n_active = sum(1 for st in state.values() if st["units"])

        for p, row in active_rows.items():
            st = state[p]
            price, high, low = float(row["close"]), float(row["high"]), float(row["low"])
            atr, sma = float(row["atr"]), float(row["sma_filter"])
            donchian_high = float(row["donchian_high"])
            in_bull = price > sma
            pct_from_sma = price / sma - 1

            if st["units"]:  # LONG only
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
                    state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
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
                # LONG-ONLY entry
                if in_bull and high >= donchian_high:
                    if exhaustion_pct > 0 and pct_from_sma > exhaustion_pct:
                        continue
                    per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                    risk = min(base_risk, per_pair_max)
                    stop_dist = atr_stop_mult * atr
                    qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                    if qty > 0 and atr > 0:
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        state[p] = {
                            "units": [{"qty": qty, "entry_price": price, "entry_atr": atr}],
                            "side": "long", "extreme": high, "trail_stop": price - stop_dist,
                        }
                        n_trades += 1

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
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 0 else 0
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (365 / max(n_days, 1)) - 1

    return {
        "n_days": n_days, "final_equity": final_eq,
        "total_return": total_return, "annualized": annualized,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades, "n_dd_kills": n_dd_kills,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("HEAD-TO-HEAD: long-only vs long+short variants")
    print("=" * 80)
    print()

    pair_data = fetch_all_with_donch(PAIRS, days_back=2500)
    end_date = max(df.index[-1] for df in pair_data.values())
    sixty_days_ago = end_date - pd.Timedelta(days=60)

    print("[1] FULL 6.3-YEAR HISTORY")
    print("-" * 80)
    print(f"{'Variant':<35s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'Trades':>7s}")
    variants = [
        ("LONG-ONLY no filter", lambda d, **kw: portfolio_run_with_exhaustion_long_only(d, exhaustion_pct=0.0, **kw)),
        ("LONG-ONLY +20% exhaustion cap", lambda d, **kw: portfolio_run_with_exhaustion_long_only(d, exhaustion_pct=0.20, **kw)),
        ("LONG-ONLY +25% exhaustion cap", lambda d, **kw: portfolio_run_with_exhaustion_long_only(d, exhaustion_pct=0.25, **kw)),
        ("LONG+SHORT no filter (current)", lambda d, **kw: portfolio_run_with_exhaustion(d, exhaustion_pct=0.0, **kw)),
        ("LONG+SHORT 20% exhaustion", lambda d, **kw: portfolio_run_with_exhaustion(d, exhaustion_pct=0.20, **kw)),
        ("LONG+SHORT 25% exhaustion", lambda d, **kw: portfolio_run_with_exhaustion(d, exhaustion_pct=0.25, **kw)),
    ]
    full_results = {}
    for label, fn in variants:
        r = fn(pair_data)
        full_results[label] = r
        print(f"{label:<35s}   {r['annualized']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']:>5.1%}   {r['n_trades']:>5d}")
    print()

    print("[2] RECENT 60-DAY WINDOW")
    print("-" * 80)
    print(f"{'Variant':<35s} {'Return':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'Trades':>7s}")
    recent_results = {}
    for label, fn in variants:
        r = fn(pair_data, date_start=sixty_days_ago, date_end=end_date)
        recent_results[label] = r
        if "error" in r:
            print(f"{label:<35s}   {r['error']}")
            continue
        print(f"{label:<35s}   {r['total_return']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']:>5.1%}   {r['n_trades']:>5d}")
    print()

    print("=" * 80)
    print("RANK BY: Full Sharpe + recent 60d performance")
    print("=" * 80)
    print(f"{'Variant':<35s} {'Full Sharpe':>11s} {'60d Return':>11s} {'Combined Score':>15s}")
    # Score = full_sharpe * (1 + recent_return) — penalizes recent losses
    ranked = sorted(full_results.items(),
                    key=lambda x: -(x[1]["sharpe"] * (1 + recent_results[x[0]].get("total_return", 0))))
    for label, r in ranked:
        rec = recent_results[label].get("total_return", 0)
        score = r["sharpe"] * (1 + rec)
        print(f"{label:<35s}   {r['sharpe']:>+8.2f}    {rec:>+8.2%}    {score:>+12.2f}")
