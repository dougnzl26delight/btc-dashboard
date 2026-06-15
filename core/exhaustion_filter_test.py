"""Test 'exhaustion filter' variants on pro_trend strategy.

The 60-day live-style sim revealed all 4 losing shorts fired when price
was 25-68% below SMA200 — the bear move was EXHAUSTED, not starting.

Filter: only allow short entries when price is within [SMA200 * (1 - EXH),
SMA200 * (1 - MIN)] — i.e., clearly below SMA200 but not too deeply.

Symmetric for longs: price within [SMA200 * (1 + MIN), SMA200 * (1 + EXH)].

Tests 4 variants on full 2300-day history + the recent 60-day window:
  v0: no filter (baseline, current production)
  v15: exhaustion at 15% (most aggressive filter)
  v20: exhaustion at 20%
  v25: exhaustion at 25%
  v30: exhaustion at 30%

Gate: must NOT degrade full-history Sharpe by more than 10%, AND must
materially improve recent 60d (less negative).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr


def fetch_all_with_donch(pairs, days_back=2500):
    out = {}
    for p in pairs:
        df = data.ohlcv_extended(p, days_back=days_back)
        if df.empty or len(df) < 250:
            continue
        df = df.copy()
        df["donchian_high"] = df["high"].rolling(20).max().shift(1)
        df["donchian_low"] = df["low"].rolling(20).min().shift(1)
        df["sma_filter"] = df["close"].rolling(200).mean()
        df["atr"] = compute_atr(df, 14)
        df = df.dropna()
        out[p] = df
    return out


PAIRS = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]


ANNUALIZATION = 365


def portfolio_run_with_exhaustion(
    pair_data: dict,
    exhaustion_pct: float = 0.0,   # 0 = no filter; 0.15 = max 15% from SMA200
    min_pct: float = 0.0,          # min |distance| from SMA200 to require freshness
    starting_equity: float = 100_000.0,
    base_risk: float = 0.04,
    portfolio_risk_cap: float = 0.15,
    atr_stop_mult: float = 4.0,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.35,
    round_trip_bps: float = 30.0,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> dict:
    """Pro_trend backtest with optional exhaustion filter."""
    if date_start or date_end:
        pair_data = {
            p: df.loc[(df.index >= (date_start or df.index[0]))
                       & (df.index <= (date_end or df.index[-1]))]
            for p, df in pair_data.items()
        }
        pair_data = {p: df for p, df in pair_data.items() if not df.empty}
        if not pair_data:
            return {"error": "no data in window"}

    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
             for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    n_trades = 0
    n_dd_kills = 0
    n_entries_filtered = 0

    for today in all_dates:
        active_rows = {p: df.loc[today] for p, df in pair_data.items() if today in df.index}
        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            sign = 1 if st["side"] == "long" else -1
            for u in st["units"]:
                unrealized += sign * u["qty"] * (price - u["entry_price"])
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        if equity_dd > drawdown_kill_pct and any(st["units"] for st in state.values()):
            for p, st in state.items():
                if not st["units"] or p not in active_rows:
                    continue
                price = float(active_rows[p]["close"])
                sign = 1 if st["side"] == "long" else -1
                for u in st["units"]:
                    pnl = sign * u["qty"] * (price - u["entry_price"])
                    cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
            n_dd_kills += 1
            equity_path.append({"ts": today, "equity": cash})
            continue

        n_active = sum(1 for st in state.values() if st["units"])

        for p, row in active_rows.items():
            st = state[p]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["atr"])
            sma = float(row["sma_filter"])
            donchian_high = float(row["donchian_high"])
            donchian_low = float(row["donchian_low"])
            in_bull = price > sma
            pct_from_sma = price / sma - 1  # negative = below SMA200

            if st["units"]:
                if st["side"] == "long":
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
                else:  # short
                    if st["extreme"] == 0 or low < st["extreme"]:
                        st["extreme"] = low
                        new_trail = low + atr_stop_mult * atr
                        if st["trail_stop"] == 0 or new_trail < st["trail_stop"]:
                            st["trail_stop"] = new_trail
                    if high >= st["trail_stop"] or price > sma:
                        exit_p = st["trail_stop"] if high >= st["trail_stop"] else price
                        for u in st["units"]:
                            pnl = -u["qty"] * (exit_p - u["entry_price"])
                            cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                            n_trades += 1
                        state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
                    elif len(st["units"]) < max_pyramid_units:
                        last_unit = st["units"][-1]
                        if low <= last_unit["entry_price"] - pyramid_atr_step * last_unit["entry_atr"]:
                            per_pair_max = portfolio_risk_cap / max(n_active, 1)
                            risk = min(base_risk, per_pair_max)
                            stop_dist = atr_stop_mult * atr
                            if stop_dist > 0:
                                qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                                cash -= qty * price * round_trip_bps / 2 / 10_000
                                st["units"].append({"qty": qty, "entry_price": price, "entry_atr": atr})
            else:
                # === ENTRY with EXHAUSTION FILTER ===
                per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                risk = min(base_risk, per_pair_max)
                stop_dist = atr_stop_mult * atr
                qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                if qty <= 0 or atr <= 0:
                    continue

                # LONG: price > SMA200 AND high >= Donchian high
                if in_bull and high >= donchian_high:
                    # Exhaustion filter: don't go long if price > SMA200 * (1 + exhaustion_pct)
                    if exhaustion_pct > 0 and pct_from_sma > exhaustion_pct:
                        n_entries_filtered += 1
                        continue
                    cash -= qty * price * round_trip_bps / 2 / 10_000
                    state[p] = {
                        "units": [{"qty": qty, "entry_price": price, "entry_atr": atr}],
                        "side": "long", "extreme": high, "trail_stop": price - stop_dist,
                    }
                    n_trades += 1
                # SHORT: price < SMA200 AND low <= Donchian low
                elif not in_bull and low <= donchian_low:
                    # Exhaustion filter: don't short if price < SMA200 * (1 - exhaustion_pct)
                    if exhaustion_pct > 0 and pct_from_sma < -exhaustion_pct:
                        n_entries_filtered += 1
                        continue
                    cash -= qty * price * round_trip_bps / 2 / 10_000
                    state[p] = {
                        "units": [{"qty": qty, "entry_price": price, "entry_atr": atr}],
                        "side": "short", "extreme": low, "trail_stop": price + stop_dist,
                    }
                    n_trades += 1

        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            sign = 1 if st["side"] == "long" else -1
            for u in st["units"]:
                unrealized += sign * u["qty"] * (price - u["entry_price"])
        equity_path.append({"ts": today, "equity": cash + unrealized})

    # Close at end
    final_day = all_dates[-1]
    for p, st in state.items():
        if st["units"] and final_day in pair_data[p].index:
            price = float(pair_data[p].loc[final_day, "close"])
            sign = 1 if st["side"] == "long" else -1
            for u in st["units"]:
                pnl = sign * u["qty"] * (price - u["entry_price"])
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
        "n_days": n_days,
        "final_equity": final_eq,
        "total_return": total_return,
        "annualized": annualized,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_dd_kills": n_dd_kills,
        "n_entries_filtered": n_entries_filtered,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("EXHAUSTION FILTER TEST")
    print("=" * 80)
    print()

    pair_data = fetch_all_with_donch(PAIRS, days_back=2500)
    print(f"Universe: {list(pair_data.keys())}")
    end_date = max(df.index[-1] for df in pair_data.values())
    sixty_days_ago = end_date - pd.Timedelta(days=60)
    print(f"End date: {end_date.date()}")
    print()

    print("[1] FULL 2300-DAY HISTORY")
    print("-" * 80)
    print(f"{'Filter':<8s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s} "
          f"{'Trades':>7s} {'Filtered':>9s}")
    full_results = {}
    for label, exh in [("v0 (none)", 0.0), ("v10", 0.10), ("v15", 0.15),
                       ("v20", 0.20), ("v25", 0.25), ("v30", 0.30)]:
        r = portfolio_run_with_exhaustion(pair_data, exhaustion_pct=exh)
        full_results[label] = r
        print(f"{label:<8s}   {r['annualized']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']:>5.1%}   {r['n_trades']:>5d}    {r['n_entries_filtered']:>5d}")
    print()

    print("[2] RECENT 60-DAY WINDOW (does it skip the 4 losing shorts?)")
    print("-" * 80)
    print(f"{'Filter':<8s} {'Return':>9s} {'Sharpe':>7s} {'MaxDD':>7s} "
          f"{'Trades':>7s} {'Filtered':>9s}")
    recent_results = {}
    for label, exh in [("v0 (none)", 0.0), ("v10", 0.10), ("v15", 0.15),
                       ("v20", 0.20), ("v25", 0.25), ("v30", 0.30)]:
        r = portfolio_run_with_exhaustion(
            pair_data, exhaustion_pct=exh,
            date_start=sixty_days_ago, date_end=end_date,
        )
        recent_results[label] = r
        if "error" in r:
            print(f"{label:<8s}   {r['error']}")
            continue
        print(f"{label:<8s}   {r['total_return']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']:>5.1%}   {r['n_trades']:>5d}    {r['n_entries_filtered']:>5d}")
    print()

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    # Find variant with: smallest 60d loss AND full Sharpe >= 0.9 * v0
    baseline_full_sharpe = full_results["v0 (none)"]["sharpe"]
    print(f"Baseline (v0) full-history Sharpe: {baseline_full_sharpe:+.2f}")
    print(f"Baseline (v0) recent 60d return: {recent_results['v0 (none)']['total_return']:+.2%}")
    print()
    print(f"{'Filter':<8s} {'Full Sharpe':>11s} {'vs v0':>8s} {'60d Return':>11s} {'Improvement':>12s}")
    for label in ["v10", "v15", "v20", "v25", "v30"]:
        full_s = full_results[label]["sharpe"]
        rec_r = recent_results[label]["total_return"]
        rec_baseline = recent_results["v0 (none)"]["total_return"]
        ratio = full_s / baseline_full_sharpe if baseline_full_sharpe != 0 else 0
        improvement = rec_r - rec_baseline
        marker = ""
        if ratio >= 0.9 and improvement > 0.05:
            marker = "  <-- PASSES (Sharpe >=90% baseline, 60d > +5pp improvement)"
        print(f"{label:<8s}   {full_s:>+8.2f}    {ratio:>5.0%}    "
              f"{rec_r:>+8.2%}    {improvement:>+9.2%}{marker}")
