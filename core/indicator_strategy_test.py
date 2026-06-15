"""Phase 3: Test top indicator combos AS ENTRY SIGNALS for pro_trend mechanics.

Hypothesis: pro_trend's wide-stop/pyramid/trail logic is the secret sauce.
The entry signal (currently Donchian-20-high) might be improvable. Replace
or augment the entry with top indicators found in Phase 1/2.

Same exit logic (4 ATR trail + SMA200 break + DD kill + pyramid at +2 ATR).
Only the ENTRY signal varies.

Variants tested:
  v0  baseline: SMA200 + Donchian-20-high (current production)
  v1  baseline + MACD_hist > 0 (require positive momentum)
  v2  baseline + CCI_20 > 0
  v3  baseline + RSI14 > 50
  v4  replace Donchian with TSMOM_30 > 0
  v5  baseline + TSMOM_30 > 0 + MACD_hist > 0 (top pairwise combo)
  v6  baseline + 'fresh' filter: pct_from_sma in (+2%, +25%)

Walk-forward 5 OOS folds on each variant on full 4-year history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr
from core.indicator_lab import (
    ind_macd_signal, ind_cci, ind_rsi, ind_tsmom,
)


ANNUALIZATION = 365
PAIRS = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]


def fetch_with_all_indicators(pairs, days_back=1700):
    out = {}
    for p in pairs:
        df = data.ohlcv_extended(p, days_back=days_back)
        if df.empty or len(df) < 250:
            continue
        df = df.copy()
        df["donchian_high"] = df["high"].rolling(20).max().shift(1)
        df["sma_filter"] = df["close"].rolling(200).mean()
        df["atr"] = compute_atr(df, 14)
        df["macd"] = ind_macd_signal(df, 12, 26, 9)
        df["cci"] = ind_cci(df, 20)
        df["rsi"] = ind_rsi(df, 14)
        df["tsmom30"] = ind_tsmom(df, 30)
        df = df.dropna()
        out[p] = df
    return out


def long_only_backtest(
    pair_data,
    entry_signal_fn,
    starting_equity=100_000.0,
    base_risk=0.04,
    portfolio_risk_cap=0.15,
    atr_stop_mult=4.0,
    pyramid_atr_step=2.0,
    max_pyramid_units=2,
    drawdown_kill_pct=0.35,
    round_trip_bps=30.0,
    date_start=None, date_end=None,
):
    """Run pro_trend mechanics with custom entry signal. LONG-ONLY."""
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
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0} for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    n_trades = 0
    n_entries = 0

    for today in all_dates:
        active_rows = {p: df.loc[today] for p, df in pair_data.items() if today in df.index}
        unrealized = sum(
            sum(u["qty"] * (float(active_rows[p]["close"]) - u["entry_price"]) for u in st["units"])
            for p, st in state.items() if st["units"] and p in active_rows
        )
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
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
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
                if entry_signal_fn(row, price, sma, high, low):
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
                        n_trades += 1
                        n_entries += 1

        unrealized = sum(
            sum(u["qty"] * (float(active_rows[p]["close"]) - u["entry_price"]) for u in st["units"])
            for p, st in state.items() if st["units"] and p in active_rows
        )
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
        "annualized": annualized, "sharpe": sharpe, "max_dd": max_dd,
        "n_trades": n_trades, "n_entries": n_entries, "n_days": n_days,
    }


# ============================================================================
# Entry signal definitions
# ============================================================================

def sig_baseline(row, price, sma, high, low):
    """Baseline: price > SMA200 AND Donchian-20-high break."""
    return price > sma and high >= row["donchian_high"]


def sig_baseline_plus_macd(row, price, sma, high, low):
    return (price > sma and high >= row["donchian_high"]
            and row["macd"] > 0)


def sig_baseline_plus_cci(row, price, sma, high, low):
    return (price > sma and high >= row["donchian_high"]
            and row["cci"] > 0)


def sig_baseline_plus_rsi(row, price, sma, high, low):
    return (price > sma and high >= row["donchian_high"]
            and row["rsi"] > 50)


def sig_tsmom_only(row, price, sma, high, low):
    """No Donchian; just TSMOM positive + SMA filter."""
    return price > sma and row["tsmom30"] > 0


def sig_baseline_plus_tsmom_macd(row, price, sma, high, low):
    """Top pairwise combo + baseline."""
    return (price > sma and high >= row["donchian_high"]
            and row["tsmom30"] > 0 and row["macd"] > 0)


def sig_baseline_fresh(row, price, sma, high, low):
    """Baseline + 'fresh long' filter: price 2-25% above SMA."""
    pct_from_sma = price / sma - 1
    return (price > sma and high >= row["donchian_high"]
            and 0.02 <= pct_from_sma <= 0.25)


VARIANTS = {
    "v0_baseline":           sig_baseline,
    "v1_+MACD":              sig_baseline_plus_macd,
    "v2_+CCI":               sig_baseline_plus_cci,
    "v3_+RSI>50":            sig_baseline_plus_rsi,
    "v4_TSMOM_only":         sig_tsmom_only,
    "v5_+TSMOM+MACD":        sig_baseline_plus_tsmom_macd,
    "v6_fresh_window":       sig_baseline_fresh,
}


# ============================================================================
def walk_forward(pair_data, signal_fn, n_folds=5):
    """N consecutive non-overlapping folds; return per-fold Sharpes."""
    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    if len(all_dates) < n_folds * 90:
        return []
    fold_size = len(all_dates) // n_folds
    results = []
    for k in range(n_folds):
        start = all_dates[k * fold_size]
        end = all_dates[(k + 1) * fold_size - 1] if k < n_folds - 1 else all_dates[-1]
        r = long_only_backtest(pair_data, signal_fn, date_start=start, date_end=end)
        if "error" not in r:
            results.append({
                "fold": k + 1,
                "start": str(start.date()), "end": str(end.date()),
                "sharpe": r["sharpe"], "annualized": r["annualized"],
                "max_dd": r["max_dd"], "n_entries": r["n_entries"],
            })
    return results


if __name__ == "__main__":
    print("=" * 90)
    print("INDICATOR STRATEGY TEST — full pro_trend mechanics, custom entry signals")
    print("=" * 90)
    print()

    pair_data = fetch_with_all_indicators(PAIRS, days_back=1700)
    print(f"Pairs: {list(pair_data.keys())}")
    print(f"Bars: {[len(df) for df in pair_data.values()]}")
    print()

    print("[1] FULL-WINDOW BACKTEST")
    print("-" * 90)
    print(f"{'Variant':<22s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'Entries':>8s}")
    full_results = {}
    for name, sig_fn in VARIANTS.items():
        r = long_only_backtest(pair_data, sig_fn)
        full_results[name] = r
        print(f"{name:<22s} {r['annualized']:>+8.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_dd']:>5.1%}    {r['n_entries']:>5d}")
    print()

    print("[2] WALK-FORWARD VALIDATION (5 folds, ~340 days each)")
    print("-" * 90)
    print(f"{'Variant':<22s} {'MeanSharpe':>11s} {'StdSharpe':>10s} {'+folds':>7s} {'MinSharpe':>10s}")
    wf_results = {}
    for name, sig_fn in VARIANTS.items():
        folds = walk_forward(pair_data, sig_fn, n_folds=5)
        if not folds:
            continue
        sharpes = [f["sharpe"] for f in folds]
        wf_results[name] = {
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes, ddof=1),
            "min_sharpe": min(sharpes),
            "n_positive": sum(s > 0 for s in sharpes),
            "folds": folds,
        }
        print(f"{name:<22s} {np.mean(sharpes):>+10.2f}  {np.std(sharpes, ddof=1):>9.2f}   "
              f"{sum(s > 0 for s in sharpes):>3d}/5    {min(sharpes):>+9.2f}")
    print()

    print("[3] RANKING — combined score (full Sharpe + WF mean - WF std)")
    print("-" * 90)
    ranked = []
    for name in VARIANTS:
        full = full_results.get(name, {})
        wf = wf_results.get(name, {})
        if not full or not wf:
            continue
        score = full["sharpe"] + wf["mean_sharpe"] - wf["std_sharpe"]
        ranked.append((name, full, wf, score))
    ranked.sort(key=lambda x: -x[3])
    print(f"{'Variant':<22s} {'Full Sh':>8s} {'WF Mean':>8s} {'WF Std':>7s} {'Score':>7s}")
    for name, full, wf, score in ranked:
        marker = "  <-- best" if name == ranked[0][0] else ""
        print(f"{name:<22s} {full['sharpe']:>+7.2f}  {wf['mean_sharpe']:>+7.2f}  "
              f"{wf['std_sharpe']:>6.2f}  {score:>+6.2f}{marker}")
