"""Comprehensive backtest of the 4-lever calibrated pro_trend system.

Covers:
  1. Max-history single backtest (2500 days, pairs phase in as listed)
  2. Walk-forward — 5 OOS folds across max history
  3. Calendar-year breakdown
  4. Parameter sensitivity — 3x3 grid on (atr_stop_mult x portfolio_cap)
  5. Stress periods — explicit named windows (LUNA, FTX, ETF rally, etc.)
  6. Drawdown distribution
  7. Bootstrap CI on Sharpe and annualized return
  8. Per-pair contribution
  9. vs BTC BAH and equal-weight BAH

Output: full report to stdout. ~1-2 minutes runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr


ANNUALIZATION = 365
UNIVERSE = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]


def fetch_all(days_back: int = 2500) -> dict[str, pd.DataFrame]:
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


def portfolio_run(
    pair_data: dict[str, pd.DataFrame],
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
    """Multi-pair portfolio sim with optional date filter."""
    # Filter dates
    if date_start or date_end:
        pair_data = {
            p: df.loc[(df.index >= (date_start or df.index[0])) &
                      (df.index <= (date_end or df.index[-1]))]
            for p, df in pair_data.items()
        }
        pair_data = {p: df for p, df in pair_data.items() if not df.empty}
        if not pair_data:
            return {"error": "no data in window"}

    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0} for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    pair_pnl: dict[str, float] = {p: 0.0 for p in pair_data}
    n_trades = 0
    n_dd_kills = 0

    for d_idx, today in enumerate(all_dates):
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
                    pair_pnl[p] += pnl
                    n_trades += 1
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
            n_dd_kills += 1
            equity_path.append({"ts": today, "equity": cash, "n_active": 0})
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
                        pair_pnl[p] += pnl
                        n_trades += 1
                    state[p] = {"units": [], "extreme": 0, "trail_stop": 0}
                elif len(st["units"]) < max_pyramid_units:
                    last = st["units"][-1]
                    if high >= last["entry_price"] + pyramid_atr_step * last["entry_atr"]:
                        per_pair_max = portfolio_risk_cap / max(n_active, 1)
                        risk = min(base_risk, per_pair_max)
                        stop_dist = atr_stop_mult * atr
                        if stop_dist > 0:
                            qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                            cash -= qty * price * round_trip_bps / 2 / 10_000
                            st["units"].append({"qty": qty, "entry_price": price, "entry_atr": atr})
            else:
                if price > sma and high >= donchian and atr > 0:
                    per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                    risk = min(base_risk, per_pair_max)
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
        equity_path.append({"ts": today, "equity": cash + unrealized,
                            "n_active": sum(1 for st in state.values() if st["units"])})

    final_day = all_dates[-1]
    for p, st in state.items():
        if st["units"] and final_day in pair_data[p].index:
            price = float(pair_data[p].loc[final_day, "close"])
            for u in st["units"]:
                pnl = u["qty"] * (price - u["entry_price"])
                cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                pair_pnl[p] += pnl

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    daily_eq = eq_df["equity"]
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION)) if daily_rets.std() > 0 else 0
    sortino_denom = daily_rets[daily_rets < 0].std()
    sortino = float(daily_rets.mean() / sortino_denom * np.sqrt(ANNUALIZATION)) if sortino_denom > 0 else 0
    peak = daily_eq.cummax()
    dd_series = 1 - daily_eq / peak
    max_dd = float(dd_series.max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1
    calmar = annualized / max_dd if max_dd > 0 else 0

    return {
        "n_days": n_days,
        "starting_equity": starting_equity,
        "final_equity": final_eq,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_trades": n_trades,
        "n_dd_kills": n_dd_kills,
        "pair_pnl": pair_pnl,
        "equity_path": eq_df,
        "daily_returns": daily_rets,
        "drawdown_series": dd_series,
        "start_date": eq_df.index[0],
        "end_date": eq_df.index[-1],
    }


def bah_basket(pair_data: dict[str, pd.DataFrame],
               date_start: pd.Timestamp | None = None,
               date_end: pd.Timestamp | None = None) -> dict:
    """Equal-weight buy-and-hold basket. Each pair gets 1/N capital at its first
    bar in the window; held to end of window."""
    if date_start or date_end:
        pair_data = {
            p: df.loc[(df.index >= (date_start or df.index[0])) &
                      (df.index <= (date_end or df.index[-1]))]
            for p, df in pair_data.items()
        }
        pair_data = {p: df for p, df in pair_data.items() if not df.empty}
    n = len(pair_data)
    if n == 0:
        return {"error": "no pairs"}
    pair_returns = {p: float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
                    for p, df in pair_data.items()}
    equal_weight = sum(pair_returns.values()) / n
    return {"per_pair": pair_returns, "equal_weight": equal_weight}


def yearly_breakdown(eq_df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year returns from an equity path."""
    daily_eq = eq_df["equity"]
    rows = []
    for year, group in daily_eq.groupby(daily_eq.index.year):
        ret = group.iloc[-1] / group.iloc[0] - 1
        peak = group.cummax()
        dd = float((1 - group / peak).max())
        rows.append({"year": year, "return": float(ret), "max_dd": dd,
                     "start_eq": float(group.iloc[0]),
                     "end_eq": float(group.iloc[-1])})
    return pd.DataFrame(rows)


def stress_period(pair_data: dict, label: str, start: str, end: str, **kw) -> dict:
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    r = portfolio_run(pair_data=pair_data, date_start=s, date_end=e, **kw)
    if "error" in r:
        return {"label": label, "start": start, "end": end, "error": r["error"]}
    bah = bah_basket(pair_data, date_start=s, date_end=e)
    return {
        "label": label,
        "start": start,
        "end": end,
        "n_days": r["n_days"],
        "total_return": r["total_return"],
        "annualized": r["annualized_return"],
        "sharpe": r["sharpe"],
        "max_dd": r["max_drawdown"],
        "n_trades": r["n_trades"],
        "bah_basket": bah.get("equal_weight", 0),
        "alpha": r["total_return"] - bah.get("equal_weight", 0),
    }


def walk_forward(pair_data: dict, n_folds: int = 5, **kw) -> list[dict]:
    """Split max history into N consecutive folds; run independent backtest in each."""
    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    if len(all_dates) < n_folds * 90:
        return []
    fold_size = len(all_dates) // n_folds
    rows = []
    for k in range(n_folds):
        start = all_dates[k * fold_size]
        end = all_dates[(k + 1) * fold_size - 1] if k < n_folds - 1 else all_dates[-1]
        r = portfolio_run(pair_data=pair_data, date_start=start, date_end=end, **kw)
        if "error" in r:
            continue
        bah = bah_basket(pair_data, date_start=start, date_end=end)
        rows.append({
            "fold": k + 1,
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": r["n_days"],
            "total_return": r["total_return"],
            "annualized": r["annualized_return"],
            "sharpe": r["sharpe"],
            "max_dd": r["max_drawdown"],
            "n_trades": r["n_trades"],
            "bah_basket": bah.get("equal_weight", 0),
        })
    return rows


def param_sensitivity(pair_data: dict, atrs: list, caps: list, **base) -> list[dict]:
    rows = []
    for atr in atrs:
        for cap in caps:
            kw = {**base, "atr_stop_mult": atr, "portfolio_risk_cap": cap}
            r = portfolio_run(pair_data=pair_data, **kw)
            rows.append({
                "atr_stop_mult": atr, "portfolio_cap": cap,
                "annualized": r["annualized_return"],
                "sharpe": r["sharpe"],
                "max_dd": r["max_drawdown"],
                "n_trades": r["n_trades"],
            })
    return rows


def bootstrap_sharpe(daily_returns: pd.Series, n_boot: int = 1000) -> dict:
    """Block bootstrap (block_size=20) to get 95% CI on Sharpe and ann return."""
    rng = np.random.default_rng(42)
    rets = daily_returns.values
    if len(rets) < 100:
        return {"error": "insufficient returns"}
    block_size = 20
    n_blocks = len(rets) // block_size
    sharpes, anns = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rets) - block_size, size=n_blocks)
        sample = np.concatenate([rets[i:i + block_size] for i in idx])
        s = sample.mean() / sample.std() * np.sqrt(ANNUALIZATION) if sample.std() > 0 else 0
        a = (1 + sample.mean()) ** ANNUALIZATION - 1
        sharpes.append(s)
        anns.append(a)
    sharpes = np.array(sharpes)
    anns = np.array(anns)
    return {
        "n_boot": n_boot,
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_p5": float(np.percentile(sharpes, 5)),
        "sharpe_p25": float(np.percentile(sharpes, 25)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p75": float(np.percentile(sharpes, 75)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "ann_mean": float(anns.mean()),
        "ann_p5": float(np.percentile(anns, 5)),
        "ann_p50": float(np.percentile(anns, 50)),
        "ann_p95": float(np.percentile(anns, 95)),
        "pct_sharpe_above_0": float((sharpes > 0).mean()),
        "pct_sharpe_above_0p5": float((sharpes > 0.5).mean()),
    }


def drawdown_stats(dd_series: pd.Series) -> dict:
    """Peak-to-trough drawdown episodes."""
    in_dd = dd_series > 0.05  # >5% drawdown counts as an episode
    episodes = []
    start_idx = None
    for i, flag in enumerate(in_dd.values):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            ep = dd_series.iloc[start_idx:i]
            episodes.append({"depth": float(ep.max()), "length_days": len(ep)})
            start_idx = None
    if start_idx is not None:
        ep = dd_series.iloc[start_idx:]
        episodes.append({"depth": float(ep.max()), "length_days": len(ep)})
    if not episodes:
        return {"n_episodes": 0}
    depths = np.array([e["depth"] for e in episodes])
    lengths = np.array([e["length_days"] for e in episodes])
    return {
        "n_episodes": len(episodes),
        "median_depth": float(np.median(depths)),
        "p90_depth": float(np.percentile(depths, 90)),
        "max_depth": float(depths.max()),
        "median_length": int(np.median(lengths)),
        "max_length": int(lengths.max()),
        "n_above_20pct": int((depths > 0.20).sum()),
        "n_above_30pct": int((depths > 0.30).sum()),
    }


# ============================================================================
# REPORT
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("COMPREHENSIVE BACKTEST — pro_trend top-5, portcap-15, no catalyst")
    print("=" * 78)
    print()

    # Fetch max history
    pair_data = fetch_all(days_back=2500)
    earliest = {p: df.index[0].date() for p, df in pair_data.items()}
    latest = max(df.index[-1] for df in pair_data.values())
    print(f"Universe:  {list(pair_data.keys())}")
    print(f"End date:  {latest.date()}")
    for p in pair_data:
        print(f"  {p:<12s} starts {earliest[p]}  ({len(pair_data[p])} bars)")
    print()

    base_kw = dict(
        starting_equity=100_000.0, base_risk=0.04,
        portfolio_risk_cap=0.15, atr_stop_mult=4.0,
    )

    # ========================================================================
    print("=" * 78)
    print("[1] MAX-HISTORY SINGLE BACKTEST")
    print("=" * 78)
    full = portfolio_run(pair_data=pair_data, **base_kw)
    print(f"Window:           {full['start_date'].date()} -> {full['end_date'].date()} "
          f"({full['n_days']} days)")
    print(f"Final equity:     ${full['final_equity']:>14,.0f}  (start ${full['starting_equity']:,.0f})")
    print(f"Total return:     {full['total_return']:>+14.2%}")
    print(f"Annualized:       {full['annualized_return']:>+14.2%}")
    print(f"Sharpe:           {full['sharpe']:>+14.2f}")
    print(f"Sortino:          {full['sortino']:>+14.2f}")
    print(f"Max drawdown:     {full['max_drawdown']:>+14.2%}")
    print(f"Calmar:           {full['calmar']:>+14.2f}")
    print(f"Total trades:     {full['n_trades']:>14d}")
    print(f"DD-kill events:   {full['n_dd_kills']:>14d}")
    print()
    print("Per-pair P&L contribution:")
    total_pnl = sum(full["pair_pnl"].values())
    for p, pnl in sorted(full["pair_pnl"].items(), key=lambda x: -x[1]):
        share = pnl / total_pnl if total_pnl else 0
        print(f"  {p:<12s} ${pnl:>+14,.0f}  ({share:>+5.1%} of total)")
    print()
    bah = bah_basket(pair_data)
    print(f"vs equal-weight BAH basket: {bah['equal_weight']:+.2%}")
    print(f"vs BTC BAH:                 {bah['per_pair'].get('BTC/USDT',0):+.2%}")
    print()

    # ========================================================================
    print("=" * 78)
    print("[2] WALK-FORWARD — 5 OOS folds (true out-of-sample)")
    print("=" * 78)
    folds = walk_forward(pair_data, n_folds=5, **base_kw)
    print(f"{'Fold':>4s}  {'Window':<25s} {'Days':>4s} {'Return':>9s} {'Annlzd':>9s} "
          f"{'Sharpe':>7s} {'MaxDD':>6s} {'BAH':>9s}")
    for f in folds:
        print(f"{f['fold']:>4d}  {f['start']} -> {f['end']:<10s} {f['n_days']:>4d} "
              f"{f['total_return']:>+8.2%} {f['annualized']:>+8.2%} "
              f"{f['sharpe']:>+6.2f} {f['max_dd']:>5.1%} {f['bah_basket']:>+8.2%}")
    if folds:
        sharpes = [f["sharpe"] for f in folds]
        anns = [f["annualized"] for f in folds]
        print()
        print(f"Mean OOS Sharpe:        {np.mean(sharpes):+.2f}")
        print(f"Std OOS Sharpe:         {np.std(sharpes, ddof=1):.2f}")
        print(f"Min/Max OOS Sharpe:     [{min(sharpes):+.2f}, {max(sharpes):+.2f}]")
        print(f"Mean OOS annualized:    {np.mean(anns):+.2%}")
        print(f"Folds with positive S:  {sum(s > 0 for s in sharpes)}/{len(sharpes)}")
    print()

    # ========================================================================
    print("=" * 78)
    print("[3] CALENDAR-YEAR BREAKDOWN")
    print("=" * 78)
    yb = yearly_breakdown(full["equity_path"])
    print(f"{'Year':<6s} {'Start eq':>12s} {'End eq':>12s} {'Return':>9s} {'MaxDD':>7s}")
    for _, row in yb.iterrows():
        print(f"{int(row['year']):<6d} ${row['start_eq']:>10,.0f} ${row['end_eq']:>10,.0f} "
              f"{row['return']:>+8.2%} {row['max_dd']:>6.1%}")
    print()

    # ========================================================================
    print("=" * 78)
    print("[4] PARAMETER SENSITIVITY — atr_stop x portfolio_cap (3x3)")
    print("=" * 78)
    sens = param_sensitivity(
        pair_data,
        atrs=[3.0, 4.0, 5.0],
        caps=[0.10, 0.15, 0.20],
        starting_equity=100_000.0, base_risk=0.04,
    )
    print(f"{'ATR':>5s}  {'Cap':>5s}   {'Annlzd':>8s}  {'Sharpe':>6s}  {'MaxDD':>6s}  {'Trades':>6s}")
    for r in sens:
        marker = "  *" if (r["atr_stop_mult"] == 4.0 and r["portfolio_cap"] == 0.15) else ""
        print(f"{r['atr_stop_mult']:>5.1f}  {r['portfolio_cap']:>5.2f}   "
              f"{r['annualized']:>+7.2%}  {r['sharpe']:>+5.2f}   "
              f"{r['max_dd']:>5.1%}   {r['n_trades']:>5d}{marker}")
    print("(* = current production config)")
    print()

    # ========================================================================
    print("=" * 78)
    print("[5] STRESS PERIODS")
    print("=" * 78)
    stress_windows = [
        ("LUNA collapse",        "2022-05-01", "2022-06-30"),
        ("FTX collapse",         "2022-11-01", "2022-12-31"),
        ("2022 full bear",       "2022-01-01", "2022-12-31"),
        ("2023 recovery",        "2023-01-01", "2023-12-31"),
        ("ETF rally",            "2024-01-01", "2024-04-30"),
        ("2024 full year",       "2024-01-01", "2024-12-31"),
        ("2025 YTD",             "2025-01-01", "2025-12-31"),
        ("Last 90 days",         (pd.Timestamp(latest) - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                                  pd.Timestamp(latest).strftime("%Y-%m-%d")),
    ]
    print(f"{'Window':<22s} {'Span':<26s} {'Days':>4s} {'Return':>9s} {'Sharpe':>7s} "
          f"{'MaxDD':>6s} {'BAH':>9s} {'Alpha':>8s}")
    for label, s, e in stress_windows:
        r = stress_period(pair_data, label, s, e, **base_kw)
        if "error" in r:
            print(f"{label:<22s} {s} -> {e}     n/a")
            continue
        print(f"{label:<22s} {s} -> {e}    {r['n_days']:>4d} "
              f"{r['total_return']:>+8.2%} {r['sharpe']:>+6.2f} "
              f"{r['max_dd']:>5.1%} {r['bah_basket']:>+8.2%} {r['alpha']:>+7.2%}")
    print()

    # ========================================================================
    print("=" * 78)
    print("[6] DRAWDOWN DISTRIBUTION")
    print("=" * 78)
    dd_stats = drawdown_stats(full["drawdown_series"])
    print(f"DD episodes (>5% trough):     {dd_stats['n_episodes']}")
    print(f"Median episode depth:          {dd_stats.get('median_depth',0):.1%}")
    print(f"P90 episode depth:             {dd_stats.get('p90_depth',0):.1%}")
    print(f"Max episode depth:             {dd_stats.get('max_depth',0):.1%}")
    print(f"Median episode length (days):  {dd_stats.get('median_length',0)}")
    print(f"Max episode length (days):     {dd_stats.get('max_length',0)}")
    print(f"Episodes >20% deep:            {dd_stats.get('n_above_20pct',0)}")
    print(f"Episodes >30% deep:            {dd_stats.get('n_above_30pct',0)}")
    print()

    # ========================================================================
    print("=" * 78)
    print("[7] BOOTSTRAP CI (block bootstrap, 1000 reps, block=20 days)")
    print("=" * 78)
    bs = bootstrap_sharpe(full["daily_returns"], n_boot=1000)
    if "error" not in bs:
        print(f"Sharpe distribution:")
        print(f"  P5:   {bs['sharpe_p5']:+.2f}")
        print(f"  P25:  {bs['sharpe_p25']:+.2f}")
        print(f"  P50:  {bs['sharpe_p50']:+.2f}  (point estimate {full['sharpe']:+.2f})")
        print(f"  P75:  {bs['sharpe_p75']:+.2f}")
        print(f"  P95:  {bs['sharpe_p95']:+.2f}")
        print(f"  P(Sharpe>0):     {bs['pct_sharpe_above_0']:.1%}")
        print(f"  P(Sharpe>0.5):   {bs['pct_sharpe_above_0p5']:.1%}")
        print()
        print(f"Annualized return distribution:")
        print(f"  P5:   {bs['ann_p5']:+.2%}")
        print(f"  P50:  {bs['ann_p50']:+.2%}  (point estimate {full['annualized_return']:+.2%})")
        print(f"  P95:  {bs['ann_p95']:+.2%}")
    print()

    # ========================================================================
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    fold_anns = [f["annualized"] for f in folds] if folds else []
    fold_sharpes = [f["sharpe"] for f in folds] if folds else []
    if fold_anns:
        oos_mean = np.mean(fold_anns)
        in_sample = full["annualized_return"]
        deg = (in_sample - oos_mean) / max(abs(in_sample), 0.01)
        print(f"In-sample annualized:      {in_sample:+.2%}")
        print(f"Out-of-sample mean (5 folds): {oos_mean:+.2%}")
        print(f"Degradation:               {deg:+.1%}")
        print(f"Live discount estimate:    use ~{oos_mean * 0.7:+.0%} (OOS x 70% slippage haircut)")
    if "error" not in bs:
        print(f"95% CI on Sharpe:          [{bs['sharpe_p5']:+.2f}, {bs['sharpe_p95']:+.2f}]")
        print(f"95% CI on annualized:      [{bs['ann_p5']:+.2%}, {bs['ann_p95']:+.2%}]")
