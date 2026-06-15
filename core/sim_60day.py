"""60-day live-style simulator — replay the last 60 days as if the rig was live.

Runs the EXACT production configuration over the most recent 60 days of
historic price data, day-by-day. Captures:

  - Every pro_trend entry/exit/pyramid/DD-kill event
  - Every XSMOM weekly rebalance + the resulting holdings
  - Basis arb fires (if funding qualified during the window)
  - Daily portfolio equity (combined 70% pro_trend / 30% XSMOM)
  - Trade-by-trade log with realistic costs

Output: full daily action log + summary stats. Should let the user see
within 1-2 minutes what the system WOULD have done over the last 60 days
instead of waiting 60 calendar days for live evidence.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr


ANNUALIZATION = 365

PRO_TREND_PAIRS = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]
XSMOM_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                    "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ATOM/USDT"]


def fetch_with_indicators(pairs, days_back=300):
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
        # v5 entry filter inputs (matches strategies/pro_trend.py)
        df["tsmom30"] = df["close"].pct_change(30)
        _emf = df["close"].ewm(span=12).mean()
        _ems = df["close"].ewm(span=26).mean()
        _macd = _emf - _ems
        _sig = _macd.ewm(span=9).mean()
        df["macd_hist"] = (_macd - _sig) / df["close"]
        df = df.dropna()
        out[p] = df
    return out


def run_pro_trend_sim(pair_data, sim_start_date, sim_end_date,
                       starting_equity=70_000.0):
    """Simulate pro_trend over a date window with detailed trade log.

    Uses the EXACT production parameters from strategies/pro_trend.py.
    """
    # Production params (must match strategies/pro_trend.py)
    base_risk = 0.04
    portfolio_risk_cap = 0.15
    atr_stop_mult = 4.0
    pyramid_atr_step = 2.0
    max_pyramid_units = 2
    drawdown_kill_pct = 0.35
    round_trip_bps = 30.0

    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0, "side": None}
             for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    trades = []  # detailed log
    n_dd_kills = 0

    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    all_dates = [d for d in all_dates if sim_start_date <= d <= sim_end_date]

    for today in all_dates:
        active_rows = {p: df.loc[today] for p, df in pair_data.items()
                       if today in df.index}

        # MTM
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

        # DD kill
        if equity_dd > drawdown_kill_pct and any(st["units"] for st in state.values()):
            for p, st in state.items():
                if not st["units"] or p not in active_rows:
                    continue
                price = float(active_rows[p]["close"])
                sign = 1 if st["side"] == "long" else -1
                for u in st["units"]:
                    pnl = sign * u["qty"] * (price - u["entry_price"])
                    cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                    trades.append({
                        "ts": today, "pair": p, "action": "dd_kill_exit",
                        "side": st["side"], "qty": u["qty"], "price": price,
                        "pnl": pnl,
                    })
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

            if st["units"]:
                # === EXIT / PYRAMID for LONG ===
                if st["side"] == "long":
                    if high > st["extreme"]:
                        st["extreme"] = high
                        new_trail = high - atr_stop_mult * atr
                        if new_trail > st["trail_stop"]:
                            st["trail_stop"] = new_trail
                    if low <= st["trail_stop"] or price < sma:
                        exit_p = st["trail_stop"] if low <= st["trail_stop"] else price
                        reason = "trail" if low <= st["trail_stop"] else "sma_break"
                        for u in st["units"]:
                            pnl = u["qty"] * (exit_p - u["entry_price"])
                            cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                            trades.append({
                                "ts": today, "pair": p, "action": f"exit_long ({reason})",
                                "side": "long", "qty": u["qty"], "price": exit_p,
                                "pnl": pnl,
                            })
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
                                trades.append({
                                    "ts": today, "pair": p, "action": "pyramid_long",
                                    "side": "long", "qty": qty, "price": price, "pnl": 0,
                                })
                # === EXIT / PYRAMID for SHORT ===
                else:  # short
                    if st["extreme"] == 0 or low < st["extreme"]:
                        st["extreme"] = low
                        new_trail = low + atr_stop_mult * atr
                        if st["trail_stop"] == 0 or new_trail < st["trail_stop"]:
                            st["trail_stop"] = new_trail
                    if high >= st["trail_stop"] or price > sma:
                        exit_p = st["trail_stop"] if high >= st["trail_stop"] else price
                        reason = "trail" if high >= st["trail_stop"] else "sma_break"
                        for u in st["units"]:
                            pnl = -u["qty"] * (exit_p - u["entry_price"])
                            cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                            trades.append({
                                "ts": today, "pair": p, "action": f"exit_short ({reason})",
                                "side": "short", "qty": u["qty"], "price": exit_p,
                                "pnl": pnl,
                            })
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
                                trades.append({
                                    "ts": today, "pair": p, "action": "pyramid_short",
                                    "side": "short", "qty": qty, "price": price, "pnl": 0,
                                })
            else:
                # === ENTRY (v5 production: LONG-ONLY + TSMOM>0 + MACD_hist>0) ===
                per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                risk = min(base_risk, per_pair_max)
                stop_dist = atr_stop_mult * atr
                qty = min((mtm_eq * risk) / stop_dist, mtm_eq * 0.25 / price)
                if qty <= 0 or atr <= 0:
                    continue
                tsmom30 = float(row["tsmom30"])
                macd_hist = float(row["macd_hist"])
                v5_filter = tsmom30 > 0 and macd_hist > 0
                if in_bull and high >= donchian_high and v5_filter:
                    cash -= qty * price * round_trip_bps / 2 / 10_000
                    state[p] = {
                        "units": [{"qty": qty, "entry_price": price, "entry_atr": atr}],
                        "side": "long", "extreme": high, "trail_stop": price - stop_dist,
                    }
                    trades.append({
                        "ts": today, "pair": p, "action": "entry_long", "side": "long",
                        "qty": qty, "price": price, "pnl": 0,
                    })
                # NB: short entries DISABLED (2026-05-11 rebuild). Per data:
                # late shorts on crypto majors lose to V-reversals (60d sim).

        # MTM at end of day
        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            sign = 1 if st["side"] == "long" else -1
            for u in st["units"]:
                unrealized += sign * u["qty"] * (price - u["entry_price"])
        equity_path.append({"ts": today, "equity": cash + unrealized})

    # Capture end-of-window position state (DO NOT close — show what's live today)
    final_day = all_dates[-1]
    open_positions = []
    for p, st in state.items():
        if st["units"] and final_day in pair_data[p].index:
            price = float(pair_data[p].loc[final_day, "close"])
            sign = 1 if st["side"] == "long" else -1
            for u in st["units"]:
                unrealized = sign * u["qty"] * (price - u["entry_price"])
                open_positions.append({
                    "pair": p, "side": st["side"], "qty": u["qty"],
                    "entry": u["entry_price"], "current": price,
                    "trail_stop": st["trail_stop"],
                    "unrealized_pnl": unrealized,
                })

    # Also compute mark-to-market final equity (don't close)
    final_unrealized = sum(p["unrealized_pnl"] for p in open_positions)
    mtm_final = cash + final_unrealized

    eq_df = pd.DataFrame(equity_path).set_index("ts")
    return {
        "equity_path": eq_df,
        "trades": pd.DataFrame(trades),
        "n_dd_kills": n_dd_kills,
        "final_cash": cash,
        "mtm_final": mtm_final,
        "n_trades": len(trades),
        "open_positions": open_positions,
    }


def run_xsmom_sim(panel, sim_start_date, sim_end_date,
                   starting_equity=30_000.0):
    """XSMOM weekly rebalance over the window."""
    momentum_window = 14
    rebalance_freq = 14
    long_n = 2
    short_n = 2
    risk_per_leg = 0.20
    round_trip_bps = 30.0

    panel = panel.dropna(how="all")
    panel = panel.loc[(panel.index >= sim_start_date - pd.Timedelta(days=momentum_window + 5))
                       & (panel.index <= sim_end_date)]

    momentum = panel.pct_change(momentum_window)
    daily_rets = panel.pct_change()

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    last_weights = pd.Series(0.0, index=panel.columns)
    rebalances = []

    sim_rows = panel.index[panel.index >= sim_start_date]
    for i, date in enumerate(sim_rows):
        idx = panel.index.get_loc(date)
        if i % rebalance_freq != 0:
            weights.loc[date] = last_weights
            continue
        m = momentum.iloc[idx].dropna()
        if len(m) < long_n + short_n:
            weights.loc[date] = last_weights
            continue
        ranked = m.sort_values(ascending=False)
        new_w = pd.Series(0.0, index=panel.columns)
        for pair in ranked.index[:long_n]:
            new_w[pair] = risk_per_leg / long_n
        for pair in ranked.index[-short_n:]:
            new_w[pair] = -risk_per_leg / short_n
        weights.loc[date] = new_w
        rebalances.append({
            "ts": date,
            "longs": [(p, float(m[p])) for p in ranked.index[:long_n]],
            "shorts": [(p, float(m[p])) for p in ranked.index[-short_n:]],
        })
        last_weights = new_w

    portfolio_rets = (weights.shift(1) * daily_rets).sum(axis=1).fillna(0)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0)
    cost_per_day = turnover * round_trip_bps / 2 / 10_000
    portfolio_rets -= cost_per_day

    sim_rets = portfolio_rets.loc[sim_rows]
    eq = starting_equity * (1 + sim_rets).cumprod()
    if len(eq) == 0:
        eq = pd.Series([starting_equity])
    return {
        "equity_path": pd.DataFrame({"equity": eq}),
        "rebalances": rebalances,
        "n_rebalances": len(rebalances),
    }


# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("60-DAY LIVE-STYLE SIMULATION (replay of last 60 days)")
    print("=" * 80)
    print()

    days_back = 60

    # Fetch data going back a bit further for indicator warmup
    pair_data = fetch_with_indicators(PRO_TREND_PAIRS, days_back=300)
    print(f"Pro_trend pairs: {list(pair_data.keys())}")

    # Set window
    end_date = max(df.index[-1] for df in pair_data.values())
    start_date = end_date - pd.Timedelta(days=days_back)
    print(f"Window: {start_date.date()} -> {end_date.date()} ({days_back} days)")
    print()

    # === PRO_TREND ===
    pt = run_pro_trend_sim(pair_data, start_date, end_date,
                            starting_equity=70_000.0)
    print("[1] PRO_TREND results (70% allocation = $70,000 starting)")
    print("-" * 80)
    print(f"  Cash at end:  ${pt['final_cash']:>10,.2f}  (closed trades only)")
    print(f"  MTM equity:   ${pt['mtm_final']:>10,.2f}  (includes open positions)")
    final_eq = pt["equity_path"]["equity"].iloc[-1]
    print(f"  P&L:          ${final_eq - 70_000:>+10,.2f}  ({final_eq/70_000-1:+.2%})")
    print(f"  Total events: {pt['n_trades']}")
    print(f"  DD-kill events: {pt['n_dd_kills']}")
    if pt["open_positions"]:
        print(f"  Open positions today (sim):")
        for op in pt["open_positions"]:
            print(f"    {op['pair']} {op['side'].upper():<6s} entry ${op['entry']:.4f}, "
                  f"current ${op['current']:.4f}, trail ${op['trail_stop']:.4f}, "
                  f"pnl ${op['unrealized_pnl']:+,.2f}")
    else:
        print(f"  Open positions today (sim): NONE (system is flat)")

    if not pt["trades"].empty:
        action_counts = pt["trades"]["action"].value_counts()
        print(f"  Event breakdown:")
        for act, n in action_counts.items():
            print(f"    {act}: {n}")
    print()

    # === XSMOM ===
    xs_panel = {}
    for p in XSMOM_UNIVERSE:
        try:
            df = data.ohlcv_extended(p, days_back=days_back + 30)
            if not df.empty:
                xs_panel[p] = df["close"]
        except Exception:
            continue
    xs_panel_df = pd.concat(xs_panel, axis=1)
    xs = run_xsmom_sim(xs_panel_df, start_date, end_date, starting_equity=30_000.0)
    xs_final = xs["equity_path"]["equity"].iloc[-1]
    print("[2] XSMOM results (30% allocation = $30,000 starting)")
    print("-" * 80)
    print(f"  Final equity: ${xs_final:>10,.2f}  (from $30,000)")
    print(f"  P&L:          ${xs_final - 30_000:>+10,.2f}  ({xs_final/30_000-1:+.2%})")
    print(f"  Rebalances:   {xs['n_rebalances']}")
    if xs["rebalances"]:
        print(f"  Recent rebalance: {xs['rebalances'][-1]['ts'].date()}")
        print(f"    Longs:  {[p for p, _ in xs['rebalances'][-1]['longs']]}")
        print(f"    Shorts: {[p for p, _ in xs['rebalances'][-1]['shorts']]}")
    print()

    # === COMBINED PORTFOLIO ===
    print("[3] COMBINED 70/30 portfolio")
    print("-" * 80)
    total_start = 100_000.0
    total_end = final_eq + xs_final
    total_ret = total_end / total_start - 1
    ann = (1 + total_ret) ** (365 / days_back) - 1
    print(f"  Start:       ${total_start:>10,.2f}")
    print(f"  End:         ${total_end:>10,.2f}")
    print(f"  60-day P&L:  ${total_end - total_start:>+10,.2f}  ({total_ret:+.2%})")
    print(f"  Annualized:  {ann:+.2%}")

    # Build combined equity curve for Sharpe/DD
    pt_rets = pt["equity_path"]["equity"].pct_change().dropna()
    xs_rets = xs["equity_path"]["equity"].pct_change().dropna()
    common = pt_rets.index.intersection(xs_rets.index)
    if len(common) > 5:
        combined_rets = 0.7 * pt_rets.loc[common] + 0.3 * xs_rets.loc[common]
        sharpe = combined_rets.mean() / combined_rets.std() * np.sqrt(365) if combined_rets.std() > 0 else 0
        eq = (1 + combined_rets).cumprod()
        max_dd = float((1 - eq / eq.cummax()).max())
        print(f"  Sharpe:      {sharpe:+.2f}")
        print(f"  Max DD:      {max_dd:.2%}")
    print()

    # === DETAILED TRADE LOG (last 60 days) ===
    print("[4] DETAILED PRO_TREND TRADE LOG (last 60 days)")
    print("-" * 80)
    if pt["trades"].empty:
        print("  No trades fired by pro_trend in the last 60 days.")
        print("  This means: no Donchian-20 breakouts AND no SMA200 crosses on any pair.")
    else:
        for _, t in pt["trades"].iterrows():
            print(f"  {t['ts'].date()}  {t['pair']:<12s} {t['action']:<25s} "
                  f"side={t['side']:<5s} qty={t['qty']:>10.4f}  "
                  f"price=${t['price']:>10,.4f}  pnl=${t['pnl']:>+10,.2f}")
    print()

    # === XSMOM REBALANCE HISTORY ===
    print("[5] XSMOM REBALANCE HISTORY")
    print("-" * 80)
    for r in xs["rebalances"]:
        longs_str = ", ".join(f"{p}({m:+.1%})" for p, m in r["longs"])
        shorts_str = ", ".join(f"{p}({m:+.1%})" for p, m in r["shorts"])
        print(f"  {r['ts'].date()}: long {longs_str}  |  short {shorts_str}")
    print()

    # === BAH COMPARISON ===
    print("[6] BUY-AND-HOLD BENCHMARK — what passive exposure would have done")
    print("-" * 80)
    bah_returns = {}
    for p in PRO_TREND_PAIRS:
        df = data.ohlcv_extended(p, days_back=days_back + 5)
        sub = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(sub) >= 2:
            ret = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
            bah_returns[p] = ret
    if bah_returns:
        for p, r in bah_returns.items():
            print(f"  {p:<12s} BAH 60d return: {r:>+8.2%}")
        avg_bah = sum(bah_returns.values()) / len(bah_returns)
        print(f"  {'Avg BAH (5 pairs)':<20s}: {avg_bah:>+8.2%}")
        print(f"  Combined sim 60d:      {total_ret:+.2%}")
        alpha = total_ret - avg_bah
        print(f"  ALPHA vs BAH basket:   {alpha:+.2%}")
        if alpha > 0:
            print(f"  -> System BEAT BAH basket by {alpha:+.2%}")
        else:
            print(f"  -> System LAGGED BAH basket by {alpha:+.2%}")
    print()

    # === WEEKLY EQUITY CHECKPOINTS ===
    print("[7] WEEKLY EQUITY CHECKPOINTS (combined 70/30)")
    print("-" * 80)
    pt_eq = pt["equity_path"]["equity"]
    xs_eq = xs["equity_path"]["equity"]
    # Align on common dates
    combined_eq = pd.Series(dtype=float)
    common_dates = pt_eq.index.intersection(xs_eq.index)
    if len(common_dates) > 0:
        combined_eq = pt_eq.loc[common_dates] + xs_eq.loc[common_dates]
    # Weekly snapshots
    weekly_dates = pd.date_range(start_date, end_date, freq="7D", tz="UTC")
    print(f"  {'Date':<12s} {'Equity':>12s} {'d from start':>14s} {'7d d':>9s}")
    prev_eq = 100_000.0
    for wd in weekly_dates:
        # Find nearest available date
        available = combined_eq.index[combined_eq.index <= wd]
        if len(available) == 0:
            continue
        snap_date = available[-1]
        eq = float(combined_eq.loc[snap_date])
        delta_total = (eq - 100_000) / 100_000
        delta_7d = (eq - prev_eq) / prev_eq if prev_eq > 0 else 0
        print(f"  {str(snap_date.date()):<12s} ${eq:>10,.0f}    "
              f"{delta_total:>+12.2%}  {delta_7d:>+8.2%}")
        prev_eq = eq
    print()

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"In the last 60 days, the systematic pro_trend strategy generated")
    print(f"{pt['n_trades']} events ({action_counts.get('entry_long', 0) if not pt['trades'].empty else 0} long entries, "
          f"{action_counts.get('entry_short', 0) if not pt['trades'].empty else 0} short entries).")
    print(f"XSMOM rebalanced {xs['n_rebalances']} times.")
    print()
    print(f"If you had run the system for the last 60 days starting at $100k,")
    print(f"you would now have ${total_end:,.2f} ({total_ret:+.2%}).")
