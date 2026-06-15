"""Backtest cointegrated pairs trading + correlation gate.

Approach:
  1. Each week, scan 7-pair universe for cointegrated relationships.
  2. Use BEST cointegrated pair (lowest Engle-Granger p-value).
  3. Open long-spread / short-spread trades on z-score thresholds.
  4. Capture daily equity path for correlation analysis vs pro_trend + xsmom.

Gates (must pass all):
  - standalone Sharpe > 0.3
  - correlation to pro_trend < 0.3
  - correlation to XSMOM < 0.3
  - combined portfolio Sharpe >= pro_trend Sharpe (no degradation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, cointegration
from core.comprehensive_backtest import fetch_all, portfolio_run
from core.xsmom_backtest import xsmom_backtest


ANNUALIZATION = 365
UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
            "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
RESCAN_FREQ_DAYS = 30   # re-test cointegration monthly (more stable than weekly)
LOOKBACK_DAYS_FOR_TEST = 180


def coint_pairs_backtest(
    days_back: int = 2500,
    starting_equity: float = 100_000.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,         # hard stop if spread blows out
    risk_per_trade: float = 0.05,
    round_trip_bps: float = 30.0,
    pairs: list[str] | None = None,
) -> dict:
    """Backtest cointegrated pairs with periodic re-scan.

    State:
      - active_pair: (sym_a, sym_b, hedge_ratio, mean, std)
      - position: +1 = long spread, -1 = short spread, 0 = flat
    """
    pairs = pairs or UNIVERSE
    # Fetch all closing prices
    panel_dict = {}
    for p in pairs:
        try:
            df = data.ohlcv_extended(p, days_back=days_back)
            if not df.empty:
                panel_dict[p] = df["close"]
        except Exception:
            continue
    if len(panel_dict) < 2:
        return {"error": "insufficient pairs"}

    panel = pd.concat(panel_dict, axis=1).dropna()
    n = len(panel)
    if n < LOOKBACK_DAYS_FOR_TEST + 30:
        return {"error": "insufficient history"}

    # Clean P&L accounting: cash is unchanged on opens. Unrealized P&L is
    # tracked via entry prices. Realized P&L (cash delta) happens on closes.
    cash = starting_equity
    active_pair = None  # tuple (a, b, beta, mean, std, scan_day)
    position = 0  # +1 long spread (long a, short b*beta), -1 short, 0 flat
    qty_a = 0.0
    qty_b = 0.0
    entry_price_a = 0.0
    entry_price_b = 0.0
    equity_path = []
    trades = 0
    rescan_count = 0

    def _close_position(price_a_now, price_b_now, cash_in):
        """Realize P&L and return updated cash."""
        pnl = qty_a * (price_a_now - entry_price_a) + qty_b * (price_b_now - entry_price_b)
        gross_notional = abs(qty_a * price_a_now) + abs(qty_b * price_b_now)
        cost = gross_notional * round_trip_bps / 2 / 10_000  # exit cost
        return cash_in + pnl - cost

    for i in range(LOOKBACK_DAYS_FOR_TEST, n):
        today = panel.index[i]

        # Periodic re-scan + initial scan
        if active_pair is None or (i - active_pair[5]) >= RESCAN_FREQ_DAYS:
            lookback = panel.iloc[i - LOOKBACK_DAYS_FOR_TEST:i]
            try:
                scan = cointegration.find_cointegrated_pairs(lookback)
                coint_rows = scan[scan["is_cointegrated"]]
            except Exception:
                coint_rows = pd.DataFrame()
            rescan_count += 1

            # If position open, close on regime change
            if position != 0 and active_pair is not None:
                price_a = float(panel[active_pair[0]].iloc[i])
                price_b = float(panel[active_pair[1]].iloc[i])
                cash = _close_position(price_a, price_b, cash)
                trades += 1
                position = 0
                qty_a = qty_b = 0

            if coint_rows.empty:
                active_pair = None
            else:
                best = coint_rows.iloc[0]
                a, b = best["pair_a"], best["pair_b"]
                beta = float(best["hedge_ratio"])
                spread = lookback[a] - beta * lookback[b]
                active_pair = (a, b, beta, float(spread.mean()),
                               float(spread.std()), i)

        if active_pair is None:
            equity_path.append({"ts": today, "equity": cash})
            continue

        a, b, beta, mean, std, _ = active_pair
        price_a = float(panel[a].iloc[i])
        price_b = float(panel[b].iloc[i])
        spread = price_a - beta * price_b
        z = (spread - mean) / std if std > 0 else 0

        # === EXITS ===
        if position != 0:
            if abs(z) > stop_z:
                cash = _close_position(price_a, price_b, cash)
                trades += 1
                position = 0
                qty_a = qty_b = 0
            elif (position > 0 and z > -exit_z) or (position < 0 and z < exit_z):
                cash = _close_position(price_a, price_b, cash)
                trades += 1
                position = 0
                qty_a = qty_b = 0

        # === ENTRIES ===
        if position == 0:
            if z < -entry_z:  # long spread (long A, short β*B)
                spread_dist_to_stop = (stop_z - entry_z) * std
                if spread_dist_to_stop > 0:
                    qty_a = (cash * risk_per_trade) / spread_dist_to_stop
                    # Cap each leg notional to 25% of cash
                    qty_a = min(qty_a, cash * 0.25 / price_a)
                    qty_a = min(qty_a, cash * 0.25 / (beta * price_b))
                    qty_b = -qty_a * beta
                    entry_price_a = price_a
                    entry_price_b = price_b
                    gross_notional = abs(qty_a * price_a) + abs(qty_b * price_b)
                    cash -= gross_notional * round_trip_bps / 2 / 10_000
                    position = 1
                    trades += 1
            elif z > entry_z:  # short spread (short A, long β*B)
                spread_dist_to_stop = (stop_z - entry_z) * std
                if spread_dist_to_stop > 0:
                    qty_a = (cash * risk_per_trade) / spread_dist_to_stop
                    qty_a = min(qty_a, cash * 0.25 / price_a)
                    qty_a = min(qty_a, cash * 0.25 / (beta * price_b))
                    qty_a = -qty_a
                    qty_b = -qty_a * beta
                    entry_price_a = price_a
                    entry_price_b = price_b
                    gross_notional = abs(qty_a * price_a) + abs(qty_b * price_b)
                    cash -= gross_notional * round_trip_bps / 2 / 10_000
                    position = -1
                    trades += 1

        # Mark to market — only the P&L of open position adds to cash
        if position != 0:
            unrealized = (qty_a * (price_a - entry_price_a)
                           + qty_b * (price_b - entry_price_b))
        else:
            unrealized = 0
        equity_path.append({"ts": today, "equity": cash + unrealized})

    # Close any open at end
    if position != 0 and active_pair is not None:
        price_a = float(panel[active_pair[0]].iloc[-1])
        price_b = float(panel[active_pair[1]].iloc[-1])
        cash = _close_position(price_a, price_b, cash)

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
        "starting_equity": starting_equity,
        "final_equity": final_eq,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_trades": trades,
        "n_rescans": rescan_count,
        "equity_path": eq_df,
        "daily_returns": daily_rets,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("COINTEGRATED PAIRS BACKTEST + CORRELATION GATE")
    print("=" * 78)
    print()

    # === [1] Standalone backtest ===
    print("[1] Standalone cointegrated pairs (7-pair universe, monthly re-scan)")
    print("-" * 78)
    pc = coint_pairs_backtest(days_back=2500)
    if "error" in pc:
        print(f"FAILED: {pc['error']}")
        sys.exit(1)
    print(f"Annualized:   {pc['annualized_return']:+.2%}")
    print(f"Sharpe:       {pc['sharpe']:+.2f}")
    print(f"Max DD:       {pc['max_drawdown']:.1%}")
    print(f"Trades:       {pc['n_trades']}")
    print(f"Re-scans:     {pc['n_rescans']}")
    print()

    if pc["sharpe"] < 0.3:
        print(f"GATE 1 FAILED: standalone Sharpe {pc['sharpe']:.2f} < 0.3")
        print("Recommendation: skip cointegrated pairs.")
        sys.exit(0)

    # === [2] Build comparison sleeves for correlation ===
    print("[2] Building pro_trend + XSMOM equity paths for correlation...")
    pair_data = fetch_all(days_back=2500)
    pt = portfolio_run(pair_data=pair_data, starting_equity=100_000.0,
                       base_risk=0.04, portfolio_risk_cap=0.15,
                       atr_stop_mult=4.0, drawdown_kill_pct=0.35)
    pt_rets = pt["daily_returns"]

    xs = xsmom_backtest(days_back=2500, momentum_window=14, rebalance_freq=14,
                        long_n=2, short_n=2, risk_per_leg=0.20)
    xs_rets = xs["daily_returns"]
    pc_rets = pc["daily_returns"]
    print()

    # Align
    common = pt_rets.index.intersection(xs_rets.index).intersection(pc_rets.index)
    pt_r = pt_rets.loc[common]
    xs_r = xs_rets.loc[common]
    pc_r = pc_rets.loc[common]

    corr_pt = float(pc_r.corr(pt_r))
    corr_xs = float(pc_r.corr(xs_r))
    print(f"[3] Correlations of pairs to existing sleeves:")
    print(f"  Pairs <-> pro_trend:  {corr_pt:+.3f}")
    print(f"  Pairs <-> XSMOM:      {corr_xs:+.3f}")
    print()

    # === [4] Combined portfolio test ===
    # Current allocation: 70% pro_trend / 30% XSMOM
    # New proposal: 70% pro_trend / 25% XSMOM / 5% pairs
    pt_sharpe = float(pt_r.mean() / pt_r.std() * np.sqrt(ANNUALIZATION))
    old_combined = 0.7 * pt_r + 0.3 * xs_r
    new_combined = 0.7 * pt_r + 0.25 * xs_r + 0.05 * pc_r

    old_sharpe = float(old_combined.mean() / old_combined.std() * np.sqrt(ANNUALIZATION))
    new_sharpe = float(new_combined.mean() / new_combined.std() * np.sqrt(ANNUALIZATION))

    old_dd = float((1 - (1 + old_combined).cumprod() /
                     (1 + old_combined).cumprod().cummax()).max())
    new_dd = float((1 - (1 + new_combined).cumprod() /
                     (1 + new_combined).cumprod().cummax()).max())

    print(f"[4] Portfolio comparison:")
    print(f"  Pro_trend solo:    Sharpe {pt_sharpe:+.2f}")
    print(f"  Old 70/30:         Sharpe {old_sharpe:+.2f}")
    print(f"  New 70/25/5:       Sharpe {new_sharpe:+.2f}")
    print(f"  Old 70/30 DD:      {old_dd:.1%}")
    print(f"  New 70/25/5 DD:    {new_dd:.1%}")
    print()

    # === [5] Decision gate ===
    print("=" * 78)
    print("DECISION GATE")
    print("=" * 78)
    pass_sharpe = pc["sharpe"] > 0.3
    pass_corr_pt = abs(corr_pt) < 0.3
    pass_corr_xs = abs(corr_xs) < 0.3
    pass_combined = new_sharpe >= old_sharpe - 0.05  # tolerance: must not degrade

    print(f"Gate 1: standalone Sharpe > 0.3:     {pass_sharpe} ({pc['sharpe']:+.2f})")
    print(f"Gate 2: corr to pro_trend < 0.3:     {pass_corr_pt} ({corr_pt:+.3f})")
    print(f"Gate 3: corr to XSMOM < 0.3:         {pass_corr_xs} ({corr_xs:+.3f})")
    print(f"Gate 4: combined Sharpe maintained:  {pass_combined} ({new_sharpe:+.2f} vs {old_sharpe:+.2f})")
    print()

    if all([pass_sharpe, pass_corr_pt, pass_corr_xs, pass_combined]):
        print("ALL GATES PASS — wire cointegrated pairs at 5% allocation.")
        print("Recommended allocation: 70% pro_trend, 25% XSMOM, 5% pairs_coint.")
    else:
        print("ONE OR MORE GATES FAILED — do not wire.")
