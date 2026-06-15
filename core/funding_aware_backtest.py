"""Backtest with realistic funding cost on the perp (1.5x leverage) leg.

The pro_trend strategy routes longs through perp at 1.5x notional. The
spot-only backtest ignores the funding rate paid (or received) on this perp
position.

Funding model:
  - Per 8h, position pays: notional × funding_rate
  - Three settlements per day, so daily drag = notional × funding × 3
  - We use a conservative constant: 0.01% per 8h on average for retail-bullish
    regimes (~11% annualized on perp notional).
  - Real funding varies dramatically: -10% in extreme bear, +50% in bull peaks.
  - We test 3 levels: 0% (baseline), -5% ann (typical), -10% ann (bull peak).

Output: how much funding drags annualized return at each leverage level.
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


def fetch_with_indicators(days_back: int = 2500) -> dict:
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


def portfolio_with_funding(
    pair_data: dict,
    starting_equity: float = 100_000.0,
    base_risk: float = 0.04,
    portfolio_risk_cap: float = 0.15,
    atr_stop_mult: float = 4.0,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.35,
    round_trip_bps: float = 30.0,
    leverage: float = 1.5,
    daily_funding_drag: float = 0.0,  # e.g. -0.0003 = -3 bps/day = ~-11%/year
) -> dict:
    """Same as portfolio_run but applies daily funding cost on the perp leg.

    Funding only applies to the LEVERED portion: notional > 1x equity.
    For 1.5x leverage on $100k position, $50k is the perp portion paying funding.
    """
    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {p: {"units": [], "extreme": 0, "trail_stop": 0} for p in pair_data}
    peak_equity = starting_equity
    equity_path = []
    n_trades = 0
    n_dd_kills = 0
    total_funding_paid = 0.0

    for d_idx, today in enumerate(all_dates):
        active_rows = {p: df.loc[today] for p, df in pair_data.items() if today in df.index}

        # MTM
        unrealized = 0.0
        total_position_value = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            for u in st["units"]:
                unrealized += u["qty"] * (price - u["entry_price"])
                total_position_value += u["qty"] * price
        mtm_eq = cash + unrealized

        # Apply funding cost — on the levered portion of position value
        if leverage > 1.0 and daily_funding_drag != 0:
            levered_notional = total_position_value * (leverage - 1.0) / leverage
            funding_today = levered_notional * daily_funding_drag
            cash += funding_today  # negative drag = subtract
            total_funding_paid += funding_today

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
            n_dd_kills += 1
            equity_path.append({"ts": today, "equity": cash})
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
                        risk = min(base_risk, per_pair_max)
                        stop_dist = atr_stop_mult * atr
                        if stop_dist > 0:
                            qty = min((mtm_eq * risk * leverage) / stop_dist,
                                      mtm_eq * 0.30 * leverage / price)
                            cash -= qty * price * round_trip_bps / 2 / 10_000
                            st["units"].append({"qty": qty, "entry_price": price, "entry_atr": atr})
            else:
                if price > sma and high >= donchian and atr > 0:
                    per_pair_max = portfolio_risk_cap / max(n_active, 1) if n_active > 0 else base_risk
                    risk = min(base_risk, per_pair_max)
                    stop_dist = atr_stop_mult * atr
                    qty = min((mtm_eq * risk * leverage) / stop_dist,
                              mtm_eq * 0.30 * leverage / price)
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
        "annualized_return": float(annualized),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "n_dd_kills": n_dd_kills,
        "total_funding_paid": total_funding_paid,
        "final_equity": final_eq,
    }


if __name__ == "__main__":
    print("=" * 78)
    print("FUNDING-COST-AWARE BACKTEST — 1.5x leverage on long perp leg")
    print("=" * 78)
    print()
    print("Annualized funding rates tested (for the LEVERED portion only):")
    print("  0%   = no cost (idealized; matches existing backtest)")
    print("  -5%  = mild bull regime (typical)")
    print("  -10% = strong retail-bullish (BTC funding 1bps/8h)")
    print("  -20% = peak frenzy (BTC funding 2-3bps/8h)")
    print()

    pair_data = fetch_with_indicators(days_back=2500)
    print(f"Universe: {list(pair_data.keys())}")
    print(f"Days: up to {max(len(df) for df in pair_data.values())}")
    print()

    print(f"{'Lev':>4s}  {'Funding':>10s}  {'Annlzd':>8s}  {'Sharpe':>6s}  "
          f"{'MaxDD':>6s}  {'Funding $':>13s}  {'Final eq':>13s}")
    for lev in [1.0, 1.5, 2.0]:
        for ann_funding in [0.0, -0.05, -0.10, -0.20]:
            daily_drag = ann_funding / 365.0 if lev > 1.0 else 0
            r = portfolio_with_funding(
                pair_data=pair_data,
                leverage=lev, daily_funding_drag=daily_drag,
            )
            label = f"{ann_funding:+.0%}" if lev > 1.0 else "n/a"
            print(f"{lev:>4.1f}   {label:>10s}   "
                  f"{r['annualized_return']:>+7.2%}  {r['sharpe']:>+5.2f}   "
                  f"{r['max_drawdown']:>5.1%}   "
                  f"${r['total_funding_paid']:>+11,.0f}   "
                  f"${r['final_equity']:>11,.0f}")
        print()

    print("INTERPRETATION:")
    print("  - 1.0x lev: no perp = no funding cost. Pure spot.")
    print("  - 1.5x with -5% funding: realistic typical regime drag.")
    print("  - 1.5x with -10%: pessimistic but plausible in bull phase.")
    print("  - 1.5x with -20%: extreme bull peak; sustained 2-3bps/8h funding.")
    print()
    print("The current production backtest assumed 0% funding — actual live")
    print("performance will be somewhere between -5% and -10% lines depending")
    print("on regime.")
