"""Test: per-pair vol-targeting and portfolio-level risk cap.

The existing strategy uses 4% risk per pair via ATR-scaled qty. ATR already
normalizes per-trade dollar risk. But two things are NOT addressed:

  1. Cross-pair vol differences. Two pairs with different long-run vols
     get the same 4% risk allocation. A natural improvement: scale by
     realized 60d vol ratio so each pair contributes equally to portfolio vol.

  2. Concurrent positions. If 5 pairs are in trend simultaneously, total
     active risk = 5 * 4% = 20%. Adding a portfolio cap normalizes this.

Tests on the top-5 universe (the lever-1 winner).
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
        df["ret"] = df["close"].pct_change()
        df["vol60"] = df["ret"].rolling(60).std()
        df["vol120"] = df["ret"].rolling(120).std()
        df = df.dropna()
        out[p] = df
    return out


def portfolio_backtest(
    pair_data: dict[str, pd.DataFrame],
    starting_equity: float = 100_000.0,
    base_risk: float = 0.04,
    vol_target_window: int | None = None,    # None = no per-pair vol scaling
    portfolio_risk_cap: float | None = None,  # e.g. 0.10 = 10% total active risk cap
    atr_stop_mult: float = 4.0,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.35,
    round_trip_bps: float = 30.0,
) -> dict:
    """Multi-pair portfolio sim that runs all pairs against shared equity."""
    all_dates = sorted(set().union(*[df.index for df in pair_data.values()]))
    cash = starting_equity
    state = {
        p: {"units": [], "extreme": 0, "trail_stop": 0, "entry_idx": -1}
        for p in pair_data
    }
    peak_equity = starting_equity
    equity_path = []
    trades_per_pair = {p: 0 for p in pair_data}
    rets_for_target = {p: 0.0 for p in pair_data}
    n_dd_kills = 0

    for d_idx, today in enumerate(all_dates):
        # Snapshot: which pairs have data for today?
        active_rows = {}
        for p, df in pair_data.items():
            if today in df.index:
                active_rows[p] = df.loc[today]

        # Mark to market across all open positions
        unrealized = 0.0
        position_value = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            row = active_rows[p]
            price = float(row["close"])
            for u in st["units"]:
                unrealized += u["qty"] * (price - u["entry_price"])
                position_value += u["qty"] * price
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        # DD kill — flatten everything
        if equity_dd > drawdown_kill_pct and any(st["units"] for st in state.values()):
            for p, st in state.items():
                if not st["units"] or p not in active_rows:
                    continue
                price = float(active_rows[p]["close"])
                for u in st["units"]:
                    pnl = u["qty"] * (price - u["entry_price"])
                    cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "entry_idx": -1}
            n_dd_kills += 1
            continue

        # Compute number of currently active pairs (for portfolio cap calc)
        n_active = sum(1 for st in state.values() if st["units"])

        # Iterate each pair
        for p, row in active_rows.items():
            st = state[p]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row["atr"])
            sma = float(row["sma_filter"])
            donchian = float(row["donchian_high"])
            vol60 = float(row.get("vol60", 0))
            vol120 = float(row.get("vol120", 0))

            # === EXIT ===
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
                        trades_per_pair[p] += 1
                    state[p] = {"units": [], "extreme": 0, "trail_stop": 0, "entry_idx": -1}
                # === PYRAMID ===
                elif len(st["units"]) < max_pyramid_units:
                    last = st["units"][-1]
                    if high >= last["entry_price"] + pyramid_atr_step * last["entry_atr"]:
                        risk_pct = _compute_risk_pct(
                            base_risk, vol_target_window, vol60, vol120,
                            portfolio_risk_cap, n_active,
                        )
                        stop_dist = atr_stop_mult * atr
                        if stop_dist > 0:
                            qty = min((mtm_eq * risk_pct) / stop_dist,
                                      mtm_eq * 0.25 / price)
                            cash -= qty * price * round_trip_bps / 2 / 10_000
                            st["units"].append({"qty": qty, "entry_price": price,
                                                 "entry_atr": atr, "entry_idx": d_idx})
            else:
                # === ENTRY ===
                if price > sma and high >= donchian and atr > 0:
                    risk_pct = _compute_risk_pct(
                        base_risk, vol_target_window, vol60, vol120,
                        portfolio_risk_cap, n_active,
                    )
                    stop_dist = atr_stop_mult * atr
                    qty = min((mtm_eq * risk_pct) / stop_dist,
                              mtm_eq * 0.25 / price)
                    if qty > 0:
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        st["units"] = [{"qty": qty, "entry_price": price,
                                         "entry_atr": atr, "entry_idx": d_idx}]
                        st["extreme"] = high
                        st["trail_stop"] = price - stop_dist

        # Daily MTM record
        unrealized = 0.0
        for p, st in state.items():
            if not st["units"] or p not in active_rows:
                continue
            price = float(active_rows[p]["close"])
            for u in st["units"]:
                unrealized += u["qty"] * (price - u["entry_price"])
        equity_path.append({"ts": today, "equity": cash + unrealized})

    # Close any open
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
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(ANNUALIZATION))
        if daily_rets.std() > 0 else 0
    )
    peak = daily_eq.cummax()
    max_dd = float((1 - daily_eq / peak).max())
    final_eq = float(daily_eq.iloc[-1])
    total_return = final_eq / starting_equity - 1
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    annualized = (1 + total_return) ** (ANNUALIZATION / max(n_days, 1)) - 1

    return {
        "n_days": n_days,
        "starting_equity": starting_equity,
        "final_equity": final_eq,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades_per_pair": trades_per_pair,
        "total_trades": sum(trades_per_pair.values()),
        "n_dd_kills": n_dd_kills,
    }


def _compute_risk_pct(
    base: float,
    vt_window: int | None,
    vol60: float,
    vol120: float,
    portfolio_cap: float | None,
    n_active: int,
) -> float:
    """Compute size for this trade given vol-target and portfolio-cap settings."""
    risk = base

    # Per-pair vol scaling — target each pair contributes ~base vol
    if vt_window == 60 and vol60 > 0:
        # BTC-equivalent vol baseline ~ 0.035 daily
        risk = base * (0.035 / vol60)
        risk = min(risk, base * 1.5)  # cap up-scale at 1.5x
        risk = max(risk, base * 0.5)  # floor at 0.5x
    elif vt_window == 120 and vol120 > 0:
        risk = base * (0.035 / vol120)
        risk = min(risk, base * 1.5)
        risk = max(risk, base * 0.5)

    # Portfolio cap — total risk across active pairs ≤ cap
    if portfolio_cap and n_active > 0:
        per_pair_max = portfolio_cap / max(n_active, 1)
        risk = min(risk, per_pair_max)

    return max(risk, 0)


if __name__ == "__main__":
    print("Fetching data for 5-pair universe...")
    pair_data = fetch_all(days_back=1500)
    print(f"  Got {len(pair_data)} pairs: {list(pair_data.keys())}")
    n = min(len(df) for df in pair_data.values())
    print(f"  Common bars (min): {n}")
    print()

    variants = [
        ("baseline-4%",            dict(base_risk=0.04)),
        ("baseline-3%",            dict(base_risk=0.03)),
        ("vt60",                   dict(base_risk=0.04, vol_target_window=60)),
        ("vt120",                  dict(base_risk=0.04, vol_target_window=120)),
        ("portcap-10%",            dict(base_risk=0.04, portfolio_risk_cap=0.10)),
        ("portcap-15%",            dict(base_risk=0.04, portfolio_risk_cap=0.15)),
        ("portcap-20%",            dict(base_risk=0.04, portfolio_risk_cap=0.20)),
        ("vt60+portcap-15",        dict(base_risk=0.04, vol_target_window=60,
                                       portfolio_risk_cap=0.15)),
        ("vt120+portcap-15",       dict(base_risk=0.04, vol_target_window=120,
                                       portfolio_risk_cap=0.15)),
    ]

    rows = []
    for label, kwargs in variants:
        r = portfolio_backtest(pair_data=pair_data, **kwargs)
        print(f"{label:<22s}  ann {r['annualized_return']:>+7.2%}  "
              f"Sharpe {r['sharpe']:>+5.2f}  DD {r['max_drawdown']:>5.1%}  "
              f"trades {r['total_trades']:>3d}  DDkills {r['n_dd_kills']}")
        rows.append({"label": label, **r})

    print()
    print("=" * 75)
    print("RANKED BY SHARPE:")
    print("=" * 75)
    rows.sort(key=lambda x: -x["sharpe"])
    for r in rows:
        print(f"{r['label']:<22s}  Sharpe {r['sharpe']:>+5.2f}  "
              f"ann {r['annualized_return']:>+7.2%}  "
              f"DD {r['max_drawdown']:>5.1%}")
