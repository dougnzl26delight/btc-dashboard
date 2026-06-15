"""Test cycle-timing BAH BTC vs plain BAH BTC.

Hypothesis: BTC cycle tops/bottoms can be timed using simple price-based
indicators. Test variants:

  v0 plain BAH (baseline)
  v1 Mayer Multiple (price/200d-SMA) — sell at >2.0, rebuy at <0.7
  v2 weekly RSI — sell at >80, rebuy at <30
  v3 Pi Cycle Top (111d SMA crosses 350d SMA*2) + 200w SMA bottom
  v4 Combined: any sell signal -> sell, any buy signal -> rebuy
  v5 Halving cycle calendar — sell 14-18mo post-halving, rebuy 24-36mo post

Historic context (what these signals saw at known tops/bottoms):
  2021-04 top ($64k): Mayer ~1.55, RSI weekly ~75
  2021-11 top ($69k): Mayer ~1.64, RSI weekly ~72
  2022-11 bottom ($15.5k): Mayer ~0.45, RSI weekly ~32
  2023-01 confirmed bottom: Mayer ~0.42
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


HALVINGS = [
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 19),
    date(2028, 4, 1),  # estimated
]


def add_indicators(df):
    df = df.copy()
    df["sma200"] = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / df["sma200"]
    # Weekly RSI on daily bars approximated using 7d returns
    df["weekly_ret"] = df["close"].pct_change(7)
    df["rsi_w"] = _wilder_rsi(df["close"], period=14 * 7) if len(df) >= 100 else 50
    # Pi Cycle: 111d SMA vs 350d SMA × 2
    df["sma111"] = df["close"].rolling(111).mean()
    df["sma350x2"] = df["close"].rolling(350).mean() * 2
    df["pi_top_signal"] = df["sma111"] >= df["sma350x2"]
    # 200-week SMA
    df["sma200w"] = df["close"].rolling(200 * 7).mean()
    return df


def _wilder_rsi(s, period=14):
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def cycle_timing_backtest(
    df,
    sell_mayer_threshold=None,   # e.g., 2.0 — sell when Mayer > this
    buy_mayer_threshold=None,    # e.g., 0.7 — rebuy when Mayer < this
    sell_pi_cycle=False,         # sell on Pi Cycle top signal
    sell_halving_window=None,    # e.g., (14, 18) — sell N months post-halving
    buy_halving_window=None,     # e.g., (28, 38) — rebuy M months post-halving
    starting_equity=100_000.0,
    transaction_cost_bps=20.0,
):
    """Backtest BTC with cycle-timing sells/buys."""
    df = add_indicators(df).dropna(subset=["sma200"])
    if df.empty:
        return None

    cash = starting_equity
    btc = 0.0
    in_btc = False
    trades = []
    equity_path = []

    # Initial buy
    btc = cash / float(df["close"].iloc[0])
    cash = 0
    in_btc = True
    initial_price = float(df["close"].iloc[0])
    trades.append({"ts": df.index[0], "action": "init_buy",
                    "price": initial_price, "btc": btc})

    for date_idx, row in df.iterrows():
        price = float(row["close"])
        mayer = float(row["mayer"]) if pd.notna(row["mayer"]) else None
        pi_top = bool(row.get("pi_top_signal", False))

        # Halving window check
        in_sell_halving = False
        in_buy_halving = False
        if sell_halving_window or buy_halving_window:
            today = date_idx.date()
            past_halvings = [h for h in HALVINGS if h <= today]
            if past_halvings:
                months_post = (today - max(past_halvings)).days / 30.4
                if sell_halving_window:
                    lo, hi = sell_halving_window
                    in_sell_halving = lo <= months_post <= hi
                if buy_halving_window:
                    lo, hi = buy_halving_window
                    in_buy_halving = lo <= months_post <= hi

        # SELL signals
        sell_signals = []
        if in_btc:
            if sell_mayer_threshold and mayer and mayer > sell_mayer_threshold:
                sell_signals.append(f"mayer>{sell_mayer_threshold}")
            if sell_pi_cycle and pi_top:
                sell_signals.append("pi_cycle_top")
            if in_sell_halving:
                sell_signals.append("halving_top_window")

            if sell_signals:
                proceeds = btc * price * (1 - transaction_cost_bps / 10_000)
                cash += proceeds
                trades.append({"ts": date_idx, "action": "sell",
                                "price": price, "btc": btc, "proceeds": proceeds,
                                "signal": ",".join(sell_signals)})
                btc = 0
                in_btc = False

        # BUY signals
        if not in_btc:
            buy_signals = []
            if buy_mayer_threshold and mayer and mayer < buy_mayer_threshold:
                buy_signals.append(f"mayer<{buy_mayer_threshold}")
            if in_buy_halving:
                buy_signals.append("halving_bottom_window")

            if buy_signals:
                buy_amount = cash * (1 - transaction_cost_bps / 10_000)
                btc = buy_amount / price
                cash = 0
                in_btc = True
                trades.append({"ts": date_idx, "action": "buy",
                                "price": price, "btc": btc,
                                "signal": ",".join(buy_signals)})

        # MTM
        equity = cash + btc * price
        equity_path.append({"ts": date_idx, "equity": equity})

    eq_df = pd.DataFrame(equity_path).set_index("ts")["equity"]
    daily_rets = eq_df.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 0 else 0
    peak = eq_df.cummax()
    max_dd = float((1 - eq_df / peak).max())
    total_ret = float(eq_df.iloc[-1] / starting_equity - 1)
    n_days = (eq_df.index[-1] - eq_df.index[0]).days
    ann = (1 + total_ret) ** (365 / max(n_days, 1)) - 1
    return {
        "sharpe": sharpe, "annualized": ann, "max_dd": max_dd,
        "total_return": total_ret, "final_equity": eq_df.iloc[-1],
        "n_trades": len(trades) - 1,  # exclude initial buy
        "trades": trades,
        "equity_path": eq_df,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("CYCLE TIMING BACKTEST — can we time BTC tops/bottoms?")
    print("=" * 90)
    print()

    df_btc = data.ohlcv_extended("BTC/USDT", days_back=2500)
    print(f"Data: {df_btc.index[0].date()} to {df_btc.index[-1].date()} "
          f"({len(df_btc)} days)")
    print()

    variants = {
        "v0 plain BAH":                 dict(),
        "v1 Mayer 2.0/0.7":             dict(sell_mayer_threshold=2.0, buy_mayer_threshold=0.7),
        "v1b Mayer 1.8/0.7":            dict(sell_mayer_threshold=1.8, buy_mayer_threshold=0.7),
        "v1c Mayer 1.6/0.6":            dict(sell_mayer_threshold=1.6, buy_mayer_threshold=0.6),
        "v1d Mayer 1.5/0.65":           dict(sell_mayer_threshold=1.5, buy_mayer_threshold=0.65),
        "v2 Pi Cycle + Mayer rebuy":    dict(sell_pi_cycle=True, buy_mayer_threshold=0.7),
        "v3 Halving cal (14-18mo/28-36)": dict(sell_halving_window=(14, 18),
                                                buy_halving_window=(28, 38)),
        "v4 Combined (Pi + cal + Mayer)": dict(sell_mayer_threshold=1.6,
                                                 sell_pi_cycle=True,
                                                 sell_halving_window=(14, 20),
                                                 buy_mayer_threshold=0.7,
                                                 buy_halving_window=(28, 38)),
    }

    print("RESULTS")
    print("-" * 90)
    print(f"{'Variant':<32s} {'Sharpe':>7s} {'Annlzd':>9s} {'TotRet':>11s} {'MaxDD':>7s} {'Trades':>6s}")
    results = {}
    for name, params in variants.items():
        r = cycle_timing_backtest(df_btc, **params)
        if r is None:
            continue
        results[name] = r
        print(f"{name:<32s} {r['sharpe']:>+6.2f}  {r['annualized']:>+8.1%}  "
              f"{r['total_return']:>+10.1%}  {r['max_dd']:>5.1%}   {r['n_trades']:>5d}")
    print()

    print("=" * 90)
    print("TRADE LOG — what signals fired for the BEST variant?")
    print("=" * 90)
    best = max(results.items(), key=lambda x: x[1]["sharpe"])
    print(f"Best Sharpe: {best[0]} (Sharpe {best[1]['sharpe']:+.2f})")
    print()
    for t in best[1]["trades"]:
        if t.get("action") == "init_buy":
            print(f"  {t['ts'].date()}  INIT BUY at ${t['price']:>10,.0f}")
        elif t.get("action") == "sell":
            print(f"  {t['ts'].date()}  SELL at ${t['price']:>10,.0f}  "
                  f"signal={t['signal']}, proceeds=${t['proceeds']:,.0f}")
        elif t.get("action") == "buy":
            print(f"  {t['ts'].date()}  REBUY at ${t['price']:>10,.0f}  "
                  f"signal={t['signal']}")
    print()

    print("=" * 90)
    print("HISTORICAL INDICATOR VALUES AT KNOWN TOPS/BOTTOMS")
    print("=" * 90)
    df_ind = add_indicators(df_btc).dropna(subset=["sma200"])
    print(f"{'Date':<12s} {'Price':>10s} {'Mayer':>7s} {'SMA200':>10s} {'PiTop':>6s}")
    known_events = [
        ("2021-04-13", "Apr 2021 top"),
        ("2021-11-09", "Nov 2021 top"),
        ("2022-06-18", "LUNA crash low"),
        ("2022-11-21", "FTX collapse low"),
        ("2023-01-04", "2023 bottom"),
        ("2024-03-13", "2024 peak"),
        ("2025-01-20", "2025 peak"),
        (df_ind.index[-1].strftime("%Y-%m-%d"), "Today"),
    ]
    for d, label in known_events:
        try:
            ts = pd.Timestamp(d, tz="UTC")
            available = df_ind.index[df_ind.index <= ts]
            if len(available) == 0:
                continue
            row = df_ind.loc[available[-1]]
            print(f"{d:<12s} ${row['close']:>9,.0f}  {row['mayer']:>5.2f}  "
                  f"${row['sma200']:>8,.0f}    {bool(row.get('pi_top_signal', False)):>5}  "
                  f"({label})")
        except Exception:
            continue
