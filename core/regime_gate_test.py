"""Test: does a BTC-regime gate improve pro_trend on the 11-pair universe?

Hypothesis: pro_trend's biggest losers happen when the broader market is
chopping or in a bear regime. Filter entries to ONLY days where BTC itself
is in a confirmed bull regime — i.e. only fish when the tide is rising.

Regimes tested:
  R0 (baseline) — no gate
  R1            — BTC above 200d SMA
  R2            — BTC 50d SMA above 200d SMA  (persistent golden cross)
  R3            — BTC 90d return > 0
  R4            — R1 AND R2 AND R3 (strict)

Soft gate: blocks NEW entries when regime is off; existing positions managed
by their normal trail/SMA exits. This preserves the captured runs.
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

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT",
    "LINK/USDT", "DOT/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT",
]


def build_btc_regime(days_back: int = 1500) -> pd.DataFrame:
    """Returns a DataFrame indexed by date with regime flags."""
    btc = data.ohlcv_extended("BTC/USDT", days_back=days_back).copy()
    btc["sma200"] = btc["close"].rolling(200).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["ret90"] = btc["close"].pct_change(90)
    btc["r1"] = btc["close"] > btc["sma200"]
    btc["r2"] = btc["sma50"] > btc["sma200"]
    btc["r3"] = btc["ret90"] > 0
    btc["r4"] = btc["r1"] & btc["r2"] & btc["r3"]
    btc["r0"] = True  # baseline always-on
    return btc[["r0", "r1", "r2", "r3", "r4"]]


def gated_pro_trend(
    pair: str,
    btc_regime: pd.DataFrame,
    regime_col: str = "r0",
    hard_gate: bool = False,
    days_back: int = 1500,
    starting_equity: float = 100_000.0,
    sma_filter: int = 200,
    donchian_window: int = 20,
    atr_period: int = 14,
    atr_stop_mult: float = 4.0,
    risk_pct_per_unit: float = 0.04,
    pyramid_atr_step: float = 2.0,
    max_pyramid_units: int = 2,
    drawdown_kill_pct: float = 0.35,
    round_trip_bps: float = 30.0,
) -> dict:
    df = data.ohlcv_extended(pair, days_back=days_back)
    if df.empty or len(df) < sma_filter * 2:
        return {"error": "insufficient data"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_window).max().shift(1)
    df["sma_filter"] = df["close"].rolling(sma_filter).mean()
    df["atr"] = compute_atr(df, atr_period)
    df = df.join(btc_regime[regime_col].rename("regime_on"), how="left")
    df["regime_on"] = df["regime_on"].fillna(False)
    df = df.dropna()
    n = len(df)

    cash = starting_equity
    units, trades, equity_path = [], [], []
    high_water = trail_stop = 0
    peak_equity = starting_equity

    for i in range(n):
        row = df.iloc[i]
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"])
        sma = float(row["sma_filter"])
        donchian = float(row["donchian_high"])
        regime_on = bool(row["regime_on"])

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        mtm_eq = cash + unrealized
        if mtm_eq > peak_equity:
            peak_equity = mtm_eq
        equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

        if equity_dd > drawdown_kill_pct and units:
            for u in units:
                pnl = u["qty"] * (price - u["entry_price"])
                cash += pnl - u["qty"] * price * round_trip_bps / 2 / 10_000
                trades.append({"pnl": pnl, "n_days": i - u["entry_idx"], "reason": "dd_kill"})
            units, high_water, trail_stop = [], 0, 0

        if units:
            if high > high_water:
                high_water = high
                new_trail = high - atr_stop_mult * atr
                if new_trail > trail_stop:
                    trail_stop = new_trail
            stop_hit = low <= trail_stop
            sma_break = price < sma
            regime_break = hard_gate and not regime_on
            if stop_hit or sma_break or regime_break:
                exit_p = trail_stop if stop_hit else price
                reason = "trail" if stop_hit else ("sma" if sma_break else "regime")
                for u in units:
                    pnl = u["qty"] * (exit_p - u["entry_price"])
                    cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000
                    trades.append({"pnl": pnl, "n_days": i - u["entry_idx"],
                                   "reason": reason})
                units, high_water, trail_stop = [], 0, 0
            elif len(units) < max_pyramid_units:
                last = units[-1]
                if high >= last["entry_price"] + pyramid_atr_step * last["entry_atr"]:
                    stop_dist = atr_stop_mult * atr
                    if stop_dist > 0:
                        qty = min((mtm_eq * risk_pct_per_unit) / stop_dist, mtm_eq * 0.25 / price)
                        cash -= qty * price * round_trip_bps / 2 / 10_000
                        units.append({"qty": qty, "entry_price": price,
                                      "entry_atr": atr, "entry_idx": i})
        else:
            # Initial entry — gated
            if regime_on and price > sma and high >= donchian and atr > 0:
                stop_dist = atr_stop_mult * atr
                qty = min((mtm_eq * risk_pct_per_unit) / stop_dist, mtm_eq * 0.25 / price)
                cash -= qty * price * round_trip_bps / 2 / 10_000
                units = [{"qty": qty, "entry_price": price, "entry_atr": atr, "entry_idx": i}]
                high_water = high
                trail_stop = price - stop_dist

        unrealized = sum(u["qty"] * (price - u["entry_price"]) for u in units)
        equity_path.append({"ts": row.name, "equity": cash + unrealized})

    if units:
        exit_p = float(df["close"].iloc[-1])
        for u in units:
            pnl = u["qty"] * (exit_p - u["entry_price"])
            cash += pnl - u["qty"] * exit_p * round_trip_bps / 2 / 10_000

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
    bah_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["pnl"] > 0) / max(n_trades, 1)

    return {
        "pair": pair, "n_trades": n_trades, "win_rate": float(win_rate),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "sharpe": float(sharpe), "max_drawdown": float(max_dd),
        "bah_return": bah_return, "alpha_vs_bah": float(total_return) - bah_return,
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    annlzd = np.array([r["annualized_return"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    dds = np.array([r["max_drawdown"] for r in results])
    alpha = np.array([r["alpha_vs_bah"] for r in results])
    n_trades = np.array([r["n_trades"] for r in results])
    return {
        "n_pairs": len(results),
        "mean_annlzd": float(annlzd.mean()),
        "median_annlzd": float(np.median(annlzd)),
        "mean_sharpe": float(sharpes.mean()),
        "n_positive": int((annlzd > 0).sum()),
        "n_beat_bah": int((alpha > 0).sum()),
        "mean_max_dd": float(dds.mean()),
        "max_max_dd": float(dds.max()),
        "mean_n_trades": float(n_trades.mean()),
        "total_n_trades": int(n_trades.sum()),
    }


if __name__ == "__main__":
    print("Building BTC regime indicators...")
    btc_regime = build_btc_regime(days_back=1500)
    pct_on = btc_regime.mean() * 100
    print(f"  R0 (always)        : {pct_on['r0']:.0f}% on")
    print(f"  R1 (price>SMA200)  : {pct_on['r1']:.0f}% on")
    print(f"  R2 (SMA50>SMA200)  : {pct_on['r2']:.0f}% on")
    print(f"  R3 (90d ret>0)     : {pct_on['r3']:.0f}% on")
    print(f"  R4 (all combined)  : {pct_on['r4']:.0f}% on")
    print()

    rows = []
    variants = [
        ("R0", "r0", False),
        ("R1-soft", "r1", False),
        ("R1-HARD", "r1", True),
        ("R2-soft", "r2", False),
        ("R2-HARD", "r2", True),
        ("R3-soft", "r3", False),
        ("R3-HARD", "r3", True),
        ("R4-soft", "r4", False),
        ("R4-HARD", "r4", True),
    ]
    for label, regime, hard in variants:
        print(f"=== {label} ===")
        results = []
        for pair in PAIRS:
            r = gated_pro_trend(pair=pair, btc_regime=btc_regime,
                                regime_col=regime, hard_gate=hard, days_back=1500)
            if "error" not in r:
                results.append(r)
        agg = aggregate(results)
        if not agg:
            print("  no successful backtests")
            continue
        print(f"  Pairs: {agg['n_pairs']}, total trades: {agg['total_n_trades']}, "
              f"per pair: {agg['mean_n_trades']:.1f}")
        print(f"  Mean annualized: {agg['mean_annlzd']:+.2%}  "
              f"Median: {agg['median_annlzd']:+.2%}")
        print(f"  Mean Sharpe:     {agg['mean_sharpe']:+.2f}")
        print(f"  Pairs positive:  {agg['n_positive']}/{agg['n_pairs']}")
        print(f"  Pairs beat BAH:  {agg['n_beat_bah']}/{agg['n_pairs']}")
        print(f"  Mean max DD:     {agg['mean_max_dd']:.1%}")
        print(f"  Max max DD:      {agg['max_max_dd']:.1%}")
        print()
        rows.append({"regime": label, **agg})

    print("=" * 70)
    print("SUMMARY (sorted by mean annualized):")
    print("=" * 70)
    rows.sort(key=lambda x: -x["mean_annlzd"])
    print(f"{'Variant':>10s}  {'MeanAnn':>8s}  {'Sharpe':>6s}  {'+/N':>6s}  "
          f"{'BeatBAH':>7s}  {'MaxDD':>6s}  {'Trades':>7s}")
    for r in rows:
        print(f"{r['regime']:>10s}  {r['mean_annlzd']:>+7.2%}  "
              f"{r['mean_sharpe']:>+5.2f}   "
              f"{r['n_positive']:>2d}/{r['n_pairs']:<2d}    "
              f"{r['n_beat_bah']:>2d}/{r['n_pairs']:<2d}    "
              f"{r['mean_max_dd']:>5.1%}   {r['total_n_trades']:>5d}")
