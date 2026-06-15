"""Decompose strategy performance by market regime.

For each strategy, bucket its P&L into:
  bull/bear (price vs 200-SMA)  x  high/low vol (rv > median)

Reports Sharpe, return, and time-fraction in each bucket. This is a
diagnostic tool, not a search — it tells you WHEN your strategy works
and when it doesn't, given an already-committed strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, data
from research import signals as res_sig
from strategies import diverse_mom_ethbtc, tsmom_v3


ANNUALIZATION = 365


def regime_buckets(prices: pd.Series) -> pd.DataFrame:
    """Compute (bull, high_vol) flags per day."""
    bull = (prices > prices.rolling(200).mean()).fillna(False)
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(30).std() * np.sqrt(ANNUALIZATION)
    high_vol = (rv > rv.median()).fillna(False)
    return pd.DataFrame({"bull": bull, "high_vol": high_vol}, index=prices.index)


def attribute(returns: pd.Series, regimes: pd.DataFrame) -> dict:
    df = pd.concat([returns.rename("ret"), regimes], axis=1).dropna()
    if df.empty:
        return {}

    out: dict = {}
    for bull_state in (True, False):
        for vol_state in (True, False):
            mask = (df["bull"] == bull_state) & (df["high_vol"] == vol_state)
            r = df.loc[mask, "ret"]
            if len(r) > 30 and r.std() > 0:
                sharpe = float(r.mean() / r.std() * np.sqrt(ANNUALIZATION))
                ann_ret = float(r.mean() * ANNUALIZATION)
            else:
                sharpe = 0.0
                ann_ret = 0.0
            label = f"{'bull' if bull_state else 'bear'}_{'highvol' if vol_state else 'lowvol'}"
            out[label] = {
                "n_obs": int(len(r)),
                "frac_of_time": float(len(r) / len(df)) if len(df) else 0.0,
                "annualized_return": ann_ret,
                "sharpe": sharpe,
            }
    return out


def rolling_sharpe(returns: pd.Series, window: int = 180) -> pd.Series:
    """180-day rolling annualized Sharpe."""
    return (
        returns.rolling(window).mean()
        / returns.rolling(window).std()
        * np.sqrt(ANNUALIZATION)
    )


def main():
    btc = data.ohlcv_extended("BTC/USDT", days_back=2000)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=2000)["close"]
    eth_btc = (eth / btc).dropna()

    regimes = regime_buckets(btc)

    # v1: continuous-weight signal applied via standard backtest
    v1_signal = diverse_mom_ethbtc._ensemble_signal(btc, eth_btc)
    v1_bt = backtest.run(btc, v1_signal)

    # v3: continuous + GARCH + DD via path-dependent backtest
    base = tsmom_v3._ensemble_base_signal(btc, eth_btc)
    vol_s = tsmom_v3._vol_scalar_series(btc)
    v3_bt = backtest.run_path_dependent(
        btc, base, use_dd_scaling=True, vol_target_series=vol_s
    )

    print("=" * 70)
    print("REGIME ATTRIBUTION — v1 (continuous-weight)")
    print("=" * 70)
    v1_attr = attribute(v1_bt["ret"], regimes)
    _print_attr(v1_attr)

    print()
    print("=" * 70)
    print("REGIME ATTRIBUTION — v3 (GARCH + drawdown scaling)")
    print("=" * 70)
    v3_attr = attribute(v3_bt["ret"], regimes)
    _print_attr(v3_attr)

    print()
    print("=" * 70)
    print("ROLLING 180-DAY SHARPE — last 5 windows")
    print("=" * 70)
    v1_roll = rolling_sharpe(v1_bt["ret"]).dropna().tail(5)
    v3_roll = rolling_sharpe(v3_bt["ret"]).dropna().tail(5)
    print("v1 last 5 rolling Sharpes:")
    for ts, s in v1_roll.items():
        print(f"  {ts.date()}: {s:+.3f}")
    print("v3 last 5 rolling Sharpes:")
    for ts, s in v3_roll.items():
        print(f"  {ts.date()}: {s:+.3f}")

    print()
    print("=" * 70)
    print("BENCHMARK — buy and hold BTC under same regimes (sanity check)")
    print("=" * 70)
    bench = btc.pct_change()
    bench_attr = attribute(bench, regimes)
    _print_attr(bench_attr)


def _print_attr(attr: dict) -> None:
    rows = []
    for k, v in attr.items():
        rows.append(
            [k, v["n_obs"], f"{v['frac_of_time']:.1%}", f"{v['annualized_return']:+.1%}", f"{v['sharpe']:+.2f}"]
        )
    df = pd.DataFrame(rows, columns=["regime", "n_days", "%time", "ann_ret", "sharpe"])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
