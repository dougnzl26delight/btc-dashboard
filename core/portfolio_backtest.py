"""Multi-strategy multi-pair portfolio backtest.

Replays the orchestrator's combine-signals-then-rebalance logic over
historical price data. This is the missing test from the JS review:
we'd validated individual signals but never the FULL stack interaction.

Reports portfolio Sharpe, alpha vs buy-and-hold benchmarks, max drawdown,
and per-pair contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from research import signals as res_sig
from signals.funding_basis import funding_signal_series


SLIPPAGE_BPS = 5
FEE_BPS = 10
ANNUALIZATION = 365
SMA_REGIME_WINDOW = 200
PER_PAIR_CAP = 0.10
COST_PER_SIDE = (SLIPPAGE_BPS + FEE_BPS) / 10_000


def _build_signals(
    prices_df: pd.DataFrame,
    funding_signals: dict[str, pd.Series] | None,
    strategy_set: str = "full",
) -> dict[str, pd.DataFrame]:
    """Compute per-strategy signal DataFrames (rows=dates, cols=pairs)."""
    out: dict[str, pd.DataFrame] = {}

    # TSMOM multi-horizon — per-pair, full history
    tsmom = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for pair in prices_df.columns:
        tsmom[pair] = res_sig.tsmom_multi(prices_df[pair], horizons=(30, 90, 180))
    out["tsmom_multi"] = tsmom.fillna(0)

    if strategy_set == "full":
        # Short-term TSMOM
        st_tsmom = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
        for pair in prices_df.columns:
            st_tsmom[pair] = res_sig.tsmom_single(prices_df[pair], lookback=10)
        out["short_term_tsmom"] = st_tsmom.fillna(0)

        # Vol breakout
        vol_bo = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
        for pair in prices_df.columns:
            vol_bo[pair] = res_sig.vol_breakout(prices_df[pair], window=30)
        out["vol_breakout"] = vol_bo.fillna(0)

    # ETH/BTC reversion (BTC-only)
    if "BTC/USDT" in prices_df.columns and "ETH/USDT" in prices_df.columns:
        ratio = (prices_df["ETH/USDT"] / prices_df["BTC/USDT"]).dropna()
        revert = res_sig.zscore_revert(ratio.reindex(prices_df.index).ffill().bfill(), window=20)
        ethbtc = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        ethbtc["BTC/USDT"] = revert.fillna(0)
        out["ethbtc_revert"] = ethbtc

    # Funding basis (per-pair)
    if funding_signals:
        funding_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        for pair, fseries in funding_signals.items():
            if pair in prices_df.columns and not fseries.empty:
                funding_df[pair] = fseries.reindex(prices_df.index).ffill().fillna(0)
        out["funding_basis"] = funding_df

    return out


def backtest_portfolio(
    universe: list[str],
    days_back: int = 730,
    starting_equity: float = 100_000.0,
    long_only: bool = False,
    apply_regime: bool = True,
    include_funding: bool = True,
    strategy_set: str = "full",
) -> dict:
    """Full-rig portfolio backtest.

    strategy_set: 'full' (all available signals) or 'pruned' (just TSMOM + reversion + funding)
    """
    prices_dict = {}
    for pair in universe:
        df = data.ohlcv_extended(pair, days_back=days_back)
        if not df.empty:
            prices_dict[pair] = df["close"]
    if not prices_dict:
        return {"error": "no price data"}

    aligned_prices = pd.DataFrame(prices_dict).dropna()
    if len(aligned_prices) < SMA_REGIME_WINDOW + 30:
        return {"error": f"insufficient aligned history: {len(aligned_prices)} bars"}

    funding_signals: dict[str, pd.Series] = {}
    if include_funding:
        for pair in universe:
            try:
                base, quote = pair.split("/")
                perp = f"{base}/{quote}:{quote}"
                fs = funding_signal_series(perp_pair=perp, days_back=days_back)
                if not fs.empty:
                    funding_signals[pair] = fs
            except Exception:
                pass

    signals_by_strategy = _build_signals(aligned_prices, funding_signals, strategy_set)
    if not signals_by_strategy:
        return {"error": "no strategy signals computed"}

    sma200 = aligned_prices.rolling(SMA_REGIME_WINDOW).mean()
    is_bull = aligned_prices > sma200

    # Average signals per pair across strategies (equal-weight)
    pair_targets = pd.DataFrame(0.0, index=aligned_prices.index, columns=aligned_prices.columns)
    for pair in aligned_prices.columns:
        contribs = []
        for sname, sdf in signals_by_strategy.items():
            if pair in sdf.columns:
                contribs.append(sdf[pair])
        if contribs:
            pair_targets[pair] = pd.concat(contribs, axis=1).mean(axis=1)

    # Per-pair regime gate
    if apply_regime:
        for pair in pair_targets.columns:
            mask_bull = is_bull[pair].fillna(False)
            pair_targets.loc[~mask_bull & (pair_targets[pair] > 0), pair] = 0
            pair_targets.loc[mask_bull & (pair_targets[pair] < 0), pair] = 0

    if long_only:
        pair_targets = pair_targets.clip(lower=0)

    pair_targets = pair_targets.clip(-PER_PAIR_CAP, PER_PAIR_CAP)

    # Daily backtest loop
    n_days = len(aligned_prices)
    equity = np.zeros(n_days)
    equity[0] = starting_equity
    weights_yesterday = pd.Series(0.0, index=aligned_prices.columns)
    pair_returns = aligned_prices.pct_change().fillna(0)

    for t in range(1, n_days):
        day_returns = pair_returns.iloc[t]
        position_pnl = (weights_yesterday * day_returns).sum() * equity[t - 1]
        targets_today = pair_targets.iloc[t]
        weight_changes = (targets_today - weights_yesterday).abs()
        cost = weight_changes.sum() * COST_PER_SIDE * equity[t - 1]
        equity[t] = equity[t - 1] + position_pnl - cost
        weights_yesterday = targets_today.copy()

    eq_series = pd.Series(equity, index=aligned_prices.index)
    rets = eq_series.pct_change().dropna()
    if rets.std() == 0:
        return {"error": "no return variation"}

    sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION))
    total_return = float(eq_series.iloc[-1] / eq_series.iloc[0] - 1)
    annualized = (1 + total_return) ** (ANNUALIZATION / n_days) - 1 if n_days > 0 else 0
    peak = eq_series.cummax()
    max_dd = float((1 - eq_series / peak).max())

    # Benchmarks
    bench_eqwt_returns = pair_returns.mean(axis=1)
    bench_eqwt_eq = starting_equity * (1 + bench_eqwt_returns).cumprod()
    bench_eqwt_sharpe = (
        float(bench_eqwt_returns.mean() / bench_eqwt_returns.std() * np.sqrt(ANNUALIZATION))
        if bench_eqwt_returns.std() > 0 else 0
    )

    btc_returns = pair_returns["BTC/USDT"] if "BTC/USDT" in pair_returns.columns else pd.Series(0)
    btc_eq = starting_equity * (1 + btc_returns).cumprod() if not btc_returns.empty else pd.Series([starting_equity])
    btc_sharpe = (
        float(btc_returns.mean() / btc_returns.std() * np.sqrt(ANNUALIZATION))
        if btc_returns.std() > 0 else 0
    )

    return {
        "n_days": int(n_days),
        "n_strategies": len(signals_by_strategy),
        "strategies_used": list(signals_by_strategy.keys()),
        "n_pairs": int(len(aligned_prices.columns)),
        "starting_equity": starting_equity,
        "ending_equity": float(eq_series.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float(annualized),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "benchmark_eqwt_total_return": float(bench_eqwt_eq.iloc[-1] / starting_equity - 1),
        "benchmark_eqwt_sharpe": bench_eqwt_sharpe,
        "benchmark_btc_total_return": float(btc_eq.iloc[-1] / starting_equity - 1),
        "benchmark_btc_sharpe": btc_sharpe,
        "alpha_vs_eqwt": total_return - float(bench_eqwt_eq.iloc[-1] / starting_equity - 1),
        "alpha_vs_btc": total_return - float(btc_eq.iloc[-1] / starting_equity - 1),
        "equity_curve": eq_series,
        "benchmark_eqwt_curve": bench_eqwt_eq,
        "benchmark_btc_curve": btc_eq,
        "pair_targets_final": pair_targets.iloc[-1].to_dict(),
        "long_only": long_only,
        "apply_regime": apply_regime,
    }


if __name__ == "__main__":
    UNIVERSE = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    ]

    print("=== FULL strategy stack (TSMOM + short-term + vol-breakout + ETH/BTC + funding) ===")
    result_full = backtest_portfolio(UNIVERSE, days_back=730, strategy_set="full")
    if "error" in result_full:
        print(f"Error: {result_full['error']}")
    else:
        for k in ["n_strategies", "n_pairs", "n_days", "total_return", "annualized_return",
                  "sharpe_annualized", "max_drawdown", "benchmark_eqwt_total_return",
                  "benchmark_eqwt_sharpe", "benchmark_btc_total_return", "benchmark_btc_sharpe",
                  "alpha_vs_eqwt", "alpha_vs_btc"]:
            v = result_full.get(k, "?")
            if isinstance(v, float):
                print(f"  {k:32s} {v:+.4f}" if abs(v) < 5 else f"  {k:32s} {v:+,.2f}")
            else:
                print(f"  {k:32s} {v}")

    print()
    print("=== PRUNED strategy stack (TSMOM + ETH/BTC + funding only) ===")
    result_pruned = backtest_portfolio(UNIVERSE, days_back=730, strategy_set="pruned")
    if "error" in result_pruned:
        print(f"Error: {result_pruned['error']}")
    else:
        for k in ["n_strategies", "n_pairs", "n_days", "total_return", "annualized_return",
                  "sharpe_annualized", "max_drawdown", "alpha_vs_eqwt", "alpha_vs_btc"]:
            v = result_pruned.get(k, "?")
            if isinstance(v, float):
                print(f"  {k:32s} {v:+.4f}" if abs(v) < 5 else f"  {k:32s} {v:+,.2f}")
            else:
                print(f"  {k:32s} {v}")
