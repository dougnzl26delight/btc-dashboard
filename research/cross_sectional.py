"""Cross-sectional momentum portfolio across BTC/ETH/SOL.

Standard formulation: at each bar, rank pairs by trailing N-day return.
Long top tercile, short bottom tercile, equal-weighted within ties. Vol-targeted.
This is dollar-neutral within the portfolio (long_w - short_w sums to 0).

Tests whether RELATIVE rank predicts forward returns — distinct from
time-series momentum which trades absolute trends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data, deflated_sharpe, factor_decomp


SLIPPAGE_BPS = 5
FEE_BPS = 10
ANNUALIZATION = 365


def _xs_portfolio_returns(
    aligned_prices: pd.DataFrame, lookback: int, target_gross: float = 0.20
) -> pd.Series:
    log_ret = np.log(aligned_prices / aligned_prices.shift(1))
    trailing = log_ret.rolling(lookback).sum()
    ranks = trailing.rank(axis=1, pct=True)

    long_pos = (ranks > 2.0 / 3.0).astype(float)
    short_pos = (ranks < 1.0 / 3.0).astype(float)
    n_long = long_pos.sum(axis=1).replace(0, np.nan)
    n_short = short_pos.sum(axis=1).replace(0, np.nan)

    long_w = long_pos.div(n_long, axis=0).fillna(0)
    short_w = short_pos.div(n_short, axis=0).fillna(0)
    weights = (long_w - short_w) * (target_gross / 2.0)  # half on each side

    daily_ret = aligned_prices.pct_change().fillna(0)
    weight_change = weights.diff().abs().fillna(weights.abs())
    cost = weight_change.sum(axis=1) * (SLIPPAGE_BPS + FEE_BPS) / 10_000.0
    portfolio_ret = (weights.shift(1) * daily_ret).sum(axis=1) - cost
    return portfolio_ret.dropna()


def evaluate_xs(
    lookback: int = 90,
    pairs: tuple = ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
    days_back: int = 2000,
    n_folds: int = 5,
    min_train: int = 365,
    num_trials: int = 30,
) -> dict:
    prices = {p: data.ohlcv_extended(p, days_back=days_back)["close"] for p in pairs}
    aligned = pd.DataFrame(prices).dropna()
    if len(aligned) < min_train + n_folds * 60:
        return {"validated": False, "reason": f"insufficient overlap: {len(aligned)} bars"}

    portfolio_ret = _xs_portfolio_returns(aligned, lookback=lookback)

    # Fold-by-fold OOS Sharpe
    n = len(portfolio_ret)
    fold_size = (n - min_train) // n_folds
    fold_sharpes: list[float] = []
    fold_returns: list[pd.Series] = []
    for k in range(n_folds):
        start = min_train + k * fold_size
        end = min(start + fold_size, n)
        chunk = portfolio_ret.iloc[start:end]
        if len(chunk) < 30:
            continue
        s = float(chunk.mean() / chunk.std() * np.sqrt(ANNUALIZATION)) if chunk.std() > 0 else 0.0
        fold_sharpes.append(s)
        fold_returns.append(chunk)

    concat = pd.concat(fold_returns) if fold_returns else pd.Series(dtype=float)

    # Alpha vs BTC benchmark
    bench = aligned[pairs[0]].pct_change().reindex(portfolio_ret.index).fillna(0)
    decomp = factor_decomp.decompose(portfolio_ret, bench)

    hurdle = (
        deflated_sharpe.passes_quant_hurdle(concat, num_trials=num_trials)
        if len(concat) >= 30
        else {"passes_combined": False, "dsr": 0.0, "t_stat": 0.0}
    )

    mean_oos = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    min_oos = float(np.min(fold_sharpes)) if fold_sharpes else 0.0

    return {
        "name": f"xs_momentum_{lookback}",
        "lookback": lookback,
        "n_obs": len(portfolio_ret),
        "n_folds": len(fold_sharpes),
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
        "mean_sharpe_oos": round(mean_oos, 3),
        "min_sharpe_oos": round(min_oos, 3),
        "alpha_ann": round(decomp.get("alpha_annualized", 0.0), 4),
        "alpha_t": round(decomp.get("alpha_t", 0.0), 2),
        "beta_to_btc": round(decomp.get("beta", 0.0), 3),
        "dsr": round(hurdle.get("dsr", 0.0), 3),
        "hurdle_t": round(hurdle.get("t_stat", 0.0), 2),
        "validated": (
            mean_oos > 0.5
            and min_oos > 0.0
            and decomp.get("passes_alpha_t", False)
            and hurdle.get("passes_combined", False)
        ),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    rows = []
    for lb in (30, 60, 90, 180):
        r = evaluate_xs(lookback=lb)
        rows.append(r)
        print(
            f"xs_momentum_{lb}: OOS={r.get('mean_sharpe_oos',0):+.2f} "
            f"(min={r.get('min_sharpe_oos',0):+.2f}) alpha_t={r.get('alpha_t',0):+.2f} "
            f"beta_to_btc={r.get('beta_to_btc',0):+.3f}"
        )
    df = pd.DataFrame(rows)
    cols = ["name", "n_obs", "mean_sharpe_oos", "min_sharpe_oos", "alpha_t", "beta_to_btc", "dsr", "validated"]
    print()
    print(df[cols].to_string(index=False))
