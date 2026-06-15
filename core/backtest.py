"""Vectorized backtest harness for signal-driven strategies.

Conventions:
  - signal in [-1, 1] is target portfolio weight, evaluated at bar CLOSE
  - traded at NEXT bar open (signal is shifted forward by 1 to prevent lookahead)
  - costs: SLIPPAGE_BPS + FEE_BPS per side, applied to absolute weight change
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import DEFAULT, RiskCaps


SLIPPAGE_BPS = 5    # per side
FEE_BPS = 10        # per side; Binance retail taker
ANNUALIZATION = 365  # crypto 24/7


def run(
    prices: pd.Series,
    signal: pd.Series,
    starting_equity: float = 100_000.0,
    caps: RiskCaps = DEFAULT,
) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices, "signal": signal}).dropna()
    if df.empty:
        return df.assign(weight=0.0, ret=0.0, equity=starting_equity)

    target = df["signal"].clip(-caps.max_position_frac, caps.max_position_frac)
    weight = target.shift(1).fillna(0.0)
    price_ret = df["price"].pct_change().fillna(0.0)
    weight_change = weight.diff().abs().fillna(weight.abs())
    cost = weight_change * (SLIPPAGE_BPS + FEE_BPS) / 10_000.0
    strat_ret = weight * price_ret - cost
    equity = starting_equity * (1.0 + strat_ret).cumprod()

    return pd.DataFrame(
        {
            "price": df["price"],
            "signal": df["signal"],
            "weight": weight,
            "ret": strat_ret,
            "equity": equity,
        }
    )


def run_path_dependent(
    prices: pd.Series,
    signal: pd.Series,
    starting_equity: float = 100_000.0,
    caps: RiskCaps = DEFAULT,
    use_dd_scaling: bool = True,
    kink_dd: float = 0.10,
    kill_dd: float = 0.30,
    vol_target_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Path-dependent backtest with optional drawdown scaling and time-varying
    vol targeting. Slower than `run()` but handles feedback effects.

    vol_target_series: optional series of (target_vol / forecast_vol) scalars
    in [0, 1] that multiplies the signal at each step (e.g. GARCH-conditional).
    """
    df = pd.DataFrame({"price": prices, "signal": signal}).dropna()
    if df.empty:
        return df.assign(weight=0.0, ret=0.0, equity=starting_equity)

    if vol_target_series is not None:
        df["vol_scale"] = vol_target_series.reindex(df.index).clip(0, 1).fillna(1.0)
    else:
        df["vol_scale"] = 1.0

    n = len(df)
    weights = np.zeros(n)
    rets = np.zeros(n)
    equity = np.zeros(n)
    equity[0] = starting_equity
    peak = starting_equity
    cap = caps.max_position_frac

    for i in range(1, n):
        price_ret = df["price"].iloc[i] / df["price"].iloc[i - 1] - 1
        prev_eq = equity[i - 1]

        target_w = float(df["signal"].iloc[i - 1]) * float(df["vol_scale"].iloc[i - 1])
        target_w = max(-cap, min(cap, target_w))

        if use_dd_scaling and peak > 0:
            dd = max(0.0, 1.0 - prev_eq / peak)
            if dd <= kink_dd:
                scale = 1.0
            elif dd >= kill_dd:
                scale = 0.0
            else:
                scale = 1.0 - (dd - kink_dd) / (kill_dd - kink_dd)
            target_w *= scale

        cost = abs(target_w - weights[i - 1]) * (SLIPPAGE_BPS + FEE_BPS) / 10_000.0
        rets[i] = target_w * price_ret - cost
        weights[i] = target_w
        equity[i] = prev_eq * (1.0 + rets[i])
        peak = max(peak, equity[i])

    return pd.DataFrame(
        {
            "price": df["price"],
            "signal": df["signal"],
            "vol_scale": df["vol_scale"],
            "weight": weights,
            "ret": rets,
            "equity": equity,
        }
    )


def summarize(bt: pd.DataFrame, periods_per_year: int = ANNUALIZATION) -> dict:
    if bt.empty or "ret" not in bt.columns:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "n_obs": 0}
    r = bt["ret"].dropna().to_numpy()
    if len(r) < 2:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "n_obs": int(len(r))}

    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
        if r.std(ddof=1) > 0
        else 0.0
    )
    eq = bt["equity"]
    peak = eq.cummax()
    dd = float((1 - eq / peak).max())
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    annual = float(eq.iloc[-1] / eq.iloc[0]) ** (periods_per_year / max(len(r), 1)) - 1
    return {
        "sharpe": sharpe,
        "total_return": total,
        "annualized_return": annual,
        "max_drawdown": dd,
        "n_obs": int(len(r)),
    }
