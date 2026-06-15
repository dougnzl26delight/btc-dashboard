"""Portfolio risk analytics.

Cohorted into three views every quant desk would want:
  1. Strategy correlation matrix — are signals genuinely independent?
  2. Position correlation matrix — are open positions actually diversified?
  3. Portfolio VaR (parametric, historical, expected shortfall)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import attribution, data


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".paper_state.json"


def strategy_signal_correlation(window_days: int = 30) -> pd.DataFrame:
    """Correlation matrix of per-strategy signals.
    Combines all (snapshot, pair) observations to maximize statistical power
    — even a couple of cycles produces 20+ observations per strategy.
    """
    log = attribution.load_log()
    if not log:
        return pd.DataFrame()

    import time
    cutoff = time.time() - window_days * 86400
    log = [e for e in log if e["ts"] >= cutoff]
    if not log:
        return pd.DataFrame()

    rows: list[dict] = []
    for entry in log:
        sigs = entry.get("signals", {})
        for pair, strat_sigs in sigs.items():
            rows.append(strat_sigs)

    if len(rows) < 3:
        return pd.DataFrame()

    df = pd.DataFrame(rows).fillna(0)
    if df.shape[1] < 2:
        return pd.DataFrame()
    return df.corr()


def position_correlation(days_back: int = 90) -> pd.DataFrame:
    """Correlation matrix of returns for currently-held assets.
    Reveals whether your "diversified" positions are actually correlated."""
    if not STATE_FILE.exists():
        return pd.DataFrame()
    state = json.loads(STATE_FILE.read_text())
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")

    held_assets = [a for a, q in positions.items() if abs(q) > 1e-9]
    if len(held_assets) < 2:
        return pd.DataFrame()

    returns = {}
    for asset in held_assets:
        try:
            df = data.ohlcv_extended(f"{asset}/{quote}", days_back=days_back)
            if not df.empty:
                returns[asset] = df["close"].pct_change()
        except Exception:
            continue

    if len(returns) < 2:
        return pd.DataFrame()
    return pd.DataFrame(returns).dropna().corr()


def portfolio_var(confidence: float = 0.99, days_back: int = 365) -> dict:
    """Parametric and historical 1-day VaR + Expected Shortfall on current portfolio."""
    if not STATE_FILE.exists():
        return {"error": "no state"}
    state = json.loads(STATE_FILE.read_text())
    cash = float(state.get("cash_quote", 0))
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")

    asset_returns = {}
    asset_values = {}
    total_exposure = 0.0
    for asset, qty in positions.items():
        if abs(qty) < 1e-9:
            continue
        pair = f"{asset}/{quote}"
        try:
            df = data.ohlcv_extended(pair, days_back=days_back)
            if df.empty:
                continue
            current_price = float(df["close"].iloc[-1])
            asset_values[asset] = qty * current_price  # signed
            asset_returns[asset] = df["close"].pct_change().dropna()
            total_exposure += abs(qty * current_price)
        except Exception:
            continue

    if not asset_returns:
        return {"error": "no positions to assess"}

    # Build portfolio P&L history: sum of (qty * price * daily_return) per day
    aligned = pd.DataFrame(asset_returns).dropna()
    if aligned.empty:
        return {"error": "no aligned return history"}

    # Each row = day; column = asset. Multiply by signed asset value to get $-P&L
    pnl_per_day = aligned.copy()
    for asset in pnl_per_day.columns:
        pnl_per_day[asset] = pnl_per_day[asset] * asset_values[asset]
    portfolio_pnl = pnl_per_day.sum(axis=1)

    if len(portfolio_pnl) < 30:
        return {"error": "insufficient history for VaR"}

    # Historical VaR
    alpha = 1 - confidence
    hist_var = float(np.percentile(portfolio_pnl, alpha * 100))
    hist_es = float(portfolio_pnl[portfolio_pnl <= hist_var].mean()) if (portfolio_pnl <= hist_var).any() else hist_var

    # Parametric (assumes normal — not great for crypto but useful comparison)
    mu = float(portfolio_pnl.mean())
    sigma = float(portfolio_pnl.std())
    from scipy.stats import norm
    param_var = float(mu + sigma * norm.ppf(alpha))

    return {
        "confidence": confidence,
        "n_obs": int(len(portfolio_pnl)),
        "current_exposure_usdt": float(total_exposure),
        "historical_var_1d": hist_var,
        "historical_es_1d": hist_es,
        "parametric_var_1d": param_var,
        "var_pct_of_exposure": float(hist_var / total_exposure) if total_exposure else 0.0,
        "daily_vol_usdt": float(sigma),
    }


if __name__ == "__main__":
    import json as _json
    print("=== Strategy correlation ===")
    print(strategy_signal_correlation().round(2).to_string())
    print()
    print("=== Position correlation ===")
    print(position_correlation().round(2).to_string())
    print()
    print("=== Portfolio VaR ===")
    print(_json.dumps(portfolio_var(), indent=2, default=str))
