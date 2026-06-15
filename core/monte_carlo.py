"""Monte Carlo P&L forecast.

Bootstrap historical daily returns of currently-held assets to simulate
the distribution of next-N-day portfolio P&L. Reports percentiles +
probability of various outcomes.

Standard practitioner risk-decision tool: "what's the chance I'm down >$X
30 days from now?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".paper_state.json"


def monte_carlo_forecast(
    horizon_days: int = 30,
    n_simulations: int = 10_000,
    history_days: int = 365,
    block_size: int = 1,
) -> dict:
    """Bootstrap simulation of portfolio P&L over horizon_days.

    block_size: 1 = i.i.d. bootstrap; > 1 = block bootstrap (preserves
    autocorrelation, Politis & Romano 1994).
    """
    if not STATE_FILE.exists():
        return {"error": "no state"}

    state = json.loads(STATE_FILE.read_text())
    cash = float(state.get("cash_quote", 0))
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")

    asset_returns = {}
    asset_values = {}
    for asset, qty in positions.items():
        if abs(qty) < 1e-9:
            continue
        try:
            df = data.ohlcv_extended(f"{asset}/{quote}", days_back=history_days)
            if df.empty:
                continue
            asset_returns[asset] = df["close"].pct_change().dropna().values
            asset_values[asset] = qty * float(df["close"].iloc[-1])
        except Exception:
            continue

    if not asset_returns:
        return {"error": "no positions"}

    aligned_len = min(len(r) for r in asset_returns.values())
    if aligned_len < 30:
        return {"error": "insufficient history"}

    return_matrix = np.array([r[-aligned_len:] for r in asset_returns.values()]).T
    asset_value_vec = np.array([asset_values[a] for a in asset_returns.keys()])
    starting_equity = cash + float(asset_value_vec.sum())

    rng = np.random.default_rng(42)
    final_equity = np.empty(n_simulations)

    for sim in range(n_simulations):
        # Block bootstrap
        if block_size <= 1:
            idx = rng.integers(0, aligned_len, size=horizon_days)
            sampled = return_matrix[idx]
        else:
            n_blocks = (horizon_days + block_size - 1) // block_size
            starts = rng.integers(0, aligned_len - block_size + 1, size=n_blocks)
            blocks = [return_matrix[s : s + block_size] for s in starts]
            sampled = np.concatenate(blocks, axis=0)[:horizon_days]

        # Position values evolve daily by the asset returns (vectorized)
        daily_pnl = sampled @ asset_value_vec
        final_equity[sim] = starting_equity + daily_pnl.sum()

    final_pnl = final_equity - starting_equity
    return {
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "starting_equity": starting_equity,
        "expected_pnl": float(np.mean(final_pnl)),
        "median_pnl": float(np.median(final_pnl)),
        "std_pnl": float(np.std(final_pnl)),
        "p1": float(np.percentile(final_pnl, 1)),
        "p5": float(np.percentile(final_pnl, 5)),
        "p25": float(np.percentile(final_pnl, 25)),
        "p75": float(np.percentile(final_pnl, 75)),
        "p95": float(np.percentile(final_pnl, 95)),
        "p99": float(np.percentile(final_pnl, 99)),
        "prob_positive": float((final_pnl > 0).mean()),
        "prob_loss_5pct_bankroll": float((final_pnl < -5_000).mean()),
        "prob_loss_10pct_bankroll": float((final_pnl < -10_000).mean()),
        "prob_loss_20pct_bankroll": float((final_pnl < -20_000).mean()),
        "prob_gain_10pct": float((final_pnl > 10_000).mean()),
        "samples": final_pnl.tolist()[:1000],  # for histogram in dashboard
    }


if __name__ == "__main__":
    import json as _json
    out = monte_carlo_forecast()
    sample_values = out.pop("samples", [])
    print(_json.dumps(out, indent=2, default=str))
