"""Per-strategy P&L attribution.

Logs each cycle's per-strategy-per-pair signals plus the realized price moves
that follow. The analyzer reads the log and computes hypothetical
contribution per strategy: if strategy S had been the only signal source on
pair P, what would its P&L have been?

Simple model:
  pnl[strategy, pair] = signal[strategy, pair] * (price_today / price_yesterday - 1) * equity

Aggregated over pairs and time, this surfaces which strategies are working
and which are dead weight. Read in the dashboard / weekly report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ATTRIBUTION_FILE = REPO_ROOT / ".strategy_attribution.jsonl"


def snapshot_signals(
    signals_per_pair_per_strategy: dict[str, dict[str, float]],
    prices: dict[str, float],
    equity: float,
) -> None:
    """Append a per-cycle snapshot to the attribution log.

    signals_per_pair_per_strategy: {pair: {strategy: signal_value}}
    prices: {pair: current_price}
    equity: current portfolio equity
    """
    entry = {
        "ts": time.time(),
        "equity": equity,
        "prices": prices,
        "signals": signals_per_pair_per_strategy,
    }
    with ATTRIBUTION_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_log() -> list[dict]:
    if not ATTRIBUTION_FILE.exists():
        return []
    out = []
    for line in ATTRIBUTION_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def compute_attribution(window_days: int | None = None) -> pd.DataFrame:
    """Hypothetical per-strategy P&L over the lookback window.

    Returns a DataFrame indexed by strategy with columns:
      total_pnl_usdt, avg_daily_pnl, win_rate, n_observations
    """
    log = load_log()
    if len(log) < 2:
        return pd.DataFrame()

    cutoff_ts = time.time() - (window_days * 86_400) if window_days else 0
    log = [e for e in log if e["ts"] >= cutoff_ts]
    if len(log) < 2:
        return pd.DataFrame()

    pnls: dict[str, list[float]] = {}
    for prev, curr in zip(log[:-1], log[1:]):
        prev_prices = prev.get("prices", {})
        curr_prices = curr.get("prices", {})
        equity = prev.get("equity", 100_000.0)

        for pair, strat_sigs in prev.get("signals", {}).items():
            if pair not in prev_prices or pair not in curr_prices:
                continue
            if prev_prices[pair] <= 0:
                continue
            price_ret = curr_prices[pair] / prev_prices[pair] - 1.0
            for strat, sig in strat_sigs.items():
                pnls.setdefault(strat, []).append(float(sig) * price_ret * equity * 0.10)
                # 0.10 = approx max_position_frac per strategy per pair

    rows = []
    for strat, vals in pnls.items():
        if not vals:
            continue
        s = pd.Series(vals)
        rows.append({
            "strategy": strat,
            "total_pnl_usdt": float(s.sum()),
            "avg_pnl_per_obs": float(s.mean()),
            "win_rate": float((s > 0).mean()),
            "n_observations": int(len(s)),
            "best": float(s.max()),
            "worst": float(s.min()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_pnl_usdt", ascending=False)
