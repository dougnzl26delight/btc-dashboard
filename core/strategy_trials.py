"""Strategy trial log for PBO computation.

Selection bias is the largest single source of fake quant edges. Computing
the Probability of Backtest Overfitting requires the FULL set of strategies
tested, not just the survivors.

This module:
    1. Provides log_trial() to record every backtest run with parameters + Sharpe
    2. Provides compute_pbo() over the recorded set via CSCV approximation
    3. Provides cli_report() to print the current overfitting health

References:
    Bailey, Borwein, Lopez de Prado, Zhu (2014):
        "The Probability of Backtest Overfitting"
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


DB_FILE = Path(__file__).resolve().parent.parent / ".pnl.db"


SCHEMA_EXTRA = """
CREATE TABLE IF NOT EXISTS strategy_trials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    variant      TEXT NOT NULL,
    params_json  TEXT,
    n_obs        INTEGER,
    sharpe       REAL,
    max_dd_pct   REAL,
    win_rate_pct REAL,
    deployed     INTEGER DEFAULT 0,
    note         TEXT
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_EXTRA)
    return c


def log_trial(strategy: str, variant: str, sharpe: float,
              n_obs: int, max_dd_pct: float = 0.0, win_rate_pct: float = 0.0,
              params: Optional[dict] = None, deployed: bool = False,
              note: str = "") -> int:
    """Record one backtest result.

    Call this from EVERY backtest run — survivors AND failures. PBO calculation
    is meaningless if only winners are recorded.
    """
    import json
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO strategy_trials"
            "(ts, strategy, variant, params_json, n_obs, sharpe, max_dd_pct, "
            "win_rate_pct, deployed, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), strategy, variant,
             json.dumps(params or {}), n_obs, sharpe, max_dd_pct,
             win_rate_pct, int(deployed), note),
        )
        return cur.lastrowid


def get_all_trials() -> list[dict]:
    """All recorded trials. Use for PBO computation."""
    with _conn() as c:
        rows = c.execute(
            "SELECT ts, strategy, variant, sharpe, max_dd_pct, win_rate_pct, "
            "deployed, note FROM strategy_trials ORDER BY ts DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def compute_pbo_approximation(sharpes: list[float], n_splits: int = 16) -> dict:
    """Approximate PBO from a list of backtest Sharpes via CSCV-style logic.

    True CSCV requires the FULL return series per trial. Without that, we
    can only compute a SIMPLIFIED PBO: rank trials by Sharpe, check whether
    the best-ranked Sharpe is statistically distinguishable from the median.

    A more rigorous PBO would partition each return series into chunks,
    interleave them across train/test splits, and check rank correlation.
    """
    if len(sharpes) < 2:
        return {"pbo": None, "n_trials": len(sharpes), "reason": "insufficient"}

    arr = np.array(sharpes)
    n = len(arr)
    rank_top = (arr == arr.max()).argmax()
    best_sharpe = float(arr.max())
    median_sharpe = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    std = float(arr.std())

    # Simplified probability: under the null (no edge), what fraction of trials
    # would BEAT the median by AT LEAST as much as the top performer did?
    # If many trials are close to best, PBO is high (top is just lucky among many).
    # If best is far above all others, PBO is low (genuinely better).
    if std > 0:
        z_top = (best_sharpe - median_sharpe) / std
        # Under N independent normal trials, P(max > z_top) ≈ 1 - Phi(z_top)^N
        from scipy.stats import norm
        prob_max_exceeds_by_chance = 1 - norm.cdf(z_top) ** n
        pbo_approx = float(prob_max_exceeds_by_chance)
    else:
        pbo_approx = 1.0

    return {
        "pbo_approximation": pbo_approx,
        "n_trials": n,
        "best_sharpe": best_sharpe,
        "median_sharpe": median_sharpe,
        "p25_sharpe": p25,
        "p75_sharpe": p75,
        "sharpe_std": std,
        "verdict": _pbo_verdict(pbo_approx, n),
        "note": "Approximation only — true CSCV requires full return series per trial.",
    }


def _pbo_verdict(pbo: float, n: int) -> str:
    if n < 5:
        return "Need >=5 trials for meaningful PBO"
    if pbo > 0.7:
        return "HIGH overfit risk — best trial likely just lucky"
    if pbo > 0.4:
        return "MODERATE overfit risk — supplement with live walk-forward"
    if pbo > 0.2:
        return "LOW overfit risk — strategy may have real edge"
    return "VERY LOW overfit risk — strong candidate for deployment"


def cli_report():
    trials = get_all_trials()
    print("=" * 100)
    print(f"STRATEGY TRIAL LOG — for PBO + selection-bias accounting")
    print("=" * 100)
    print(f"Total trials recorded: {len(trials)}")
    if not trials:
        print()
        print("No trials logged yet. Add calls to log_trial() in your backtest scripts.")
        print()
        print("Example:")
        print('  from core.strategy_trials import log_trial')
        print('  log_trial("pro_trend", "v5_atr_2.5_donch_20", sharpe=1.45, n_obs=730,')
        print('           max_dd_pct=18.5, win_rate_pct=42.0,')
        print('           params={"atr_mult": 2.5, "donchian": 20})')
        return

    print()
    print(f"{'Strategy':<22s} {'Variant':<30s} {'Sharpe':>8s} {'DD%':>6s} {'WR%':>6s} {'Dep':<4s}")
    print("-" * 100)
    for t in trials[:30]:
        dep = "YES" if t["deployed"] else "no"
        print(f"  {t['strategy']:<20s} {t['variant'][:28]:<30s} "
              f"{t['sharpe']:>+7.2f} {t['max_dd_pct']:>5.1f} {t['win_rate_pct']:>5.1f} {dep:<4s}")

    # Compute PBO across the full set
    sharpes = [t["sharpe"] for t in trials if t["sharpe"] is not None]
    print()
    print("=" * 60)
    print("PBO COMPUTATION")
    print("=" * 60)
    r = compute_pbo_approximation(sharpes)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:<30s}  {v:+.3f}")
        else:
            print(f"  {k:<30s}  {v}")


if __name__ == "__main__":
    cli_report()
