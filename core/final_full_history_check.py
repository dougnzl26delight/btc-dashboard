"""Verify DD kill 0.30 vs 0.35 on FULL 6.3-year history (the reference window)."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run

if __name__ == "__main__":
    pair_data = fetch_all(days_back=2500)
    print(f"Universe: {list(pair_data.keys())}")
    print(f"Bars: {[len(df) for df in pair_data.values()]}")
    print()

    print(f"{'Config':<30s} {'Annlzd':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'DDkills':>7s}")
    for dd_kill in [0.30, 0.35, 0.40]:
        r = portfolio_run(
            pair_data=pair_data,
            starting_equity=100_000.0,
            base_risk=0.04, portfolio_risk_cap=0.15,
            atr_stop_mult=4.0, drawdown_kill_pct=dd_kill,
        )
        marker = "  *" if dd_kill == 0.30 else ""
        print(f"DD kill = {dd_kill:.2f}                "
              f"{r['annualized_return']:>+8.2%} {r['sharpe']:>+6.2f}   "
              f"{r['max_drawdown']:>5.1%}    {r['n_dd_kills']:>4d}{marker}")
    print()
    print("(* = newly applied production config)")
