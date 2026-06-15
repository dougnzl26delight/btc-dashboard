"""Final-config backtest: top-5 universe + portcap-15% + no catalyst overlay.

This is the production config after all 4 levers have been tested.
Reports the full equity curve and key stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vol_targeted_test import fetch_all, portfolio_backtest


if __name__ == "__main__":
    print("FINAL CONFIG: top-5 universe + portfolio-cap 15% + catalyst OFF")
    print()
    pair_data = fetch_all(days_back=1500)
    print(f"Universe: {list(pair_data.keys())}")
    print(f"Days of data: {min(len(df) for df in pair_data.values())}")
    print()

    r = portfolio_backtest(
        pair_data=pair_data,
        starting_equity=100_000.0,
        base_risk=0.04,
        portfolio_risk_cap=0.15,
        atr_stop_mult=4.0,
        pyramid_atr_step=2.0,
        max_pyramid_units=2,
        drawdown_kill_pct=0.35,  # validated robust to chop (param_sweep 2026-05-10)
    )

    print(f"Starting equity:    ${r['starting_equity']:>12,.2f}")
    print(f"Final equity:       ${r['final_equity']:>12,.2f}")
    print(f"Total return:        {r['total_return']:>+12.2%}")
    print(f"Annualized:          {r['annualized_return']:>+12.2%}")
    print(f"Sharpe:              {r['sharpe']:>+12.2f}")
    print(f"Max drawdown:        {r['max_drawdown']:>+12.2%}")
    print(f"Total trades:        {r['total_trades']:>13d}")
    print(f"DD-kill events:      {r['n_dd_kills']:>13d}")
    print(f"Trades per pair:")
    for p, n in r["trades_per_pair"].items():
        print(f"  {p:<12s} {n:>4d}")
