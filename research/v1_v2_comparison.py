"""Side-by-side comparison of v1 (continuous-weight) and v2 (event-driven).

Reports both per-trade and daily-MTM metrics so the comparison is fair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies import diverse_mom_ethbtc, tsmom_v2


def main():
    print("=" * 70)
    print("v1 (diverse_mom_ethbtc — continuous weights)")
    print("=" * 70)
    v1 = diverse_mom_ethbtc.evaluate_strict()
    wf = v1.get("walk_forward", {})
    fd = v1.get("factor_decomp", {})
    h = v1.get("hurdle_oos", {})
    print(f"  walk-forward mean OOS Sharpe: {wf.get('mean_sharpe_oos', 0):+.3f}")
    print(f"  walk-forward min fold:        {wf.get('min_sharpe_oos', 0):+.3f}")
    print(f"  alpha annualized:             {fd.get('alpha_annualized', 0):+.4f}")
    print(f"  alpha t-stat:                 {fd.get('alpha_t', 0):+.3f}")
    print(f"  beta to BTC:                  {fd.get('beta', 0):+.3f}")
    print(f"  hurdle DSR:                   {h.get('dsr', 0):.3f}")
    print(f"  hurdle t-stat:                {h.get('t_stat', 0):+.3f}")
    print(f"  validated:                    {v1.get('validated', False)}")

    print()
    print("=" * 70)
    print("v2 (tsmom_v2 — event-driven, triple-barrier, fractional Kelly)")
    print("=" * 70)
    v2 = tsmom_v2.evaluate_strict()
    if "reason" in v2:
        print(f"  evaluation skipped: {v2['reason']}")
        return

    pt = v2["per_trade"]
    fd2 = v2["factor_decomp"]
    h2 = v2["hurdle_oos"]
    print(f"  n trades:                     {pt['n_trades']}")
    print(f"  win rate:                     {pt['win_rate']:.1%}")
    print(f"  avg return per trade:         {pt['avg_return_per_trade']:+.2%}")
    print(f"  avg duration:                 {pt['avg_duration_days']:.1f} days")
    print(f"  trades per year:              {pt['trades_per_year']:.1f}")
    print(f"  per-trade annualized Sharpe:  {pt['sharpe_per_trade_ann']:+.3f}")
    print(f"  daily-MTM Sharpe:             {v2['daily_mtm_sharpe']:+.3f}")
    print(f"  total return (whole sample):  {pt['total_return']:+.2%}")
    print(f"  annualized return:            {pt['annualized_return']:+.2%}")
    print(f"  max drawdown:                 {pt['max_drawdown_per_trade']:.2%}")
    print(f"  fractional Kelly size:        {v2['fractional_kelly_size']:+.4f} of capital")
    print(f"  alpha annualized:             {fd2.get('alpha_annualized', 0):+.4f}")
    print(f"  alpha t-stat:                 {fd2.get('alpha_t', 0):+.3f}")
    print(f"  beta to BTC:                  {fd2.get('beta', 0):+.3f}")
    print(f"  hurdle DSR:                   {h2.get('dsr', 0):.3f}")
    print(f"  hurdle t-stat:                {h2.get('t_stat', 0):+.3f}")
    print(f"  validated:                    {v2.get('validated', False)}")

    print()
    print("=" * 70)
    print("HEAD-TO-HEAD")
    print("=" * 70)
    rows = [
        ["Sharpe (v1: WF mean OOS, v2: daily MTM)", wf.get('mean_sharpe_oos', 0), v2['daily_mtm_sharpe']],
        ["Alpha annualized", fd.get('alpha_annualized', 0), fd2.get('alpha_annualized', 0)],
        ["Alpha t-stat", fd.get('alpha_t', 0), fd2.get('alpha_t', 0)],
        ["Beta to BTC", fd.get('beta', 0), fd2.get('beta', 0)],
        ["Hurdle t-stat", h.get('t_stat', 0), h2.get('t_stat', 0)],
        ["Validated", v1.get('validated', False), v2.get('validated', False)],
    ]
    df = pd.DataFrame(rows, columns=["metric", "v1", "v2"])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
