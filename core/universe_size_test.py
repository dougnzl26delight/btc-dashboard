"""Test: which subset of the 11-pair universe is optimal?

The universe expansion test showed top 11 > top 25 > top 50 > top 74. Now go
the other direction: maybe top 5-7 concentrates the alpha even better.

Tests every reasonable subset by ranking pairs on:
  - Their individual annualized return
  - Their individual Sharpe
  - Their alpha vs BAH

Then forms portfolios of size 3, 5, 7, 9, 11 from the top of each ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pro_trend_backtest import pro_trend_backtest


PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT",
    "LINK/USDT", "DOT/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT",
]

KW = dict(
    days_back=1500,
    atr_stop_mult=4.0, max_pyramid_units=2,
    risk_pct_per_unit=0.04, drawdown_kill_pct=0.35,
)


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    annlzd = np.array([r["annualized_return"] for r in results])
    sharpes = np.array([r["sharpe"] for r in results])
    dds = np.array([r["max_drawdown"] for r in results])
    alpha = np.array([r["alpha_vs_bah"] for r in results])
    return {
        "n": len(results),
        "mean_annlzd": float(annlzd.mean()),
        "median_annlzd": float(np.median(annlzd)),
        "mean_sharpe": float(sharpes.mean()),
        "n_positive": int((annlzd > 0).sum()),
        "n_beat_bah": int((alpha > 0).sum()),
        "mean_max_dd": float(dds.mean()),
    }


if __name__ == "__main__":
    print("Running per-pair backtests on all 11...")
    per_pair = {}
    for p in PAIRS:
        r = pro_trend_backtest(pair=p, **KW)
        if "error" not in r:
            per_pair[p] = r
            print(f"  {p:<12s} ann {r['annualized_return']:>+7.2%} "
                  f"Sharpe {r['sharpe']:>+5.2f} DD {r['max_drawdown']:>5.1%} "
                  f"alpha {r['alpha_vs_bah']:>+7.1%}")
    print()

    rankings = {
        "by_annlzd": sorted(per_pair.items(), key=lambda x: -x[1]["annualized_return"]),
        "by_sharpe": sorted(per_pair.items(), key=lambda x: -x[1]["sharpe"]),
        "by_alpha":  sorted(per_pair.items(), key=lambda x: -x[1]["alpha_vs_bah"]),
    }

    for rank_name, ranked in rankings.items():
        print(f"=== Ranked {rank_name} ===")
        print(f"  {[p for p,_ in ranked]}")
        print()

    print("=" * 75)
    print("PORTFOLIO ASSEMBLED FROM TOP-K BY EACH RANKING:")
    print("=" * 75)
    print(f"{'Rank by':>10s}  {'K':>3s}  {'MeanAnn':>8s}  {'MedAnn':>8s}  "
          f"{'Sharpe':>6s}  {'+/N':>6s}  {'BeatBAH':>7s}  {'MaxDD':>6s}")

    for rank_name, ranked in rankings.items():
        for k in [3, 5, 7, 9, 11]:
            if k > len(ranked):
                continue
            subset = [pair for pair, _ in ranked[:k]]
            results = [per_pair[p] for p in subset]
            agg = aggregate(results)
            print(f"{rank_name:>10s}  {k:>3d}  "
                  f"{agg['mean_annlzd']:>+7.2%}  "
                  f"{agg['median_annlzd']:>+7.2%}  "
                  f"{agg['mean_sharpe']:>+5.2f}   "
                  f"{agg['n_positive']:>2d}/{agg['n']:<2d}    "
                  f"{agg['n_beat_bah']:>2d}/{agg['n']:<2d}    "
                  f"{agg['mean_max_dd']:>5.1%}")
        print()

    # Also: does removing the worst pairs improve the whole?
    print("=" * 75)
    print("DROP-WORST EXPERIMENT (start with all 11, remove worst by alpha):")
    print("=" * 75)
    drop_order = [p for p, _ in rankings["by_alpha"][::-1]]
    current = set(PAIRS)
    for to_drop in [None] + drop_order:
        if to_drop:
            current.discard(to_drop)
        if len(current) < 3:
            break
        results = [per_pair[p] for p in current if p in per_pair]
        agg = aggregate(results)
        label = "all 11" if to_drop is None else f"drop {to_drop}"
        print(f"{label:<22s}  N={agg['n']:>2d}  "
              f"mean ann {agg['mean_annlzd']:>+7.2%}  "
              f"Sharpe {agg['mean_sharpe']:>+5.2f}  "
              f"DD {agg['mean_max_dd']:>5.1%}")
