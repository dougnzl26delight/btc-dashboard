"""Critical test: does XSMOM diversify pro_trend?

Standalone Sharpe 0.45 isn't impressive on its own. But if the daily
return correlation to pro_trend is < 0.3, even a modest-Sharpe sleeve
materially increases portfolio Sharpe.

Combined Sharpe formula:
  S_combined = (w1*r1 + w2*r2) / sqrt(w1^2*v1^2 + w2^2*v2^2 + 2*w1*w2*cov)

For uncorrelated strategies (cov ≈ 0), variance is reduced; Sharpe is the
weighted average of individual Sharpes weighted by sigma — roughly
maintained, BUT drawdown reduces and tail risk improves.

Tests several allocation splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.comprehensive_backtest import fetch_all, portfolio_run
from core.xsmom_backtest import xsmom_backtest


ANNUALIZATION = 365


def stats(rets: pd.Series, label: str = "") -> dict:
    if len(rets) < 30:
        return {"label": label}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION)) if rets.std() > 0 else 0
    eq = (1 + rets).cumprod()
    total = float(eq.iloc[-1] - 1)
    n_days = (rets.index[-1] - rets.index[0]).days
    ann = (1 + total) ** (ANNUALIZATION / max(n_days, 1)) - 1
    peak = eq.cummax()
    max_dd = float((1 - eq / peak).max())
    return {"label": label, "sharpe": sharpe, "ann": ann, "max_dd": max_dd,
            "total": total, "n_days": n_days}


if __name__ == "__main__":
    print("=" * 78)
    print("XSMOM × PRO_TREND CORRELATION + COMBINED PORTFOLIO TEST")
    print("=" * 78)
    print()

    # Pro_trend daily returns
    print("Generating pro_trend equity path...")
    pair_data = fetch_all(days_back=2500)
    pt = portfolio_run(
        pair_data=pair_data,
        starting_equity=100_000.0,
        base_risk=0.04, portfolio_risk_cap=0.15,
        atr_stop_mult=4.0, drawdown_kill_pct=0.35,
    )
    pt_rets = pt["daily_returns"]

    # XSMOM daily returns
    print("Generating XSMOM equity path...")
    xs = xsmom_backtest(
        days_back=2500,
        momentum_window=14, rebalance_freq=14,
        long_n=2, short_n=2,
        risk_per_leg=0.20,    # meaningful sized for combined test
    )
    xs_rets = xs["daily_returns"]
    print()

    # Align dates
    common = pt_rets.index.intersection(xs_rets.index)
    pt_rets = pt_rets.loc[common]
    xs_rets = xs_rets.loc[common]

    print(f"Common dates: {len(common)} days")
    print()

    # Correlation
    correlation = float(pt_rets.corr(xs_rets))
    print(f"Daily P&L correlation:  {correlation:+.3f}")
    interp = ("genuinely uncorrelated" if abs(correlation) < 0.2
              else "weakly correlated" if abs(correlation) < 0.4
              else "moderately correlated" if abs(correlation) < 0.6
              else "highly correlated")
    print(f"  Interpretation: {interp}")
    print()

    # Standalone stats
    pt_stats = stats(pt_rets, "pro_trend solo")
    xs_stats = stats(xs_rets, "XSMOM solo")
    print(f"{'Strategy':<28s} {'Annlzd':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for s in [pt_stats, xs_stats]:
        print(f"{s['label']:<28s} {s['ann']:>+7.2%} {s['sharpe']:>+6.2f} "
              f"{s['max_dd']:>6.1%}")
    print()

    # Combined portfolios at different splits
    print("Combined portfolios:")
    print(f"{'Allocation':<28s} {'Annlzd':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    splits = [
        (0.7, 0.3, "70% pro_trend / 30% XSMOM"),
        (0.6, 0.4, "60/40"),
        (0.5, 0.5, "50/50"),
        (0.8, 0.2, "80/20"),
        (0.9, 0.1, "90/10"),
    ]
    best_combo_sharpe = -10
    best_split = None
    for w_pt, w_xs, label in splits:
        combined = w_pt * pt_rets + w_xs * xs_rets
        s = stats(combined, label)
        marker = ""
        if s["sharpe"] > best_combo_sharpe:
            best_combo_sharpe = s["sharpe"]
            best_split = (w_pt, w_xs, s)
            marker = "  <-- best"
        print(f"{s['label']:<28s} {s['ann']:>+7.2%} {s['sharpe']:>+6.2f} "
              f"{s['max_dd']:>6.1%}{marker}")
    print()

    # Verdict
    print("=" * 78)
    print("DECISION GATE")
    print("=" * 78)
    pass_corr = abs(correlation) < 0.3
    pass_sharpe = xs_stats["sharpe"] > 0.3
    sharpe_uplift = best_split[2]["sharpe"] - pt_stats["sharpe"] if best_split else 0
    pass_uplift = sharpe_uplift > 0.1

    print(f"Gate 1 — correlation < 0.3:           {pass_corr} ({correlation:+.3f})")
    print(f"Gate 2 — XSMOM Sharpe > 0.3:          {pass_sharpe} ({xs_stats['sharpe']:+.2f})")
    print(f"Gate 3 — combined Sharpe > pro_trend: {pass_uplift} (uplift {sharpe_uplift:+.2f})")
    print()
    if pass_corr and pass_sharpe and pass_uplift:
        if best_split:
            wpt, wxs, _ = best_split
            print(f"PASSES ALL — wire XSMOM at {wpt:.0%} pro_trend / {wxs:.0%} XSMOM split")
    elif pass_corr and pass_sharpe:
        print("Diversification works (low correlation + positive Sharpe) but no Sharpe uplift")
        print("RECOMMEND: still wire it as drawdown reducer, expect modest return uplift")
    else:
        print("FAILS — XSMOM does not meaningfully diversify pro_trend")
