"""Monthly OOS (out-of-sample) revalidation.

First Sunday of each month. Re-runs walk-forward on the latest data window
and confirms parameters are still in their optimal neighborhood.

Specifically:
  - Pull max history (2500 days)
  - Walk-forward 5 OOS folds, current params
  - Compute mean OOS Sharpe vs the original (1.27)
  - Re-run the parameter neighborhood (3x3 grid: ATR x portfolio_cap)
    to check the local optimum hasn't drifted
  - Output a written report to monthly_reports/

Alerts if:
  - Mean OOS Sharpe drops below 0.6 (originally 1.27, ~50% degradation)
  - Most recent fold Sharpe < 0
  - Current production params no longer at local optimum

Scheduled as Crypto_monthly_oos (first Sunday at 14:45 NZ).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.comprehensive_backtest import (
    fetch_all, portfolio_run, walk_forward, param_sensitivity,
)
from ops.alerts import alert


REPORTS_DIR = REPO_ROOT / "monthly_reports"

# Reference values from the 2026-05-10 baseline
BASELINE_OOS_SHARPE = 1.27
BASELINE_LATEST_FOLD_SHARPE = -0.12  # was bad even at baseline
BASELINE_PRODUCTION_PARAMS = {"atr_stop_mult": 4.0, "portfolio_cap": 0.15}


def is_first_sunday() -> bool:
    """True only on first Sunday of the month."""
    today = datetime.now(timezone.utc)
    return today.weekday() == 6 and today.day <= 7  # Sunday=6


def build_report(force: bool = False) -> str:
    if not force and not is_first_sunday():
        return ("Skipping: not first Sunday of month "
                f"(today={datetime.now(timezone.utc).date().isoformat()})")

    now = datetime.now(timezone.utc)
    lines = [f"# Monthly OOS Revalidation - {now.date().isoformat()}", ""]

    base_kw = dict(
        starting_equity=100_000.0, base_risk=0.04,
        portfolio_risk_cap=0.15, atr_stop_mult=4.0, drawdown_kill_pct=0.35,
    )

    # === [1] Fetch + max-history single backtest ===
    lines.append("## [1] Max-history single backtest")
    lines.append("")
    pair_data = fetch_all(days_back=2500)
    full = portfolio_run(pair_data=pair_data, **base_kw)
    lines.append(f"- Window: {full['start_date'].date()} -> {full['end_date'].date()} "
                 f"({full['n_days']} days)")
    lines.append(f"- Final equity: ${full['final_equity']:,.0f}")
    lines.append(f"- Annualized:   {full['annualized_return']:+.2%}")
    lines.append(f"- Sharpe:       {full['sharpe']:+.2f}")
    lines.append(f"- Max DD:       {full['max_drawdown']:+.2%}")
    lines.append(f"- Trades:       {full['n_trades']}")
    lines.append("")

    # === [2] Walk-forward 5 folds ===
    lines.append("## [2] Walk-forward 5 OOS folds")
    lines.append("")
    folds = walk_forward(pair_data, n_folds=5, **base_kw)
    if folds:
        lines.append(f"| Fold | Window | Days | Annlzd | Sharpe | DD | BAH |")
        lines.append(f"|---|---|---|---|---|---|---|")
        for f in folds:
            lines.append(f"| {f['fold']} | {f['start']} -> {f['end']} | {f['n_days']} | "
                         f"{f['annualized']:+.1%} | {f['sharpe']:+.2f} | "
                         f"{f['max_dd']:.0%} | {f['bah_basket']:+.1%} |")
        sharpes = [f["sharpe"] for f in folds]
        anns = [f["annualized"] for f in folds]
        mean_sharpe = float(np.mean(sharpes))
        latest_sharpe = sharpes[-1]
        lines.append("")
        lines.append(f"- Mean OOS Sharpe:    {mean_sharpe:+.2f}  "
                     f"(baseline: {BASELINE_OOS_SHARPE:+.2f})")
        lines.append(f"- Latest fold Sharpe: {latest_sharpe:+.2f}")
        lines.append(f"- Mean OOS annualized: {np.mean(anns):+.2%}")
        lines.append(f"- Folds positive:     {sum(s>0 for s in sharpes)}/{len(sharpes)}")
        lines.append("")

        # Alerts
        triggers = []
        if mean_sharpe < 0.6:
            triggers.append(("DEGRADATION",
                             f"Mean OOS Sharpe {mean_sharpe:.2f} < 0.6 "
                             f"(baseline {BASELINE_OOS_SHARPE:.2f})"))
        if latest_sharpe < 0:
            triggers.append(("RECENT_NEGATIVE",
                             f"Latest fold Sharpe {latest_sharpe:.2f} < 0"))
        for severity, msg in triggers:
            lines.append(f"- **ALERT [{severity}]**: {msg}")
            alert(f"Monthly OOS [{severity}]: {msg}", level="warning")
    lines.append("")

    # === [3] Parameter neighborhood ===
    lines.append("## [3] Parameter neighborhood check (3x3 grid)")
    lines.append("")
    sens = param_sensitivity(
        pair_data, atrs=[3.0, 4.0, 5.0], caps=[0.10, 0.15, 0.20],
        starting_equity=100_000.0, base_risk=0.04,
    )
    lines.append(f"| ATR | Cap | Annlzd | Sharpe | DD |")
    lines.append(f"|---|---|---|---|---|")
    best = max(sens, key=lambda r: r["sharpe"])
    for r in sens:
        is_prod = (r["atr_stop_mult"] == 4.0 and r["portfolio_cap"] == 0.15)
        is_best = (r["atr_stop_mult"] == best["atr_stop_mult"]
                   and r["portfolio_cap"] == best["portfolio_cap"])
        marker = " (production)" if is_prod else ""
        if is_best and not is_prod:
            marker = " (NEW OPTIMUM)"
        lines.append(f"| {r['atr_stop_mult']:.1f} | {r['portfolio_cap']:.2f} | "
                     f"{r['annualized']:+.1%} | {r['sharpe']:+.2f} | "
                     f"{r['max_dd']:.0%} |{marker}")
    lines.append("")

    if best["atr_stop_mult"] != 4.0 or best["portfolio_cap"] != 0.15:
        msg = (f"Local optimum has drifted: best is "
               f"ATR={best['atr_stop_mult']} cap={best['portfolio_cap']:.2f} "
               f"(Sharpe {best['sharpe']:.2f}); production is "
               f"4.0 / 0.15 (Sharpe {next(r['sharpe'] for r in sens if r['atr_stop_mult']==4.0 and r['portfolio_cap']==0.15):.2f})")
        lines.append(f"- **NOTE**: {msg}")
        alert(f"Monthly OOS [PARAM_DRIFT]: {msg}", level="warning")
    else:
        lines.append("- Production params remain at local optimum.")
    lines.append("")

    # === [4] Reminder ===
    lines.append("## Reminder")
    lines.append("")
    lines.append("This is REPORTING ONLY. Do NOT change production parameters")
    lines.append("based on one monthly drift signal. Per Charter:")
    lines.append("- Wait for 2 consecutive months of drift before considering change")
    lines.append("- All parameter changes require fresh backtest + 60-day paper test")

    return "\n".join(lines)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    # Force=True for manual runs; production schedule only on first Sunday
    force = "--force" in sys.argv
    report = build_report(force=force)
    print(report)
    if "first Sunday" not in report:
        out_file = REPORTS_DIR / f"monthly_oos_{datetime.now(timezone.utc).date().isoformat()}.md"
        out_file.write_text(report, encoding="utf-8")
        print()
        print(f"=> saved to {out_file}")
