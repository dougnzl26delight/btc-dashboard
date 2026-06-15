"""Three-way comparison: v1 (continuous), v2 (event-driven), v3 (continuous + GARCH/DD)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies import diverse_mom_ethbtc, tsmom_v2, tsmom_v3


def main():
    print("=" * 70)
    print("v1: continuous-weight ensemble (mom + ethbtc revert)")
    print("=" * 70)
    v1 = diverse_mom_ethbtc.evaluate_strict()
    wf1 = v1.get("walk_forward", {})
    fd1 = v1.get("factor_decomp", {})
    h1 = v1.get("hurdle_oos", {})
    v1_metrics = {
        "Sharpe": wf1.get("mean_sharpe_oos", 0),
        "min_fold": wf1.get("min_sharpe_oos", 0),
        "alpha_ann": fd1.get("alpha_annualized", 0),
        "alpha_t": fd1.get("alpha_t", 0),
        "beta": fd1.get("beta", 0),
        "hurdle_t": h1.get("t_stat", 0),
        "validated": v1.get("validated", False),
    }
    print(_fmt(v1_metrics))

    print()
    print("=" * 70)
    print("v2: discrete event-driven, triple-barrier exits, fractional Kelly")
    print("=" * 70)
    v2 = tsmom_v2.evaluate_strict()
    if "reason" in v2:
        print(f"  skipped: {v2['reason']}")
        v2_metrics = {}
    else:
        pt = v2["per_trade"]
        fd2 = v2["factor_decomp"]
        h2 = v2["hurdle_oos"]
        v2_metrics = {
            "Sharpe (daily MTM)": v2["daily_mtm_sharpe"],
            "Sharpe (per-trade)": pt["sharpe_per_trade_ann"],
            "alpha_ann": fd2.get("alpha_annualized", 0),
            "alpha_t": fd2.get("alpha_t", 0),
            "beta": fd2.get("beta", 0),
            "hurdle_t": h2.get("t_stat", 0),
            "win_rate": pt.get("win_rate", 0),
            "n_trades": pt.get("n_trades", 0),
            "validated": v2.get("validated", False),
        }
        print(_fmt(v2_metrics))

    print()
    print("=" * 70)
    print("v3: v1 ensemble + GARCH vol-targeting + drawdown scaling")
    print("=" * 70)
    v3 = tsmom_v3.evaluate_strict()
    s3 = v3.get("summary", {})
    fd3 = v3.get("factor_decomp", {})
    h3 = v3.get("hurdle_oos", {})
    v3_metrics = {
        "Sharpe (full sample)": s3.get("sharpe", 0),
        "total_return": s3.get("total_return", 0),
        "annualized_return": s3.get("annualized_return", 0),
        "max_drawdown": s3.get("max_drawdown", 0),
        "alpha_ann": fd3.get("alpha_annualized", 0),
        "alpha_t": fd3.get("alpha_t", 0),
        "beta": fd3.get("beta", 0),
        "hurdle_t": h3.get("t_stat", 0),
        "vol_scalar_mean": v3["vol_scalar_stats"]["mean"],
        "vol_scalar_frac_full": v3["vol_scalar_stats"]["frac_full_size"],
        "validated": v3.get("validated", False),
    }
    print(_fmt(v3_metrics))

    print()
    print("=" * 70)
    print("HEAD-TO-HEAD-TO-HEAD")
    print("=" * 70)
    rows = [
        ["Sharpe (full-sample / WF mean)",
         wf1.get("mean_sharpe_oos", 0),
         v2.get("daily_mtm_sharpe", 0),
         s3.get("sharpe", 0)],
        ["Alpha annualized",
         fd1.get("alpha_annualized", 0),
         v2.get("factor_decomp", {}).get("alpha_annualized", 0) if "factor_decomp" in v2 else 0,
         fd3.get("alpha_annualized", 0)],
        ["Alpha t-stat",
         fd1.get("alpha_t", 0),
         v2.get("factor_decomp", {}).get("alpha_t", 0) if "factor_decomp" in v2 else 0,
         fd3.get("alpha_t", 0)],
        ["Beta to BTC",
         fd1.get("beta", 0),
         v2.get("factor_decomp", {}).get("beta", 0) if "factor_decomp" in v2 else 0,
         fd3.get("beta", 0)],
        ["Hurdle t-stat",
         h1.get("t_stat", 0),
         v2.get("hurdle_oos", {}).get("t_stat", 0) if "hurdle_oos" in v2 else 0,
         h3.get("t_stat", 0)],
        ["Validated",
         v1.get("validated", False),
         v2.get("validated", False),
         v3.get("validated", False)],
    ]
    df = pd.DataFrame(rows, columns=["metric", "v1", "v2", "v3"])
    print(df.to_string(index=False))


def _fmt(metrics: dict) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, bool):
            lines.append(f"  {k:30s} {v}")
        elif isinstance(v, int):
            lines.append(f"  {k:30s} {v}")
        else:
            try:
                if abs(v) >= 1:
                    lines.append(f"  {k:30s} {v:+.3f}")
                else:
                    lines.append(f"  {k:30s} {v:+.4f}")
            except Exception:
                lines.append(f"  {k:30s} {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
