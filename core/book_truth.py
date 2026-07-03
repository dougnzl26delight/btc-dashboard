"""book_truth — the honest state of the book, in one place.

Not another alpha signal. This is the "truth-telling" layer a serious quant
insists on BEFORE adding features: it answers three questions the rig's
sophistication can hide —

  1. EFFECTIVE BETS  — how many *independent* bets is the book really running?
     (Grinold: IR = IC*sqrt(breadth). 8 sleeves that are one crypto-beta bet
     => breadth ~1 and your achievable IR is capped no matter how good a signal
     is.) Computed from the eigenvalue participation ratio of active sleeves'
     realized daily returns.

  2. LIVE vs BACKTEST — per active sleeve, realized annualized Sharpe vs the
     backtest reference (ops.kill_criteria.BACKTEST_SHARPE). The gap is the
     overfit/decay tax; a large negative gap is a kill flag.

  3. DATA SUFFICIENCY — can we even assess edge yet? If most sleeves are dormant
     or the history is short, the honest answer is "not yet" — and that is the
     single most important output, not a number to trust prematurely.

Pure function -> dict, fail-safe (returns {"status":"unavailable"} on error).
Reads .equity_log.jsonl (per_sleeve_pnl, total_equity, position_value_long).
Reuses core.cost_model + ops.kill_criteria (no duplication).

Dashboard wiring (like btc_alpha_regime):
    from core import book_truth
    r = book_truth.compute()
    st.metric("Effective bets", f"{r['effective_bets']:.1f} / {r['n_sleeves_active']}")
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_DAYS_ASSESS = 30          # below this, don't pretend to judge a sleeve
MIN_ACTIVE_DAYS = 5           # a sleeve counts as "active" with >= this many nonzero P&L days

try:
    from ops.kill_criteria import BACKTEST_SHARPE as _BACKTEST_SHARPE
except Exception:
    _BACKTEST_SHARPE = 1.40

try:
    from core import cost_model as _cm
    def _ann_sharpe(rets):
        return float(_cm.annualized_sharpe(rets, periods_per_year=365))
except Exception:
    def _ann_sharpe(rets):
        a = np.asarray(rets, float)
        return float(a.mean() / a.std() * np.sqrt(365)) if a.std() > 0 else 0.0


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute(equity_log_path: str | None = None) -> dict:
    try:
        path = equity_log_path or (REPO / ".equity_log.jsonl")
        rows = _load(path)
        rows = [r for r in rows if r.get("total_equity")]
        if len(rows) < 3:
            return {"status": "unavailable", "reason": "equity log too short",
                    "summary": "book_truth: insufficient equity history."}

        eq = np.array([float(r["total_equity"]) for r in rows])
        n_days = len(rows)
        # all sleeves ever seen
        sleeves = sorted({s for r in rows for s in (r.get("per_sleeve_pnl") or {})})
        # per-sleeve daily return series (pnl / equity that day)
        sret, active = {}, []
        for s in sleeves:
            pnl = np.array([float((r.get("per_sleeve_pnl") or {}).get(s, 0.0)) for r in rows])
            ret = pnl / eq
            nz = int((pnl != 0).sum())
            sret[s] = ret
            if nz >= MIN_ACTIVE_DAYS:
                active.append(s)

        # per-sleeve stats
        per_sleeve = {}
        for s in sleeves:
            pnl_nz = int((sret[s] * eq != 0).sum())
            is_active = s in active
            # assessable only if the sleeve has traded on enough of its OWN days
            # (a 7-day Sharpe is small-sample noise, not evidence)
            assessable = is_active and pnl_nz >= MIN_DAYS_ASSESS
            sh = _ann_sharpe(sret[s]) if is_active else None
            gap = (sh - _BACKTEST_SHARPE) if sh is not None else None
            per_sleeve[s] = {
                "active_days": pnl_nz,
                "status": ("active" if is_active else "dormant"),
                "live_sharpe": (round(sh, 2) if sh is not None else None),
                "vs_backtest_gap": (round(gap, 2) if gap is not None else None),
                "assessable": bool(assessable),
            }

        # effective number of independent bets (participation ratio of eigenvalues)
        if len(active) >= 2:
            R = np.column_stack([sret[s] for s in active])
            # keep only days where at least one active sleeve moved
            R = R[np.abs(R).sum(axis=1) > 0]
            if R.shape[0] >= 3 and R.std(axis=0).min() > 0:
                C = np.corrcoef(R.T)
                ev = np.linalg.eigvalsh(C)
                ev = ev[ev > 1e-9]
                eff = float(ev.sum() ** 2 / (ev ** 2).sum())
                top_share = float(ev.max() / ev.sum())
            else:
                eff, top_share = float(len(active)), None
        else:
            eff, top_share = float(len(active)), None

        # portfolio metrics
        peak = float(np.maximum.accumulate(eq)[-1])
        cur_dd = float(eq[-1] / peak - 1.0)
        pos_val = float(rows[-1].get("position_value_long") or 0.0)
        net_exposure = pos_val / float(eq[-1]) if eq[-1] else 0.0
        port_ret = np.diff(eq) / eq[:-1]
        port_sharpe = _ann_sharpe(port_ret) if port_ret.std() > 0 else None

        # honest verdict — can only assess if some sleeve has enough of ITS OWN days
        can_assess = any(per_sleeve[s]["assessable"] for s in active)
        if len(active) == 0:
            verdict = "NO LIVE EDGE DATA — every sleeve dormant. Cannot assess anything."
        elif len(active) == 1:
            verdict = (f"EFFECTIVELY 1 BET — only '{active[0]}' has traded "
                       f"({per_sleeve[active[0]]['active_days']}d). The multi-sleeve "
                       f"book is a single live bet; diversification is theoretical.")
        else:
            verdict = (f"{eff:.1f} effective bets from {len(active)} active sleeves"
                       + (f" (top eigen-factor = {top_share*100:.0f}% of variance)" if top_share else ""))
        if not can_assess:
            verdict += (f"  |  NO sleeve has >= {MIN_DAYS_ASSESS} trading days yet — "
                        f"live Sharpes are small-sample noise, not evidence "
                        f"(kill_criteria wants 180d before a kill decision).")

        summary = (f"Book truth: {len(active)}/{len(sleeves)} sleeves active, "
                   f"{eff:.1f} effective bets, portfolio DD {cur_dd*100:+.1f}%, "
                   f"net long {net_exposure*100:.0f}%. Backtest Sharpe ref {_BACKTEST_SHARPE}. {verdict}")

        return {
            "status": "ok",
            "n_days": n_days,
            "n_sleeves": len(sleeves),
            "n_sleeves_active": len(active),
            "active_sleeves": active,
            "effective_bets": round(eff, 2),
            "top_factor_variance_share": (round(top_share, 2) if top_share else None),
            "per_sleeve": per_sleeve,
            "portfolio_drawdown": round(cur_dd, 4),
            "net_long_exposure": round(net_exposure, 4),
            "portfolio_live_sharpe": (round(port_sharpe, 2) if port_sharpe is not None else None),
            "backtest_sharpe_ref": _BACKTEST_SHARPE,
            "can_assess_edge": can_assess,
            "verdict": verdict,
            "summary": summary,
        }
    except Exception as e:
        return {"status": "unavailable", "error": f"{type(e).__name__}: {e}",
                "summary": "book_truth unavailable."}


if __name__ == "__main__":
    import json as _j
    r = compute()
    if r["status"] != "ok":
        print(_j.dumps(r, indent=2)); raise SystemExit
    print("=" * 66)
    print("BOOK TRUTH — the honest state of the book")
    print("=" * 66)
    print(f"  History: {r['n_days']} days   Sleeves: {r['n_sleeves']}  "
          f"(active: {r['n_sleeves_active']})")
    print(f"  EFFECTIVE BETS: {r['effective_bets']}  "
          f"(vs {r['n_sleeves']} nominal sleeves)")
    if r['top_factor_variance_share']:
        print(f"  Top hidden factor = {r['top_factor_variance_share']*100:.0f}% of book variance")
    print(f"  Portfolio drawdown: {r['portfolio_drawdown']*100:+.1f}%   "
          f"net long: {r['net_long_exposure']*100:.0f}%   "
          f"live Sharpe: {r['portfolio_live_sharpe']}")
    print(f"  Backtest Sharpe reference: {r['backtest_sharpe_ref']}")
    print("-" * 66)
    print("  Per sleeve:")
    for s, d in r["per_sleeve"].items():
        sh = d["live_sharpe"]
        gap = d["vs_backtest_gap"]
        line = f"    {s:22s} {d['status']:8s} days={d['active_days']:>3}"
        if sh is not None:
            line += f"  liveSharpe={sh:+.2f}  gap={gap:+.2f}"
            if not d["assessable"]:
                line += "  (n<30: small-sample noise, NOT evidence)"
        print(line)
    print("-" * 66)
    print(f"  VERDICT: {r['verdict']}")
    print("=" * 66)
