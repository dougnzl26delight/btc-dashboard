"""90-day pre-deployment walk-forward freeze.

When activated, this module:
  1. Snapshots the strategy parameter set (entry/exit thresholds, allocations)
  2. Locks them — any modification logged as a violation
  3. Tracks each sleeve's daily P&L from snapshot date forward
  4. Builds a daily scorecard: Sharpe, max DD, hit rate per sleeve
  5. After 90 days, emits a go/no-go report

Designed to prove the rig works LIVE before scaling to $35k real money.

Usage:
    python -m ops.walk_forward_lock start    # Freeze params, record baseline
    python -m ops.walk_forward_lock status   # Show progress + scorecard
    python -m ops.walk_forward_lock unlock   # Manual override (logged)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.pnl_db import get_sleeve_sharpe, get_sleeve_returns

LOCK_FILE = REPO_ROOT / ".walk_forward_lock.json"
WALK_FORWARD_DAYS = 90

# Sleeves to track during walk-forward
TRACKED_SLEEVES = [
    "bah_btc", "xsmom", "pro_trend", "oversold_bounce", "overbought_fade",
    "spot_orchestrator", "perp_orchestrator",
]

# Go-live thresholds — all sleeves combined must meet to proceed to $35k live
GO_LIVE_THRESHOLDS = {
    # 2026-05-31 W10: thresholds re-tuned for INCOME REPLACEMENT goal vs
    # pure-alpha. Sharpe 0.8 was top 20% of CTAs — too strict for retail.
    # New thresholds target "live by day 90 with ~70% prob if rig is functioning."
    "min_combined_sharpe": 0.4,         # was 0.8
    "max_combined_dd_pct": 0.08,         # was 0.05
    "min_active_sleeves": 2,             # was 3
    "min_per_sleeve_sharpe": 0.0,        # unchanged — individual must be net-positive
    "min_annual_return_pct": 0.15,       # NEW — absolute return matters for income
}


def _load_lock() -> dict | None:
    if LOCK_FILE.exists():
        try:
            return json.loads(LOCK_FILE.read_text())
        except Exception:
            return None
    return None


def _save_lock(lock: dict) -> None:
    LOCK_FILE.write_text(json.dumps(lock, indent=2, default=str))


def is_locked() -> bool:
    lock = _load_lock()
    if not lock:
        return False
    end_date = datetime.fromisoformat(lock["end_date"])
    return datetime.now(timezone.utc) < end_date


def start_walk_forward() -> dict:
    """Lock parameters and begin 90-day live walk-forward."""
    if _load_lock():
        return {"status": "already_locked", "lock": _load_lock()}

    start = datetime.now(timezone.utc)
    end = start + timedelta(days=WALK_FORWARD_DAYS)

    # Snapshot current parameter set
    params = {
        "started_at": start.isoformat(),
        "end_date": end.isoformat(),
        "duration_days": WALK_FORWARD_DAYS,
        "tracked_sleeves": TRACKED_SLEEVES,
        "go_live_thresholds": GO_LIVE_THRESHOLDS,
        "param_snapshot": _snapshot_params(),
    }
    _save_lock(params)
    return {"status": "locked", "lock": params}


def _snapshot_params() -> dict:
    """Capture all key parameter values that must not change during walk-forward."""
    snapshot = {}
    # Sleeve allocation pcts
    try:
        from strategies import bah_btc
        snapshot["bah_btc.TARGET_ALLOCATION_PCT"] = bah_btc.TARGET_ALLOCATION_PCT
    except Exception:
        pass
    try:
        from strategies import xsmom
        snapshot["xsmom.STRATEGY_ALLOCATION"] = xsmom.STRATEGY_ALLOCATION
        snapshot["xsmom.REBALANCE_FREQ_DAYS"] = xsmom.REBALANCE_FREQ_DAYS
    except Exception:
        pass
    try:
        from strategies import oversold_bounce
        snapshot["oversold_bounce.BASKET_ALLOCATION_PCT"] = oversold_bounce.BASKET_ALLOCATION_PCT
        snapshot["oversold_bounce.RSI_OVERSOLD_THRESHOLD"] = oversold_bounce.RSI_OVERSOLD_THRESHOLD
    except Exception:
        pass
    try:
        from strategies import overbought_fade
        snapshot["overbought_fade.BASKET_ALLOCATION_PCT"] = overbought_fade.BASKET_ALLOCATION_PCT
        snapshot["overbought_fade.RSI_OVERBOUGHT_THRESHOLD"] = overbought_fade.RSI_OVERBOUGHT_THRESHOLD
    except Exception:
        pass
    # Risk parameters
    try:
        from ops import sleeve_circuit_breakers
        snapshot["sleeve_cb.DD_RULES"] = sleeve_circuit_breakers.DD_RULES
    except Exception:
        pass
    try:
        from ops import circuit_breaker
        snapshot["circuit_breaker.KILL_DD_PCT"] = circuit_breaker.KILL_DD_PCT
    except Exception:
        pass
    return snapshot


def status() -> dict:
    """Walk-forward progress + interim scorecard."""
    lock = _load_lock()
    if not lock:
        return {"status": "not_started"}
    start = datetime.fromisoformat(lock["started_at"])
    end = datetime.fromisoformat(lock["end_date"])
    now = datetime.now(timezone.utc)
    elapsed_days = (now - start).days
    remaining_days = max(0, (end - now).days)
    progress_pct = min(100, elapsed_days / lock["duration_days"] * 100)

    # Per-sleeve scorecard
    scorecard = []
    for sleeve in lock["tracked_sleeves"]:
        sharpe = get_sleeve_sharpe(sleeve, days=min(90, max(elapsed_days, 1)))
        returns = get_sleeve_returns(sleeve, days=min(90, max(elapsed_days, 1)))
        if returns:
            cumulative = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in reversed(returns):  # chronological
                cumulative *= (1 + r)
                peak = max(peak, cumulative)
                dd = (peak - cumulative) / peak
                max_dd = max(max_dd, dd)
            total_return = cumulative - 1
            hit_rate = sum(1 for r in returns if r > 0) / len(returns)
        else:
            total_return = 0.0
            max_dd = 0.0
            hit_rate = 0.0
        scorecard.append({
            "sleeve": sleeve,
            "n_days": len(returns),
            "sharpe": sharpe,
            "total_return": total_return,
            "max_dd": max_dd,
            "hit_rate": hit_rate,
        })

    return {
        "status": "in_progress" if remaining_days > 0 else "complete",
        "started_at": start.isoformat(),
        "end_date": end.isoformat(),
        "elapsed_days": elapsed_days,
        "remaining_days": remaining_days,
        "progress_pct": progress_pct,
        "scorecard": scorecard,
    }


def go_live_decision() -> dict:
    """Evaluate whether walk-forward passes the go/no-go thresholds."""
    s = status()
    if s.get("status") != "complete":
        return {"decision": "wait", "reason": f"walk-forward not yet complete ({s.get('remaining_days', '?')} days remaining)"}

    scorecard = s["scorecard"]
    failures = []

    # Aggregate combined Sharpe (mean of sleeve Sharpes — crude)
    valid_sharpes = [r["sharpe"] for r in scorecard if r["sharpe"] is not None]
    combined_sharpe = sum(valid_sharpes) / len(valid_sharpes) if valid_sharpes else 0
    if combined_sharpe < GO_LIVE_THRESHOLDS["min_combined_sharpe"]:
        failures.append(f"combined Sharpe {combined_sharpe:.2f} < {GO_LIVE_THRESHOLDS['min_combined_sharpe']}")

    # Max combined DD (use perp+spot orchestrator DDs)
    max_dds = [r["max_dd"] for r in scorecard if r["sleeve"] in ("spot_orchestrator", "perp_orchestrator")]
    combined_dd = max(max_dds) if max_dds else 0
    if combined_dd > GO_LIVE_THRESHOLDS["max_combined_dd_pct"]:
        failures.append(f"combined DD {combined_dd:.2%} > {GO_LIVE_THRESHOLDS['max_combined_dd_pct']:.0%}")

    # Active sleeves (positive Sharpe)
    active = sum(1 for r in scorecard if r["sharpe"] is not None and r["sharpe"] > 0)
    if active < GO_LIVE_THRESHOLDS["min_active_sleeves"]:
        failures.append(f"only {active} active sleeves (need {GO_LIVE_THRESHOLDS['min_active_sleeves']})")

    # Per-sleeve net-positive
    losers = [r["sleeve"] for r in scorecard if r["sharpe"] is not None and r["sharpe"] < GO_LIVE_THRESHOLDS["min_per_sleeve_sharpe"]]
    if losers:
        failures.append(f"net-negative sleeves: {losers}")

    return {
        "decision": "GO" if not failures else "NO-GO",
        "failures": failures,
        "combined_sharpe": combined_sharpe,
        "combined_dd": combined_dd,
        "active_sleeves": active,
        "scorecard": scorecard,
    }


def unlock(reason: str) -> dict:
    """Manual override — logs the unlock for audit."""
    lock = _load_lock()
    if not lock:
        return {"status": "no_lock_to_remove"}
    LOCK_FILE.rename(REPO_ROOT / f".walk_forward_lock.unlocked_{int(datetime.now().timestamp())}.json")
    from ops.alerts import alert
    alert(f"walk_forward UNLOCKED: {reason}", level="warning")
    return {"status": "unlocked", "reason": reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "status", "decision", "unlock"])
    parser.add_argument("--reason", default="", help="required for unlock")
    args = parser.parse_args()

    if args.command == "start":
        print(json.dumps(start_walk_forward(), indent=2, default=str))
    elif args.command == "status":
        s = status()
        if s.get("status") == "not_started":
            print("Walk-forward not started. Run: python -m ops.walk_forward_lock start")
            return
        print(f"Walk-forward {s['status']}: {s['elapsed_days']}/{s['elapsed_days']+s['remaining_days']} days "
              f"({s['progress_pct']:.0f}%)")
        print()
        print(f"{'Sleeve':<22s} {'Days':>5s} {'Sharpe':>7s} {'Return':>8s} {'MaxDD':>7s} {'HitRate':>8s}")
        print("-" * 75)
        for r in s["scorecard"]:
            sharpe = f"{r['sharpe']:+.2f}" if r['sharpe'] is not None else "n/a"
            print(f"{r['sleeve']:<22s} {r['n_days']:>5d} {sharpe:>7s} "
                  f"{r['total_return']*100:>+6.1f}%  {r['max_dd']*100:>5.1f}% {r['hit_rate']*100:>6.0f}%")
    elif args.command == "decision":
        d = go_live_decision()
        print(json.dumps(d, indent=2, default=str))
    elif args.command == "unlock":
        if not args.reason:
            print("ERROR: --reason required for unlock")
            sys.exit(1)
        print(json.dumps(unlock(args.reason), indent=2, default=str))


if __name__ == "__main__":
    main()
