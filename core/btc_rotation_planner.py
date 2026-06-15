"""Rotation Execution Planner — actionable equities → BTC rotation.

This module turns the rotation indicator's recommendation into concrete,
tradeable tickets you can click in your brokerage:

  - "Sell $X of QQQ, buy $Y of IBIT this week"
  - Tracks what you've already deployed vs the indicator's pace
  - Detects phase changes (WATCH → ACTIVE → AGGRESSIVE) and emails you
  - Maintains a rotation log so you can see your actual execution vs plan

What this DOES:
  ✓ Generate this week's specific trade tickets
  ✓ Log your manual executions to track progress
  ✓ Email you when rotation phase changes (your trigger to deploy more)
  ✓ Show deployed vs planned vs remaining

What this does NOT do:
  ✗ Auto-execute trades on your broker (security boundary)
  ✗ Handle broker credentials
  ✗ Override your judgment — these are RECOMMENDATIONS

Run via Windows scheduled task `Crypto_rotation_check` daily to get
phase-change alerts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / ".btc_rotation_log.json"
PHASE_STATE_FILE = REPO_ROOT / ".btc_rotation_phase_state.json"


# ============================================================
# Rotation log (persistent state)
# ============================================================

def _load_log() -> dict:
    """Load the rotation log, initializing if first run."""
    if not LOG_FILE.exists():
        return {
            "stake_total_nzd": 130000,
            "rotation_started_at": None,
            "trades": [],
            "totals": {
                "btc_deployed_nzd": 0,
                "equity_sold_nzd": 0,
                "tranches_executed": 0,
            },
        }
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return _load_log.__defaults__[0] if hasattr(_load_log, "__defaults__") else {}


def _save_log(log: dict) -> None:
    try:
        LOG_FILE.write_text(json.dumps(log, indent=2, default=str))
    except Exception:
        pass


def log_rotation_trade(date: str, sell_ticker: str, sell_nzd: float,
                       buy_ticker: str, buy_nzd: float,
                       notes: str = "") -> dict:
    """Record a manually-executed rotation trade.

    Call from CLI: `python -m core.btc_rotation_planner log --date 2026-06-04 --sell SPY --sell-nzd 2600 --buy IBIT --buy-nzd 2600`
    """
    log = _load_log()
    if not log.get("rotation_started_at"):
        log["rotation_started_at"] = datetime.now(timezone.utc).isoformat()
    trade = {
        "date":          date,
        "sell_ticker":   sell_ticker,
        "sell_nzd":      sell_nzd,
        "buy_ticker":    buy_ticker,
        "buy_nzd":       buy_nzd,
        "notes":         notes,
        "logged_at":     datetime.now(timezone.utc).isoformat(),
    }
    log["trades"].append(trade)
    log["totals"]["equity_sold_nzd"] += sell_nzd
    log["totals"]["btc_deployed_nzd"] += buy_nzd
    log["totals"]["tranches_executed"] += 1
    _save_log(log)
    return {"trade_logged": trade, "totals": log["totals"]}


# ============================================================
# Weekly rotation plan
# ============================================================

def weekly_rotation_plan(rotation: Optional[dict] = None) -> dict:
    """Generate THIS WEEK's concrete trade tickets.

    Output:
        - exact NZD to deploy this week (from DCA pace)
        - specific tickers to sell (high-beta first per PTJ)
        - specific BTC ETF tickers to buy
        - cumulative progress vs plan
        - whether you're ahead/on-track/behind schedule
    """
    if rotation is None:
        from core.btc_macro_rotation import rotation_phase
        rotation = rotation_phase()
    log = _load_log()

    dca = rotation.get("dca", {})
    risk = rotation.get("risk_management", {})
    sells = rotation.get("what_to_sell", {})

    # This week's tranche amount
    total_plan_nzd = risk.get("deploy_nzd", 0)
    tranches_total = dca.get("tranches", 1)
    pct_per_tranche = dca.get("pct_per_tranche", 0)
    nzd_per_tranche = total_plan_nzd / max(1, tranches_total)
    frequency = dca.get("frequency", "weekly")

    # Progress
    deployed_so_far = log["totals"]["btc_deployed_nzd"]
    tranches_done = log["totals"]["tranches_executed"]
    deployed_pct = (deployed_so_far / max(1, total_plan_nzd) * 100) if total_plan_nzd else 0

    # Expected progress (based on elapsed time)
    start_str = log.get("rotation_started_at")
    weeks_elapsed = 0
    expected_deployed_pct = 0
    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_elapsed = max(0, (now - start_dt).days)
            weeks_elapsed = days_elapsed / 7
            weeks_total = dca.get("weeks", 8)
            expected_deployed_pct = min(100, weeks_elapsed / weeks_total * 100)
        except Exception: pass

    on_track = abs(deployed_pct - expected_deployed_pct) < 10
    pace_status = ("ON TRACK" if on_track
                   else "AHEAD" if deployed_pct > expected_deployed_pct
                   else "BEHIND")

    # Recommended sells this week (high-beta first)
    sell_recommendations = []
    for s in sells.get("sell_first", [])[:2]:  # top 2 categories
        sell_recommendations.append({
            "category": s["category"],
            "examples": s["examples"],
            "rationale": s["rationale"][:80],
        })

    # Buy recommendations (BTC ETFs)
    buy_recommendations = [
        {"ticker": "IBIT", "name": "iShares Bitcoin Trust", "expense_ratio": "0.25%"},
        {"ticker": "FBTC", "name": "Fidelity Wise Origin Bitcoin Fund", "expense_ratio": "0.25%"},
        {"ticker": "BITB", "name": "Bitwise Bitcoin ETF", "expense_ratio": "0.20%"},
    ]

    return {
        "this_week_action": (f"Sell ${nzd_per_tranche:,.0f} NZD of high-beta equities, "
                              f"buy ${nzd_per_tranche:,.0f} NZD of BTC ETF (split across IBIT/FBTC/BITB)"
                              if nzd_per_tranche > 0 else "Hold position — no rotation this week"),
        "tranche_amount_nzd": nzd_per_tranche,
        "frequency":          frequency,
        "rotation_phase":     rotation.get("phase_id", "?"),
        "kelly_deploy_pct":   rotation.get("kelly_pct", 0),
        "total_plan_nzd":     total_plan_nzd,
        "deployed_so_far_nzd": deployed_so_far,
        "deployed_pct":       deployed_pct,
        "weeks_elapsed":      weeks_elapsed,
        "expected_deployed_pct": expected_deployed_pct,
        "pace_status":        pace_status,
        "tranches_done":      tranches_done,
        "tranches_total":     tranches_total,
        "sell_recommendations": sell_recommendations,
        "buy_recommendations": buy_recommendations,
        "execution_checklist": [
            f"1. Open your brokerage (Tiger/Sharesies/Hatch/IBKR)",
            f"2. Sell ${nzd_per_tranche:,.0f} NZD of high-beta equity (e.g. QQQ/ARKK/SOXX)",
            f"3. Wait for settlement (T+1 or T+2)",
            f"4. Buy ${nzd_per_tranche:,.0f} NZD of BTC ETF (IBIT preferred)",
            f"5. Log the trade: `python -m core.btc_rotation_planner log "
            f"--date YYYY-MM-DD --sell TICKER --sell-nzd {nzd_per_tranche:.0f} "
            f"--buy IBIT --buy-nzd {nzd_per_tranche:.0f}`",
        ],
    }


# ============================================================
# Phase change detection
# ============================================================

def _load_phase_state() -> dict:
    if not PHASE_STATE_FILE.exists():
        return {"last_phase": None, "last_check": None, "last_alert_sent": None}
    try:
        return json.loads(PHASE_STATE_FILE.read_text())
    except Exception:
        return {"last_phase": None, "last_check": None, "last_alert_sent": None}


def _save_phase_state(state: dict) -> None:
    try:
        PHASE_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def check_phase_change(send_email: bool = True) -> dict:
    """Check if rotation phase changed since last check. If yes, send alert.

    Run daily via Windows scheduled task.
    """
    from core.btc_macro_rotation import rotation_phase
    rotation = rotation_phase()
    if rotation.get("error"):
        return {"error": rotation["error"]}

    current_phase = rotation.get("phase_id")
    current_action = rotation.get("action", "")
    current_kelly = rotation.get("kelly_pct", 0)

    state = _load_phase_state()
    last_phase = state.get("last_phase")

    # Update state regardless
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    state["last_phase"] = current_phase
    _save_phase_state(state)

    # Check for change
    changed = last_phase is not None and last_phase != current_phase
    if not changed:
        return {
            "phase":         current_phase,
            "last_phase":    last_phase,
            "changed":       False,
            "alert_sent":    False,
            "message":       f"No change. Still in {current_phase}.",
        }

    # PHASE CHANGED — build alert email
    btc = rotation.get("btc", {})
    spy = rotation.get("spy", {})
    risk = rotation.get("risk_management", {})

    subject = (f"!! BTC Rotation Phase Changed: {last_phase} -> {current_phase} !!")
    body = f"""BTC ROTATION PHASE CHANGE

Phase shifted: {last_phase} -> {current_phase}

================================================================
ACTION
================================================================
New verdict:       {current_action}
Kelly deploy %:    {current_kelly}%
Deploy NZD:        ${risk.get('deploy_nzd', 0):,.0f}
Stop loss:         -{risk.get('stop_loss_pct', 0):.0f}% from entry

================================================================
WHY THIS CHANGED
================================================================
{rotation.get('rationale', '')}

================================================================
CURRENT MARKET STATE
================================================================
BTC:  ${btc.get('current_price', 0):,.0f}  drawdown {btc.get('drawdown_pct', 0):+.1f}%
SPY:  ${spy.get('current_price', 0):,.2f}  drawdown {spy.get('drawdown_pct', 0):+.1f}%

VIX term structure: {(rotation.get('vix_term_structure') or {}).get('phase', '?')}
HY credit spreads:  {(rotation.get('hy_credit_spreads') or {}).get('phase', '?')}
Liquidity phase:    {(rotation.get('liquidity') or {}).get('phase', '?')}
Yield curve:        {(rotation.get('yield_curve') or {}).get('phase', '?')}

================================================================
NEXT STEPS
================================================================
Open dashboard:  http://localhost:8511
Weekly plan:     python -m core.btc_rotation_planner plan
Log a trade:     python -m core.btc_rotation_planner log --date YYYY-MM-DD ...
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            state["last_alert_sent"] = datetime.now(timezone.utc).isoformat()
            _save_phase_state(state)
            return {
                "phase":         current_phase,
                "last_phase":    last_phase,
                "changed":       True,
                "alert_sent":    True,
                "subject":       subject,
            }
        except Exception as e:
            return {
                "phase":         current_phase,
                "last_phase":    last_phase,
                "changed":       True,
                "alert_sent":    False,
                "error":         str(e),
            }
    return {
        "phase":      current_phase,
        "last_phase": last_phase,
        "changed":    True,
        "alert_sent": False,
        "preview":    body,
    }


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC rotation execution planner")
    sub = parser.add_subparsers(dest="cmd")

    # plan
    sub.add_parser("plan", help="Show this week's rotation plan")

    # log a trade
    log_p = sub.add_parser("log", help="Log an executed rotation trade")
    log_p.add_argument("--date", required=True)
    log_p.add_argument("--sell", required=True, help="Ticker sold")
    log_p.add_argument("--sell-nzd", required=True, type=float)
    log_p.add_argument("--buy", required=True, help="Ticker bought")
    log_p.add_argument("--buy-nzd", required=True, type=float)
    log_p.add_argument("--notes", default="")

    # check phase change
    check_p = sub.add_parser("check", help="Check for phase change and alert")
    check_p.add_argument("--no-email", action="store_true")

    # show log
    sub.add_parser("log-show", help="Display the rotation log")

    args = parser.parse_args()

    if args.cmd == "plan":
        from core.btc_macro_rotation import rotation_phase
        rotation = rotation_phase()
        plan = weekly_rotation_plan(rotation)
        print("\n" + "=" * 70)
        print("THIS WEEK'S ROTATION PLAN")
        print("=" * 70)
        print(f"  Phase:             {plan['rotation_phase']}")
        print(f"  Kelly deploy %:    {plan['kelly_deploy_pct']}%")
        print(f"  This tranche:      ${plan['tranche_amount_nzd']:,.0f} NZD")
        print(f"  Frequency:         {plan['frequency']}")
        print(f"  Pace status:       {plan['pace_status']} "
              f"({plan['deployed_pct']:.0f}% deployed vs {plan['expected_deployed_pct']:.0f}% expected)")
        print()
        print(f"  ACTION:")
        print(f"    {plan['this_week_action']}")
        print()
        print("  CHECKLIST:")
        for step in plan.get("execution_checklist", []):
            print(f"    {step}")

    elif args.cmd == "log":
        result = log_rotation_trade(args.date, args.sell, args.sell_nzd,
                                     args.buy, args.buy_nzd, args.notes)
        print(f"Trade logged: {result}")

    elif args.cmd == "log-show":
        log = _load_log()
        print(f"Stake total: NZ${log['stake_total_nzd']:,.0f}")
        print(f"Started at:  {log.get('rotation_started_at', 'NEVER')}")
        print(f"Totals: {log['totals']}")
        print()
        print(f"Trades ({len(log['trades'])}):")
        for t in log["trades"]:
            print(f"  {t['date']}  SELL ${t['sell_nzd']:,.0f} {t['sell_ticker']:<6} "
                  f"-> BUY ${t['buy_nzd']:,.0f} {t['buy_ticker']:<6}")

    elif args.cmd == "check":
        result = check_phase_change(send_email=not args.no_email)
        print(json.dumps(result, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
