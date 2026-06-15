"""BTC bottom countdown email alerts.

Runs daily via Windows Task Scheduler. Sends emails at three milestones
relative to the pattern-projected cycle 5 bottom:

    T-60 days: "Bottom in 2 months — start watching signals"
    T-30 days: "Bottom in 1 month — prepare capital"
    T-10 days: "Bottom in 10 days — final prep, watch scorecard daily"

PLUS — immediate "BOTTOM CONFIRMED" email any day the scorecard reaches
6/8 hard criteria, regardless of how many days from projected date.

Idempotent: maintains state file to prevent duplicate sends.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STATE_FILE = REPO_ROOT / ".btc_bottom_countdown_state.json"
MILESTONES_DAYS = [60, 30, 10]   # days before pattern bottom to send email


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"sent_milestones": [], "sent_confirmations": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"sent_milestones": [], "sent_confirmations": []}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _build_milestone_email(days_to_bottom: int,
                            milestone: int,
                            state_data: dict) -> tuple[str, str]:
    """Build subject + body for a countdown milestone email."""
    btc_price = state_data["btc_price"]
    regime = state_data["regime"]
    sc = state_data["scorecard"]
    rcd_dd = state_data.get("rcap_drawdown_pct", "?")
    pattern_bottom_date = state_data["pattern_bottom_date"]
    lth_cb = state_data.get("lth_cost_basis", "?")
    sth_cb = state_data.get("sth_cost_basis", "?")
    ev_bottom = state_data.get("ev_bottom_price", "?")

    if milestone == 60:
        title = "BTC bottom in ~2 months — start watching"
        guidance = (
            "Action: light scrutiny phase. Watch dashboard daily. "
            "Confirm capital is liquid and ready. Don't deploy yet — "
            "we're 2 months early. Cost basis hasn't capitulated.\n\n"
            "Specific watchlist:\n"
            "  - Realized Cap drawdown — needs to deepen toward -15%\n"
            "  - MVRV Z-Score — needs to drop below -1.0\n"
            "  - Coinbase Premium — needs to flip positive\n"
            "  - Hashrate Ribbon — watch for cross-up after compression\n"
            "  - 3-week MACD — Jesse Olson's bottom signal\n"
        )
    elif milestone == 30:
        title = "BTC bottom in ~1 month — prepare capital"
        guidance = (
            "Action: medium scrutiny phase. Dashboard refresh every other day. "
            "Have cash positioned for deployment within 24h notice.\n\n"
            "If 3-4 hard criteria firing already: start tier-1 scale-in (25%).\n"
            "If 0-2 criteria: keep waiting. Patience is the edge.\n\n"
            "Heads-up: actual bottom could land anywhere Aug 2026 - Feb 2027, "
            "centered on the projection. Don't anchor on a calendar date."
        )
    elif milestone == 10:
        title = "BTC bottom in ~10 days — final prep"
        guidance = (
            "Action: high scrutiny phase. Dashboard refresh daily.\n\n"
            "Deployment plan:\n"
            "  - Scorecard >= 6/8: aggressive deploy (75-100% of stack)\n"
            "  - Scorecard 4-5/8: tier-1 scale-in (25%)\n"
            "  - Scorecard < 4/8: continue waiting; pattern may be wrong\n\n"
            "Reminder: cost basis confirmation matters more than calendar date. "
            "Realized Cap drawdown is THE indicator. Need at least -15%.\n\n"
            "Don't panic-deploy on the pattern date. Trust the signals."
        )
    else:
        title = f"BTC bottom in {days_to_bottom} days"
        guidance = "Generic countdown alert."

    try:
        from core.plain_email import plain_lead, plain_bottom_level
        _months = ("about 2 months" if milestone == 60 else
                   "about 1 month" if milestone == 30 else
                   "about 10 days" if milestone == 10 else f"{days_to_bottom} days")
        _lead = plain_lead(
            f"Bitcoin's expected bottom (the best time to buy heavily) is {_months} away.",
            [f"Bitcoin is ${btc_price:,.0f} right now.",
             f"Bottom checklist: {plain_bottom_level(sc.get('verdict'))} "
             f"({sc['n_met']}/{sc['n_total']} boxes ticked).",
             "The exact date can drift weeks either way - the signals matter more than the calendar."],
            {60: "Just start watching. Don't buy yet - make sure your cash is liquid and ready.",
             30: "Get cash ready to deploy on 24h notice; only start buying if the checklist is well-filled.",
             10: "Final prep - watch daily. How much to deploy depends on the checklist (see below)."
             }.get(milestone, "Watch the dashboard daily."),
            mood="watch")
    except Exception:
        _lead = ""
    body = _lead + f"""BTC BOTTOM COUNTDOWN — T-{milestone} days

{guidance}

================================================================
CURRENT STATE
================================================================
BTC price:                 ${btc_price:,.0f}
Regime:                    {regime}
Pattern bottom date:       {pattern_bottom_date}
Days to pattern bottom:    {days_to_bottom}

Bottom scorecard:          {sc['n_met']}/{sc['n_total']} hard criteria met
Scorecard verdict:         {sc['verdict']}

Realized Cap drawdown:     {rcd_dd}%  (need -15% min for bottom zone)
LTH cost basis (support):  ${lth_cb}
STH cost basis:            ${sth_cb}
Probability EV bottom:     ${ev_bottom}

================================================================
DASHBOARD
================================================================
Live dashboard: http://localhost:8511
Run weekly report: python -m core.btc_weekly_report
"""
    return title, body


def _build_confirmation_email(state_data: dict) -> tuple[str, str]:
    """Build email for ETF-aware trigger firing (1A, 1B, or 2)."""
    sc = state_data["scorecard"]
    btc = state_data["btc_price"]
    trigger = state_data.get("trigger", {})
    deploy_pct = trigger.get("deploy_pct", 0)
    trigger_name = trigger.get("trigger_name", "BOTTOM TRIGGER")
    entry_zone = trigger.get("entry_zone", "?")

    title = (f"!! BTC {trigger_name} — DEPLOY {deploy_pct}% !!")
    try:
        from core.plain_email import plain_lead
        _lead = plain_lead(
            "This is the big one - a Bitcoin BUY signal just fired.",
            [f"The system suggests putting {deploy_pct}% of your Bitcoin stake in now.",
             f"Bitcoin is ${btc:,.0f} right now.",
             f"Bottom checklist: {sc['n_met']}/{sc['n_total']} boxes ticked."],
            f"Deploy {deploy_pct}% now - spread over 3-5 buys across ~7 days using limit orders. "
            f"Keep {100 - deploy_pct}% in reserve in case it drops further.",
            mood="act")
    except Exception:
        _lead = ""
    body = _lead + f"""BTC {trigger_name}

A deployment trigger has fired. This is the signal you've been waiting for.

================================================================
ACTIVE TRIGGER
================================================================
Trigger ID:       {trigger.get('trigger_id', '?')}
Verdict:          {trigger.get('verdict_label', '?')}
Recommended deploy: {deploy_pct}% of stake
Entry zone:       {entry_zone}

BTC price now:    ${btc:,.0f}
Regime:           {state_data['regime']}
Scorecard:        {sc['n_met']}/{sc['n_total']} hard criteria met
ETF status:       {trigger.get('etf_status', '?')} ({trigger.get('etf_5d_M', 0):+,.0f}M 5d)

================================================================
WHY THIS TRIGGER FIRED
================================================================
{trigger.get('rationale', '')}

================================================================
CRITERIA STATUS
================================================================
"""
    for c in sc.get("criteria", []):
        mark = "[FIRING]  " if c["met"] else "[not yet] "
        body += f"  {mark}{c['label']}\n"
        body += f"           {c['status']}\n"
    body += f"""
================================================================
DEPLOYMENT INSTRUCTIONS
================================================================
Trigger {trigger.get('trigger_id', '?')} recommends DEPLOY {deploy_pct}% NOW.

  - Scale across 3-5 tranches over the next 7 days (not single bullet)
  - Use limit orders, not market orders (avoid slippage)
  - Reserve %: {100-deploy_pct}% kept for potential Trigger 2 / deeper drop

NEXT LEVELS:
  {trigger.get('next_levels', '')}

================================================================
DASHBOARD
================================================================
Live dashboard: http://localhost:8511
Weekly report:  python -m core.btc_weekly_report
"""
    return title, body


def _gather_state() -> dict:
    """Build state snapshot for email content."""
    from core.btc_prediction import state_of_btc
    from core.btc_bottom_scorecard import bottom_confirmation_scorecard
    from core.halving_clock import current_halving_position
    from core.btc_cost_basis import (
        realized_price, sth_cost_basis,
        realized_cap_drawdown_depth, bottom_probability_distribution,
    )
    from core.btc_etf_aware_trigger import etf_aware_bottom_trigger

    s = state_of_btc()
    sc = bottom_confirmation_scorecard(s)
    pos = current_halving_position()
    rp = realized_price()
    sth = sth_cost_basis()
    rcd = realized_cap_drawdown_depth()
    pdb = bottom_probability_distribution()
    trigger = etf_aware_bottom_trigger(s)

    return {
        "btc_price":           s.get("btc_price", 0),
        "regime":              s.get("regime", "?"),
        "scorecard":           sc,
        "trigger":             trigger,
        "pattern_bottom_date": str(pos.get("projected_bottom_date", "?")),
        "days_to_pattern_bottom": pos.get("days_to_pattern_bottom", -999),
        "rcap_drawdown_pct":  f"{rcd['current_drawdown_pct']:+.1f}" if rcd and not rcd.get("error") else "?",
        "lth_cost_basis":     f"{rp['value']:,.0f}" if rp and not rp.get("error") else "?",
        "sth_cost_basis":     f"{sth['value']:,.0f}" if sth and not sth.get("error") else "?",
        "ev_bottom_price":    f"{pdb['expected_value_price']:,.0f}" if pdb and not pdb.get("error") else "?",
    }


def run() -> None:
    """Daily check. Send appropriate emails. Idempotent."""
    print(f"[{datetime.now().isoformat()}] BTC bottom countdown check starting...")

    state_data = _gather_state()
    days_to_bottom = state_data["days_to_pattern_bottom"]
    sc = state_data["scorecard"]
    persisted = _load_state()
    sent_milestones = set(persisted.get("sent_milestones", []))
    sent_confirmations = set(persisted.get("sent_confirmations", []))

    print(f"  BTC: ${state_data['btc_price']:,.0f}")
    print(f"  Days to pattern bottom: {days_to_bottom}")
    print(f"  Scorecard: {sc['n_met']}/{sc['n_total']}")

    sent_anything = False

    # --- Milestone countdown emails ---
    for milestone in MILESTONES_DAYS:
        # Fire if days_to_bottom is at or just below the milestone, and we
        # haven't sent this milestone yet. Allow 5-day tolerance window in
        # case the cron run misses a specific day.
        if milestone in sent_milestones:
            continue
        if days_to_bottom <= milestone and days_to_bottom > (milestone - 5):
            subject, body = _build_milestone_email(days_to_bottom, milestone, state_data)
            try:
                from ops.alerts import alert
                alert(body, level="warning", subject=subject, email=True)
                sent_milestones.add(milestone)
                sent_anything = True
                print(f"  [SENT] Milestone T-{milestone} email")
            except Exception as e:
                print(f"  [FAIL] Could not send T-{milestone} email: {e}")

    # --- TRIGGER ALERTS (ETF-aware) ---
    # Trigger 2 (scorecard >= 6): traditional bottom confirmed
    # Trigger 1B (scorecard >= 4 + ETF outflows): real bottom forming
    # Trigger 1A (scorecard >= 4 + ETF inflows): shallow ETF-era bottom
    today_str = datetime.now(timezone.utc).date().isoformat()
    trigger = state_data.get("trigger", {})
    trigger_id = trigger.get("trigger_id", "WAIT")
    if trigger_id in ("2", "1B", "1A") and today_str not in sent_confirmations:
        subject, body = _build_confirmation_email(state_data)
        try:
            from ops.alerts import alert
            level = "critical" if trigger_id == "2" else "warning"
            alert(body, level=level, subject=subject, email=True)
            sent_confirmations.add(today_str)
            sent_anything = True
            print(f"  [SENT] {trigger.get('trigger_name', 'TRIGGER')} alert (deploy {trigger.get('deploy_pct', 0)}%)")
        except Exception as e:
            print(f"  [FAIL] Could not send trigger alert: {e}")

    # Persist state
    persisted["sent_milestones"] = sorted(sent_milestones)
    persisted["sent_confirmations"] = sorted(sent_confirmations)
    _save_state(persisted)

    if not sent_anything:
        print(f"  No emails needed today. T-{days_to_bottom}d, scorecard {sc['n_met']}/{sc['n_total']}.")


if __name__ == "__main__":
    run()
