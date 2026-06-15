"""Daily top-scorecard check — email alerts when phase changes.

Runs via Crypto_top_check Windows scheduled task. Watches for transitions
between HOLD → TRIM → DEFENSIVE → BEAR_CONFIRMED → FULL_ROTATION.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_top_phase_state.json"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_phase": None, "last_check": None, "last_n_met": 0}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"last_phase": None, "last_check": None, "last_n_met": 0}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2))
    except Exception: pass


def check_top_phase(send_email: bool = True) -> dict:
    """Check if top scorecard phase changed since last check."""
    from core.btc_top_scorecard import top_confirmation_scorecard, phased_exit_recommendation

    sc = top_confirmation_scorecard()
    rec = phased_exit_recommendation(current_equity_pct=70)
    current_phase = sc["verdict_level"]
    n_met = sc["n_met"]

    state = _load_state()
    last_phase = state.get("last_phase")
    last_n_met = state.get("last_n_met", 0)

    state["last_check"] = datetime.now(timezone.utc).isoformat()
    state["last_phase"] = current_phase
    state["last_n_met"] = n_met
    _save_state(state)

    # Phase changed OR criteria firing significantly increased?
    changed = last_phase is not None and last_phase != current_phase
    big_increase = n_met >= last_n_met + 2

    if not (changed or big_increase):
        return {"phase": current_phase, "n_met": n_met,
                "changed": False, "alert_sent": False,
                "message": f"No change. Still in {current_phase} ({n_met}/{sc['n_total']})."}

    # Send alert
    subject = (f"!! Equity TOP scorecard: {last_phase or 'first run'} -> {current_phase} "
                f"({n_met}/{sc['n_total']}) — {rec['verdict_level']} !!")
    body = f"""EQUITY TOP SCORECARD UPDATE

Phase shifted: {last_phase or 'first observation'} -> {current_phase}
Criteria firing: {n_met}/{sc['n_total']} (was {last_n_met})

================================================================
ACTION
================================================================
{sc['verdict']}

{rec['rationale']}

================================================================
CRITERIA FIRING
================================================================
"""
    for c in sc["criteria"]:
        mark = "[FIRING] " if c["met"] else "[not yet]"
        body += f"  {mark} {c['label']}\n"
        body += f"           {c['status']}\n"

    body += f"""
================================================================
WHY EACH ONE MATTERS
================================================================
"""
    for c in sc["criteria"]:
        if c["met"]:
            body += f"  - {c['label']}\n    Rationale: {c['rationale']}\n"

    body += f"""
================================================================
NEXT STEPS
================================================================
Dashboard:  http://localhost:8511 (Overview tab → Top Confirmation Scorecard)
Re-check:   python -m core.btc_top_check
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            return {"phase": current_phase, "n_met": n_met,
                    "changed": True, "alert_sent": True, "subject": subject}
        except Exception as e:
            return {"phase": current_phase, "n_met": n_met,
                    "changed": True, "alert_sent": False, "error": str(e)}
    return {"phase": current_phase, "n_met": n_met,
            "changed": True, "alert_sent": False, "preview": body}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()
    result = check_top_phase(send_email=not args.no_email)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
