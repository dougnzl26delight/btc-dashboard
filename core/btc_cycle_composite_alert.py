"""I2: Cycle composite alert — Pi Cycle approach + technical divergence.

Fires when:
  Pi Cycle 111d MA within 5% of 350d×2 line, AND
  EITHER weekly RSI bearish divergence forming
  OR 3-week MACD bearish cross within 4 weeks

The "approach" trigger fires BEFORE the cross itself, giving you 1-3
weeks of warning. Combined with technical divergence = high-confidence
top signal.

Runs via Crypto_cycle_composite_check daily.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_cycle_composite_state.json"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_pi_ratio": None, "last_alert_level": None}
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return {"last_pi_ratio": None, "last_alert_level": None}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2))
    except Exception: pass


def check_cycle_composite(send_email: bool = True) -> dict:
    """Check the cycle composite. Email on regime change."""
    from core.btc_native_top_scorecard import (
        pi_cycle_top, weekly_rsi_divergence, macd_3w_bearish_cross
    )

    pi = pi_cycle_top()
    rsi = weekly_rsi_divergence()
    macd = macd_3w_bearish_cross()

    pi_ratio = pi.get("value", 0) or 0
    pi_approaching = pi_ratio > 0.95 and pi_ratio < 1.10  # within 5% either side
    pi_crossed = pi.get("met", False)

    rsi_div = rsi.get("met", False)
    macd_bear = macd.get("met", False)

    # Composite level
    level = "OK"
    if pi_crossed and (rsi_div or macd_bear):
        level = "TOP_CONFIRMED"
    elif pi_crossed:
        level = "PI_CROSSED"
    elif pi_approaching and (rsi_div or macd_bear):
        level = "EARLY_WARNING"
    elif pi_approaching:
        level = "PI_APPROACHING"
    elif rsi_div or macd_bear:
        level = "TECH_BEAR"

    state = _load_state()
    last_level = state.get("last_alert_level")
    state["last_alert_level"] = level
    state["last_pi_ratio"] = pi_ratio
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    # Should we email?
    LEVEL_ORDER = ["OK", "TECH_BEAR", "PI_APPROACHING", "EARLY_WARNING",
                    "PI_CROSSED", "TOP_CONFIRMED"]
    escalated = (LEVEL_ORDER.index(level) > LEVEL_ORDER.index(last_level or "OK"))
    if not escalated:
        return {"level": level, "last_level": last_level, "alert_sent": False,
                "pi_ratio": pi_ratio, "message": "no escalation"}

    # Build email
    subject = f"!! BTC CYCLE COMPOSITE: {last_level or 'first run'} -> {level} !!"
    body = f"""BTC CYCLE COMPOSITE ALERT

Composite level: {last_level or 'first observation'} -> {level}

================================================================
SIGNALS
================================================================
  Pi Cycle ratio (111d / 350d*2):  {pi_ratio:.3f}
    Approaching cross:   {pi_approaching}
    Crossed:             {pi_crossed}
    Status: {pi.get('status', '')}

  Weekly RSI bearish divergence:   {rsi_div}
    Status: {rsi.get('status', '')}

  3-week MACD bearish cross:       {macd_bear}
    Status: {macd.get('status', '')}

================================================================
INTERPRETATION
================================================================
"""
    if level == "TOP_CONFIRMED":
        body += "TOP CONFIRMED. Pi Cycle crossed + technical divergence. Execute exit plan.\n"
    elif level == "PI_CROSSED":
        body += "Pi Cycle has crossed. Historical: top within 0-3 days. Reduce BTC 50% now.\n"
    elif level == "EARLY_WARNING":
        body += "Pi Cycle approaching + technical divergence. Top likely within 2-4 weeks.\n"
    elif level == "PI_APPROACHING":
        body += "Pi Cycle within 5% of cross. Watch closely, prepare exit plan.\n"
    elif level == "TECH_BEAR":
        body += "Technical divergence forming. Not yet confirmed by Pi cycle.\n"
    body += """
================================================================
ACTIONS
================================================================
Dashboard:  http://localhost:8511
Re-check:   python -m core.btc_cycle_composite_alert
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            return {"level": level, "last_level": last_level,
                    "alert_sent": True, "subject": subject}
        except Exception as e:
            return {"level": level, "alert_sent": False, "error": str(e)}
    return {"level": level, "alert_sent": False, "preview": body}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-email", action="store_true")
    a = p.parse_args()
    r = check_cycle_composite(send_email=not a.no_email)
    print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
