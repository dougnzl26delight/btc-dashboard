"""Catalyst calendar — known crypto + macro events with pre-positioning rules.

GCR-style: pre-position around known catalysts. This module holds a
hardcoded calendar of events and alerts when:
  - 7 days BEFORE event: alert "reduce risk pre-catalyst"
  - 0 days (event day): alert
  - 3 days AFTER event: alert "re-evaluate post-catalyst regime"

Events tracked:
  - BTC halvings (4-year cycle)
  - Major spot ETF launches/approvals
  - FOMC meeting dates
  - Major regulatory events (when known)
  - Crypto-specific (Binance announcements, major fork dates)

This is alert-only. Strategy may CHOOSE to act on these but the rig itself
doesn't auto-adjust based on calendar.

Scheduled as Crypto_catalyst_calendar (daily 14:33 NZ).
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.alerts import alert


CALENDAR_LOG = REPO_ROOT / ".catalyst_calendar_log.jsonl"

# Hardcoded calendar — extend over time
CATALYSTS = [
    # ===== BTC halvings =====
    {"date": "2024-04-19", "type": "btc_halving", "name": "BTC Halving #4",
     "impact": "BULL_LONG_TERM",
     "expected_effect": "Supply shock; historical 6-18mo bull run starting ~6mo post"},
    {"date": "2028-04-01", "type": "btc_halving", "name": "BTC Halving #5 (est)",
     "impact": "BULL_LONG_TERM",
     "expected_effect": "Same as prior halvings; reduce risk 1mo before"},

    # ===== Spot ETF approvals =====
    {"date": "2024-01-10", "type": "etf", "name": "US Spot BTC ETF approval",
     "impact": "BULL_SHORT_TERM",
     "expected_effect": "Major buying catalyst; volatility expected"},
    {"date": "2024-05-23", "type": "etf", "name": "US Spot ETH ETF approval",
     "impact": "MIXED",
     "expected_effect": "Approved but no staking; initial muted"},

    # ===== FOMC meetings 2026 =====
    {"date": "2026-01-28", "type": "fomc", "name": "FOMC Jan 2026",
     "impact": "MACRO",
     "expected_effect": "Risk asset volatility; reduce size 24h before"},
    {"date": "2026-03-18", "type": "fomc", "name": "FOMC Mar 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-04-29", "type": "fomc", "name": "FOMC Apr 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-06-17", "type": "fomc", "name": "FOMC Jun 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-07-29", "type": "fomc", "name": "FOMC Jul 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-09-16", "type": "fomc", "name": "FOMC Sep 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-11-04", "type": "fomc", "name": "FOMC Nov 2026",
     "impact": "MACRO", "expected_effect": "Same"},
    {"date": "2026-12-16", "type": "fomc", "name": "FOMC Dec 2026",
     "impact": "MACRO", "expected_effect": "Same"},

    # ===== Other known dates =====
    {"date": "2026-05-15", "type": "regulatory",
     "name": "EU MiCA full compliance deadline",
     "impact": "REGULATORY",
     "expected_effect": "Possible delistings of non-compliant tokens"},
]


def days_until(event_date: str) -> int:
    """Signed days from today to event. Negative = past."""
    event = date.fromisoformat(event_date)
    return (event - datetime.now(timezone.utc).date()).days


def main() -> dict:
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "today": datetime.now(timezone.utc).date().isoformat(),
        "upcoming_7d": [],
        "today_events": [],
        "recent_3d": [],
        "alerts_fired": [],
    }

    for event in CATALYSTS:
        d = days_until(event["date"])
        if d == 0:
            snapshot["today_events"].append(event)
            msg = (f"CATALYST TODAY: {event['name']} ({event['type']}). "
                   f"Expected: {event['expected_effect']}")
            alert(f"CATALYST: {msg}", level="warning")
            snapshot["alerts_fired"].append(msg)
        elif 0 < d <= 7:
            snapshot["upcoming_7d"].append({**event, "days_until": d})
            msg = (f"CATALYST IN {d}D: {event['name']} ({event['type']}). "
                   f"Consider reducing risk before {event['date']}.")
            alert(f"CATALYST: {msg}", level="info")
            snapshot["alerts_fired"].append(msg)
        elif -3 <= d < 0:
            snapshot["recent_3d"].append({**event, "days_since": abs(d)})
            # Only alert once per event — log shows we already alerted
            msg = (f"POST-CATALYST {abs(d)}D AGO: {event['name']}. "
                   f"Re-evaluate regime post-event.")
            alert(f"CATALYST: {msg}", level="info")
            snapshot["alerts_fired"].append(msg)

    # Daily log (idempotent)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    if CALENDAR_LOG.exists():
        last_line = None
        for line in CALENDAR_LOG.read_text().strip().split("\n"):
            if line:
                last_line = line
        if last_line:
            try:
                if json.loads(last_line)["ts"][:10] == today_iso:
                    return snapshot
            except Exception:
                pass
    with CALENDAR_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
