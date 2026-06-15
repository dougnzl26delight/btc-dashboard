"""High-volatility event calendar — pause new entries during scheduled events.

Macro events reliably produce 3-10% intraday moves with adverse selection
for systematic strategies. Pausing new entries 2h before/after each event
avoids the worst slippage and false signals.

Events tracked:
    - US Fed meetings (FOMC + minutes release)
    - US CPI release (typically 12:30 UTC on 2nd Tue of month)
    - US PCE release
    - Bitcoin halvings (every ~4 years)
    - Major scheduled crypto events (ETF decisions, exchange listings on Coinbase)

This is a HARDCODED calendar — update annually. For dynamic event scraping,
add a calendar API integration later (Polygon, Tradingeconomics, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional


# Pre-loaded events from now through 2027.
# Format: (datetime_utc, name, vol_score)
#   vol_score: 1 = mild, 2 = moderate, 3 = severe (FOMC, CPI surprises)
SCHEDULED_EVENTS = [
    # === 2026 events ===
    # Fed FOMC meetings (8 per year, ~6 weeks apart)
    (datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc), "FOMC June 2026", 3),
    (datetime(2026, 7, 30, 18, 0, tzinfo=timezone.utc), "FOMC July 2026", 3),
    (datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc), "FOMC Sept 2026", 3),
    (datetime(2026, 11, 5, 19, 0, tzinfo=timezone.utc), "FOMC Nov 2026", 3),
    (datetime(2026, 12, 17, 19, 0, tzinfo=timezone.utc), "FOMC Dec 2026", 3),

    # CPI release (typically 2nd Tue, 12:30 UTC)
    (datetime(2026, 6, 9, 12, 30, tzinfo=timezone.utc), "CPI June 2026", 3),
    (datetime(2026, 7, 14, 12, 30, tzinfo=timezone.utc), "CPI July 2026", 3),
    (datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc), "CPI Aug 2026", 3),
    (datetime(2026, 9, 8, 12, 30, tzinfo=timezone.utc), "CPI Sept 2026", 3),
    (datetime(2026, 10, 13, 12, 30, tzinfo=timezone.utc), "CPI Oct 2026", 3),
    (datetime(2026, 11, 10, 13, 30, tzinfo=timezone.utc), "CPI Nov 2026", 3),
    (datetime(2026, 12, 9, 13, 30, tzinfo=timezone.utc), "CPI Dec 2026", 3),

    # NFP (Non-farm payrolls, 1st Friday of month)
    (datetime(2026, 6, 5, 12, 30, tzinfo=timezone.utc), "NFP June 2026", 2),
    (datetime(2026, 7, 3, 12, 30, tzinfo=timezone.utc), "NFP July 2026", 2),
    (datetime(2026, 8, 7, 12, 30, tzinfo=timezone.utc), "NFP Aug 2026", 2),
    (datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc), "NFP Sept 2026", 2),
    (datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc), "NFP Oct 2026", 2),
    (datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc), "NFP Nov 2026", 2),
    (datetime(2026, 12, 4, 13, 30, tzinfo=timezone.utc), "NFP Dec 2026", 2),

    # === 2027 events ===
    (datetime(2027, 1, 27, 19, 0, tzinfo=timezone.utc), "FOMC Jan 2027", 3),
    (datetime(2027, 3, 17, 18, 0, tzinfo=timezone.utc), "FOMC Mar 2027", 3),
    (datetime(2027, 4, 28, 18, 0, tzinfo=timezone.utc), "FOMC Apr 2027", 3),

    # Bitcoin halving 5 (projected April 2028)
    (datetime(2028, 4, 1, 0, 0, tzinfo=timezone.utc), "BTC Halving 5", 3),
]

# Pause window around each event
PAUSE_HOURS_BEFORE = 2
PAUSE_HOURS_AFTER = 2


def is_high_vol_window(now: Optional[datetime] = None) -> dict:
    """Returns {in_window: bool, event: dict or None, mins_to_event: int or None}.

    True if NOW is within [event_time - 2h, event_time + 2h] for any event.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    for event_time, name, vol_score in SCHEDULED_EVENTS:
        window_start = event_time - timedelta(hours=PAUSE_HOURS_BEFORE)
        window_end = event_time + timedelta(hours=PAUSE_HOURS_AFTER)
        if window_start <= now <= window_end:
            mins_to_event = (event_time - now).total_seconds() / 60
            return {
                "in_window": True,
                "event": {
                    "name": name, "time": event_time.isoformat(),
                    "vol_score": vol_score,
                },
                "mins_to_event": int(mins_to_event),
            }
    return {"in_window": False, "event": None, "mins_to_event": None}


def next_event(now: Optional[datetime] = None, lookahead_days: int = 14) -> Optional[dict]:
    """Return the next upcoming event within lookahead_days."""
    if now is None:
        now = datetime.now(timezone.utc)
    upcoming = [(t, n, v) for t, n, v in SCHEDULED_EVENTS if t > now]
    if not upcoming:
        return None
    upcoming.sort(key=lambda x: x[0])
    t, name, vol_score = upcoming[0]
    hours_until = (t - now).total_seconds() / 3600
    if hours_until > lookahead_days * 24:
        return None
    return {
        "name": name, "time": t.isoformat(), "vol_score": vol_score,
        "hours_until": hours_until,
    }


def main():
    """CLI: show current window status + next 3 events."""
    status = is_high_vol_window()
    if status["in_window"]:
        e = status["event"]
        print(f"!! HIGH-VOL WINDOW ACTIVE: {e['name']} (vol_score={e['vol_score']})")
        print(f"   Mins to event: {status['mins_to_event']:+d}")
        print(f"   Strategies should NOT enter new positions during this window.")
    else:
        print("Current status: NO high-vol event window — normal trading.")
    print()
    nxt = next_event()
    if nxt:
        print(f"Next event: {nxt['name']}")
        print(f"   Time:  {nxt['time']}")
        print(f"   Hours until: {nxt['hours_until']:.1f}")
        print(f"   Vol score: {nxt['vol_score']}")
    print()
    print(f"Total scheduled events in calendar: {len(SCHEDULED_EVENTS)}")


if __name__ == "__main__":
    main()
