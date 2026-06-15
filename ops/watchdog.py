"""Heartbeat watchdog — alerts if running strategies have gone silent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.alerts import alert


# 2026-05-31 W13: align file names. ops/heartbeat.py reads .watchdog_beat,
# so this module now writes to BOTH paths for backward compatibility.
HEARTBEAT_FILE = Path(__file__).resolve().parent.parent / ".heartbeat"
WATCHDOG_BEAT_FILE = Path(__file__).resolve().parent.parent / ".watchdog_beat"
STALE_AFTER_S = 6 * 3600  # 6 hours


def beat() -> None:
    """Touch BOTH heartbeat files so all consumers see fresh beat.

    Called by every cron sleeve runner + the realtime_monitor service.
    """
    now = str(time.time())
    HEARTBEAT_FILE.write_text(now)
    WATCHDOG_BEAT_FILE.write_text(now)


def check() -> dict:
    """Inspect heartbeat freshness; alert if stale or missing."""
    if not HEARTBEAT_FILE.exists():
        alert("Crypto watchdog: no heartbeat file — system never ran", level="warning")
        return {"status": "missing", "age_s": None}
    age = time.time() - float(HEARTBEAT_FILE.read_text())
    if age > STALE_AFTER_S:
        alert(f"Crypto watchdog: stale heartbeat ({age/3600:.1f}h)", level="critical")
        return {"status": "stale", "age_s": int(age)}
    return {"status": "fresh", "age_s": int(age)}


if __name__ == "__main__":
    print(check())
