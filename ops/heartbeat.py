"""SMS heartbeat + remote kill switch.

Two functions:
  1. Heartbeat watchdog: if watchdog.beat() hasn't been called in N hours,
     fire an SMS alert via CallMeBot (free, personal use). Catches the case
     where the entire rig has frozen — Telegram bot can't tell you because
     it's running in the same process.

  2. Telegram /halt command poller: long-polls Telegram bot API for /halt
     messages from your chat_id and writes the kill switch file. Lets you
     halt the rig from your phone with no laptop access.

Run as a separate scheduled task (Crypto_heartbeat_5min) — must not share
process with anything that could deadlock the main rig.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

WATCHDOG_FILE = REPO_ROOT / ".watchdog_beat"
KILL_FILE = REPO_ROOT / ".kill_switch.json"
TELEGRAM_OFFSET_FILE = REPO_ROOT / ".telegram_update_offset"

HEARTBEAT_TIMEOUT_HOURS = 6

_CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE", "").strip()
_CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY", "").strip()
_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_sms(message: str) -> bool:
    """Send SMS via CallMeBot. Free for personal use after one-time setup at
    https://www.callmebot.com/blog/free-api-text-messages/
    """
    if not _CALLMEBOT_PHONE or not _CALLMEBOT_APIKEY:
        return False
    try:
        encoded = urllib.parse.quote(message[:160])
        url = (
            f"https://api.callmebot.com/whatsapp.php?phone={_CALLMEBOT_PHONE}"
            f"&text={encoded}&apikey={_CALLMEBOT_APIKEY}"
        )
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def check_heartbeat() -> dict:
    """Return whether watchdog has beaten recently. Fires SMS if not.

    Note: if watchdog file doesn't exist, the rig has never started yet
    (e.g. pre-deployment) — do NOT alert. Only alert on STALE heartbeat
    (file exists but is older than threshold).
    """
    if not WATCHDOG_FILE.exists():
        return {"alive": None, "age_hours": None,
                "reason": "watchdog file not yet created — rig not started"}
    mtime = WATCHDOG_FILE.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    alive = age_hours < HEARTBEAT_TIMEOUT_HOURS
    if not alive:
        send_sms(
            f"CRYPTO RIG HEARTBEAT FAILED: last beat {age_hours:.1f}h ago. "
            f"Check immediately. Run: python run.py to restart."
        )
    return {
        "alive": alive,
        "age_hours": age_hours,
        "reason": f"last beat {age_hours:.1f}h ago",
    }


def _read_telegram_offset() -> int:
    if TELEGRAM_OFFSET_FILE.exists():
        try:
            return int(TELEGRAM_OFFSET_FILE.read_text().strip() or "0")
        except Exception:
            return 0
    return 0


def _write_telegram_offset(offset: int) -> None:
    TELEGRAM_OFFSET_FILE.write_text(str(offset))


def poll_telegram_halt(timeout_seconds: int = 5) -> dict:
    """Check Telegram for /halt commands. Writes kill switch if found.

    Returns: {received_count: int, halted: bool, messages: list}
    """
    if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
        return {"received_count": 0, "halted": False, "messages": [],
                "note": "Telegram not configured"}

    offset = _read_telegram_offset()
    url = (
        f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/getUpdates"
        f"?offset={offset + 1}&timeout={timeout_seconds}&limit=10"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds + 5) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"received_count": 0, "halted": False, "messages": [],
                "error": str(e)}

    msgs = data.get("result", [])
    halted = False
    handled = []
    new_offset = offset
    for u in msgs:
        new_offset = max(new_offset, int(u.get("update_id", 0)))
        msg = u.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = msg.get("text", "")
        # Authentication: only respond to YOUR chat_id
        if chat_id != _TELEGRAM_CHAT_ID:
            continue
        handled.append(text)
        if text.lower().strip() in ("/halt", "/kill", "/stop"):
            # Write kill switch
            KILL_FILE.write_text(json.dumps({
                "killed_at": datetime.now(timezone.utc).isoformat(),
                "reason": "Telegram /halt command",
                "via": "remote",
            }, indent=2))
            halted = True
            # Confirm to user
            confirm_url = (
                f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage"
            )
            data = urllib.parse.urlencode({
                "chat_id": _TELEGRAM_CHAT_ID,
                "text": "🛑 KILL SWITCH ACTIVATED. Rig will skip all cycles until manual reset.",
            }).encode()
            try:
                urllib.request.urlopen(
                    urllib.request.Request(confirm_url, data=data), timeout=10
                )
            except Exception:
                pass

    _write_telegram_offset(new_offset)
    return {
        "received_count": len(handled),
        "halted": halted,
        "messages": handled,
    }


def main():
    """Run both checks. Designed for 5-minute scheduled execution."""
    print(f"=== Heartbeat check @ {datetime.now(timezone.utc).isoformat()} ===")
    hb = check_heartbeat()
    print(f"  Heartbeat: {hb}")
    tg = poll_telegram_halt()
    print(f"  Telegram poll: {tg}")
    if tg["halted"]:
        print(f"  KILL SWITCH WRITTEN by remote command")


if __name__ == "__main__":
    main()
