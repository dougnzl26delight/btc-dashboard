"""Real-time notifications — Telegram primary, email fallback, stdout always.

Reuses the same env var names as the stocks rig's alerts.py so you can copy
SMTP/Telegram creds directly:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    EMAIL_FROM, EMAIL_TO, EMAIL_PASS
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
_EMAIL_FROM = os.getenv("EMAIL_FROM", "").strip()
_EMAIL_TO = os.getenv("EMAIL_TO", "").strip()
_EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip()

_LEVEL_PREFIX = {
    "info": "[INFO]",
    "trade": "[TRADE]",
    "warning": "[WARN]",
    "critical": "[CRITICAL]",
    "success": "[OK]",
}


def _send_telegram(text: str) -> bool:
    if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": _TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        ).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _send_email(subject: str, body: str) -> bool:
    if not _EMAIL_FROM or not _EMAIL_TO or not _EMAIL_PASS:
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = _EMAIL_FROM
        msg["To"] = _EMAIL_TO
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls()
            s.login(_EMAIL_FROM, _EMAIL_PASS)
            s.send_message(msg)
        return True
    except Exception:
        return False


def alert(message: str, level: str = "info", subject: str | None = None,
          email: bool = False) -> None:
    """Send via best available channel. Always logs to stdout.

    email=True forces an email regardless of level — used by the BTC-dashboard
    alerts (digest, bottom countdown, guru, rotation warming/execute) so they
    reach the inbox even though Telegram isn't configured. Other warning/info
    alerts stay quiet (Telegram + stdout only) to avoid the email spam that was
    silenced 2026-06-03.
    """
    prefix = _LEVEL_PREFIX.get(level, "[INFO]")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    formatted = f"{prefix} <b>{ts}</b>\n{message}"
    plain = f"{prefix} {ts} | {message}"

    try:
        print(plain)
    except UnicodeEncodeError:
        # Windows console (cp1252) can't encode some Unicode chars. Fall back
        # to ASCII-safe encoding so the rig doesn't crash on alert printing.
        print(plain.encode("ascii", errors="replace").decode("ascii"))

    # Telegram is attempted for everything (no-op if unconfigured). Email fires
    # for "critical" OR an explicit opt-in (email=True) — the latter is how the
    # BTC-dashboard alerts reach the inbox without un-silencing the whole rig.
    _send_telegram(formatted)
    if level == "critical" or email:
        _send_email(subject or f"Crypto {level.upper()}: {message[:60]}", plain)


def send_email_report(subject: str, body: str) -> bool:
    """Send a long-form report by email. Returns True if sent, False otherwise.

    Use for daily reports, weekly reviews — anything too long for Telegram.
    Configured via EMAIL_FROM / EMAIL_TO / EMAIL_PASS env vars (Gmail SMTP).

    OPT-IN: silenced 2026-06-03 per user request. Reports now only send when
    SEND_REPORT_EMAILS=1 is set in .env. Set to 1 to restore daily/weekly emails.
    """
    if os.getenv("SEND_REPORT_EMAILS", "").strip() != "1":
        return False
    return _send_email(subject, body)


def alert_status() -> dict:
    return {
        "telegram_configured": bool(_TELEGRAM_TOKEN and _TELEGRAM_CHAT_ID),
        "email_configured": bool(_EMAIL_FROM and _EMAIL_TO and _EMAIL_PASS),
        "channels_active": int(bool(_TELEGRAM_TOKEN)) + int(bool(_EMAIL_FROM)),
    }


if __name__ == "__main__":
    print("Channel status:", alert_status())
    alert("Test alert from crypto alerts.py — INFO", level="info")
    alert("Test alert from crypto alerts.py — CRITICAL", level="critical")
