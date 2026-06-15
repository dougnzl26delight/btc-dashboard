"""Weekly red-team email runner — the in-house devil's advocate.

Emails the strongest bear case against the rotation campaign once a week so
conviction is stress-tested, not reinforced. Scheduled task: Crypto_red_team_weekly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from core.red_team import red_team_email_body
    from ops.alerts import alert
    subject, body = red_team_email_body()
    alert(body, level="info", subject=subject, email=True)
    try:
        print("[red_team] emailed:", subject.encode("ascii", "replace").decode())
    except Exception:
        print("[red_team] emailed.")


if __name__ == "__main__":
    main()
