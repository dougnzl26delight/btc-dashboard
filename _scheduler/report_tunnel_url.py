"""Report current Cloudflare Tunnel URL.

Runs daily — if the URL has changed since yesterday, sends an email
with the new URL. This handles the "quick tunnel URL changes on restart"
limitation of Cloudflare's free tier.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "cloudflared.log"
STATE = REPO / ".tunnel_url_state.json"


def find_current_url() -> str | None:
    """Extract latest trycloudflare URL from cloudflared log."""
    if not LOG.exists(): return None
    try:
        text = LOG.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # Get the most recent URL — the log accumulates so the latest is at the end
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
    return urls[-1] if urls else None


def main():
    current_url = find_current_url()
    if not current_url:
        print(json.dumps({"status": "no_url_found"}))
        return

    if STATE.exists():
        try:
            saved = json.loads(STATE.read_text())
            last_url = saved.get("url")
        except Exception:
            last_url = None
    else:
        last_url = None

    STATE.write_text(json.dumps({
        "url": current_url,
        "updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    if current_url == last_url:
        print(json.dumps({"status": "unchanged", "url": current_url}))
        return

    # URL changed — send email
    subject = "!! BTC Dashboard tunnel URL changed !!"
    body = f"""Your Cloudflare Tunnel URL has changed.

NEW URL:
  {current_url}

PREVIOUS URL:
  {last_url or 'first observation'}

This typically happens after a reboot or when cloudflared restarts.
The free Cloudflare quick tunnels assign a random URL each time.

To get a STABLE URL: set up a named tunnel with a Cloudflare account
(free, but requires a domain or workers.dev subdomain — see
https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/).

Dashboard is accessible at the URL above from anywhere with internet.
"""
    try:
        from ops.alerts import alert
        alert(body, level="info", subject=subject)
        print(json.dumps({"status": "url_changed_email_sent", "url": current_url}))
    except Exception as e:
        print(json.dumps({"status": "url_changed_email_failed",
                            "url": current_url, "error": str(e)}))


if __name__ == "__main__":
    main()
