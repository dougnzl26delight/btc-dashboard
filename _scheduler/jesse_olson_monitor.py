"""Monitor Jesse Olson (@JesseOlson) tweets via Nitter RSS.

Twitter/X is hostile to scraping in 2026 (API $100+/mo, Nitter mostly dead),
but nitter.net STILL provides a working public RSS feed. We poll every 2h.

Behavior:
  - Fetch nitter.net/JesseOlson/rss
  - Diff against state file -- new tweets only
  - Score each new tweet for relevance:
      * HIGH: mentions QQQ/BTC + specific price level OR call to action
      * MEDIUM: BTC/crypto/equity macro analysis, no specific call
      * LOW: replies, off-topic
  - Email HIGH-relevance tweets immediately
  - Email digest of MEDIUM tweets daily (only when >= 2 accumulated)
  - LOW skipped silently

State file persists last-seen tweet IDs to prevent re-sends.

Cache the latest tweets to disk so the dashboard can show them in the
EQUITY TOP WATCH section.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
STATE_FILE = REPO / ".jesse_olson_monitor_state.json"
CACHE_FILE = REPO / ".jesse_olson_tweets_cache.json"

NITTER_INSTANCES = [
    "nitter.net",            # confirmed working 2026-06
    "nitter.privacydev.net",
    "nitter.poast.org",
]

HANDLE = "JesseOlson"


# ─── relevance scoring ────────────────────────────────────────────────────
# HIGH: specific price call + asset
HIGH_PATTERNS = [
    r"\bqqq\b.*\$?\d{3,}",                       # "QQQ 589"
    r"\bbtc\b.*\$?\d{2,3}k",                      # "BTC 100k"
    r"\b(spy|nasdaq|sp500|nq)\b.*\$?\d{2,4}",     # "SPY 600"
    r"\b(buy|sell|exit|scale|short|long)\b.*\$?\d",
    r"\b200[\s-]?(week|wk|day|d)?\s*(sma|ma)\b", # "200 week MA"
    r"\b(macd|rsi)\b.*(cross|divergence|flip)",   # MACD cross
    r"\bgap (down|up|fill|below|above)\b",
    r"\b(top|bottom|peak|crash) (signal|call|in)\b",
    r"\b(capitulation|euphoria|extreme)\b",
]
# MEDIUM: relevant topics but no specific call
MEDIUM_PATTERNS = [
    r"\bqqq\b", r"\bbtc\b", r"\bbitcoin\b",
    r"\bnasdaq\b", r"\bspy\b", r"\bsp500\b",
    r"\bmarket\b", r"\bmacro\b", r"\bfed\b",
    r"\bliquidity\b", r"\brecession\b",
    r"\bcorrection\b", r"\bbreakout\b", r"\bsupport\b",
    r"\bresistance\b",
]


def _classify(text: str) -> str:
    t = text.lower()
    for p in HIGH_PATTERNS:
        if re.search(p, t):
            return "HIGH"
    for p in MEDIUM_PATTERNS:
        if re.search(p, t):
            return "MEDIUM"
    return "LOW"


# ─── fetch ────────────────────────────────────────────────────────────────
def _fetch_rss() -> str | None:
    """Try each Nitter instance until one works."""
    for host in NITTER_INSTANCES:
        url = f"https://{host}/{HANDLE}/rss"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
            with urllib.request.urlopen(req, timeout=15) as r:
                content = r.read().decode("utf-8", errors="ignore")
                if "<item>" in content:
                    return content
        except Exception:
            continue
    return None


def _parse_rss(rss_text: str) -> list[dict]:
    """Parse RSS into list of tweet dicts."""
    try:
        root = ET.fromstring(rss_text)
    except ET.ParseError:
        return []

    tweets = []
    # RSS items under channel
    channel = root.find("channel")
    if channel is None: return []
    for item in channel.findall("item"):
        try:
            link = (item.find("link").text or "").strip()
            # Tweet ID is the last part of the URL (after status/)
            m = re.search(r"/status/(\d+)", link)
            tweet_id = m.group(1) if m else link
            title = (item.find("title").text or "").strip()
            desc = (item.find("description").text or "").strip()
            # Strip HTML from description
            desc_text = re.sub(r"<[^>]+>", " ", desc)
            desc_text = re.sub(r"\s+", " ", desc_text).strip()
            pub = (item.find("pubDate").text or "").strip()
            tweets.append({
                "id":      tweet_id,
                "title":   title[:280],
                "text":    desc_text[:500],
                "link":    link.replace("nitter.net",  "x.com")
                             .replace("nitter.privacydev.net", "x.com")
                             .replace("nitter.poast.org", "x.com"),
                "pub":     pub,
                "relevance": _classify(f"{title} {desc_text}"),
            })
        except Exception:
            continue
    return tweets


# ─── state ────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"seen_ids": [], "pending_medium": []}
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return {"seen_ids": [], "pending_medium": []}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2, default=str))
    except Exception: pass


def _save_cache(tweets: list) -> None:
    """Cache latest 20 tweets for the dashboard."""
    try:
        CACHE_FILE.write_text(json.dumps({
            "tweets":   tweets[:20],
            "updated":  datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))
    except Exception: pass


# ─── email ────────────────────────────────────────────────────────────────
def _email_high(tweet: dict) -> bool:
    """Send immediate email for a single HIGH-relevance tweet."""
    try:
        from ops.alerts import alert
    except Exception:
        return False
    body = (
        "Jesse Olson - a Bitcoin & US-stocks analyst the dashboard tracks - just posted\n"
        "something worth a look (he named a specific price level or made a clear call).\n\n"
        "WHAT HE SAID:\n"
        f"  \"{tweet.get('text', tweet.get('title', ''))[:400]}\"\n\n"
        f"  Posted: {tweet.get('pub')}\n"
        f"  Link:   {tweet.get('link')}\n\n"
        "WHY YOU'RE GETTING THIS:\n"
        "  It's flagged important because it mentions Bitcoin/stocks AND a specific price\n"
        "  or action - the kind of post most likely to matter for the rotation plan.\n\n"
        "  It's information, not an instruction. The dashboard's own signals still decide\n"
        "  what (if anything) you do - one analyst's view never moves the plan by itself.")
    subject = "Jesse Olson said something worth a look (BTC/stocks)"
    try:
        alert(body, level="warning", subject=subject, email=True)
        return True
    except Exception:
        return False


def _email_digest(tweets: list[dict]) -> bool:
    """Send digest of accumulated MEDIUM tweets."""
    if not tweets: return True
    try:
        from ops.alerts import alert
    except Exception:
        return False
    lines = [
        "Jesse Olson (a Bitcoin & US-stocks analyst the dashboard tracks) - recent posts.",
        "These are lower-priority FYI posts. Nothing here needs any action from you.",
        "",
    ]
    for t in tweets[:12]:
        rel = str(t.get("relevance", "?")).upper()
        _tag = {"HIGH": "important", "MEDIUM": "notable", "LOW": "minor"}.get(rel, rel.lower())
        lines.append(f"  ({_tag}) {t.get('text', t.get('title', ''))[:200]}")
        lines.append(f"          {t.get('link')}")
        lines.append("")
    try:
        alert("\n".join(lines), level="info", email=True,
              subject=f"Jesse Olson - {len(tweets)} recent posts (FYI, no action)")
        return True
    except Exception:
        return False


# ─── on-demand refresh (dashboard button) ──────────────────────────────────
def refresh_cache_only() -> dict:
    """Force a live nitter fetch + cache update, with NO email/state side effects.
    For the dashboard 'Refresh feed' button — a user click must never blast emails.
    Returns {ok, n, updated, error}."""
    rss = _fetch_rss()
    if not rss:
        return {"ok": False, "n": 0, "error": "nitter unreachable (try again shortly)"}
    tweets = _parse_rss(rss)
    if not tweets:
        return {"ok": False, "n": 0, "error": "no tweets parsed"}
    _save_cache(tweets)
    return {"ok": True, "n": len(tweets),
            "updated": datetime.now(timezone.utc).isoformat(), "error": ""}


# ─── main ─────────────────────────────────────────────────────────────────
def main():
    rss = _fetch_rss()
    if not rss:
        print(json.dumps({"status": "no_nitter_available"}))
        return

    tweets = _parse_rss(rss)
    if not tweets:
        print(json.dumps({"status": "no_tweets_parsed"}))
        return

    _save_cache(tweets)  # always update cache for dashboard

    state = _load_state()
    seen = set(state.get("seen_ids", []))
    pending = state.get("pending_medium", [])

    # New tweets only
    new = [t for t in tweets if t["id"] not in seen]

    if not new:
        print(json.dumps({"status": "no_new_tweets",
                            "n_tweets_cached": len(tweets)}))
        return

    # Sort by relevance: HIGH first
    high = [t for t in new if t["relevance"] == "HIGH"]
    medium = [t for t in new if t["relevance"] == "MEDIUM"]
    low = [t for t in new if t["relevance"] == "LOW"]

    # Email HIGH immediately
    sent_high = 0
    for t in high:
        if _email_high(t):
            sent_high += 1

    # Accumulate MEDIUM; send digest if >= 2 pending
    pending.extend(medium)
    sent_digest = 0
    if len(pending) >= 2:
        if _email_digest(pending):
            sent_digest = len(pending)
            pending = []

    # Mark all new as seen
    seen.update(t["id"] for t in new)
    # Keep only last 200 IDs (LRU-ish)
    seen_list = list(seen)[-200:]

    _save_state({
        "seen_ids":        seen_list,
        "pending_medium":  pending[-20:],   # cap pending
        "last_check":      datetime.now(timezone.utc).isoformat(),
    })

    print(json.dumps({
        "status":           "OK",
        "n_new":            len(new),
        "n_high":           len(high),
        "n_medium":         len(medium),
        "n_low":            len(low),
        "high_emailed":     sent_high,
        "digest_sent":      sent_digest,
        "pending_medium":   len(pending),
    }))


if __name__ == "__main__":
    main()
