"""Generic guru-Twitter monitor — supports multiple handles in one job.

Parameterizes the original jesse_olson_monitor.py to track N analysts
via nitter.net RSS. Each guru gets its own state + cache file so the
dashboard can show their tweets independently.

Configured handles (add more here):
  - JesseOlson         (already monitored separately, but included for unity)
  - PositiveCrypto     (Phillip Swift — LookIntoBitcoin founder)
  - benjamincowen      (Benjamin Cowen — IntoTheCryptoverse risk metric)

Behavior per guru (same as jesse_olson_monitor.py):
  - Fetch nitter.net/{HANDLE}/rss
  - Diff vs state — new tweets only
  - HIGH/MEDIUM/LOW scoring
  - Email HIGH immediately, batch MEDIUM into daily digest
  - Cache top 20 tweets to disk for dashboard

Runs every 2 hours by scheduled task.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

# Add more gurus here. Each gets independent state + cache.
GURUS = [
    {
        "handle":    "PositiveCrypto",
        "name":      "Phillip Swift",
        "url_label": "LookIntoBitcoin founder",
    },
    {
        "handle":    "benjamincowen",
        "name":      "Benjamin Cowen",
        "url_label": "IntoTheCryptoverse — risk metric / log regression",
        # X handle has historically been @intocryptoverse too; try both so a
        # rename or redirect doesn't leave the feed permanently empty.
        "alt_handles": ["intocryptoverse"],
    },
    # JesseOlson handled by jesse_olson_monitor.py separately (keep that running)
]

NITTER_INSTANCES = [
    "nitter.net",
    "nitter.privacydev.net",
    "nitter.poast.org",
    "nitter.tiekoetter.com",
    "nitter.kavin.rocks",
]

# ─── relevance scoring (same patterns as Olson monitor) ──────────────────
HIGH_PATTERNS = [
    r"\bqqq\b.*\$?\d{3,}",
    r"\bbtc\b.*\$?\d{2,3}k",
    r"\b(spy|nasdaq|sp500|nq)\b.*\$?\d{2,4}",
    r"\b(buy|sell|exit|scale|short|long)\b.*\$?\d",
    r"\b200[\s-]?(week|wk|day|d)?\s*(sma|ma)\b",
    r"\b(macd|rsi|mvrv|nupl|reserve risk|risk index)\b.*(cross|divergence|flip|peak|bottom)",
    r"\bgap (down|up|fill|below|above)\b",
    r"\b(top|bottom|peak|crash|capitulation|euphoria) (signal|call|in|zone|level)\b",
    r"\b(pi cycle|golden ratio|mayer|rainbow|hodl wave)\b",
]
MEDIUM_PATTERNS = [
    r"\bqqq\b", r"\bbtc\b", r"\bbitcoin\b",
    r"\bnasdaq\b", r"\bspy\b", r"\bsp500\b",
    r"\bmarket\b", r"\bmacro\b", r"\bfed\b",
    r"\bliquidity\b", r"\brecession\b",
    r"\bcorrection\b", r"\bbreakout\b", r"\bsupport\b",
    r"\bresistance\b", r"\bmvrv\b", r"\bnupl\b",
    r"\bcycle\b", r"\bhalving\b", r"\bbull\b", r"\bbear\b",
]


def _classify(text: str) -> str:
    t = text.lower()
    for p in HIGH_PATTERNS:
        if re.search(p, t): return "HIGH"
    for p in MEDIUM_PATTERNS:
        if re.search(p, t): return "MEDIUM"
    return "LOW"


def _fetch_rss(handle: str, alt_handles=None) -> str | None:
    """Try each (handle, instance). alt_handles covers renamed/redirected accounts.
    Reject the xcancel-style 'RSS reader not yet whitelisted!' placeholder, which
    contains <item> but no real tweets."""
    for h in [handle] + list(alt_handles or []):
        for host in NITTER_INSTANCES:
            url = f"https://{host}/{h}/rss"
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    content = r.read().decode("utf-8", errors="ignore")
                    if "<item>" in content and "not yet whitelisted" not in content.lower():
                        return content
            except Exception:
                continue
    return None


def _parse_rss(rss_text: str, handle: str) -> list[dict]:
    try:
        root = ET.fromstring(rss_text)
    except ET.ParseError:
        return []

    tweets = []
    channel = root.find("channel")
    if channel is None: return []
    for item in channel.findall("item"):
        try:
            link = (item.find("link").text or "").strip()
            m = re.search(r"/status/(\d+)", link)
            tweet_id = m.group(1) if m else link
            title = (item.find("title").text or "").strip()
            desc = (item.find("description").text or "").strip()
            desc_text = re.sub(r"<[^>]+>", " ", desc)
            desc_text = re.sub(r"\s+", " ", desc_text).strip()
            pub = (item.find("pubDate").text or "").strip()
            x_link = re.sub(r"https?://nitter\.[^/]+",
                              "https://x.com", link)
            tweets.append({
                "id":         tweet_id,
                "handle":     handle,
                "title":      title[:280],
                "text":       desc_text[:500],
                "link":       x_link,
                "pub":        pub,
                "relevance":  _classify(f"{title} {desc_text}"),
            })
        except Exception:
            continue
    return tweets


def _state_files(handle: str) -> tuple[Path, Path]:
    """Return (state_file, cache_file) for a given guru handle."""
    return (
        REPO / f".guru_{handle.lower()}_monitor_state.json",
        REPO / f".guru_{handle.lower()}_tweets_cache.json",
    )


def _load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"seen_ids": [], "pending_medium": []}
    try: return json.loads(state_file.read_text())
    except Exception: return {"seen_ids": [], "pending_medium": []}


def _save_state(state_file: Path, s: dict) -> None:
    try: state_file.write_text(json.dumps(s, indent=2, default=str))
    except Exception: pass


def _save_cache(cache_file: Path, tweets: list, handle: str, name: str) -> None:
    try:
        cache_file.write_text(json.dumps({
            "handle":    handle,
            "name":      name,
            "tweets":    tweets[:20],
            "updated":   datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))
    except Exception: pass


def _email_high(tweet: dict, guru_name: str) -> bool:
    try:
        from ops.alerts import alert
    except Exception:
        return False
    body = (
        f"{guru_name} - a Bitcoin analyst the dashboard tracks - just posted something\n"
        f"worth a look (they named a specific price level or made a clear call).\n\n"
        f"WHAT THEY SAID:\n"
        f"  \"{tweet.get('text', tweet.get('title', ''))[:400]}\"\n\n"
        f"  Posted: {tweet.get('pub')}\n"
        f"  Link:   {tweet.get('link')}\n\n"
        f"WHY YOU'RE GETTING THIS:\n"
        f"  It's flagged important because it mentions Bitcoin AND a specific price or\n"
        f"  action - the kind of post most likely to matter for the plan.\n\n"
        f"  It's information, not an instruction. The dashboard's own signals still\n"
        f"  decide what (if anything) you do.")
    subject = f"{guru_name} said something worth a look (BTC)"
    try:
        alert(body, level="warning", subject=subject, email=True)
        return True
    except Exception:
        return False


def _email_digest(tweets: list, guru_name: str, handle: str) -> bool:
    if not tweets: return True
    try:
        from ops.alerts import alert
    except Exception:
        return False
    lines = [
        f"{guru_name} (a Bitcoin analyst the dashboard tracks) - recent posts.",
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
              subject=f"{guru_name} - {len(tweets)} recent posts (FYI, no action)")
        return True
    except Exception:
        return False


def _process_guru(guru: dict) -> dict:
    handle = guru["handle"]
    name = guru["name"]
    state_file, cache_file = _state_files(handle)

    rss = _fetch_rss(handle, guru.get("alt_handles"))
    if not rss:
        return {"handle": handle, "status": "no_nitter_available"}

    tweets = _parse_rss(rss, handle)
    if not tweets:
        return {"handle": handle, "status": "no_tweets_parsed"}

    _save_cache(cache_file, tweets, handle, name)

    state = _load_state(state_file)
    seen = set(state.get("seen_ids", []))
    pending = state.get("pending_medium", [])

    new = [t for t in tweets if t["id"] not in seen]
    if not new:
        return {"handle": handle, "status": "no_new_tweets",
                "n_cached": len(tweets)}

    high = [t for t in new if t["relevance"] == "HIGH"]
    medium = [t for t in new if t["relevance"] == "MEDIUM"]

    sent_high = 0
    for t in high:
        if _email_high(t, name):
            sent_high += 1

    pending.extend(medium)
    sent_digest = 0
    if len(pending) >= 2:
        if _email_digest(pending, name, handle):
            sent_digest = len(pending)
            pending = []

    seen.update(t["id"] for t in new)
    _save_state(state_file, {
        "seen_ids":        list(seen)[-200:],
        "pending_medium":  pending[-20:],
        "last_check":      datetime.now(timezone.utc).isoformat(),
    })

    return {
        "handle":     handle,
        "status":     "OK",
        "n_new":      len(new),
        "n_high":     len(high),
        "n_medium":   len(medium),
        "emailed_high": sent_high,
        "digest_sent":  sent_digest,
    }


def main():
    results = [_process_guru(g) for g in GURUS]
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
