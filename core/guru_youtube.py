"""Guru YouTube feed — a free, no-auth, IP-unblocked replacement for the dying
Nitter/Twitter scrape.

These analysts publish their actual thesis on YouTube. YouTube's per-channel RSS
(`/feeds/videos.xml?channel_id=...`) needs no key and is NOT IP-blocked, so it
works from the US cloud precompute — unlike Nitter, which X has strangled (only
nitter.net survives and it blocks datacenter IPs).

Output is PURE PYTHON (dicts of str — no numpy/objects), so the pickled panel is
version-proof on the cloud. Add a guru by dropping their channel_id below.
"""
from __future__ import annotations

import html
import re
import urllib.request
from datetime import datetime, timezone

# name shown -> (youtube channel_id, x_handle for cross-link / track-record join)
GURU_YT_CHANNELS = {
    "Benjamin Cowen": ("UCRvqjQPSeaWn-uEx-w0XOIg", "benjamincowen"),
    "James Check":    ("UCGldOK1JzL_SMYwpie4A5fQ", "_Checkmatey_"),
}

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _fetch_channel(channel_id: str, limit: int = 4) -> list:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers=_UA), timeout=15
        ).read().decode("utf-8", "replace")
    except Exception:
        return []
    vids = []
    for ent in re.findall(r"<entry>(.*?)</entry>", raw, re.S)[:limit]:
        vid = re.search(r"<yt:videoId>(.*?)</yt:videoId>", ent)
        title = re.search(r"<title>(.*?)</title>", ent)
        pub = re.search(r"<published>(.*?)</published>", ent)
        if not (vid and title):
            continue
        vids.append({
            "title": html.unescape(title.group(1) or "").strip(),
            "date":  (pub.group(1)[:10] if pub else ""),
            "url":   f"https://www.youtube.com/watch?v={vid.group(1)}",
        })
    return vids


def guru_youtube_feed(per_guru: int = 3) -> dict:
    """Return {'gurus': [{name, x_handle, channel_id, videos:[{title,date,url}]}], 'asof': str}.

    Pure-Python; safe to pickle as a dashboard panel.
    """
    gurus = []
    for name, (cid, handle) in GURU_YT_CHANNELS.items():
        videos = _fetch_channel(cid, limit=per_guru)
        if videos:
            gurus.append({
                "name":       name,
                "x_handle":   handle,
                "channel_id": cid,
                "videos":     videos,
            })
    return {
        "gurus": gurus,
        "asof":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(guru_youtube_feed(), indent=2))
