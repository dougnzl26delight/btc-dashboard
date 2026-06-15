"""Summarise Jesse Olson's latest X feed into a readable digest.

Built from the cached nitter feed (`.jesse_olson_tweets_cache.json`) — no LLM, no
live fetch (the dashboard's Refresh button updates the cache; this just reads it).
Produces: a one-line current read, per-asset stance (bull/bear/mixed from the
direction of his recent tweets), the price levels he's naming, and his top recent
tweets verbatim so you read his ACTUAL ideas, not a paraphrase. NOT advice.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".jesse_olson_tweets_cache.json"

_BEAR = ("bearish", "bear ", "below", "breakdown", "lower high", "gap below", "sell",
         "short", "topping", "top is", "decline", "downside", "drop", "rejected",
         "resistance", "lookout below")
_BULL = ("bullish", "bull ", "above", "breakout", "higher", "gap above", "bottom is in",
         "reversal", "support holds", "buy", "accumulate", "rally", "upside", "reclaim",
         "bottom in")


def _direction(t: str) -> int:
    t = (t or "").lower()
    bear = sum(1 for k in _BEAR if k in t)
    bull = sum(1 for k in _BULL if k in t)
    return -1 if bear > bull else 1 if bull > bear else 0


def _assets(t: str):
    out = set()
    for m in re.findall(r"\$([A-Za-z]{2,5})\b", t or ""):
        out.add(m.upper())
    tl = (t or "").lower()
    for sym, name in (("BTC", "bitcoin"), ("QQQ", "qqq"), ("SPY", "spy"),
                      ("NQ", "nasdaq"), ("ETH", "ethereum")):
        if name in tl:
            out.add(sym)
    return out


def _levels(t: str):
    # conservative: $-prefixed amounts only (avoids catching years / counts);
    # decimal must be followed by digits so we don't capture a trailing period.
    lv = re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?\s?[kK]?", t or "")
    seen, out = set(), []
    for x in lv:
        x = x.strip()
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:4]


def olson_feed_summary(max_tweets: int = 6) -> dict:
    out = {"ok": False, "updated": "", "n": 0, "by_asset": {}, "tweets": [],
           "read": ""}
    if not CACHE.exists():
        out["read"] = "No cached feed yet — hit Refresh to pull @JesseOlson."
        return out
    try:
        data = json.loads(CACHE.read_text())
    except Exception:
        out["read"] = "Feed cache unreadable."
        return out

    tweets = data.get("tweets", []) or []
    out["updated"] = (data.get("updated", "") or "")[:16]
    out["n"] = len(tweets)

    # Per-CORE-asset price LEVELS he's naming. We deliberately DON'T auto-assert a
    # bull/bear stance: keyword sentiment can't read negation ("no bull run",
    # "bottom is NOT in" both look bullish to a keyword counter) and a wrong
    # directional call is worse than none. The levels are reliable; his actual
    # direction is carried by his verbatim tweets below.
    CORE = ("BTC", "QQQ", "SPY", "NQ", "ETH")
    asset_levels = {}
    for t in tweets:
        txt = t.get("text") or t.get("title") or ""
        present = _assets(txt) & set(CORE)
        lv = _levels(txt)
        for a in present:
            bucket = asset_levels.setdefault(a, [])
            for l in lv:
                if l not in bucket:
                    bucket.append(l)
    for a, lv in asset_levels.items():
        out["by_asset"][a] = {"levels": lv[:5]}

    # top tweets (HIGH, then MEDIUM, then LOW), verbatim
    ranked = ([t for t in tweets if t.get("relevance") == "HIGH"] +
              [t for t in tweets if t.get("relevance") == "MEDIUM"] +
              [t for t in tweets if t.get("relevance") == "LOW"])
    for t in ranked[:max_tweets]:
        txt = (t.get("text") or t.get("title") or "")[:300]
        out["tweets"].append({
            "text": txt, "link": t.get("link", ""), "pub": (t.get("pub", "") or "")[:16],
            "relevance": t.get("relevance", "?"),
            "assets": sorted(_assets(txt)), "levels": _levels(txt),
        })

    bits = []
    for a in CORE:
        lv = out["by_asset"].get(a, {}).get("levels", [])
        if lv:
            bits.append(f"{a} {', '.join(lv[:2])}")
    out["read"] = ("Levels he's naming — " + "; ".join(bits)) if bits else "Read his recent posts below for his current view."
    out["ok"] = bool(tweets)
    return out


if __name__ == "__main__":
    s = olson_feed_summary()
    print(f"updated {s['updated']} · {s['n']} tweets · {s['read']}")
    for t in s["tweets"]:
        print(f"  [{t['relevance']}] {t['text'][:90]} "
              f"{('lvls:'+','.join(t['levels'])) if t['levels'] else ''}")
