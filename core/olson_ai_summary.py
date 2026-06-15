"""AI-written summary of Jesse Olson's latest X feed (Anthropic Haiku).

Turns his recent tweets into a short prose read of his CURRENT stance + ideas —
the thing keyword parsing can't do (it reads negation/sarcasm/context, e.g.
"bottom is not in" = bearish).

COST + SAFETY (this is a PUBLIC dashboard):
  - the LLM is called ONLY from the background (precompute / CLI), NEVER on a page
    render or the public Refresh button — visitors cannot spend your tokens.
  - cached by a HASH of the feed content: it runs at most once per NEW feed
    (~once / 2h when the monitor pulls fresh tweets). Same feed -> cached, 0 calls.
  - hard guards: >= 5 min between calls + a daily call cap.
  - only PUBLIC tweet text is sent (no personal / portfolio data ever).

KEY: reads ANTHROPIC_API_KEY from the environment or the gitignored .env file.
Add `ANTHROPIC_API_KEY=sk-ant-...` to .env to enable; absent -> degrades cleanly
to the levels digest with a one-line note. NOT advice.
"""
from __future__ import annotations
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FEED = REPO / ".jesse_olson_tweets_cache.json"
CACHE = REPO / ".olson_ai_summary.json"
MODEL = "claude-haiku-4-5-20251001"
ENDPOINT = "https://api.anthropic.com/v1/messages"   # explicit — never the CC proxy
MIN_INTERVAL_SEC = 300
DAILY_CAP = 50


def _api_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    envf = REPO / ".env"
    if envf.exists():
        try:
            for line in envf.read_text().splitlines():
                s = line.strip()
                if s.startswith("ANTHROPIC_API_KEY="):
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(d: dict):
    try:
        CACHE.write_text(json.dumps(d, indent=2, default=str))
    except Exception:
        pass


def _feed_tweets():
    if not FEED.exists():
        return [], ""
    try:
        data = json.loads(FEED.read_text())
    except Exception:
        return [], ""
    tw = data.get("tweets", []) or []
    texts = [(t.get("text") or t.get("title") or "").strip() for t in tw]
    texts = [x for x in texts if x][:15]
    return texts, (data.get("updated", "") or "")[:16]


def _call_anthropic(key: str, tweets: list) -> str:
    prompt = (
        "Here are recent public X/Twitter posts from Jesse Olson, a Bitcoin & "
        "US-equities technical analyst. Summarise his CURRENT views in 3-5 short "
        "bullet points: his directional stance on BTC and on US equities/QQQ, the key "
        "price levels he is watching, and any clear call or warning. Read negation and "
        "sarcasm correctly (e.g. \"bottom is not in\" = bearish; mocking bottom-callers "
        "= bearish). Be concise and neutral, plain English, no preamble, no advice. "
        "If the posts are mostly off-topic, say that briefly.\n\nPOSTS:\n"
        + "\n".join(f"- {t}" for t in tweets)
    )
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 450,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read().decode("utf-8"))
    parts = out.get("content", []) or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()


def olson_ai_summary(force: bool = False) -> dict:
    """Generate-or-return the cached AI summary. Call from BACKGROUND ONLY
    (precompute / CLI) — it may spend a token. The dashboard must use
    olson_ai_summary_cached() instead."""
    tweets, updated = _feed_tweets()
    if not tweets:
        return {"ok": False, "reason": "no_feed", "summary": "", "updated": updated}
    digest = hashlib.sha1("||".join(tweets).encode("utf-8")).hexdigest()[:16]
    cache = _load_cache()

    if not force and cache.get("hash") == digest and cache.get("summary"):
        return {"ok": True, "reason": "cached", "summary": cache["summary"],
                "updated": updated, "model": cache.get("model", MODEL),
                "generated": cache.get("generated", "")}

    key = _api_key()
    if not key:
        return {"ok": False, "reason": "no_key", "summary": cache.get("summary", ""),
                "updated": updated}

    now = datetime.now(timezone.utc)
    last = cache.get("last_call_ts")
    if last and not force:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < MIN_INTERVAL_SEC:
                return {"ok": bool(cache.get("summary")), "reason": "rate_limited",
                        "summary": cache.get("summary", ""), "updated": updated}
        except Exception:
            pass
    today = now.date().isoformat()
    n_today = cache.get("calls_today", 0) if cache.get("call_day") == today else 0
    if n_today >= DAILY_CAP:
        return {"ok": bool(cache.get("summary")), "reason": "daily_cap",
                "summary": cache.get("summary", ""), "updated": updated}

    try:
        summary = _call_anthropic(key, tweets)
    except Exception as e:
        return {"ok": bool(cache.get("summary")),
                "reason": f"api_error:{type(e).__name__}",
                "summary": cache.get("summary", ""), "updated": updated}

    cache.update({"hash": digest, "summary": summary, "model": MODEL,
                  "generated": now.isoformat(), "last_call_ts": now.isoformat(),
                  "call_day": today, "calls_today": n_today + 1})
    _save_cache(cache)
    return {"ok": True, "reason": "generated", "summary": summary, "updated": updated,
            "model": MODEL, "generated": now.isoformat()}


def olson_ai_summary_cached() -> dict:
    """READ-ONLY: the last cached AI summary; NEVER calls the LLM. For the render path."""
    _, updated = _feed_tweets()
    cache = _load_cache()
    return {"ok": bool(cache.get("summary")), "summary": cache.get("summary", ""),
            "updated": updated, "model": cache.get("model", ""),
            "generated": cache.get("generated", ""),
            "enabled": _api_key() is not None}


if __name__ == "__main__":
    import sys
    r = olson_ai_summary(force="--force" in sys.argv)
    print(f"ok={r['ok']} reason={r['reason']} model={r.get('model','')}")
    print(r.get("summary", "")[:800] or "(no summary)")
