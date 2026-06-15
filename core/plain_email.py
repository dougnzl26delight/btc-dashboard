"""Plain-English email helpers — lead every BTC-dashboard alert with a human summary.

Goal: the reader sees the top few lines and instantly knows (a) what's going on
and (b) whether they need to do anything. The raw numbers/jargon stay BELOW for
reference, never leading. ASCII-only (email transport + Windows console safe).
"""
from __future__ import annotations


# ── The lead block ───────────────────────────────────────────────────────────
_MOOD_TAG = {
    "calm":  "ALL CALM - nothing to do",
    "watch": "WORTH A LOOK",
    "act":   "ACTION NEEDED",
    "good":  "GOOD NEWS",
}


def plain_lead(headline: str, bullets=None, todo: str = "", mood: str = "watch") -> str:
    """Build a plain-English TL;DR block to lead an email body.

    headline : one-line plain summary, no jargon.
    bullets  : optional list of plain-language status lines.
    todo     : 'what to do' line; defaults to a stay-aware nudge.
    mood     : calm | watch | act | good -> a leading tag so the gist lands instantly.
    """
    tag = _MOOD_TAG.get(mood, "UPDATE")
    out = [
        "==================================================",
        f"  IN PLAIN ENGLISH  -  {tag}",
        "==================================================",
        "",
        f"  {headline}",
    ]
    if bullets:
        out.append("")
        out.extend(f"   - {b}" for b in bullets if b)
    out += [
        "",
        f"  WHAT TO DO:  {todo or 'Nothing - just stay aware.'}",
        "",
        "  - - - - - - the detail (numbers) is below - - - - - -",
        "",
    ]
    return "\n".join(out)


# ── Jargon -> plain translators ──────────────────────────────────────────────
def plain_zone(zone: str) -> str:
    """Swift Risk Index zone -> plain words."""
    return {
        "MAX BUY":  "screaming cheap (a rare 'back up the truck' zone)",
        "BUY":      "cheap",
        "NEUTRAL":  "around fair value",
        "SELL":     "getting expensive",
        "MAX SELL": "very expensive (danger zone)",
    }.get((zone or "").upper(), zone or "unknown")


def plain_bottom_level(lvl: str) -> str:
    """Bottom-scorecard verdict -> plain words."""
    return {
        "HOLD":       "not a bottom yet",
        "WATCH":      "a few early bottom signs, but not there yet",
        "ACCUMULATE": "starting to look genuinely cheap",
        "STRONG_BUY": "looking like a real bottom",
        "DEEP_VALUE": "deep value - a rare buying zone",
        "EXTREME":    "once-a-cycle bargain",
    }.get((lvl or "").upper(), lvl or "unknown")


def plain_top_level(lvl: str) -> str:
    """Top-scorecard verdict -> plain words."""
    return {
        "HOLD":         "no sell signal",
        "WATCH":        "a few early 'getting toppy' warnings",
        "TRIM_25":      "time to trim a bit",
        "SCALE_OUT_50": "time to sell about half",
        "EXIT_75":      "time to get mostly out",
    }.get((lvl or "").upper(), lvl or "unknown")


def plain_qqq(tier: str) -> str:
    """Olson QQQ (US tech stocks) tier -> plain words."""
    return {
        "SAFE":       "calm, near highs",
        "CAUTION":    "showing the first cracks",
        "WATCH":      "near a key trapdoor level",
        "GAP_BROKEN": "breaking down",
        "RETESTING":  "in serious trouble",
        "DATA_GAP":   "no data right now",
    }.get((tier or "").upper(), tier or "unknown")


def plain_plan(status: str) -> str:
    """Rotation-trigger status -> plain words."""
    return {
        "ARMED":   "waiting - nothing to do",
        "WARMING": "early warning - the first signal has tripped",
        "FIRED":   "FIRED - time to move shares into Bitcoin",
    }.get((status or "").upper(), status or "unknown")
