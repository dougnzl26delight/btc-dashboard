"""Guru intelligence layer — track records, recent calls, narrative summarizer.

Closes Anya's UX gap "show me how reliable each guru actually is":

  1. Guru track record       — historical hit/miss on documented calls
  2. Recent call summary     — pulls last 24h of HIGH-relevance tweets
  3. WoC narrative pointer   — link to current Glassnode editorial
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Documented public calls — hand-curated. Update as new calls land.
GURU_TRACK_RECORD = {
    "JesseOlson": {
        "name":          "Jesse Olson",
        "url":            "https://x.com/JesseOlson",
        "specialty":      "Crypto technical analyst (multi-week timeframes)",
        "calls": [
            {"date": "2025-10-01", "asset": "BTC",
              "call": "Cycle 5 top imminent via 3wk MACD divergence",
              "outcome": "RIGHT",
              "evidence": "BTC peaked Oct 6 2025 $124,659"},
            {"date": "2025-10-08", "asset": "BTC",
              "call": "Binance cascade incoming",
              "outcome": "RIGHT",
              "evidence": "Cascade occurred Oct 10 2025"},
            {"date": "2024-09-15", "asset": "BTC",
              "call": "Bottom at $40k, bull resumption",
              "outcome": "WRONG",
              "evidence": "BTC went to $124k peak instead, not bottoming"},
            {"date": "2026-06-01", "asset": "BTC",
              "call": "Bearish W target $52-57k",
              "outcome": "PENDING",
              "evidence": "BTC currently $63k, target not hit yet"},
            {"date": "2026-06-05", "asset": "QQQ",
              "call": "QQQ gap below 589, 200wMA = -36%",
              "outcome": "PENDING",
              "evidence": "QQQ at $716, not breached yet"},
        ],
    },
    "PositiveCrypto": {
        "name":          "Phillip Swift",
        "url":            "https://x.com/PositiveCrypto",
        "specialty":      "BTC cycle math + 200wMA framework (LookIntoBitcoin)",
        "calls": [
            {"date": "2018-12-15", "asset": "BTC",
              "call": "200wMA bottom signal at $3,200",
              "outcome": "RIGHT",
              "evidence": "Cycle 3 bottom confirmed"},
            {"date": "2022-11-09", "asset": "BTC",
              "call": "200wMA bottom signal at $15,500",
              "outcome": "RIGHT",
              "evidence": "Cycle 4 bottom confirmed"},
            {"date": "2026-06-05", "asset": "BTC",
              "call": "200wMA touch = generational buy",
              "outcome": "PENDING",
              "evidence": "Cycle 5 in progress; ETF era may invalidate pattern"},
            {"date": "2021-11", "asset": "BTC",
              "call": "Pi Cycle Top fired, near peak",
              "outcome": "RIGHT",
              "evidence": "Cycle 4 peaked Nov 2021"},
        ],
    },
    "Checkmatey": {
        "name":          "James Check",
        "url":            "https://x.com/_Checkmatey_",
        "specialty":      "Glassnode lead on-chain analyst (cost basis + supply)",
        "calls": [
            {"date": "2022-06", "asset": "BTC",
              "call": "Bear market underway via LTH-NUPL capitulation",
              "outcome": "RIGHT",
              "evidence": "Bear deepened, bottomed Nov 2022"},
            {"date": "2023-01", "asset": "BTC",
              "call": "Bull resumption via STH/LTH cost basis flip",
              "outcome": "RIGHT",
              "evidence": "BTC rallied from $16k to $73k peak"},
            {"date": "2025-08", "asset": "BTC",
              "call": "Late cycle - distribution phase",
              "outcome": "RIGHT",
              "evidence": "Peak Oct 6 2025"},
        ],
    },
    "benjamincowen": {
        "name":          "Benjamin Cowen",
        "url":            "https://x.com/benjamincowen",
        "specialty":      "Quant macro / risk-metric analyst (log regression, diminishing returns)",
        "calls": [
            {"date": "2025-09-20", "asset": "BTC",
              "call": "Q4 2025 cycle top forming — late-cycle distribution",
              "outcome": "RIGHT",
              "evidence": "BTC peaked $124,753 on Oct 6 2025 (Q4 timing right; his $131-154k price band ran a bit high)"},
            {"date": "2026-02-15", "asset": "BTC",
              "call": "Bottom NOT in — ~75% odds the cycle low is still ahead, ~Oct 2026",
              "outcome": "PENDING",
              "evidence": "BTC ~$63k, -49% from top; thesis unresolved (Q2-2026 risk memo)"},
            {"date": "2026-03-10", "asset": "BTC",
              "call": "2026 likely a 'red year' for BTC even if the economy holds up",
              "outcome": "PENDING",
              "evidence": "Year in progress; his ISM-up / BTC-down analog post on X"},
            {"date": "2026-04-11", "asset": "BTC",
              "call": "Sub-$40k remains on the table before the cycle low",
              "outcome": "PENDING",
              "evidence": "BTC ~$63k now; level not tested ('5 reasons not bottomed')"},
        ],
    },
}


def guru_hit_rates() -> dict:
    """Compute historical hit rate per guru."""
    out = {}
    for handle, info in GURU_TRACK_RECORD.items():
        calls = info.get("calls", [])
        scored = [c for c in calls if c.get("outcome") in ("RIGHT", "WRONG")]
        n_right = sum(1 for c in scored if c["outcome"] == "RIGHT")
        n_total = len(scored)
        hit_rate = (n_right / n_total * 100) if n_total else 0
        pending = [c for c in calls if c.get("outcome") == "PENDING"]
        out[handle] = {
            "name":          info["name"],
            "url":            info["url"],
            "specialty":      info["specialty"],
            "n_calls_scored": n_total,
            "n_right":        n_right,
            "n_wrong":        n_total - n_right,
            "hit_rate_pct":   round(hit_rate, 0),
            # < 3 graded calls is too small to assign a confident tier — flag it
            # so a single right call doesn't masquerade as a HIGH-tier track record.
            "tier":           ("INSUFFICIENT" if n_total < 3 else
                                "HIGH"   if hit_rate >= 75 else
                                "MODERATE" if hit_rate >= 50 else "LOW"),
            "pending_calls":   pending,
            "all_calls":       calls,
        }
    return out


def recent_high_calls(hours_back: int = 48) -> list[dict]:
    """Pull all HIGH-relevance tweets from last N hours across monitored gurus."""
    REPO = Path(__file__).resolve().parent.parent
    cache_files = [
        ("JesseOlson", REPO / ".jesse_olson_tweets_cache.json"),
        ("PositiveCrypto", REPO / ".guru_positivecrypto_tweets_cache.json"),
        ("benjamincowen", REPO / ".guru_benjamincowen_tweets_cache.json"),
    ]

    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    for handle, cache_path in cache_files:
        if not cache_path.exists(): continue
        try:
            data = json.loads(cache_path.read_text())
            for t in data.get("tweets", []):
                if t.get("relevance") != "HIGH": continue
                # Parse pub date (RFC 822 style: "Tue, 09 Jun 2026 04:02:25 GMT")
                pub = t.get("pub", "")
                try:
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pub) if pub else None
                    if pub_dt and pub_dt > cutoff:
                        out.append({
                            "handle":     handle,
                            "name":       GURU_TRACK_RECORD.get(handle, {}).get("name", handle),
                            "text":       (t.get("text") or "")[:280],
                            "link":       t.get("link", ""),
                            "pub":        pub[:16],
                        })
                except Exception:
                    pass
        except Exception: continue

    # Most recent first
    return sorted(out, key=lambda x: x.get("pub", ""), reverse=True)[:8]


def all_guru_intelligence() -> dict:
    """Compute everything in one call."""
    return {
        "track_records":   guru_hit_rates(),
        "recent_calls":    recent_high_calls(48),
        "computed_at":     datetime.now(timezone.utc).isoformat(),
    }


def main():
    r = all_guru_intelligence()
    print("=== GURU TRACK RECORDS ===")
    for h, info in r["track_records"].items():
        print(f"\n  {info['name']} (@{h})")
        print(f"    Hit rate: {info['hit_rate_pct']:.0f}% ({info['n_right']}/{info['n_calls_scored']})")
        print(f"    Tier: {info['tier']}")
        print(f"    Pending: {len(info['pending_calls'])} calls")
    print(f"\n=== RECENT HIGH CALLS (48h): {len(r['recent_calls'])} ===")
    for c in r["recent_calls"][:3]:
        print(f"  {c['name']}: {c['text'][:100]}")


if __name__ == "__main__":
    main()
