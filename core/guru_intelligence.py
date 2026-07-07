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
    # 2026-07-08: each call now carries STRUCTURED grading metadata (kind +
    # target/direction/horizon) so guru_grader.grade_call() can derive RIGHT/
    # WRONG from actual forward price action, not a hand-typed outcome. The
    # `evidence` line is kept as human context. Survivorship (which calls are
    # listed) is author-selected and disclosed in the UI.
    "JesseOlson": {
        "name": "Jesse Olson", "url": "https://x.com/JesseOlson",
        "specialty": "Crypto technical analyst (multi-week timeframes)",
        "calls": [
            {"date": "2025-10-01", "asset": "BTC", "kind": "TOP", "horizon_days": 90,
             "call": "Cycle 5 top imminent via 3wk MACD divergence",
             "evidence": "BTC peaked Oct 6 2025 ~$124,700"},
            {"date": "2025-10-08", "asset": "BTC", "kind": "DIRECTION", "direction": "DOWN",
             "horizon_days": 21,
             "call": "Binance cascade incoming",
             "evidence": "Cascade occurred Oct 10 2025"},
            {"date": "2024-09-15", "asset": "BTC", "kind": "TARGET",
             "target_low": 38000, "target_high": 42000, "bias": "DOWN", "horizon_days": 180,
             "call": "Bottom at $40k, bull resumption (called a drop to $40k first)",
             "evidence": "BTC never revisited $40k — rose toward the $124k peak"},
            {"date": "2026-06-01", "asset": "BTC", "kind": "TARGET",
             "target_low": 52000, "target_high": 57000, "bias": "DOWN", "horizon_days": 120,
             "call": "Bearish W target $52-57k",
             "evidence": "Whether price tags the band decides it"},
            {"date": "2026-06-05", "asset": "QQQ", "kind": "TARGET",
             "target_low": 560, "target_high": 589, "bias": "DOWN", "horizon_days": 90,
             "call": "QQQ gap below 589 toward 200wMA",
             "evidence": "QQQ ~$716 at the call"},
        ],
    },
    "PositiveCrypto": {
        "name": "Phillip Swift", "url": "https://x.com/PositiveCrypto",
        "specialty": "BTC cycle math + 200wMA framework (LookIntoBitcoin)",
        "calls": [
            {"date": "2018-12-15", "asset": "BTC", "kind": "BOTTOM", "horizon_days": 180,
             "call": "200wMA bottom signal ~$3,200",
             "evidence": "Cycle 3 bottom region"},
            {"date": "2022-11-09", "asset": "BTC", "kind": "BOTTOM", "horizon_days": 180,
             "call": "200wMA bottom signal ~$15,500",
             "evidence": "Cycle 4 bottom confirmed Nov 2022"},
            {"date": "2021-11-01", "asset": "BTC", "kind": "TOP", "horizon_days": 90,
             "call": "Pi Cycle Top fired, near peak",
             "evidence": "Cycle 4 peaked Nov 2021"},
            {"date": "2026-06-05", "asset": "BTC", "kind": "BOTTOM", "horizon_days": 180,
             "call": "200wMA touch = generational buy",
             "evidence": "Cycle 5; ETF era may invalidate the pattern"},
        ],
    },
    "Checkmatey": {
        "name": "James Check", "url": "https://x.com/_Checkmatey_",
        "specialty": "Glassnode lead on-chain analyst (cost basis + supply)",
        "calls": [
            {"date": "2022-06-01", "asset": "BTC", "kind": "DIRECTION", "direction": "DOWN",
             "horizon_days": 150,
             "call": "Bear market underway via LTH-NUPL capitulation",
             "evidence": "Bear deepened, bottomed Nov 2022"},
            {"date": "2023-01-01", "asset": "BTC", "kind": "DIRECTION", "direction": "UP",
             "horizon_days": 180,
             "call": "Bull resumption via STH/LTH cost-basis flip",
             "evidence": "BTC rallied off $16k"},
            {"date": "2025-08-01", "asset": "BTC", "kind": "TOP", "horizon_days": 120,
             "call": "Late cycle - distribution phase",
             "evidence": "Peak Oct 6 2025"},
        ],
    },
    "benjamincowen": {
        "name": "Benjamin Cowen", "url": "https://x.com/benjamincowen",
        "specialty": "Quant macro / risk-metric analyst (log regression, diminishing returns)",
        "calls": [
            {"date": "2025-09-20", "asset": "BTC", "kind": "TOP", "horizon_days": 90,
             "call": "Q4 2025 cycle top forming - late-cycle distribution",
             "evidence": "BTC peaked ~$124,700 Oct 6 2025 (price band ran high)"},
            {"date": "2026-02-15", "asset": "BTC", "kind": "DIRECTION", "direction": "DOWN",
             "horizon_days": 240,
             "call": "Bottom NOT in - ~75% odds the cycle low is still ahead, ~Oct 2026",
             "evidence": "Thesis resolves ~Oct 2026"},
            {"date": "2026-03-10", "asset": "BTC", "kind": "DIRECTION", "direction": "DOWN",
             "horizon_days": 296,
             "call": "2026 likely a 'red year' for BTC",
             "evidence": "Resolves at 2026 year-end"},
            {"date": "2026-04-11", "asset": "BTC", "kind": "TARGET",
             "target_low": 30000, "target_high": 40000, "bias": "DOWN", "horizon_days": 240,
             "call": "Sub-$40k remains on the table before the cycle low",
             "evidence": "Level not tested at ~$63k"},
        ],
    },
}

def guru_hit_rates() -> dict:
    """Per-guru hit rate, graded OBJECTIVELY from forward price action.

    2026-07-08 rebuild: outcomes come from guru_grader.grade_call() (measured
    against yfinance price), not a hand-typed field. hit_rate = right/(right+
    wrong) over DECISIVE calls only; MARGINAL (resolved-inconclusive) and
    PENDING (window open) are excluded from the ratio and reported separately.
    Fail-safe: if price data is unavailable a call is UNGRADED and excluded.
    """
    from core.guru_grader import grade_call
    out = {}
    for handle, info in GURU_TRACK_RECORD.items():
        graded = [grade_call(c) for c in info.get("calls", [])]
        n_right = sum(1 for c in graded if c.get("outcome_objective") == "RIGHT")
        n_wrong = sum(1 for c in graded if c.get("outcome_objective") == "WRONG")
        n_marg = sum(1 for c in graded if c.get("outcome_objective") == "MARGINAL")
        n_pend = sum(1 for c in graded if c.get("outcome_objective") == "PENDING")
        n_ungraded = sum(1 for c in graded if c.get("outcome_objective") == "UNGRADED")
        n_decisive = n_right + n_wrong
        hit_rate = (n_right / n_decisive * 100) if n_decisive else None
        pending = [c for c in graded if c.get("outcome_objective") in ("PENDING", "UNGRADED")]
        out[handle] = {
            "name":          info["name"],
            "url":            info["url"],
            "specialty":      info["specialty"],
            "n_calls_scored": n_decisive,       # decisive (right+wrong) only
            "n_right":        n_right,
            "n_wrong":        n_wrong,
            "n_marginal":     n_marg,
            "n_pending":      n_pend + n_ungraded,
            "hit_rate_pct":   None if hit_rate is None else round(hit_rate, 0),
            # < 3 DECISIVE graded calls is too small to rate. Objective grading
            # means these are measured, but survivorship (call selection) remains.
            "tier":           ("INSUFFICIENT" if n_decisive < 3 or hit_rate is None else
                                "HIGH"   if hit_rate >= 75 else
                                "MODERATE" if hit_rate >= 50 else "LOW"),
            "graded":         True,
            "pending_calls":   pending,
            "all_calls":       graded,
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
        _hr = info['hit_rate_pct']
        print(f"    Hit rate: {'n/a' if _hr is None else str(int(_hr))+'%'} "
              f"({info['n_right']}/{info['n_calls_scored']} decisive; "
              f"{info.get('n_marginal',0)} marginal, {info['n_pending']} pending)")
        print(f"    Tier: {info['tier']}")
        print(f"    Pending: {len(info['pending_calls'])} calls")
    print(f"\n=== RECENT HIGH CALLS (48h): {len(r['recent_calls'])} ===")
    for c in r["recent_calls"][:3]:
        print(f"  {c['name']}: {c['text'][:100]}")


if __name__ == "__main__":
    main()
