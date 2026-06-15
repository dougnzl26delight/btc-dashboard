"""Data-health monitor — so no indicator ever goes stale or drifts silently.

Guards two failure modes the dashboard otherwise hides:

1. STALE DATA — `disk_cached` returns the last-good value if a refresh fails, so a
   feed can be hours/days old with NO visible sign. This flags any cache older
   than its freshness budget.

2. DENOMINATOR DRIFT — scorecards grow (8->10, 15->16) but hardcoded "/N" labels
   don't follow (the bug class the operator kept hitting). This compares each
   scorecard's LIVE n_total to its canonical value and flags any mismatch, so a
   drift is caught automatically instead of by eye.

Plus a dead-feed scan (criteria reading "data unavailable"). NOT advice — infra.
"""
from __future__ import annotations
import time

from core.dashboard_cache import cache_age_seconds, get_cached

# All caches the dashboard depends on (from the live .panel_cache set).
_TRACKED_KEYS = [
    "unified_decision", "predictor_engine", "btc_native_top_scorecard",
    "btc_native_bottom_scorecard", "swift_watch", "swift_dials", "swift_charts",
    "swift_indicators", "cycle_dials", "equity_olson", "equity_semis",
    "rotation_trigger", "rotation_validation", "scale_out_trigger",
    "date_predictions", "realized_price", "realized_cap_drawdown",
    "sth_cost_basis", "top_scorecard", "macro_layer", "regime", "early_rotation",
    "rotation", "etf_regime", "glassnode_proxies", "free_proxies",
    "bottom_signals", "state_of_btc", "guru_intelligence", "pattern_zones",
    "olson", "ohlcv_90d",
]

# Per-key freshness budget (hours). Beyond this = STALE. Slow/low-cadence feeds
# get a longer budget. precompute refreshes most every few hours.
_FRESH_BUDGET_H = {
    "equity_olson": 14, "equity_semis": 14,      # daily-close driven
    "date_predictions": 26, "guru_intelligence": 14,
    "rotation_validation": 26,
}
_DEFAULT_BUDGET_H = 8

# Canonical scorecard totals. If the LIVE n_total drifts from these, hardcoded
# "/N" labels elsewhere need updating — this catches it automatically.
_CANONICAL_TOTALS = {
    "btc_native_bottom_scorecard": 16,
    "btc_native_top_scorecard": 16,
    "top_scorecard": 10,    # equity macro-top (under .scorecard)
    "cycle_dials": 7,
}


def _scorecard_total(key: str, v: dict):
    if not isinstance(v, dict):
        return None
    if key == "cycle_dials":
        s = (v.get("summary") or {})
        return s.get("n_total") or (len(v.get("dials") or {}) or None)
    sc = v.get("scorecard", v)
    return sc.get("n_total") if isinstance(sc, dict) else None


def data_health() -> dict:
    items = []
    n_stale = n_aging = n_missing = 0
    for k in _TRACKED_KEYS:
        age = cache_age_seconds(k)
        budget_h = _FRESH_BUDGET_H.get(k, _DEFAULT_BUDGET_H)
        if age is None:
            status, age_h = "MISSING", None
            n_missing += 1
        else:
            age_h = age / 3600.0
            if age_h > budget_h:
                status = "STALE"; n_stale += 1
            elif age_h > budget_h * 0.6:
                status = "AGING"; n_aging += 1
            else:
                status = "FRESH"
        items.append({"key": k, "age_h": age_h, "budget_h": budget_h, "status": status})

    # Denominator-drift check
    drift = []
    for k, expected in _CANONICAL_TOTALS.items():
        actual = _scorecard_total(k, get_cached(k) or {})
        if actual is not None and actual != expected:
            drift.append({"key": k, "expected": expected, "actual": actual})

    # Dead-feed scan (paywalled/unavailable criteria)
    dead = []
    for k in ("btc_native_bottom_scorecard", "top_scorecard"):
        v = get_cached(k) or {}
        sc = v.get("scorecard", v) if isinstance(v, dict) else {}
        for c in (sc.get("criteria") or []):
            if isinstance(c, dict):
                stt = str(c.get("status") or "").lower()
                if "unavail" in stt or "data gap" in stt or "data_gap" in stt:
                    dead.append({"key": k, "label": c.get("label", "?")})

    if n_stale or n_missing:
        verdict, color = "STALE DATA", "#ef4444"
    elif drift:
        verdict, color = "DENOMINATOR DRIFT", "#ef4444"
    elif dead or n_aging:
        verdict, color = "DEGRADED", "#f0b90b"
    else:
        verdict, color = "ALL FRESH", "#22c55e"

    return {
        "verdict": verdict, "color": color,
        "n_tracked": len(_TRACKED_KEYS),
        "n_stale": n_stale, "n_aging": n_aging, "n_missing": n_missing,
        "drift": drift, "dead_feeds": dead, "items": items,
        "ts": time.time(),
    }


if __name__ == "__main__":
    h = data_health()
    print(f"VERDICT: {h['verdict']}  ({h['n_stale']} stale, {h['n_aging']} aging, "
          f"{h['n_missing']} missing of {h['n_tracked']})")
    if h["drift"]:
        print("DRIFT:", h["drift"])
    if h["dead_feeds"]:
        print("DEAD FEEDS:", [d["label"] for d in h["dead_feeds"]])
    for it in sorted(h["items"], key=lambda x: (x["age_h"] or 0), reverse=True)[:8]:
        a = f"{it['age_h']:.1f}h" if it["age_h"] is not None else "n/a"
        print(f"  [{it['status']:<7}] {it['key']:<30} {a} (budget {it['budget_h']}h)")
