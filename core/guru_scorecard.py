"""Generic guru signal-scorecard — auto-log ANY monitored analyst's directional
calls and grade them honestly by forward return.

This is the parameterized engine behind the Olson scorecard. Point it at a guru
config and it does the same thing for that handle:

  1. capture each NEW directional call from their cached tweets (asset + bull/bear
     + the price when they said it),
  2. auto-grade it by the signed forward return over a fixed horizon (default 30d),
  3. report hit-rate + PAYOFF RATIO (avg win / avg loss = "R") + EXPECTANCY/call.

Each guru gets its own log file so records grow independently. Small samples mean
little — the value is after ~20-30 graded calls. NOT advice.

  core.olson_scorecard  → thin wrapper, OLSON_CFG (seeds from JesseOlson record)
  GURU_CFGS["benjamincowen"] → Benjamin Cowen (seeds from his curated record)
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HORIZON_DAYS = 30          # multi-week timeframe — matches how these analysts call
WIN_THRESHOLD = 0.01       # |move| < 1% over the horizon = "flat", not scored

_BEAR = ("bearish", "bear ", "below", "breakdown", "lower high", "gap below",
         "lookout below", "sell signal", "engulfing", "not in", "downside",
         "drop", "rejected", "resistance", "short", "topping", "top is", "decline",
         "red year", "dead cat", "capitulation", "lower", "distribution")
_BULL = ("bullish", "bull ", "above", "breakout", "higher", "gap above",
         "bottom is in", "reversal", "golden cross", "support holds", "buy signal",
         "accumulate", "rally", "upside", "reclaim", "bottom in", "low is in")


# ── guru configs ─────────────────────────────────────────────────────────────
# log: per-guru persisted call log.  seed_handle: GURU_TRACK_RECORD key to seed
# the curated baseline from.  cache: the monitor's tweet cache (handle.lower()).
GURU_CFGS = {
    "benjamincowen": {
        "handle":      "benjamincowen",
        "name":        "Benjamin Cowen",
        "seed_handle": "benjamincowen",
        "log":         REPO / ".cowen_scorecard_log.json",
        "horizon_days": HORIZON_DAYS,
    },
}


# ── price helpers ────────────────────────────────────────────────────────────
def _yf(asset: str):
    sym = "BTC-USD" if asset.upper() in ("BTC", "BITCOIN") else asset.upper()
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="3y", interval="1d")["Close"].dropna()
        if h.empty:
            return None
        h.index = h.index.tz_localize(None)
        return h
    except Exception:
        return None


def _close_on(series, d: datetime):
    if series is None:
        return None
    try:
        sub = series[series.index <= d]
        return float(sub.iloc[-1]) if len(sub) else None
    except Exception:
        return None


# ── call parsing ─────────────────────────────────────────────────────────────
def _direction(text: str):
    t = (text or "").lower()
    bear = sum(1 for k in _BEAR if k in t)
    bull = sum(1 for k in _BULL if k in t)
    if bear > bull:
        return -1
    if bull > bear:
        return 1
    return 0


def _asset(text: str):
    m = re.search(r"\$([A-Za-z]{2,5})\b", text or "")
    if m:
        return m.group(1).upper()
    if "bitcoin" in (text or "").lower():
        return "BTC"
    return None


def _parse_pub(pub: str):
    """Tweet pubDate -> date (best-effort). Cached pub is often date-only
    ('Wed, 10 Jun 2026'), so a STABLE parse matters — else the dedup key drifts
    daily and the same tweet gets re-logged every run."""
    p = (pub or "").strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%a, %d %b %Y",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(p, fmt)
        except Exception:
            continue
    return None


# ── persistence ──────────────────────────────────────────────────────────────
def _load_log(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_log(path: Path, calls: list) -> None:
    try:
        path.write_text(json.dumps(calls, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _seed(calls: list, seed_handle: str) -> list:
    """Seed once from the hand-curated guru_intelligence calls (verified outcomes)."""
    if calls:
        return calls
    try:
        from core.guru_intelligence import guru_hit_rates
        rec = (guru_hit_rates() or {}).get(seed_handle, {})
        for c in rec.get("all_calls", []):
            calls.append({
                "date": c.get("date"), "asset": c.get("asset"),
                "thesis": c.get("call"), "direction": None,
                "source": "curated", "price_at_call": None,
                "outcome": c.get("outcome"), "fwd_return": None,
                "evidence": c.get("evidence"),
            })
    except Exception:
        pass
    return calls


# ── main pipeline ────────────────────────────────────────────────────────────
def update_and_grade(cfg: dict, now: datetime | None = None) -> dict:
    """Capture new calls from this guru's tweets, grade matured ones, persist."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    handle = cfg["handle"]
    log_path = cfg["log"]
    horizon = cfg.get("horizon_days", HORIZON_DAYS)
    calls = _seed(_load_log(log_path), cfg["seed_handle"])

    # 1) capture NEW directional calls from the live tweet cache
    try:
        from core.dashboard_cache import get_cached
        gi = get_cached("guru_intelligence") or {}
        tweets = [c for c in (gi.get("recent_calls") or [])
                  if (c.get("handle") or "").lower() == handle.lower()]
    except Exception:
        tweets = []
    seen = {(c.get("date"), (c.get("thesis") or "")[:60]) for c in calls}
    price_cache = {}
    for tw in tweets:
        text = tw.get("text") or tw.get("title") or ""
        d = _parse_pub(tw.get("pub", "")) or now
        asset = _asset(text)
        direction = _direction(text)
        if not asset or direction == 0:
            continue                      # only score CLEAR directional calls
        key = (d.strftime("%Y-%m-%d"), text[:60])
        if key in seen:
            continue
        seen.add(key)
        if asset not in price_cache:
            price_cache[asset] = _yf(asset)
        calls.append({
            "date": d.strftime("%Y-%m-%d"), "asset": asset,
            "thesis": text[:140], "direction": direction, "source": "auto",
            "price_at_call": _close_on(price_cache[asset], d),
            "outcome": "PENDING", "fwd_return": None, "evidence": "",
        })

    # 2) grade matured PENDING auto-calls by forward return
    for c in calls:
        if c.get("source") != "auto" or c.get("outcome") != "PENDING":
            continue
        if c.get("direction") is None or not c.get("price_at_call"):
            continue
        try:
            call_d = datetime.strptime(c["date"], "%Y-%m-%d")
        except Exception:
            continue
        if (now - call_d).days < horizon:
            continue                       # not matured yet
        asset = c["asset"]
        if asset not in price_cache:
            price_cache[asset] = _yf(asset)
        px_after = _close_on(price_cache[asset], call_d + timedelta(days=horizon))
        if not px_after:
            continue
        raw = px_after / c["price_at_call"] - 1.0
        signed = raw * c["direction"]      # + if their direction was right
        c["fwd_return"] = round(signed * 100, 1)
        c["outcome"] = ("RIGHT" if signed > WIN_THRESHOLD else
                        "WRONG" if signed < -WIN_THRESHOLD else "FLAT")

    _save_log(log_path, calls)
    return _aggregate(calls, horizon)


def _aggregate(calls: list, horizon: int = HORIZON_DAYS) -> dict:
    scored = [c for c in calls if c.get("outcome") in ("RIGHT", "WRONG")]
    auto_scored = [c for c in scored if c.get("source") == "auto" and c.get("fwd_return") is not None]
    n_right = sum(1 for c in scored if c["outcome"] == "RIGHT")
    n_total = len(scored)
    hit = (n_right / n_total) if n_total else 0.0

    wins = [c["fwd_return"] for c in auto_scored if c["outcome"] == "RIGHT"]
    losses = [abs(c["fwd_return"]) for c in auto_scored if c["outcome"] == "WRONG"]
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None
    payoff = (avg_win / avg_loss) if (avg_win and avg_loss) else None
    expectancy = None
    if auto_scored:
        ah = sum(1 for c in auto_scored if c["outcome"] == "RIGHT") / len(auto_scored)
        if avg_win is not None and avg_loss is not None:
            expectancy = round(ah * avg_win - (1 - ah) * avg_loss, 2)

    return {
        "n_logged": len(calls),
        "n_scored": n_total,
        "n_pending": sum(1 for c in calls if c.get("outcome") == "PENDING"),
        "n_auto_scored": len(auto_scored),
        "hit_rate_pct": round(hit * 100, 0),
        "avg_win_pct": round(avg_win, 1) if avg_win is not None else None,
        "avg_loss_pct": round(avg_loss, 1) if avg_loss is not None else None,
        "payoff_R": round(payoff, 2) if payoff is not None else None,
        "expectancy_pct": expectancy,
        "calls": sorted(calls, key=lambda c: c.get("date", ""), reverse=True),
        "horizon_days": horizon,
        "verdict": _verdict(n_total, hit, expectancy),
        "ts": now_str(),
    }


def _verdict(n_scored, hit, expectancy):
    if n_scored < 8:
        return "TOO EARLY — need ~20+ graded calls to judge"
    if expectancy is not None and expectancy > 0.5:
        return "POSITIVE EDGE so far"
    if expectancy is not None and expectancy < -0.5:
        return "NEGATIVE EDGE so far"
    return "FLAT / inconclusive"


def now_str():
    try:
        return datetime.now(timezone.utc).isoformat()
    except Exception:
        return ""


def scorecard(cfg: dict) -> dict:
    """Read-only aggregate of a guru's current log (no capture/grade — dashboard)."""
    return _aggregate(_seed(_load_log(cfg["log"]), cfg["seed_handle"]),
                      cfg.get("horizon_days", HORIZON_DAYS))


# ── convenience wrappers per guru ────────────────────────────────────────────
def cowen_scorecard() -> dict:
    return scorecard(GURU_CFGS["benjamincowen"])


def update_cowen() -> dict:
    return update_and_grade(GURU_CFGS["benjamincowen"])


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    for _key, _cfg in GURU_CFGS.items():
        r = update_and_grade(_cfg)
        print(f"[{_cfg['name']}] VERDICT: {r['verdict']}")
        print(f"  logged={r['n_logged']} scored={r['n_scored']} pending={r['n_pending']} "
              f"auto-scored={r['n_auto_scored']}")
        print(f"  hit-rate={r['hit_rate_pct']}%  payoff(R)={r['payoff_R']}  "
              f"expectancy={r['expectancy_pct']}%/call")
