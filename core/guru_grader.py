"""Objective guru-call grader — data-derived RIGHT/WRONG, not hand-asserted.

2026-07-08: built to replace the hand-assigned `outcome` field in
guru_intelligence.GURU_TRACK_RECORD (a claim-validity-audit finding: the old
hit-rates were asserted, not measured). Each call is graded against ACTUAL
forward price action via yfinance, with transparent, pre-declared rules and the
computed forward return exposed so every verdict is auditable.

What this fixes: the RIGHT/WRONG is now measured from price.
What this does NOT fix: SURVIVORSHIP — the *set* of calls is still author-
selected. That is disclosed in the UI; a full fix needs comprehensive call
capture, which is a separate ingestion project.

Grading rules (pre-declared, applied uniformly):
  TOP call      RIGHT if price `horizon` days later is >= DECISIVE below the
                call-date price; WRONG if >= DECISIVE above; else MARGINAL.
  BOTTOM call   mirror of TOP (RIGHT if price rose >= DECISIVE).
  DIRECTION     UP/DOWN — RIGHT if forward return over horizon matches sign
                beyond a DEADBAND; WRONG if it goes the other way beyond it.
  TARGET        a price band + bias — RIGHT if the band is touched within the
                horizon; if the horizon elapsed untouched and price moved
                decisively against the bias, WRONG; otherwise still open.
  PENDING       the grading horizon has not elapsed yet (window still open).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
_CACHE = _REPO / ".guru_grader_prices.json"

DECISIVE = 0.10   # >=10% move = a decisive resolution of a top/bottom call
DEADBAND = 0.03   # <3% forward move = too small to score a direction call

# asset symbol -> yfinance ticker
_TICKER = {"BTC": "BTC-USD", "QQQ": "QQQ", "ETH": "ETH-USD", "SPY": "SPY"}


def _load_price_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except Exception:
        return {}


def _save_price_cache(c: dict) -> None:
    try:
        _CACHE.write_text(json.dumps(c))
    except Exception:
        pass


def _price_series(asset: str) -> Optional[dict]:
    """Return {isodate: close} for the asset, cached 12h. yfinance daily."""
    tkr = _TICKER.get(asset.upper())
    if not tkr:
        return None
    cache = _load_price_cache()
    ent = cache.get(tkr)
    now = datetime.now(timezone.utc).timestamp()
    if ent and (now - ent.get("ts", 0)) < 43200 and ent.get("data"):
        return ent["data"]
    try:
        import yfinance as yf
        # 'max' so 2018-era calls have coverage; daily closes.
        h = yf.Ticker(tkr).history(period="max", interval="1d")
        if h is None or h.empty:
            return ent.get("data") if ent else None
        close = h["Close"]
        data = {d.strftime("%Y-%m-%d"): float(v)
                for d, v in close.items() if v == v}  # drop NaN
        cache[tkr] = {"ts": now, "data": data}
        _save_price_cache(cache)
        return data
    except Exception:
        return ent.get("data") if ent else None


def _norm_date(s: str) -> Optional[str]:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM' (-> first of month)."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _price_on_or_after(series: dict, iso: str) -> Optional[tuple]:
    """First available (date, close) at or after `iso` (handles weekends/gaps)."""
    if not series:
        return None
    for i in range(0, 8):
        d = (datetime.strptime(iso, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
        if d in series:
            return d, series[d]
    return None


def _window(series: dict, start_iso: str, end_iso: str) -> list:
    """Closes with start <= date <= end, ascending."""
    return [v for d, v in sorted(series.items()) if start_iso <= d <= end_iso]


def grade_call(call: dict, today: Optional[str] = None) -> dict:
    """Grade one call from price data. Returns the call enriched with:
       outcome (RIGHT/WRONG/MARGINAL/PENDING/UNGRADED), method, fwd_return_pct,
       detail. Fail-safe: UNGRADED (falls back to any hand `outcome`) on data gap.
    """
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = dict(call)
    asset = call.get("asset", "BTC")
    cdate = _norm_date(call.get("date", ""))
    kind = call.get("kind")
    horizon = int(call.get("horizon_days", 120))

    def _fallback(reason):
        out["outcome_objective"] = "UNGRADED"
        out["grade_detail"] = reason
        # preserve any hand label so the UI can still show something
        out["outcome"] = call.get("outcome", "PENDING")
        return out

    if not cdate or not kind:
        return _fallback("no structured date/kind")
    series = _price_series(asset)
    if not series:
        return _fallback(f"no price series for {asset}")
    p0 = _price_on_or_after(series, cdate)
    if not p0:
        return _fallback("no price at call date")
    _, px0 = p0
    end_iso = (datetime.strptime(cdate, "%Y-%m-%d") + timedelta(days=horizon)).strftime("%Y-%m-%d")
    window_open = end_iso > today

    def _finish(outcome, method, fwd, detail):
        out["outcome_objective"] = outcome
        out["outcome"] = outcome if outcome in ("RIGHT", "WRONG") else \
            ("PENDING" if outcome == "PENDING" else call.get("outcome", "PENDING"))
        out["method"] = method
        out["fwd_return_pct"] = None if fwd is None else round(fwd * 100, 1)
        out["grade_detail"] = detail
        return out

    # ---- TARGET: band touched within horizon? -----------------------------
    if kind == "TARGET":
        lo = call.get("target_low"); hi = call.get("target_high")
        bias = call.get("bias", "DOWN")  # which way the target implies price goes
        if lo is None or hi is None:
            return _fallback("target band missing")
        end_for_scan = today if window_open else end_iso
        wl = _window(series, cdate, end_for_scan)
        touched = any(lo <= v <= hi for v in wl)
        cur = wl[-1] if wl else px0
        fwd = (cur / px0 - 1) if px0 else None
        if touched:
            return _finish("RIGHT", "target-touched",
                           fwd, f"price entered {lo:g}-{hi:g} within window")
        if window_open:
            return _finish("PENDING", "target-open", fwd,
                           f"band {lo:g}-{hi:g} not yet touched; window open to {end_iso}")
        # horizon elapsed untouched: wrong if price moved decisively against bias
        moved_against = (bias == "DOWN" and fwd is not None and fwd > DECISIVE) or \
                        (bias == "UP" and fwd is not None and fwd < -DECISIVE)
        return _finish("WRONG" if moved_against else "MARGINAL", "target-missed",
                       fwd, f"band {lo:g}-{hi:g} not touched by {end_iso}")

    # ---- price at horizon end (or latest, if window still open) -----------
    if window_open:
        # can only resolve a top/bottom/direction early if it ALREADY moved decisively
        wl = _window(series, cdate, today)
        cur = wl[-1] if wl else px0
        fwd_now = (cur / px0 - 1) if px0 else 0.0
        decided_early = abs(fwd_now) >= DECISIVE
        if not decided_early:
            return _finish("PENDING", "window-open", fwd_now,
                           f"{fwd_now*100:+.1f}% so far; horizon to {end_iso}")
        pxN, fwd = cur, fwd_now
        asof = "so-far"
    else:
        pN = _price_on_or_after(series, end_iso)
        if not pN:
            # window closed but no price (very recent gap) — use latest
            wl = _window(series, cdate, today)
            if not wl:
                return _fallback("no forward price")
            pxN = wl[-1]
        else:
            pxN = pN[1]
        fwd = (pxN / px0 - 1) if px0 else 0.0
        asof = f"at +{horizon}d"

    if kind == "TOP":
        if fwd <= -DECISIVE:
            return _finish("RIGHT", "top", fwd, f"fell {fwd*100:+.1f}% {asof} after top call")
        if fwd >= DECISIVE:
            return _finish("WRONG", "top", fwd, f"rose {fwd*100:+.1f}% {asof} — top call early/wrong")
        return _finish("MARGINAL", "top", fwd, f"{fwd*100:+.1f}% {asof} — inconclusive")
    if kind == "BOTTOM":
        if fwd >= DECISIVE:
            return _finish("RIGHT", "bottom", fwd, f"rose {fwd*100:+.1f}% {asof} after bottom call")
        if fwd <= -DECISIVE:
            return _finish("WRONG", "bottom", fwd, f"fell {fwd*100:+.1f}% {asof} — bottom call wrong")
        return _finish("MARGINAL", "bottom", fwd, f"{fwd*100:+.1f}% {asof} — inconclusive")
    if kind == "DIRECTION":
        want_up = call.get("direction", "UP").upper() == "UP"
        if abs(fwd) < DEADBAND:
            return _finish("MARGINAL", "direction", fwd, f"{fwd*100:+.1f}% {asof} — within deadband")
        right = (fwd > 0) == want_up
        return _finish("RIGHT" if right else "WRONG", "direction", fwd,
                       f"{fwd*100:+.1f}% {asof} vs {'UP' if want_up else 'DOWN'} call")
    return _fallback(f"unknown kind {kind}")
