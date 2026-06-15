# -*- coding: utf-8 -*-
"""Simpleton daily brief — a plain-English 'what changed in the last 24 hours' note.

Generated once a day at 6am NZ (by simpleton_brief_run.py). It snapshots the
handful of signals the Simpleton Summary tab cares about, diffs against the
previous day's snapshot, and writes a friendly narrative to
.simpleton_daily_brief.json which the dashboard simply reads (frozen until the
next 6am run, so the note is stable all day).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / ".simpleton_brief_state.json"      # snapshot history (one per day)
OUT = REPO / ".simpleton_daily_brief.json"         # the rendered brief the tab reads


def _live_price() -> float:
    try:
        from core import data
        return data.btc_spot()
    except Exception:
        return 0.0


def _btc_24h() -> float:
    try:
        from core import data
        return float(data.btc_ticker().get("percentage") or 0)
    except Exception:
        return 0.0


def _g(key: str) -> dict:
    try:
        from core.dashboard_cache import get_cached
        return get_cached(key) or {}
    except Exception:
        return {}


def _snapshot() -> dict:
    cd = _g("cycle_dials").get("summary", {})
    nb = _g("btc_native_bottom_scorecard")
    tsc = _g("top_scorecard").get("scorecard", {})
    rt = _g("rotation_trigger")
    olq = _g("equity_olson")
    sem = _g("equity_semis")
    ud = _g("unified_decision")

    fire = rt.get("firing_paths", []) or []
    plan = "WAITING" if not fire else ("TIME TO ACT" if len(fire) >= 2 else "GETTING READY")
    hl = cd.get("headline", "") or ""
    cheap = ("cheap" if "ACCUMULATION" in hl
             else "expensive" if "DISTRIBUTION" in hl else "around fair value")
    return {
        "date": datetime.now().date().isoformat(),          # local (NZ) date
        "price": _live_price(),
        "cheap": cheap,
        "cycle_buy": cd.get("n_buy"),
        "cycle_total": cd.get("n_total"),
        "bottom_n": nb.get("n_met"),
        "bottom_total": nb.get("n_total"),
        "stocks_n": tsc.get("n_met"),
        "stocks_total": tsc.get("n_total"),
        "qqq_tier": olq.get("tier"),
        "semis_tier": sem.get("tier"),
        "plan": plan,
        "regime": ud.get("regime"),
    }


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"snaps": []}


def _int_delta(a, b) -> int:
    try:
        return int(a) - int(b)
    except Exception:
        return 0


def _prev_from_change_log() -> dict | None:
    """Bridge: until this module has its own multi-day history, borrow the most
    recent PRIOR-day snapshot from the existing btc_change_log (which has been
    running for days) so we can show real changes today. Only maps the fields
    the two share — simpleton-only fields (cheap/stocks/plan) are left absent so
    they never produce a fabricated diff."""
    try:
        f = REPO / ".btc_change_log_state.json"
        if not f.exists():
            return None
        snaps = json.loads(f.read_text(encoding="utf-8")).get("snapshots", [])
        tdate = datetime.now().date().isoformat()
        for s in reversed(snaps):
            if s.get("date") and s.get("date") != tdate and s.get("price"):
                return {
                    "date": s.get("date"),
                    "price": s.get("price"),
                    "bottom_n": s.get("native_bottom_n"),
                    "regime": s.get("regime"),
                }
    except Exception:
        pass
    return None


def build_daily_brief() -> dict:
    state = _load_state()
    snaps = state.get("snaps", [])
    today = _snapshot()
    tdate = today["date"]

    prev = None
    for s in reversed(snaps):
        if s.get("date") != tdate:
            prev = s
            break

    # store today's snapshot (replace same-day), keep ~40 days
    snaps = [s for s in snaps if s.get("date") != tdate]
    snaps.append(today)
    state["snaps"] = snaps[-40:]
    try:
        STATE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    btc24 = _btc_24h()
    lines = []
    if today.get("price"):
        if abs(btc24) < 0.3:
            lines.append(f"₿ Bitcoin was roughly flat over the last 24 hours, around **${today['price']:,.0f}**.")
        else:
            move = "rose" if btc24 > 0 else "fell"
            lines.append(f"₿ Bitcoin **{move} {abs(btc24):.1f}%** in the last 24 hours, to **${today['price']:,.0f}**.")

    if prev is None:
        prev = _prev_from_change_log()      # bridge to existing multi-day history

    if prev is None:
        return _write(today, lines,
                      "First daily update — from tomorrow I'll tell you exactly what changed overnight.",
                      btc24, first=True)

    n = 0
    # cheap / expensive
    if prev.get("cheap") is not None and today.get("cheap") != prev.get("cheap"):
        lines.append(f"📊 On the cycle gauges, Bitcoin shifted from **{prev.get('cheap')}** to **{today.get('cheap')}**.")
        n += 1
    elif (prev.get("cycle_buy") is not None and today.get("cycle_buy") is not None
            and today.get("cycle_buy") != prev.get("cycle_buy")):
        lines.append(f"📊 The cycle gauges now read **{today['cycle_buy']}/{today.get('cycle_total')}** saying 'cheap' "
                     f"(was {prev.get('cycle_buy')}).")
        n += 1

    # bottom checklist
    bd = _int_delta(today.get("bottom_n"), prev.get("bottom_n"))
    if bd:
        verb = "fired" if bd > 0 else "switched off"
        s = "" if abs(bd) == 1 else "s"
        lines.append(f"🟢 {abs(bd)} bottom signal{s} {verb} — the bottom checklist is now "
                     f"**{today.get('bottom_n')}/{today.get('bottom_total')}** (was {prev.get('bottom_n')}).")
        n += 1

    # stocks
    sd = _int_delta(today.get("stocks_n"), prev.get("stocks_n"))
    if sd:
        word = "more stretched" if sd > 0 else "calmer"
        lines.append(f"📉 Stocks look **{word}**: {today.get('stocks_n')}/{today.get('stocks_total')} "
                     f"'toppy' signs now (was {prev.get('stocks_n')}).")
        n += 1
    elif (prev.get("qqq_tier") is not None and today.get("qqq_tier")
            and today.get("qqq_tier") != prev.get("qqq_tier")):
        lines.append(f"📉 The stock-market tier moved from **{prev.get('qqq_tier')}** to **{today.get('qqq_tier')}**.")
        n += 1

    # the plan
    if prev.get("plan") is not None and today.get("plan") != prev.get("plan"):
        lines.append(f"🧭 **The plan changed: now {today.get('plan')}** (was {prev.get('plan')}) — worth a look.")
        n += 1

    # macro backdrop
    if prev.get("regime") is not None and today.get("regime") and today.get("regime") != prev.get("regime"):
        lines.append(f"🌐 The market backdrop shifted from **{prev.get('regime')}** to **{today.get('regime')}**.")
        n += 1

    if n == 0:
        summary = "A quiet 24 hours — the key signals held steady."
    else:
        summary = f"{n} thing{'' if n == 1 else 's'} changed in the last 24 hours."
    return _write(today, lines, summary, btc24, first=False)


def _write(today, lines, summary, btc24, first) -> dict:
    out = {
        "generated_local": datetime.now().isoformat(),
        "date": today["date"],
        "date_friendly": datetime.now().strftime("%a %d %b %Y"),
        "summary": summary,
        "lines": lines,
        "btc_24h_pct": btc24,
        "first": first,
        "plan": today.get("plan"),
    }
    try:
        OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return out


def main():
    r = build_daily_brief()
    try:
        print("SIMPLETON BRIEF:", r["summary"])
        for ln in r["lines"]:
            print("  -", ln)
    except UnicodeEncodeError:
        print("SIMPLETON BRIEF:", r["summary"].encode("ascii", "replace").decode())
        for ln in r["lines"]:
            print("  -", ln.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
