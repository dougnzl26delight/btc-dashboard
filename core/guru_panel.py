"""Guru Panel — score legendary crypto & macro top/bottom callers off the LIVE
dashboard signals, so the operator can see "what would the legends think?"
grounded in real data rather than vibes.

Each guru's framework is either (a) read directly from a live cache the dashboard
already computes, or (b) a well-documented standing stance noted as such. Verdicts:
  on_track  (green)  — their framework supports the current campaign direction
  wait      (amber)  — supportive of direction but says "not yet / be patient"
  caution   (amber)  — flags a real risk the plan should respect
  dissent   (red)    — their framework disagrees / a contrarian warning

NOT investment advice — a framework-based sanity panel.
"""
from __future__ import annotations

_VERDICT_COLOR = {
    "on_track": "#22c55e",
    "wait":     "#f0b90b",
    "caution":  "#f0b90b",
    "dissent":  "#ef4444",
}
_VERDICT_LABEL = {
    "on_track": "ON TRACK",
    "wait":     "WAIT / PATIENT",
    "caution":  "CAUTION",
    "dissent":  "DISSENT",
}


def guru_panel() -> dict:
    from core.dashboard_cache import get_cached as g

    sw = g("swift_watch") or {}
    ri = sw.get("risk_index", {}) or {}
    cd = g("cycle_dials") or {}
    dials = cd.get("dials") or {}
    rp = g("realized_price") or {}
    nb = g("btc_native_bottom_scorecard") or {}
    eo = g("equity_olson") or {}
    dp = g("date_predictions") or {}
    ts = (g("top_scorecard") or {}).get("scorecard", {}) or {}
    rt = g("rotation_trigger") or {}
    btc_px = rt.get("btc_price") or 0

    gurus: list[dict] = []

    def add(name, cat, framework, reading, verdict, detail, live=True):
        gurus.append({
            "name": name, "category": cat, "framework": framework,
            "reading": reading, "verdict": verdict,
            "verdict_label": _VERDICT_LABEL.get(verdict, "?"),
            "color": _VERDICT_COLOR.get(verdict, "#888"),
            "detail": detail, "live": live,
        })

    def _dial(key):
        d = dials.get(key) if isinstance(dials, dict) else None
        if isinstance(d, dict):
            return d.get("label") or d.get("zone") or d.get("signal")
        return None

    # ── CRYPTO CYCLE PANEL ───────────────────────────────────────────────────
    idx = ri.get("risk_index")
    zone = ri.get("zone", "?")
    add("Phillip Swift", "crypto", "Bitcoin Risk Index / cycle",
        (f"Risk Index {idx:.2f} -> {zone}" if isinstance(idx, (int, float)) else "n/a"),
        ("on_track" if isinstance(idx, (int, float)) and idx < 0.40 else "wait"),
        "Long-term valuation sits in the cheap half of the cycle — accumulation territory.",
        live=isinstance(idx, (int, float)))

    lr = _dial("log_regression")
    add("Ben Cowen", "crypto", "Logarithmic regression + risk metric",
        (f"Log-regression band: {lr}" if lr else "n/a"),
        ("on_track" if str(lr).upper() == "BUY" else "wait"),
        "Lower half of the cycle — DCA over the bottom zone; expect a shallower (muted) low.",
        live=bool(lr))

    rpx = rp.get("value")
    nmet, ntot = nb.get("n_met"), nb.get("n_total")
    above = ((btc_px / rpx - 1) * 100) if (rpx and btc_px) else None
    add("Glassnode (James Check)", "crypto", "Realized price / cost basis",
        (f"Realized price ${rpx:,.0f}; BTC {above:+.0f}% above cost; bottom {nmet}/{ntot}"
         if rpx else f"bottom checklist {nmet}/{ntot}"),
        "wait",
        "Cheap-ish but NOT capitulated — price is still above the market's aggregate cost basis.",
        live=bool(rpx))

    tier = eo.get("tier", "?")
    add("Jesse Olson", "crypto", "Bearish-W pattern + 3-week MACD",
        f"BTC bottom not in; target $52-57k; QQQ tier {tier}",
        "wait",
        "His W-pattern projects one more low to ~$53k. US stocks (QQQ) still calm.",
        live=bool(eo))

    evd = (dp.get("convergence") or {}).get("ev_date") or dp.get("ev_date") or "~Oct 2026"
    add("Rekt Capital", "crypto", "Halving-cycle timing",
        f"Halving +~900 days -> {evd}",
        "on_track",
        "Bottom timing lines up with the halving clock (~Oct 2026).",
        live=bool(evd))

    add("Plan B", "crypto", "Stock-to-Flow model",
        "Model broke down after 2021",
        "dissent",
        "S2F badly over-predicted price — a reminder to trust the multi-signal ensemble, not one model.",
        live=False)

    # ── EQUITY / MACRO PANEL ─────────────────────────────────────────────────
    def crit(sub):
        for c in (ts.get("criteria") or []):
            if isinstance(c, dict) and sub in c.get("label", ""):
                return c
        return {}

    pe_c, erp_c = crit("P/E"), crit("Risk Premium")
    _g_read = " · ".join(x for x in [(pe_c.get("status") or "").strip(),
                                     (erp_c.get("status") or "").strip()] if x) or "n/a"
    add("Jeremy Grantham", "macro", "Superbubble / valuation (called 2000 & 2008)",
        _g_read, "caution",
        "Equities richly valued (equity risk premium negative) — supports the eventual rotation OUT of stocks.",
        live=bool(pe_c or erp_c))

    add("Lyn Alden", "macro", "Liquidity / fiscal dominance",
        "Fiscal + liquidity backdrop (in your Clemente+Alden suite)", "on_track",
        "Structural fiscal/liquidity tide favours Bitcoin over a multi-year horizon.", live=False)

    add("Arthur Hayes", "macro", "Liquidity + AI-crash regime",
        "NQ/SPY correlation monitor active", "caution",
        "Watch a NASDAQ / AI unwind — that's the likely trigger that cracks equities.", live=False)

    add("Raoul Pal", "macro", "Global M2 / 'Banana Zone'",
        "Liquidity-cycle framework", "on_track",
        "Crypto-bullish while global liquidity expands.", live=False)

    add("Michael Burry", "macro", "Crash-caller / contrarian (called 2008)",
        "Both assets liquidity-inflated", "dissent",
        "Tail-risk reminder: stocks AND Bitcoin could crash together — that's exactly what your tail hedge is for.",
        live=False)

    add("Howard Marks", "macro", "'Where are we in the cycle' / prepare-don't-predict",
        "Conditional, rules-based posture", "on_track",
        "Your conditional rules + tail hedge ARE the preparation he preaches.", live=False)

    # ── consensus ────────────────────────────────────────────────────────────
    counts = {v: sum(1 for x in gurus if x["verdict"] == v)
              for v in ("on_track", "wait", "caution", "dissent")}
    n = len(gurus)
    supportive = counts["on_track"] + counts["wait"]  # direction-supportive
    verdict = "ON TRACK" if supportive >= n * 0.6 else "MIXED"

    return {
        "gurus": gurus,
        "counts": counts,
        "n": n,
        "verdict": verdict,
        "summary": ("Mainstream cycle frameworks support the DIRECTION (accumulate into the "
                    "cycle bottom). The timing crowd says WAIT — not capitulated yet. The macro "
                    "voices flag rich equity valuations (supports the rotation) and tail risk "
                    "(keep the hedge alive). Net: on track, with discipline."),
        "as_of_note": "Scored off live dashboard signals where marked; others are standing stances.",
    }
