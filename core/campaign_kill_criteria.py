"""Campaign kill-criteria + cycle-death monitor.

Intellectual honesty for a multi-year thesis: decide IN ADVANCE what would prove
the "rotate equity -> BTC at the cycle bottom (~$53k, ~Oct 2026)" campaign WRONG,
then watch for it. A green dashboard is comforting; pre-registered falsification
is what keeps you honest. NOT investment advice.

Each criterion -> INTACT / WARNING / TRIPPED. Overall: THESIS INTACT / WATCH / BROKEN.
"""
from __future__ import annotations
from datetime import datetime, timezone


def _btc_price_and_ath():
    """Live BTC price + all-time high (max daily close)."""
    px, ath = 0.0, 0.0
    try:
        import yfinance as yf
        h = yf.Ticker("BTC-USD").history(period="max", interval="1d")["Close"].dropna()
        if len(h):
            ath = float(h.max())
            px = float(h.iloc[-1])
    except Exception:
        pass
    try:
        from core import data
        live = data.btc_spot()  # region-resilient (Kraken/Coinbase/Binance/Bitstamp)
        if live > 0:
            px = live
            ath = max(ath, live)
    except Exception:
        pass
    return px, ath


def campaign_thesis_check() -> dict:
    from core.dashboard_cache import get_cached as g

    px, ath = _btc_price_and_ath()
    nb = g("btc_native_bottom_scorecard") or {}
    bot_n = nb.get("n_met") or 0
    bot_tot = nb.get("n_total") or 16
    dp = g("date_predictions") or {}
    ev_date = (dp.get("convergence") or {}).get("ev_date") or dp.get("ev_date")
    rp = g("realized_price") or {}
    rpx = rp.get("value")

    crits = []

    def add(name, status, detail, current=""):
        crits.append({"name": name, "status": status, "detail": detail, "current": current})

    # 1. NEW ATH BEFORE THE BOTTOM -> "the bottom is still ahead" premise is dead
    if ath and px:
        ratio = px / ath
        status = ("TRIPPED" if ratio >= 0.98 else "WARNING" if ratio >= 0.85 else "INTACT")
        add("New ATH before the $52-57k retest", status,
            "If BTC prints a new all-time high before the bottom retest, we're in a fresh bull leg, "
            "not pre-bottom — the entire 'buy the cycle bottom' premise is wrong; switch to the exit plan.",
            f"BTC ${px:,.0f} = {ratio*100:.0f}% of ATH ${ath:,.0f}")

    # 2. TIMING MODEL BLOWN -> well past projected bottom with no capitulation
    days_past = None
    if ev_date:
        try:
            evd = datetime.strptime(str(ev_date)[:10], "%Y-%m-%d").date()
            days_past = (datetime.now(timezone.utc).date() - evd).days
        except Exception:
            pass
    if days_past is not None:
        confirmed = bot_n >= max(6, bot_tot // 2)
        status = ("TRIPPED" if (days_past > 120 and not confirmed)
                  else "WARNING" if (days_past > 0 and not confirmed) else "INTACT")
        add("Bottom-timing model blown", status,
            "If we blow >120 days past the projected bottom date with no capitulation, the "
            "three-way convergence timing model is wrong — re-derive the date or abandon it.",
            f"{days_past:+d} days vs projected bottom ({str(ev_date)[:10]}); scorecard {bot_n}/{bot_tot}")

    # 3. 4-YEAR CYCLE MUTATING (ETF-muting / institutionalisation)
    try:
        from core.rotation_validation import cycle6_modifier
        cyc = cycle6_modifier() or {}
        era = cyc.get("era", "?")
        scale = cyc.get("suggested_scale", 1.0)
        status = "WARNING" if era == "ETF_MUTED" else "INTACT"
        add("4-year cycle mutating (ETF era)", status,
            "If ETF/institutional flows keep compressing the cycle, the historical halving playbook "
            "the whole campaign rests on may be breaking — bottoms get shallower and signals never "
            "reach historical extremes. Already auto-scaling thresholds x{:.2f}.".format(scale),
            f"era: {era}, threshold scale x{scale:.2f}")
    except Exception:
        pass

    # 4. BOTTOM MAY HAVE BEEN MISSED -> strong recovery without confirmation
    if rpx and px:
        above = px / rpx - 1
        status = "WARNING" if (above > 0.60 and bot_n < 6) else "INTACT"
        add("Bottom may have already passed", status,
            "If BTC sits well above the market's cost basis and is rising, yet the bottom scorecard "
            "never confirmed, the low may have been shallower than expected and already happened — "
            "the BTC-LED BACKSTOP is the safety net for exactly this.",
            f"BTC {above*100:+.0f}% above realized price ${rpx:,.0f}; scorecard {bot_n}/{bot_tot}")

    # 5. THESIS DECAY BY ATTRITION — the shallow grind that dissolves a point-bottom
    #    thesis (the likeliest ETF-era failure, and the one the other 4 don't catch).
    #    Capped at WARNING on purpose: a grind means CHANGE TACTICS (DCA the band),
    #    not "thesis falsified" — so it must NOT flip the campaign to BROKEN (which
    #    would block the time-based deploy that's the correct response to a grind).
    try:
        import yfinance as yf
        w = yf.Ticker("BTC-USD").history(period="2y", interval="1wk")["Close"].dropna()
        if len(w) >= 30:
            recent = w.tail(39)                                   # ~9 months
            rng = float(recent.max() / recent.min() - 1)
            no_new_ath = (float(recent.max()) < ath * 0.98) if ath else True
            rets = w.pct_change().dropna()
            vol_recent = float(rets.tail(13).std())
            vol_prior = float(rets.tail(26).head(13).std())
            vol_falling = vol_recent < vol_prior
            weeks_mid = int(((recent >= 48_000) & (recent <= 82_000)).sum())
            grind = (weeks_mid >= 26 and rng < 0.60 and no_new_ath
                     and vol_falling and bot_n < 5)
            status = "WARNING" if grind else "INTACT"
            add("Thesis decay by attrition (shallow grind)", status,
                "The likeliest ETF-era failure is NOT a clean $52k capitulation - it's a shallow "
                "grind that never fires a bottom signal. If BTC range-trades mid-band ~9 months "
                "with falling volatility and no scorecard escalation, the point-bottom thesis is "
                "EXPIRING: switch to DCA-the-band (the time-based deploy), stop waiting for a "
                "confirmation that may never come. This is a TACTIC change, not a buy/no-buy call.",
                f"{weeks_mid}/39 wk mid-band, range {rng*100:.0f}%, "
                f"vol {'falling' if vol_falling else 'rising'}, scorecard {bot_n}/{bot_tot}")
    except Exception:
        pass

    n_tripped = sum(1 for c in crits if c["status"] == "TRIPPED")
    n_warning = sum(1 for c in crits if c["status"] == "WARNING")
    verdict = ("THESIS BROKEN" if n_tripped else "WATCH" if n_warning else "THESIS INTACT")
    color = ("#ef4444" if n_tripped else "#f0b90b" if n_warning else "#22c55e")

    return {
        "criteria": crits,
        "n_tripped": n_tripped,
        "n_warning": n_warning,
        "verdict": verdict,
        "color": color,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    r = campaign_thesis_check()
    print(f"VERDICT: {r['verdict']}  ({r['n_tripped']} tripped, {r['n_warning']} warning)")
    for c in r["criteria"]:
        print(f"  [{c['status']:<8}] {c['name']}")
        print(f"             {c['current']}".encode("ascii", "replace").decode())
