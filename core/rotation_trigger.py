"""Single-shot equity→BTC rotation trigger.

Fires when any one of three paths confirms. On fire: sell all equity,
buy BTC with the proceeds, same trading window.

Paths:

  PATH 1 (BTC-led) — BTC overwhelmingly bottoming
    - BTC bottom scorecard >= 8/16
    - BTC price <= $57k (in/below Olson target zone, near Swift's 200wMA)

  PATH 2 (Equity-led) — Forced rotation from equity crash
    - QQQ closes below 589 for 2 consecutive sessions
    - Equity top scorecard >= 5/10 hard criteria

  PATH 3 (Balanced) — Moderate confirmation on both sides
    - BTC bottom scorecard >= 6/16
    - AND any one of:
       * QQQ Olson tier >= CAUTION
       * QQQ price < 200-day SMA
       * Equity top scorecard >= 3/10

Discipline:
  - 2-of-2 (or 3-of-3 in P3) means no single signal can fire alone
  - Three paths cover scenarios where one side leads
  - Historical backtest fires ~1x per 3-4 years on real bottoms
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Baseline thresholds (historical cycle calibration — 2018 & 2022)
BTC_BOTTOM_OVERWHELMING = 8     # Path 1: BTC scorecard baseline threshold
BTC_PRICE_TARGET = 57_000        # Path 1: BTC price ceiling (Olson upper band) — NOT scaled
QQQ_GAP_LEVEL = 589              # Path 2: QQQ trapdoor (Olson) — NOT scaled
QQQ_BREAK_DAYS = 2               # Path 2: consecutive sessions below 589 required
EQUITY_TOP_HARD = 5              # Path 2: macro scorecard threshold
BTC_BOTTOM_MODERATE = 6          # Path 3: BTC scorecard moderate threshold
EQUITY_TOP_LIGHT = 3             # Path 3: equity scorecard moderate threshold

# QQQ Olson tier ladder — CAUTION or worse triggers Path 3 equity side
QQQ_OLSON_BAD_TIERS = {"CAUTION", "WATCH", "GAP_BROKEN", "RETESTING"}


def _cycle_scaled_thresholds() -> dict:
    """Auto-scale BTC scorecard thresholds based on cycle-6 ETF detector.

    If cycle is ETF_MUTED, the historical 8/16 'overwhelming' baseline never
    fires because the drawdown is shallower than past bears. Scale thresholds
    DOWN to match the muted cycle's signal density.

    Returns dict with scaled thresholds + scale factor.
    """
    try:
        from core.rotation_validation import cycle6_modifier
        cyc = cycle6_modifier()
        scale = cyc.get("suggested_scale", 1.0)
        era = cyc.get("era", "HISTORICAL_BEAR")
    except Exception:
        scale = 1.0
        era = "UNKNOWN"

    return {
        "scale":              scale,
        "era":                era,
        "btc_overwhelming":   max(1, round(BTC_BOTTOM_OVERWHELMING * scale)),
        "btc_moderate":       max(1, round(BTC_BOTTOM_MODERATE * scale)),
        # Equity scorecard thresholds also scale (macro is also muted in ETF era)
        "equity_hard":        max(1, round(EQUITY_TOP_HARD * scale)),
        "equity_light":       max(1, round(EQUITY_TOP_LIGHT * scale)),
    }

# Portfolio assumptions for the action email (user's NZ stake)
TOTAL_STAKE_NZD = 130_000
EQUITY_PCT_START = 0.70
CASH_RESERVE_PCT = 0.05  # keep 5% as emergency reserve
NZD_USD_RATE = 0.59       # rough — recompute if needed


def _qqq_recent_closes(n: int = 5):
    """Last N daily closes of QQQ via yfinance."""
    try:
        import yfinance as yf
        qqq = yf.Ticker("QQQ")
        h = qqq.history(period="10d", interval="1d")
        if h.empty: return []
        # dropna: yfinance returns NaN for partial/in-progress sessions, which
        # would poison the "below 589 for 2 sessions" comparison
        return h["Close"].dropna().tail(n).tolist()
    except Exception:
        return []


def _btc_price() -> float:
    try:
        from core import data
        return data.btc_spot()
    except Exception:
        return 0.0


# Honest deploy BAND (wider than the 52-57k point estimate) + time-deploy params
BTC_BAND_LO = 45_000
BTC_BAND_HI = 58_000
TIME_DEPLOY_WEEKS = 8     # weeks inside the band before a time-based starter tranche


def _weeks_in_band(lo: int = BTC_BAND_LO, hi: int = BTC_BAND_HI) -> int:
    """Consecutive most-recent weekly closes inside the deploy band."""
    try:
        import yfinance as yf
        w = yf.Ticker("BTC-USD").history(period="1y", interval="1wk")["Close"].dropna()
        n = 0
        for c in reversed(list(w)):
            if lo <= float(c) <= hi:
                n += 1
            else:
                break
        return n
    except Exception:
        return 0


def _fast_equity_stress():
    """Fast credit/vol early-warning that LEADS the slow 200dMA/MACD legs: VIX
    term structure inverted (VIX > VIX3M) OR HY credit selling off hard
    (HYG -3%/20d). Counts toward WARMING only — never toward execute. (bool, detail)."""
    try:
        import yfinance as yf
        def cl(t, p="3mo"):
            return yf.Ticker(t).history(period=p)["Close"].dropna()
        vix, vix3m = cl("^VIX"), cl("^VIX3M")
        inverted = bool(len(vix) and len(vix3m) and vix.iloc[-1] > vix3m.iloc[-1])
        hyg = cl("HYG", "2mo")
        hy_stress = bool(len(hyg) > 21 and (hyg.iloc[-1] / hyg.iloc[-21] - 1) < -0.03)
        bits = []
        if inverted:  bits.append("VIX term inverted")
        if hy_stress: bits.append("HY credit selling off")
        return (inverted or hy_stress), (", ".join(bits) if bits else "calm")
    except Exception:
        return False, "n/a"


def btc_deploy_plan(btc_px: float, kill_broken: bool = False) -> dict:
    """Banded scale-in plan for the BTC leg — replaces the all-or-nothing backstop.

    Combines price-in-band, theme-breadth confirmation, a price-turn, and a
    TIME-based starter (>= N weeks in band past the cycle window) so a SHALLOW
    grind that never triggers a capitulation signal still gets deployed instead
    of waiting for confirmation forever. Advisory fraction; operator executes."""
    from core.dashboard_cache import get_cached
    in_band = (btc_px > 0 and BTC_BAND_LO <= btc_px <= BTC_BAND_HI)
    below_band = (btc_px > 0 and btc_px < BTC_BAND_LO)   # cheaper than band = even better

    bc = get_cached("bottom_confirmation") or get_cached("btc_bottom_scorecard") or {}
    sc = bc.get("scorecard", bc) if isinstance(bc, dict) else {}
    themes_met = sc.get("themes_met") or 0
    deploy_level = sc.get("deploy_level")
    momentum = bool(sc.get("momentum_met"))

    weeks = _weeks_in_band()
    try:
        from core.halving_clock import current_halving_position
        days_post = current_halving_position().get("days_post_halving") or 0
    except Exception:
        days_post = 0
    past_window = days_post >= 850
    time_deploy = ((in_band or below_band) and weeks >= TIME_DEPLOY_WEEKS
                   and past_window and not kill_broken)

    frac, reason = 0, "wait — not in band / no confirmation"
    if in_band or below_band:
        if deploy_level == "DEPLOY" and momentum:
            frac, reason = 100, "full breadth + price turned + in band"
        elif themes_met >= 3 and (momentum or past_window):
            frac, reason = 60, "broad confirmation + a timing signal"
        elif themes_met >= 2:
            frac, reason = 30, "cheap + early confirmation (first tranche)"
        elif time_deploy:
            frac, reason = 30, f"TIME-based: {weeks}w in band, clock past window, thesis intact"
    elif time_deploy:
        frac, reason = 30, f"TIME-based starter: {weeks}w in band past window"

    return {"fraction_pct": frac, "reason": reason, "in_band": in_band,
            "below_band": below_band, "weeks_in_band": weeks, "themes_met": themes_met,
            "momentum": momentum, "past_cycle_window": past_window, "time_deploy": time_deploy}


def evaluate_rotation_trigger() -> dict:
    """EQUITY-PRIORITY single-shot rotation trigger.

    Per user requirement: priority is AVOIDING equity drawdown, not catching
    the absolute BTC bottom. Fires on EARLY equity-weakness confirmation.

    Logic: 2-of-4 equity stress signals fire → ALL-IN rotation
       (sell all equity → same-day buy all BTC, no cash in middle)

    The 4 signals are independent equity-side indicators:
      1. QQQ Olson tier reaches CAUTION or worse
      2. Equity top scorecard hits cycle-6-scaled threshold (>=2/10)
      3. QQQ 3-week MACD bearish cross
      4. QQQ closes below 200-day SMA

    User accepts the cost: BTC may not be at perfect bottom when rotation
    fires. The asymmetry math (equity DD impact >> BTC timing impact)
    favors firing early on equity weakness rather than waiting for BTC.
    """
    from core.dashboard_cache import get_cached

    # --- Build 1: Apply cycle-6 era scaling auto-magically ---
    scaled = _cycle_scaled_thresholds()
    threshold_btc_overwhelming = scaled["btc_overwhelming"]
    threshold_btc_moderate = scaled["btc_moderate"]
    threshold_equity_hard = scaled["equity_hard"]
    threshold_equity_light = scaled["equity_light"]
    cycle_era = scaled["era"]
    cycle_scale = scaled["scale"]

    # --- Pull all required signals ---
    # BTC bottom scorecard
    nb = get_cached("btc_native_bottom_scorecard") or {}
    btc_bot_n = nb.get("n_met") or 0
    btc_bot_total = nb.get("n_total") or 16
    btc_bot_level = nb.get("verdict_level") or "?"

    # BTC live price
    btc_px = _btc_price()

    # QQQ recent closes (for 2-day break check)
    qqq_closes = _qqq_recent_closes(5)
    qqq_now = qqq_closes[-1] if qqq_closes else 0
    qqq_2d_below_589 = (
        len(qqq_closes) >= 2 and
        qqq_closes[-1] < QQQ_GAP_LEVEL and
        qqq_closes[-2] < QQQ_GAP_LEVEL
    )

    # Equity top scorecard
    ts = get_cached("top_scorecard") or {}
    eq_sc = ts.get("scorecard", {}) if isinstance(ts, dict) else {}
    eq_n = eq_sc.get("n_met") or 0
    eq_total = eq_sc.get("n_total") or 10
    eq_verdict = eq_sc.get("verdict_level") or "HOLD"

    # QQQ Olson tier + 200dMA
    eq_ol = get_cached("equity_olson") or {}
    qqq_tier = eq_ol.get("tier") or "?"
    qqq_dma200 = eq_ol.get("dma200") or 0
    qqq_below_dma200 = (qqq_now > 0 and qqq_dma200 > 0 and qqq_now < qqq_dma200)

    # ─── NEW EQUITY-PRIORITY LOGIC ───
    # 4 independent equity-stress signals. 2/4 = all-in rotate.
    # BTC bottom signals are IGNORED for the fire decision (user priority).

    # Pull QQQ 3-week MACD from equity_olson cache
    macd_3w = (eq_ol.get("signals", {}) or {}).get("macd_3w", {}) or {}
    qqq_macd_bearish = not bool(macd_3w.get("bullish", True))

    # 4 conditions (all equity-side)
    c1 = qqq_tier in QQQ_OLSON_BAD_TIERS              # Olson tier
    c2 = eq_n >= threshold_equity_light               # Macro scorecard (cycle-scaled)
    c3 = qqq_macd_bearish                              # 3-week MACD bearish
    c4 = qqq_below_dma200                              # Below 200dMA

    n_firing = int(c1) + int(c2) + int(c3) + int(c4)
    equity_fired = n_firing >= 2

    # ── BTC-LED BACKSTOP (added 2026-06-12) ──────────────────────────────────
    # Equity-priority is deliberate, but on its own it would MISS the BTC bottom
    # entirely if equities never crack (the AI-melt-up case). This backstop fires
    # a rotation when BTC actually reaches its bottom zone AND the bottom
    # scorecard moderately confirms — regardless of equity state — so a strong
    # stock market can't cost us the cycle bottom we've waited years for.
    # Conservative: BTC (~$63.7k now) is far above the <=$57k zone, so dormant.
    # Banded scale-in plan + kill status + fast warning leg (all additive)
    try:
        from core.campaign_kill_criteria import campaign_thesis_check
        kill_broken = (campaign_thesis_check().get("verdict") == "THESIS BROKEN")
    except Exception:
        kill_broken = False
    deploy_plan = btc_deploy_plan(btc_px, kill_broken=kill_broken)
    fast_stress, fast_detail = _fast_equity_stress()

    btc_in_zone   = (btc_px > 0 and btc_px <= BTC_PRICE_TARGET)        # <= $57k
    # Backstop now fires on EITHER the native-16 count OR the theme-breadth FULL
    # deploy (price-turned + 4 orthogonal themes) — two independent confirmation
    # methods (OR), looser than the old single-count AND so a muted cycle that
    # confirms by breadth-not-depth can't be missed.
    btc_confirmed = (btc_bot_n >= threshold_btc_overwhelming) or (deploy_plan["fraction_pct"] >= 100)
    btc_backstop_fired = btc_in_zone and btc_confirmed

    any_fired = equity_fired or btc_backstop_fired
    firing_paths = []
    if equity_fired:       firing_paths.append("EQUITY-PRIORITY ROTATION")
    if btc_backstop_fired: firing_paths.append("BTC-LED BACKSTOP")

    # Build single "path" for backward-compat with downstream code
    p1 = {
        "name":     "EQUITY-PRIORITY ROTATION (2-of-4 equity signals)",
        "fired":     any_fired,
        "score":     f"{n_firing}/4",
        "conditions": [
            {"label":   "QQQ Olson tier reaches CAUTION or worse",
              "current": f"tier: {qqq_tier}",
              "met":     c1},
            {"label":   f"Equity top scorecard >= {threshold_equity_light}/10"
                         + (f" (scaled from {EQUITY_TOP_LIGHT})"
                              if threshold_equity_light != EQUITY_TOP_LIGHT else ""),
              "current": f"{eq_n}/{eq_total}",
              "met":     c2},
            {"label":   "QQQ 3-week MACD bearish cross",
              "current": f"trend: {macd_3w.get('trend', '?')}",
              "met":     c3},
            {"label":   f"QQQ closes below 200-day SMA (${qqq_dma200:,.0f})",
              "current": f"${qqq_now:,.0f}",
              "met":     c4},
        ],
    }
    # Path 2 is now LIVE: the BTC-led backstop (was retired).
    p2 = {
        "name":  "BTC-LED BACKSTOP (catch the bottom even if equities stay calm)",
        "fired":  btc_backstop_fired,
        "score":  f"{int(btc_in_zone) + int(btc_confirmed)}/2",
        "conditions": [
            {"label":   f"BTC at/below Olson bottom zone (<= ${BTC_PRICE_TARGET:,})",
              "current": f"${btc_px:,.0f}", "met": btc_in_zone},
            {"label":   f"BTC bottom scorecard >= {threshold_btc_moderate} (cycle-scaled)",
              "current": f"{btc_bot_n}/{btc_bot_total}", "met": btc_confirmed},
        ],
    }
    p3 = {"name": "(retired)", "fired": False, "score": "n/a", "conditions": []}

    best_score = n_firing
    if any_fired:
        overall = "FIRED"
        color = "#ef4444"
    elif n_firing >= 1 or btc_in_zone or fast_stress or deploy_plan["fraction_pct"] > 0:
        overall = "WARMING"
        color = "#f0b90b"
    else:
        overall = "ARMED"
        color = "#22c55e"

    # NZ$ rotation amounts (for the action email)
    equity_to_sell_nzd = int(TOTAL_STAKE_NZD * EQUITY_PCT_START)
    cash_reserve_nzd = int(TOTAL_STAKE_NZD * CASH_RESERVE_PCT)
    btc_to_buy_nzd = equity_to_sell_nzd - cash_reserve_nzd
    # USD equivalents (rough — show user both)
    equity_usd = int(equity_to_sell_nzd * NZD_USD_RATE)
    btc_usd = int(btc_to_buy_nzd * NZD_USD_RATE)

    # --- Build 2/3/4: Compute confidence + effective signal count for output ---
    try:
        from core.rotation_validation import confidence_score, signal_correlation
        conf = confidence_score()
        corr = signal_correlation()
        confidence_pct = conf.get("confidence_pct", 0)
        confidence_tier = conf.get("tier", "?")
        effective_signals = corr.get("clusters_firing", "0/6")
    except Exception:
        confidence_pct = 0
        confidence_tier = "?"
        effective_signals = "?"

    return {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "overall":         overall,
        "color":           color,
        "fired":           any_fired,
        "firing_paths":    firing_paths,
        "best_score":      best_score,
        "paths":           [p1, p2, p3],
        # --- Build 1: Cycle scaling applied (auto) ---
        "cycle_era":         cycle_era,
        "cycle_scale":       cycle_scale,
        "scaled_thresholds": {
            "btc_overwhelming":   threshold_btc_overwhelming,
            "btc_moderate":       threshold_btc_moderate,
            "equity_hard":        threshold_equity_hard,
            "equity_light":       threshold_equity_light,
        },
        # --- Build 2/3: Confidence + effective signal count ---
        "confidence_pct":    confidence_pct,
        "confidence_tier":   confidence_tier,
        "effective_signals": effective_signals,
        # Action instructions if fired
        "action": {
            "sell_equity_nzd":     equity_to_sell_nzd,
            "sell_equity_usd":     equity_usd,
            "cash_reserve_nzd":    cash_reserve_nzd,
            "buy_btc_nzd":         btc_to_buy_nzd,
            "buy_btc_usd":         btc_usd,
            "btc_amount_estimate": btc_usd / btc_px if btc_px > 0 else 0,
        },
        # Live values
        "btc_price":         btc_px,
        "btc_bottom_n":      btc_bot_n,
        "btc_bottom_total":  btc_bot_total,
        "qqq_close":         qqq_now,
        "qqq_dma200":        qqq_dma200,
        "qqq_tier":          qqq_tier,
        "equity_top_n":      eq_n,
        "equity_top_total":  eq_total,
        # banded scale-in deploy plan (replaces all-or-nothing) + fast warning leg
        "deploy_plan":       deploy_plan,
        "fast_stress":       fast_stress,
        "fast_stress_detail": fast_detail,
    }


# ── Historical lead-time playbook ───────────────────────────────────────────
# Backtested on real QQQ daily data at the last three NASDAQ tops. FIXED facts:
#   EXECUTE = earlier of {3-week MACD bear, close < 200-day SMA}
#   WARMING = weekly RSI bearish divergence (the earlier, less-reliable tell)
# exec_off/warn_off = % below the prior peak when each fired; dodged = the
# further fall AVOIDED after execute; lead_days = calendar days before the low.
LEADTIME_EPISODES = [
    {"year": "2022", "kind": "slow grind bear", "total_dd": -35,
     "warn_off": 0,    "exec_off": -6,  "dodged": -31, "lead_days": 298},
    {"year": "2018", "kind": "ordinary bear",   "total_dd": -23,
     "warn_off": None, "exec_off": -9,  "dodged": -15, "lead_days": 74},
    {"year": "2020", "kind": "crash (COVID)",   "total_dd": -29,
     "warn_off": -1,   "exec_off": -18, "dodged": -13, "lead_days": 7},
]
# 2026-07-07 claim-validity audit: LEADTIME_EPISODES above are HAND-COMPILED
# historical constants, NOT a reproducible backtest — nothing here re-simulates
# QQQ history. Labeled honestly so the rotation email doesn't present n=3
# stylized numbers as measured edge.
LEADTIME_SOURCE = ("Illustrative, not a backtest — hand-compiled from 3 past "
                   "NASDAQ tops (2018/2020/2022), in-sample, n=3. A guide to "
                   "shape, not a measured edge. Past performance is no guarantee.")


def leadtime_context(status: str) -> dict:
    """Historical lead-time narrative tailored to the live trigger status.
    Accepts ARMED/WARMING/FIRED or the Simpleton WAITING/GETTING READY/TIME TO ACT."""
    s = (status or "").upper()
    fired = s in ("FIRED", "TIME TO ACT")
    warming = s in ("WARMING", "GETTING READY")
    if fired:
        headline = "What this has meant historically"
        lead = ("At past tops the full signal rotated you out after the first ~6-18% of the "
                "fall, then avoided a further 13-31% of the decline, weeks to months before the "
                "bottom. It does not catch the exact top - it dodges the deep part.")
        plain = ("In plain terms: you've probably given up the first part of the drop, but "
                 "side-stepped the much bigger fall that usually follows.")
    elif warming:
        headline = "What an early warning has meant historically"
        lead = ("This is the early tell. At past tops it pinned the peak within days (2022: to "
                "the day; COVID: 3 days before) and bought 3-7 extra weeks of notice - but it is "
                "NOT certain: there was no clean signal in 2018, and it sometimes fizzles. "
                "Eyes-up, not act-now; the full 2-of-4 signal still has to confirm.")
        plain = ("In plain terms: this often shows up right at the top and gives a few weeks of "
                 "notice - but it's only a heads-up, and sometimes it fades with no drop at all. "
                 "Watch closely; don't act yet.")
    else:
        headline = "What will happen when this fires"
        lead = ("Nothing to do while this is green. When it fires, history says you'd exit after "
                "the first ~6-18% and dodge a further 13-31% - earlier in a slow bear, only just "
                "in time in a crash (where the tail-hedge matters more).")
        plain = ("Nothing to do right now. When the plan fires, in the past it got you out before "
                 "the worst of the fall - sooner in a slow decline, barely in time in a crash.")
    return {"status": s, "fired": fired, "warming": warming,
            "headline": headline, "lead": lead, "plain": plain,
            "episodes": LEADTIME_EPISODES, "source": LEADTIME_SOURCE}


def leadtime_email_block(status: str) -> str:
    """Plain-text (ASCII) historical-context block for trigger emails."""
    ctx = leadtime_context(status)
    out = [
        "--- HISTORICAL CONTEXT (what this has meant before) ---",
        "  " + ctx["lead"],
        "",
        "  Past NASDAQ tops (real QQQ backtest):",
    ]
    for e in ctx["episodes"]:
        tell = "no early tell" if e["warn_off"] is None else f"early tell ~{e['warn_off']:+d}% off peak"
        out.append(
            f"    {e['year']} ({e['kind']}): execute {e['exec_off']:+d}% off peak, "
            f"avoided a further {e['dodged']:+d}%, {e['lead_days']}d before the low; {tell}.")
    out += ["", "  " + ctx["source"]]
    return "\n".join(out)


def action_email_body(state: dict) -> str:
    """Build the BIG email body when trigger fires.
    Includes validation summary (cycle era, scale, confidence, effective signals)."""
    a = state.get("action", {}) or {}
    btc_px = state.get("btc_price", 0)
    fired = state.get("firing_paths", [])
    body = [
        f"========================================================",
        f"   ROTATION TRIGGER FIRED — EXECUTE TODAY",
        f"========================================================",
        f"",
        f"  Trigger path(s): {', '.join(fired)}",
        f"  Timestamp:       {state.get('ts', '')[:16]}",
        f"  BTC price:       ${btc_px:,.0f}",
        f"  Confidence:      {state.get('confidence_pct', 0):.0f}% "
            f"({state.get('confidence_tier', '?')})",
        f"  Effective signals: {state.get('effective_signals', '?')} clusters firing",
        f"  Cycle era:       {state.get('cycle_era', '?')} "
            f"(thresholds x {state.get('cycle_scale', 1.0):.2f})",
        f"",
        f"--- LEG 1: SELL EQUITY ---",
        f"  Amount:    NZ${a.get('sell_equity_nzd', 0):,} "
            f"(US${a.get('sell_equity_usd', 0):,})",
        f"  Action:    Market-sell ALL equity holdings via your NZ broker.",
        f"             Hold proceeds in broker cash temporarily.",
        f"",
        f"--- LEG 2: BUY BTC ---",
        f"  Amount:    NZ${a.get('buy_btc_nzd', 0):,} "
            f"(US${a.get('buy_btc_usd', 0):,})",
        f"  Estimate:  ~{a.get('btc_amount_estimate', 0):.4f} BTC at ${btc_px:,.0f}",
        f"  Action:    Transfer USD via Wise/Hatch to Binance.",
        f"             Market-buy BTC in 2-3 orders over 6h for execution avg.",
        f"  Reserve:   Keep NZ${a.get('cash_reserve_nzd', 0):,} as emergency cash.",
        f"",
        f"--- AFTER EXECUTION ---",
        f"  Target allocation: ~93% BTC · ~5% cash · ~2% existing positions",
        f"",
        f"=== WHY THIS FIRED (live state) ===",
        f"  BTC bottom scorecard:   {state.get('btc_bottom_n')}/{state.get('btc_bottom_total')}",
        f"  Equity top scorecard:   {state.get('equity_top_n')}/{state.get('equity_top_total')}",
        f"  QQQ:                    ${state.get('qqq_close', 0):,.0f}",
        f"  QQQ Olson tier:         {state.get('qqq_tier')}",
        f"",
        leadtime_email_block("FIRED"),
        f"",
        f"Open dashboard for full breakdown:",
        f"  Remote: https://btcdelight.com",
        f"",
        f"!!! Do not over-think. Execute today within trading window. !!!",
    ]
    return "\n".join(body)


def main():
    s = evaluate_rotation_trigger()
    print(f"OVERALL: {s['overall']} (best score {s['best_score']}/2)")
    print(f"Fired: {s['fired']}  Paths firing: {s['firing_paths']}")
    print()
    for p in s["paths"]:
        flag = "[FIRE]" if p["fired"] else "      "
        print(f"  {flag} {p['name']}  score={p['score']}")
        for c in p["conditions"]:
            mark = "OK" if c["met"] else "no"
            print(f"        [{mark}] {c['label']}")
            print(f"             current: {c['current']}")
    print()
    if s["fired"]:
        print(action_email_body(s))


if __name__ == "__main__":
    main()
