"""BOTTOM CONFIRMATION SCORECARD — hard-criteria checklist.

The prediction engine can show many bullish signals while the ACTUAL cycle
bottom has not yet occurred. Soft signals (Reserve Risk proxy, halving clock
forward outlook) can fire as soon as the pattern says a bottom is *projected*,
even though hard confirmation signals (Realized Cap drawdown, MVRV-Z deep
value, Coinbase Premium positive, hashrate ribbon cross-up) have not.

This module produces a no-soft-bullshit scorecard: how many HARD criteria are
actually met right now? An actual cycle bottom historically requires most of these. The tier ladder
(2026: 10 criteria) is 3=EARLY, 5=BOTTOM_FORMING, 7=BOTTOM_IN. NOTE the
gate now also reports n_firm / n_mechanisms_firm — the count that EXCLUDES
hair-over-the-line criteria and dedups the 2 hashrate + 2 premium pairs.
Capital deployment keys off firm mechanisms, not the raw met count, so the
deploy gate does not flicker on daily BTC noise.
"""

from __future__ import annotations

from typing import Optional

import json
from datetime import datetime, timezone
from pathlib import Path

# ── 2026-07-08 bulletproof gate: persistence debounce ────────────────────────
# The raw firm-mechanism count can still nick a threshold for a single day
# (BTC daily vol ~2-3%). To stop the CONFIRMED gate level flickering, a level
# change must PERSIST for GATE_PERSIST_DAYS distinct calendar days before it's
# accepted. State is local runtime (.gate_state.json, gitignored); the accepted
# level rides to the cloud inside the bottom_confirmation panel cache. Fully
# fail-safe: any error -> fall back to the raw level, never crash the scorecard.
_GATE_STATE = Path(__file__).resolve().parent.parent / ".gate_state.json"
GATE_PERSIST_DAYS = 2


def _level_from_mech(n_mech_firm: int) -> str:
    if n_mech_firm >= 6: return "CONFIRMED"
    if n_mech_firm >= 4: return "SCALE_IN"
    if n_mech_firm >= 3: return "EARLY"
    return "NOT_CONFIRMED"


def _gate_debounce(n_mech_firm: int) -> dict:
    """Return the DEBOUNCED gate level. A new level must hold GATE_PERSIST_DAYS
    distinct days before it's accepted; otherwise the last confirmed level
    holds and the pending change is reported. Stateless-safe on any failure."""
    raw = _level_from_mech(n_mech_firm)
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        st = {}
        if _GATE_STATE.exists():
            st = json.loads(_GATE_STATE.read_text()) or {}
        hist = [h for h in st.get("history", []) if h.get("date") != today]
        hist.append({"date": today, "mech_firm": int(n_mech_firm), "level": raw})
        hist = sorted(hist, key=lambda h: h.get("date", ""))[-14:]

        confirmed = st.get("confirmed_level", raw)
        since = st.get("confirmed_since", today)
        pending = False
        days_at_raw = 0
        if raw != confirmed:
            for h in reversed(hist):              # trailing run of days at `raw`
                if h.get("level") == raw: days_at_raw += 1
                else: break
            if days_at_raw >= GATE_PERSIST_DAYS:
                confirmed, since, pending = raw, today, False
            else:
                pending = True                    # building, not yet accepted
        _GATE_STATE.write_text(json.dumps({
            "confirmed_level": confirmed, "confirmed_since": since,
            "history": hist,
        }, indent=1))
        return {"confirmed_level": confirmed, "raw_level": raw,
                "pending_change": pending, "days_at_raw": days_at_raw,
                "confirmed_since": since}
    except Exception as e:
        return {"confirmed_level": raw, "raw_level": raw,
                "pending_change": False, "days_at_raw": 0, "error": str(e)[:60]}




# ============================================================
# Hard criteria for an ACTUAL cycle bottom
# ============================================================

CRITERIA_DEFS = [
    {"id": "price_drawdown", "mechanism": "price", "firm_buffer": 1.0,
     "label": "Price -50% or worse from ATH",
     "rationale": "Historical median cycle drawdown -50% to -85%",
     "threshold": -50.0, "comparator": "less_than"},
    {"id": "realized_cap_drawdown", "mechanism": "realized_cap", "firm_buffer": 1.5,
     "label": "Realized Cap drawdown -15% or worse",
     "rationale": "Checkmate's #1 bottom indicator — coins recapitulated at loss",
     "threshold": -15.0, "comparator": "less_than"},
    {"id": "mvrv_z", "mechanism": "mvrv", "firm_buffer": 0.1,
     "label": "MVRV Z-Score below -1.0",
     "rationale": "Deep value zone — coins trading below realized price avg",
     "threshold": -1.0, "comparator": "less_than"},
    {"id": "coinbase_premium", "mechanism": "cb_premium", "firm_buffer": 5.0,
     # 2026-07-09 sense-check audit: value is in BASIS POINTS and oscillates
     # around zero (+0.8bps at audit time = coin-flip noise). ">0.0" fired on
     # any microscopic tick. Now requires a MEANINGFUL premium: met > +5bps,
     # firm > +10bps.
     "label": "Coinbase Premium > +5bps (meaningful US bid)",
     "rationale": "US institutional buying confirmed",
     "threshold": 5.0, "comparator": "greater_than"},
    {"id": "hashrate_ribbon", "mechanism": "hashrate",
     "label": "Hashrate Ribbon cross-up",
     "rationale": "Woo's signal — miner capitulation done, recovery confirmed",
     "threshold": None, "comparator": "ribbon_cross_up"},
    {"id": "halving_day", "mechanism": "cycle_time", "firm_buffer": 5,
     "label": "Days post-halving between 850-920",
     "rationale": "Historical bottom window (cycle 3: 889d, cycle 4: 912d)",
     "threshold": (850, 920), "comparator": "in_range"},
    {"id": "puell_multiple", "mechanism": "miner_rev", "firm_buffer": 0.05,
     "label": "Puell Multiple below 0.5",
     "rationale": "Miner revenue capitulation",
     "threshold": 0.5, "comparator": "less_than"},
    {"id": "sth_mvrv_reclaim", "mechanism": "sth_mvrv",
     "label": "STH-MVRV reclaim of 1.0 (after extended below)",
     "rationale": "Short-term holders back in profit = recovery confirmed",
     "threshold": 1.0, "comparator": "reclaim"},
    # === Added 2026-06-03 from Clemente+Alden review ===
    {"id": "hashrate_drawdown", "mechanism": "hashrate", "firm_buffer": 2.0,
     "label": "Hashrate drawdown -25% or worse from 365d peak",
     "rationale": "Miner capitulation — every prior bottom required this",
     "threshold": -25.0, "comparator": "less_than"},
    {"id": "cb_premium_streak", "mechanism": "cb_premium", "firm_buffer": 3,
     "label": "Coinbase Premium negative streak 21+ days",
     "rationale": "Clemente's 2024 bottom signal — fired within a week of low",
     "threshold": 21, "comparator": "greater_than"},
]


def _check_criterion(crit: dict, value, extra: Optional[dict] = None) -> tuple[bool, str]:
    """Return (met, status_text) for a criterion against a value."""
    if value is None:
        return False, "data unavailable"
    cmp_type = crit["comparator"]
    th = crit["threshold"]
    if cmp_type == "less_than":
        met = value < th
        return met, f"current: {value:+.2f}, need: <{th}"
    if cmp_type == "greater_than":
        met = value > th
        return met, f"current: {value:+.2f}, need: >{th}"
    if cmp_type == "in_range":
        lo, hi = th
        met = lo <= value <= hi
        return met, f"current: {value:.0f}, need: {lo}-{hi}"
    if cmp_type == "ribbon_cross_up":
        # value is the phase string from hashrate_ribbon_cross signal
        phase = str(value).upper()
        met = "CROSS_UP" in phase or "RECOVERY" in phase
        return met, f"current phase: {phase}, need: cross_up/recovery"
    if cmp_type == "reclaim":
        # value should be {"current": float, "extended_below": bool}
        if not isinstance(extra, dict):
            return False, "structure check unavailable"
        current = extra.get("current", value)
        extended = extra.get("extended_below", False)
        met = current >= th and extended
        status = f"current: {current:.2f}, "
        status += "extended below" if extended else "no extended period below"
        return met, status
    return False, "unknown comparator"


def _criterion_firm(crit: dict, value, met: bool) -> tuple:
    """2026-07-08 gate audit: is a MET criterion firmly met, or a hair over the
    line? Returns (is_firm, is_marginal, margin). Prevents the capital gate
    flickering on daily BTC noise: e.g. price -50.19% (threshold -50) or MVRV
    -1.01 (threshold -1.0) are 'met' but a +0.5% BTC day un-fires them. A
    criterion only counts toward the deploy-unlock when it clears its threshold
    by `firm_buffer`. Non-numeric comparators (ribbon/reclaim) are firm if met.
    """
    if not met:
        return False, False, None
    cmp_type = crit.get("comparator")
    buf = crit.get("firm_buffer")
    th = crit.get("threshold")
    try:
        if cmp_type == "less_than" and buf is not None:
            margin = th - value                 # how far below threshold
            return (margin >= buf), (margin < buf), round(margin, 3)
        if cmp_type == "greater_than" and buf is not None:
            margin = value - th
            return (margin >= buf), (margin < buf), round(margin, 3)
        if cmp_type == "in_range" and buf is not None:
            lo, hi = th
            margin = min(value - lo, hi - value)
            return (margin >= buf), (margin < buf), round(margin, 3)
    except (TypeError, ValueError):
        pass
    return True, False, None   # boolean/structural criteria: firm when met


# ── injected theme inputs (momentum + derivatives) the 10 criteria lack ──────
def _btc_weekly_price_turn():
    """True if BTC has RECLAIMED a rising 10-week EMA — a 'don't catch the knife'
    price-turn confirmation for the momentum theme. None if data unavailable."""
    try:
        import yfinance as yf
        w = yf.Ticker("BTC-USD").history(period="2y", interval="1wk")["Close"].dropna()
        if len(w) < 14:
            return None
        ema = w.ewm(span=10, adjust=False).mean()
        return bool(w.iloc[-1] > ema.iloc[-1] and ema.iloc[-1] > ema.iloc[-4])
    except Exception:
        return None


def _deriv_reset(sigs):
    """True if derivatives have RESET (funding flat/negative). Best-effort; None
    if no funding/OI read available (theme stays 'unknown', not penalised)."""
    for cat in ("flows", "fundamentals", "onchain", "derivatives"):
        d = sigs.get(cat, {}) or {}
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if "funding" in k.lower() and isinstance(v, dict) and not v.get("error"):
                val = v.get("value", v.get("rate"))
                if isinstance(val, (int, float)):
                    return bool(val <= 0.0001)
    return None


def bottom_confirmation_scorecard(state: Optional[dict] = None,
                                  compute_breadth: bool = True) -> dict:
    """Run the 8-point hard-criteria scorecard.

    Returns dict with:
        criteria: list of {id, label, met, value, status, rationale}
        n_met: int
        n_total: int
        verdict: text
        verdict_level: "BOTTOM_IN" | "BOTTOM_FORMING" | "EARLY" | "NO_BOTTOM"
    """
    if state is None:
        from core.btc_prediction import state_of_btc
        state = state_of_btc()

    sigs = state.get("signals", {})
    btc_price = state.get("btc_price", 0)
    cycle5_ath = 124659

    # Extract values for each criterion
    pct_drawdown = (btc_price / cycle5_ath - 1) * 100 if btc_price else None

    rcap_dd = sigs.get("onchain", {}).get("realized_cap_drawdown", {})
    rcap_val = rcap_dd.get("value") if isinstance(rcap_dd, dict) and not rcap_dd.get("error") else None

    mvrvz = sigs.get("onchain", {}).get("mvrv_z_score", {})
    mvrvz_val = mvrvz.get("value") if isinstance(mvrvz, dict) and not mvrvz.get("error") else None

    cb = sigs.get("flows", {}).get("coinbase_premium_gap", {})
    cb_val = cb.get("premium_bps") if isinstance(cb, dict) and not cb.get("error") else None

    hrr = sigs.get("fundamentals", {}).get("hashrate_ribbon_cross", {})
    hrr_phase = hrr.get("phase") if isinstance(hrr, dict) else None

    try:
        from core.halving_clock import current_halving_position
        pos = current_halving_position()
        days_post = pos.get("days_post_halving")
    except Exception:
        days_post = None

    puell = sigs.get("fundamentals", {}).get("puell_multiple", {})
    puell_val = puell.get("value") if isinstance(puell, dict) and not puell.get("error") else None

    sth = sigs.get("onchain", {}).get("sth_mvrv_cross", {})
    sth_val = sth.get("value") if isinstance(sth, dict) else None
    sth_phase = sth.get("phase") if isinstance(sth, dict) else None
    sth_extra = {
        "current": sth_val if isinstance(sth_val, (int, float)) else 0,
        "extended_below": str(sth_phase).upper() in ("RECLAIM", "RECOVERY", "BULL_CONFIRMED"),
    }

    # === Clemente+Alden additions (2026-06-03) ===
    hd_sig = sigs.get("fundamentals", {}).get("hashrate_drawdown", {})
    hd_val = hd_sig.get("value") if isinstance(hd_sig, dict) and not hd_sig.get("error") else None
    cb_streak_sig = sigs.get("flows", {}).get("cb_premium_streak", {})
    cb_streak_val = cb_streak_sig.get("value") if isinstance(cb_streak_sig, dict) and not cb_streak_sig.get("error") else None

    value_map = {
        "price_drawdown":         (pct_drawdown, None),
        "realized_cap_drawdown":  (rcap_val, None),
        "mvrv_z":                 (mvrvz_val, None),
        "coinbase_premium":       (cb_val, None),
        "hashrate_ribbon":        (hrr_phase, None),
        "halving_day":            (days_post, None),
        "puell_multiple":         (puell_val, None),
        "sth_mvrv_reclaim":       (sth_val, sth_extra),
        "hashrate_drawdown":      (hd_val, None),
        "cb_premium_streak":      (cb_streak_val, None),
    }

    results = []
    for crit in CRITERIA_DEFS:
        val, extra = value_map.get(crit["id"], (None, None))
        met, status = _check_criterion(crit, val, extra)
        firm, marginal, margin = _criterion_firm(crit, val, bool(met))
        results.append({
            "id":         crit["id"],
            "label":      crit["label"],
            "rationale":  crit["rationale"],
            "mechanism":  crit.get("mechanism", crit["id"]),
            "value":      val,
            "met":        bool(met),
            "firm":       bool(firm),
            "marginal":   bool(marginal),
            "margin":     margin,
            "status":     status + ("  [MARGINAL — within threshold buffer]" if marginal else ""),
        })

    n_met = sum(1 for r in results if r["met"])
    n_firm = sum(1 for r in results if r["firm"])
    n_marginal = sum(1 for r in results if r["marginal"])
    n_total = len(results)
    # distinct underlying mechanisms (dedups the 2 hashrate + 2 premium pairs)
    mechs_met = {r["mechanism"] for r in results if r["met"]}
    mechs_firm = {r["mechanism"] for r in results if r["firm"]}
    n_mech_met = len(mechs_met)
    n_mech_firm = len(mechs_firm)

    # ── Theme-breadth overlay: orthogonal-theme confirmation (the real gate) ──
    # Raw count over-credits correlated criteria (3 miner, 2 cost-basis, 2 flow).
    # Group into 6 orthogonal themes + require a price-turn for full deploy. Additive.
    # NOTE: does a yfinance weekly call (price-turn) — skipped on the hot render
    # path via compute_breadth=False; precompute computes it for the guru panel.
    breadth = {}
    gate = {}
    if compute_breadth:
        try:
            from core.btc_signal_themes import theme_breadth
            _extra = {"price_turn": _btc_weekly_price_turn(), "deriv_reset": _deriv_reset(sigs)}
            breadth = theme_breadth(results, extra_met=_extra)
        except Exception:
            breadth = {}
        # 2026-07-08 bulletproof gate: debounce the firm-mechanism level so a
        # single-day threshold nick can't move the confirmed gate (persist path).
        gate = _gate_debounce(n_mech_firm)

    # Updated thresholds for 10-criterion scorecard (was 8)
    if n_met >= 7:
        verdict_level = "BOTTOM_IN"
        verdict = f"BOTTOM CONFIRMED ({n_met}/{n_total} hard criteria met) — deploy capital."
    elif n_met >= 5:
        verdict_level = "BOTTOM_FORMING"
        verdict = f"BOTTOM LIKELY FORMING ({n_met}/{n_total} criteria) — begin scaling in."
    elif n_met >= 3:
        verdict_level = "EARLY"
        verdict = f"EARLY BOTTOM SIGNALS ({n_met}/{n_total}) — premature to deploy."
    else:
        verdict_level = "NO_BOTTOM"
        verdict = f"BOTTOM NOT CONFIRMED ({n_met}/{n_total} criteria met) — pattern projection only."

    # 2026-07-07 logic audit (F5): the theme-breadth overlay computes deploy_action
    # from cheapness breadth ALONE, so it could say "EARLY — deploy first tranche"
    # while the hard verdict is NO_BOTTOM ("premature to deploy") — mixed message.
    # The hard n_met gate is the deploy AUTHORITY: when it says no bottom, breadth
    # is context (good price), not a deploy instruction.
    _deploy_action = breadth.get("deploy_action")
    _deploy_level = breadth.get("deploy_level")
    if verdict_level == "NO_BOTTOM" and _deploy_level in ("EARLY", "SCALE_IN", "DEPLOY"):
        _deploy_action = (f"HOLD — {breadth.get('themes_met', 0)}/"
                          f"{breadth.get('themes_total', 6)} cheapness themes present "
                          f"but hard bottom NOT confirmed ({n_met}/{n_total}); "
                          f"breadth is context, not yet a deploy signal")
        _deploy_level = "HOLD"

    return {
        "criteria":      results,
        "n_met":         n_met,
        "n_firm":        n_firm,          # met AND decisively past threshold
        "n_marginal":    n_marginal,      # met but within the threshold buffer (fragile)
        "n_total":       n_total,
        "n_mechanisms_met":  n_mech_met,  # distinct underlying mechanisms (dedup'd)
        "n_mechanisms_firm": n_mech_firm, # distinct mechanisms firmly met -> the honest count
        # bulletproof gate: debounced level (persists GATE_PERSIST_DAYS before a
        # change is accepted). The capital gate reads gate_level, not the raw count.
        "gate_level":        gate.get("confirmed_level"),
        "gate_raw_level":    gate.get("raw_level"),
        "gate_pending":      gate.get("pending_change", False),
        "gate_days_at_raw":  gate.get("days_at_raw", 0),
        "verdict":       verdict,
        "verdict_level": verdict_level,
        # theme-breadth overlay (orthogonal-theme confirmation; momentum mandatory)
        "themes":            breadth.get("themes", []),
        "themes_met":        breadth.get("themes_met"),
        "themes_total":      breadth.get("themes_total"),
        "momentum_met":      breadth.get("momentum_met"),
        "robust_themes_met": breadth.get("robust_themes_met"),
        "deploy_action":     _deploy_action,
        "deploy_level":      _deploy_level,
        "deploy_breadth_raw": breadth.get("deploy_action"),  # ungated breadth read, kept for reference
        "breadth_summary":   breadth.get("summary"),
    }


def main():
    print("\n" + "=" * 78)
    print("BOTTOM CONFIRMATION SCORECARD — hard-criteria checklist")
    print("=" * 78)
    sc = bottom_confirmation_scorecard()
    print()
    for r in sc["criteria"]:
        mark = "[YES]" if r["met"] else "[NO ]"
        print(f"  {mark}  {r['label']}")
        print(f"           {r['status']}")
        print(f"           ({r['rationale']})")
        print()
    print("=" * 78)
    print(f"VERDICT: {sc['verdict']}")
    print(f"Level: {sc['verdict_level']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
