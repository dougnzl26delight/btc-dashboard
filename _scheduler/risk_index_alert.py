"""Risk Index + scorecard verdict-level cross alert.

Phillip Swift's #1 requested addition: realtime push when the Risk Index
crosses a zone boundary. "A real trader has the dashboard open or has
an alert tripped. No third option."

Watches state across:
- Swift's Risk Index zone           (MAX BUY -> MAX SELL, 5 zones)
- Native top scorecard verdict      (HOLD -> WATCH -> TRIM_25 -> SCALE_OUT_50 -> EXIT_75)
- Native bottom scorecard verdict   (HOLD -> WATCH -> ACCUMULATE -> STRONG_BUY -> DEEP_VALUE -> EXTREME)
- BTC state                          (shallow/deep/normal)
- Macro regime                       (risk-on/risk-off/etc)

On ANY level change vs last check -> email alert.

Designed to run hourly via scheduled task. No-ops if state unchanged.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
STATE_FILE = REPO / ".risk_index_alert_state.json"


# Zone severity ranking — used to determine "escalation vs de-escalation"
RISK_ZONE_RANK = {
    "MAX BUY": 0, "BUY": 1, "NEUTRAL": 2, "SELL": 3, "MAX SELL": 4,
}
TOP_LEVEL_RANK = {
    "HOLD": 0, "WATCH": 1, "TRIM_25": 2, "SCALE_OUT_50": 3, "EXIT_75": 4,
}
BOTTOM_LEVEL_RANK = {
    "HOLD": 0, "WATCH": 1, "ACCUMULATE": 2,
    "STRONG_BUY": 3, "DEEP_VALUE": 4, "EXTREME": 5,
}
# Jesse Olson's QQQ tier ladder (SAFE -> CAUTION -> WATCH -> GAP_BROKEN -> RETESTING)
QQQ_OLSON_RANK = {
    "SAFE": 0, "CAUTION": 1, "WATCH": 2, "GAP_BROKEN": 3, "RETESTING": 4,
}
# BTC top scale-out ladder (next-bull exit system)
SCALE_OUT_RANK = {
    "DORMANT": 0, "ARMED": 1, "TRIM_25": 2, "SCALE_OUT_50": 3, "EXIT_75": 4,
}


def _capture_state() -> dict:
    """Read current state from disk cache (populated by precompute job)."""
    from core.dashboard_cache import get_cached

    state = {
        "ts":   datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
    }

    try:
        sw = get_cached("swift_watch") or {}
        ri = sw.get("risk_index", {}) or {}
        state["risk_index"]   = ri.get("risk_index")
        state["risk_zone"]    = ri.get("zone")
    except Exception: pass

    try:
        nt = get_cached("btc_native_top_scorecard") or {}
        state["top_level"]   = nt.get("verdict_level")
        state["top_n_met"]   = nt.get("n_met")
    except Exception: pass

    try:
        nb = get_cached("btc_native_bottom_scorecard") or {}
        state["bottom_level"] = nb.get("verdict_level")
        state["bottom_n_met"] = nb.get("n_met")
        state["bottom_n_total"] = nb.get("n_total")   # live count — never hardcode /15
    except Exception: pass

    try:
        ud = get_cached("unified_decision") or {}
        state["regime"]       = ud.get("regime")
        state["btc_state"]    = (ud.get("btc_state") or {}).get("state") \
                                 if isinstance(ud.get("btc_state"), dict) \
                                 else ud.get("btc_state")
    except Exception: pass

    try:
        pe = get_cached("predictor_engine") or {}
        bs = pe.get("btc_state", {}) or {}
        if "btc_state" not in state or state["btc_state"] is None:
            state["btc_state"] = bs.get("state")
        state["btc_bottom_prob"] = bs.get("bottom_probability")
    except Exception: pass

    # Olson's QQQ tier
    try:
        eq = get_cached("equity_olson") or {}
        state["qqq_olson_tier"]      = eq.get("tier")
        state["qqq_close"]           = eq.get("last_close")
        state["qqq_pct_to_gap"]      = eq.get("pct_to_gap")
        state["qqq_pct_to_wma200"]   = eq.get("pct_to_wma200")
    except Exception: pass

    # Semis leading tell (SOXX) — early warning ahead of QQQ
    try:
        sem = get_cached("equity_semis") or {}
        state["semis_tier"] = sem.get("tier")
    except Exception: pass

    # Olson's BTC bearish W pattern target window: $52k-$57k
    # When BTC drops INTO this band -> Olson's call is hitting; flag as TARGET_ZONE
    try:
        import ccxt
        t = ccxt.binance().fetch_ticker("BTC/USDT")
        px = float(t.get("last") or 0)
        state["btc_price"] = px
        if 52_000 <= px <= 57_000:
            state["olson_btc_target_status"] = "IN_BAND"
        elif px > 57_000:
            state["olson_btc_target_status"] = "ABOVE"
        else:
            state["olson_btc_target_status"] = "BELOW"
    except Exception: pass

    # Rotation trigger — single-shot equity->BTC. Captures `overall` (FIRED/WARMING/ARMED)
    # and the firing paths so we can detect FIRED transitions and send the big email.
    try:
        rt = get_cached("rotation_trigger") or {}
        state["rotation_trigger_status"]  = rt.get("overall")
        state["rotation_firing_paths"]    = ",".join(rt.get("firing_paths", []) or [])
        state["rotation_fired"]            = bool(rt.get("fired"))
        # For the 1-of-4 EARLY-WARNING email: capture score + which signal tripped.
        state["rotation_score"]            = rt.get("best_score", 0)
        state["rotation_qqq_dma200"]       = rt.get("qqq_dma200")
        _rconds = (rt.get("paths") or [{}])[0].get("conditions", []) or []
        state["rotation_tripped"]          = " · ".join(
            c.get("label", "?") for c in _rconds if c.get("met"))
    except Exception: pass

    # BTC top scale-out ladder (DORMANT/ARMED/TRIM_25/SCALE_OUT_50/EXIT_75)
    try:
        so = get_cached("scale_out_trigger") or {}
        state["scale_out_tier"] = so.get("tier")
    except Exception: pass

    # Data-health verdict — so STALE DATA / DENOMINATOR DRIFT surfaces + emails.
    try:
        from core.data_health import data_health
        state["data_health"] = data_health().get("verdict")
    except Exception: pass

    return state


def _load_last() -> dict | None:
    if not STATE_FILE.exists(): return None
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return None


def _save(state: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception: pass


def _direction(prev: str | None, curr: str | None, rank: dict) -> str:
    """Was this a more-bullish, more-bearish, or sideways change?"""
    if prev is None or curr is None: return "?"
    pr, cr = rank.get(prev, -1), rank.get(curr, -1)
    if pr == -1 or cr == -1: return "?"
    if cr > pr: return "ESCALATED"
    if cr < pr: return "DEESCALATED"
    return "="


def _diff(prev: dict, curr: dict) -> list[str]:
    """Build human-readable list of meaningful changes."""
    out = []

    # Risk Index zone
    p_zone, c_zone = prev.get("risk_zone"), curr.get("risk_zone")
    if p_zone != c_zone and c_zone is not None:
        dir_ = _direction(p_zone, c_zone, RISK_ZONE_RANK)
        arrow = "->" if dir_ != "DEESCALATED" else "<-"
        out.append(f"** RISK INDEX zone {arrow} {dir_}: {p_zone} -> {c_zone}")

    # Top scorecard level
    p_top, c_top = prev.get("top_level"), curr.get("top_level")
    if p_top != c_top and c_top is not None:
        dir_ = _direction(p_top, c_top, TOP_LEVEL_RANK)
        flag = "**" if dir_ == "ESCALATED" else "  "
        out.append(f"{flag} TOP scorecard {dir_}: {p_top} -> {c_top} "
                   f"({prev.get('top_n_met','?')}/16 -> {curr.get('top_n_met','?')}/16)")

    # Bottom scorecard level
    p_bot, c_bot = prev.get("bottom_level"), curr.get("bottom_level")
    if p_bot != c_bot and c_bot is not None:
        dir_ = _direction(p_bot, c_bot, BOTTOM_LEVEL_RANK)
        flag = "**" if dir_ == "ESCALATED" else "  "
        out.append(f"{flag} BOTTOM scorecard {dir_}: {p_bot} -> {c_bot} "
                   f"({prev.get('bottom_n_met','?')} -> {curr.get('bottom_n_met','?')}"
                   f"/{curr.get('bottom_n_total', 16)})")

    # Regime + BTC state changes (no direction)
    if prev.get("regime") != curr.get("regime") and curr.get("regime"):
        out.append(f"   MACRO regime: {prev.get('regime')} -> {curr.get('regime')}")
    if prev.get("btc_state") != curr.get("btc_state") and curr.get("btc_state"):
        out.append(f"   BTC state: {prev.get('btc_state')} -> {curr.get('btc_state')}")

    # Olson's QQQ tier (equity-side trapdoor watch)
    p_qqq, c_qqq = prev.get("qqq_olson_tier"), curr.get("qqq_olson_tier")
    if p_qqq != c_qqq and c_qqq is not None:
        dir_ = _direction(p_qqq, c_qqq, QQQ_OLSON_RANK)
        flag = "**" if dir_ == "ESCALATED" else "  "
        qpct = curr.get("qqq_pct_to_gap", 0) or 0
        out.append(f"{flag} QQQ Olson tier {dir_}: {p_qqq} -> {c_qqq} "
                    f"(QQQ ${curr.get('qqq_close', 0):,.0f}, "
                    f"{qpct:+.1f}% to 589 gap)")

    # Rotation trigger status (ARMED -> WARMING -> FIRED)
    p_rt, c_rt = prev.get("rotation_trigger_status"), curr.get("rotation_trigger_status")
    if p_rt != c_rt and c_rt is not None:
        if c_rt == "FIRED":
            out.append(f"** ROTATION TRIGGER FIRED: {p_rt} -> {c_rt} "
                        f"(paths: {curr.get('rotation_firing_paths', '?')})")
        elif c_rt == "WARMING":
            out.append(f"   ROTATION TRIGGER warming ({curr.get('rotation_score', 1)}/4 "
                        f"equity signals): {p_rt} -> {c_rt} — first tripped: "
                        f"{curr.get('rotation_tripped') or 'an equity signal'}")
        else:
            out.append(f"   ROTATION TRIGGER de-escalated: {p_rt} -> {c_rt}")

    # Semis leading tell (SOXX) — INTACT -> WEAKENING -> BREAKDOWN. This is an
    # EARLY pre-warning that precedes the QQQ-based rotation trigger (semis lead).
    SEMIS_RANK = {"INTACT": 0, "WEAKENING": 1, "BREAKDOWN": 2}
    p_sem, c_sem = prev.get("semis_tier"), curr.get("semis_tier")
    if p_sem != c_sem and c_sem is not None:
        dir_ = _direction(p_sem, c_sem, SEMIS_RANK)
        flag = "**" if dir_ == "ESCALATED" else "  "
        out.append(f"{flag} SEMIS TELL (SOXX, leads QQQ) {dir_}: {p_sem} -> {c_sem}"
                    + (" — equity weakness building EARLY, ahead of rotation trigger"
                        if dir_ == "ESCALATED" else ""))

    # BTC top scale-out ladder — escalation to an action tier is URGENT
    p_so, c_so = prev.get("scale_out_tier"), curr.get("scale_out_tier")
    if p_so != c_so and c_so is not None:
        dir_ = _direction(p_so, c_so, SCALE_OUT_RANK)
        action_tier = c_so in ("TRIM_25", "SCALE_OUT_50", "EXIT_75")
        flag = "**" if (dir_ == "ESCALATED" and action_tier) else "  "
        out.append(f"{flag} BTC TOP SCALE-OUT {dir_}: {p_so} -> {c_so}" +
                    (f" — sell {c_so.split('_')[-1]}% of BTC holdings TODAY"
                      if action_tier and dir_ == "ESCALATED" else ""))

    # Olson's BTC bearish W pattern target band ($52k-$57k)
    p_tgt, c_tgt = prev.get("olson_btc_target_status"), curr.get("olson_btc_target_status")
    if p_tgt != c_tgt and c_tgt is not None:
        px = curr.get("btc_price", 0) or 0
        if c_tgt == "IN_BAND":
            out.append(f"** OLSON BTC TARGET BAND ENTERED: BTC ${px:,.0f} "
                        f"is now within Olson's bearish W target $52k-$57k zone")
        elif p_tgt == "IN_BAND" and c_tgt == "BELOW":
            out.append(f"** OLSON BTC TARGET BAND BROKEN BELOW: BTC ${px:,.0f} "
                        f"fell below $52k -- Olson's target zone exceeded")
        elif p_tgt == "IN_BAND" and c_tgt == "ABOVE":
            out.append(f"   OLSON BTC TARGET BAND EXITED ABOVE: BTC recovered "
                        f"above $57k to ${px:,.0f}")
        # Note: ABOVE -> BELOW skipping band is rare; ABOVE -> IN_BAND covered

    # Data health — alert on ACTIONABLE states only (stale/drift), not the
    # steady-state DEGRADED (known paywalled feeds NVT + funding).
    p_dh, c_dh = prev.get("data_health"), curr.get("data_health")
    if p_dh != c_dh and c_dh in ("STALE DATA", "DENOMINATOR DRIFT"):
        out.append(f"** DATA HEALTH: {p_dh} -> {c_dh} — an indicator went stale or a "
                   f"scorecard total drifted; open Guru tab > Data Health.")
    elif p_dh in ("STALE DATA", "DENOMINATOR DRIFT") and c_dh == "ALL FRESH":
        out.append(f"   DATA HEALTH recovered: {p_dh} -> {c_dh}")

    return out


def _severity(diffs: list[str]) -> str:
    """Decide email severity for subject line emoji."""
    if any("**" in d and "ESCALATED" in d for d in diffs): return "URGENT"
    if any("**" in d for d in diffs):                       return "HIGH"
    if diffs:                                                return "INFO"
    return "QUIET"


def _send_big_rotation_email():
    """Send the EXECUTE TODAY email with NZ$ amounts + leg instructions."""
    try:
        from core.rotation_trigger import evaluate_rotation_trigger, action_email_body
        from ops.alerts import alert
        s = evaluate_rotation_trigger()
        body = action_email_body(s)
        subject = "!!! ROTATION TRIGGER FIRED — EXECUTE EQUITY -> BTC TODAY !!!"
        alert(body, level="warning", subject=subject, email=True)
        return True
    except Exception as e:
        return False


def _send_warming_email(curr: dict) -> bool:
    """EARLY-WARNING heads-up: the FIRST equity-stress signal has tripped
    (rotation trigger ARMED -> WARMING, 1-of-4). Deliberately NOT the rotate-now
    email — the disciplined 2-of-4 trigger still governs execution. Gives the
    operator 'earlier eyes' to watch the equity leg develop a few days sooner."""
    try:
        from ops.alerts import alert
        # Historical context block — defensive: a failure drops the context,
        # never the whole heads-up email.
        try:
            from core.rotation_trigger import leadtime_email_block as _ltb
            _leadtime_block = _ltb("WARMING")
        except Exception:
            _leadtime_block = ""
        score   = curr.get("rotation_score", 1) or 1
        tripped = curr.get("rotation_tripped") or "an equity-stress signal"
        qqq     = curr.get("qqq_close", 0) or 0
        dma     = curr.get("rotation_qqq_dma200", 0) or 0
        body = "\n".join([
            "EARLY WARNING — first equity-stress signal has tripped.",
            "",
            f"  Rotation trigger:  ARMED -> WARMING ({score}/4 equity signals)",
            f"  First tripped:     {tripped}",
            f"  QQQ:               ${qqq:,.0f}"
                + (f"   (200-day SMA ${dma:,.0f})" if dma else ""),
            "",
            "  >>> This is NOT the rotate-now signal. <<<",
            "  The disciplined trigger still needs 2-of-4 equity signals before it",
            "  emails the EXECUTE instruction. Treat this as earlier eyes: watch the",
            "  equity leg (QQQ, semis, 3-week MACD) develop over the coming days.",
            "",
            "  The 4 equity signals tracked:",
            "    1. QQQ Olson tier reaches CAUTION or worse",
            "    2. Macro top scorecard hits its (cycle-scaled) threshold",
            "    3. QQQ 3-week MACD bearish cross",
            "    4. QQQ closes below its 200-day SMA",
            "",
            _leadtime_block,
            "",
            "Open dashboard for the full breakdown:",
            "  Remote: https://btcdelight.com",
        ])
        alert(body, level="warning", email=True,
              subject="[WARMING 1/4] First equity-stress signal tripped — NOT execute yet")
        return True
    except Exception:
        return False


def main():
    curr = _capture_state()
    prev = _load_last()

    # ===== CRITICAL: ROTATION TRIGGER detection =====
    # If prev was not-fired and curr IS fired -> send the big rotation email
    # immediately, separately from the normal change-log email below.
    if prev is not None:
        was_fired = prev.get("rotation_fired", False)
        is_fired = curr.get("rotation_fired", False)
        if is_fired and not was_fired:
            _send_big_rotation_email()
        else:
            # EARLY WARNING: first equity signal trips (ARMED -> WARMING).
            # Edge-triggered on the transition INTO warming, so it won't re-send
            # every hour while it sits at 1/4. Not the execute signal — earlier eyes.
            was_warming = prev.get("rotation_trigger_status") == "WARMING"
            is_warming  = curr.get("rotation_trigger_status") == "WARMING"
            if is_warming and not was_warming:
                _send_warming_email(curr)

    if prev is None:
        # First observation -- baseline only, no email
        _save(curr)
        print(json.dumps({"status": "baseline_saved", "state": {
            "risk_zone": curr.get("risk_zone"),
            "top_level": curr.get("top_level"),
            "bottom_level": curr.get("bottom_level"),
        }}))
        return

    diffs = _diff(prev, curr)
    if not diffs:
        print(json.dumps({"status": "unchanged", "checks_run": 5}))
        return

    severity = _severity(diffs)

    # Live BTC price for context
    btc_price = "n/a"
    try:
        import ccxt
        t = ccxt.binance().fetch_ticker("BTC/USDT")
        btc_price = f"${float(t.get('last') or 0):,.0f}"
    except Exception: pass

    # Email body
    subject_emoji = {"URGENT": "!! URGENT", "HIGH": "** HIGH",
                     "INFO": "[INFO]", "QUIET": "[QUIET]"}.get(severity, "[INFO]")
    subject = f"{subject_emoji} BTC dashboard verdict change ({severity})"

    # Plain-English lead — what's going on + what to do, in normal words.
    try:
        from core.plain_email import (plain_lead, plain_zone, plain_bottom_level,
                                       plain_qqq, plain_plan)
        _mood = {"URGENT": "act", "HIGH": "watch", "INFO": "watch",
                 "QUIET": "calm"}.get(severity, "watch")
        _head = {
            "URGENT": "Something important just changed - worth acting on.",
            "HIGH":   "Something notable changed - worth a look.",
            "INFO":   "A small change since the last check - nothing urgent.",
            "QUIET":  "Markets are quiet - nothing important changed.",
        }.get(severity, "Something changed since the last check.")
        _pbul = []
        if curr.get("btc_price"):
            _pbul.append(f"Bitcoin is ${curr.get('btc_price', 0) or 0:,.0f}, and on the "
                         f"cheap/expensive scale it looks {plain_zone(curr.get('risk_zone'))}.")
        if curr.get("bottom_level") is not None:
            _pbul.append(f"Bottom checklist: {plain_bottom_level(curr.get('bottom_level'))} "
                         f"({curr.get('bottom_n_met', '?')}/{curr.get('bottom_n_total', 16)} boxes ticked).")
        if curr.get("qqq_olson_tier"):
            _pbul.append(f"US tech stocks (QQQ): {plain_qqq(curr.get('qqq_olson_tier'))}.")
        if curr.get("rotation_trigger_status"):
            _pbul.append(f"Shares-into-Bitcoin plan: {plain_plan(curr.get('rotation_trigger_status'))}.")
        _todo = {
            "URGENT": "Open the dashboard and read the change below - it may need action.",
            "HIGH":   "Have a look at what changed below when you get a moment.",
            "INFO":   "Nothing required - just so you know.",
            "QUIET":  "Nothing - sit tight.",
        }.get(severity, "Have a look below.")
        _lead = plain_lead(_head, _pbul, _todo, mood=_mood)
    except Exception:
        _lead = "BTC dashboard state changed since last check."

    body_lines = [
        _lead,
        f"WHAT CHANGED since last check (the specifics):",
        f"  BTC price now: {btc_price}",
    ]
    body_lines.extend(f"  {d}" for d in diffs)
    body_lines += [
        f"",
        f"CURRENT STATE:",
        f"  Risk Index zone:        {curr.get('risk_zone')} "
            f"(score={curr.get('risk_index')})",
        f"  Top scorecard:          {curr.get('top_level')} "
            f"({curr.get('top_n_met')}/16)",
        f"  Bottom scorecard:       {curr.get('bottom_level')} "
            f"({curr.get('bottom_n_met')}/{curr.get('bottom_n_total', 16)})",
        f"  Macro regime:           {curr.get('regime')}",
        f"  BTC state:              {curr.get('btc_state')}",
        f"  Olson QQQ tier:         {curr.get('qqq_olson_tier')} "
            f"(QQQ ${curr.get('qqq_close', 0) or 0:,.0f}, "
            f"{curr.get('qqq_pct_to_gap') or 0:+.1f}% to 589 gap)",
        f"  Olson BTC tgt $52-57k:  {curr.get('olson_btc_target_status')} "
            f"(BTC ${curr.get('btc_price', 0) or 0:,.0f})",
        f"",
        f"Open dashboard for full breakdown:",
        f"  Local:  http://localhost:8511",
        f"  Remote: https://btcdelight.com",
    ]
    body = "\n".join(body_lines)

    # Save new baseline AFTER constructing message (so an email failure
    # doesn't lose the prev state and re-fire next run)
    try:
        from ops.alerts import alert
        level = "warning" if severity in ("URGENT", "HIGH") else "info"
        alert(body, level=level, subject=subject, email=True)
        _save(curr)
        print(json.dumps({"status": "email_sent", "severity": severity,
                            "n_diffs": len(diffs)}))
    except Exception as e:
        # Don't save state on email failure -- retry next hour
        print(json.dumps({"status": "email_failed", "error": str(e),
                            "n_diffs": len(diffs)}))


if __name__ == "__main__":
    main()
