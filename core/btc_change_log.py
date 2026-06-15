"""Daily change log — tracks day-over-day movements in key indicators.

Phillip Swift insisted on this: "What's DIFFERENT since yesterday?"
Without this, the user has to mentally compare with yesterday's numbers.

Stores yesterday's snapshot on each successful call. Diff against today's
to produce a "what changed" digest at the top of Overview.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_change_log_state.json"


def _live_btc_price() -> float:
    try:
        import ccxt
        t = ccxt.binance().fetch_ticker("BTC/USDT")
        return float(t.get("last") or 0)
    except Exception:
        return 0.0


def _btc_24h_change() -> float:
    try:
        import ccxt
        t = ccxt.binance().fetch_ticker("BTC/USDT")
        return float(t.get("percentage", 0))
    except Exception:
        return 0.0


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"snapshots": []}
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return {"snapshots": []}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2, default=str))
    except Exception: pass


def _take_snapshot() -> dict:
    """Capture current state across all key indicators."""
    from core.dashboard_cache import get_cached

    snap = {
        "ts":     datetime.now(timezone.utc).isoformat(),
        "date":   datetime.now(timezone.utc).date().isoformat(),
        "price":  _live_btc_price(),
    }

    try:
        ud = get_cached("unified_decision")
        if ud:
            snap["regime"]              = ud.get("regime")
            snap["top_n_met"]            = ud["scorecards"]["top"]["n_met"]
            snap["early_n_firing"]       = ud["scorecards"]["early"]["n_firing"]
            snap["bottom_n_met"]         = ud["scorecards"]["bottom"]["n_met"]
            snap["vetoes_count"]         = len(ud.get("vetoes_active", []))
    except Exception: pass

    try:
        pe = get_cached("predictor_engine")
        if pe:
            dec = pe.get("decision_composites", {})
            snap["top_z"]                = dec.get("top")
            snap["early_z"]              = dec.get("early")
            snap["bottom_z"]             = dec.get("bottom")
            snap["btc_state"]             = pe.get("btc_state", {}).get("state")
            snap["btc_bottom_prob"]      = pe.get("btc_state", {}).get("bottom_probability")
    except Exception: pass

    try:
        native_top = get_cached("btc_native_top_scorecard")
        if native_top:
            snap["native_top_n"]         = native_top.get("n_met")
            snap["native_top_level"]     = native_top.get("verdict_level")
    except Exception: pass

    try:
        native_bot = get_cached("btc_native_bottom_scorecard")
        if native_bot:
            snap["native_bottom_n"]      = native_bot.get("n_met")
            snap["native_bottom_level"]  = native_bot.get("verdict_level")
    except Exception: pass

    try:
        sw = get_cached("swift_watch")
        if sw:
            snap["risk_index"]           = sw.get("risk_index", {}).get("risk_index")
            snap["risk_zone"]            = sw.get("risk_index", {}).get("zone")
            snap["wma_pct"]              = sw.get("two_hundred_wma", {}).get("pct_vs_ma")
    except Exception: pass

    return snap


def get_diff() -> dict:
    """Capture current snapshot, diff against yesterday's, save."""
    state = _load_state()
    today_snap = _take_snapshot()
    today_date = today_snap["date"]

    # Find yesterday's snapshot (latest one < today)
    snapshots = state.get("snapshots", [])
    yesterday_snap = None
    for s in reversed(snapshots):
        if s.get("date") != today_date:
            yesterday_snap = s
            break

    # Save today's snapshot (replacing if same day)
    snapshots = [s for s in snapshots if s.get("date") != today_date]
    snapshots.append(today_snap)
    snapshots = snapshots[-30:]  # keep 30 days
    state["snapshots"] = snapshots
    _save_state(state)

    if not yesterday_snap:
        return {"first_observation": True, "today": today_snap,
                "diffs": [], "btc_24h_change": _btc_24h_change()}

    diffs = []

    def _diff_num(key, label, today_val, yest_val, fmt=".2f"):
        if today_val is None or yest_val is None: return None
        try: delta = float(today_val) - float(yest_val)
        except Exception: return None
        if abs(delta) < 0.005: return None  # too small to bother
        arrow = "↑" if delta > 0 else "↓"
        return {
            "label": label, "key": key,
            "today": today_val, "yesterday": yest_val,
            "delta": delta, "arrow": arrow,
            "text": f"{arrow} {label}: {yest_val:{fmt}} → {today_val:{fmt}} ({delta:+{fmt}})",
        }

    def _diff_int(key, label, today_val, yest_val):
        if today_val is None or yest_val is None: return None
        try:
            delta = int(today_val) - int(yest_val)
        except Exception: return None
        if delta == 0: return None
        arrow = "↑" if delta > 0 else "↓"
        return {
            "label": label, "key": key,
            "today": today_val, "yesterday": yest_val,
            "delta": delta, "arrow": arrow,
            "text": f"{arrow} {label}: {yest_val} → {today_val} ({delta:+d})",
        }

    def _diff_str(key, label, today_val, yest_val):
        if today_val == yest_val or today_val is None: return None
        return {
            "label": label, "key": key,
            "today": today_val, "yesterday": yest_val,
            "text": f"≠ {label}: {yest_val} → {today_val}",
        }

    candidates = [
        _diff_int("top_n_met", "Top scorecard", today_snap.get("top_n_met"), yesterday_snap.get("top_n_met")),
        _diff_int("early_n_firing", "Early rotation", today_snap.get("early_n_firing"), yesterday_snap.get("early_n_firing")),
        _diff_int("bottom_n_met", "Bottom scorecard", today_snap.get("bottom_n_met"), yesterday_snap.get("bottom_n_met")),
        _diff_int("vetoes_count", "Vetoes active", today_snap.get("vetoes_count"), yesterday_snap.get("vetoes_count")),
        _diff_int("native_top_n", "Native top scorecard", today_snap.get("native_top_n"), yesterday_snap.get("native_top_n")),
        _diff_int("native_bottom_n", "Native bottom scorecard", today_snap.get("native_bottom_n"), yesterday_snap.get("native_bottom_n")),
        _diff_num("top_z", "Top composite z", today_snap.get("top_z"), yesterday_snap.get("top_z")),
        _diff_num("bottom_z", "Bottom composite z", today_snap.get("bottom_z"), yesterday_snap.get("bottom_z")),
        _diff_num("risk_index", "Risk Index", today_snap.get("risk_index"), yesterday_snap.get("risk_index")),
        _diff_num("btc_bottom_prob", "Bottom probability", today_snap.get("btc_bottom_prob"), yesterday_snap.get("btc_bottom_prob"), ".3f"),
        _diff_num("wma_pct", "vs 200wMA", today_snap.get("wma_pct"), yesterday_snap.get("wma_pct"), ".1f"),
        _diff_str("regime", "Macro regime", today_snap.get("regime"), yesterday_snap.get("regime")),
        _diff_str("btc_state", "BTC state", today_snap.get("btc_state"), yesterday_snap.get("btc_state")),
        _diff_str("risk_zone", "Risk zone", today_snap.get("risk_zone"), yesterday_snap.get("risk_zone")),
        _diff_str("native_top_level", "Native top level", today_snap.get("native_top_level"), yesterday_snap.get("native_top_level")),
        _diff_str("native_bottom_level", "Native bottom level", today_snap.get("native_bottom_level"), yesterday_snap.get("native_bottom_level")),
    ]
    diffs = [d for d in candidates if d]

    # Also include BTC price delta
    today_price = today_snap.get("price", 0)
    yest_price = yesterday_snap.get("price", 0)
    price_diff = None
    if today_price > 0 and yest_price > 0:
        pct = (today_price / yest_price - 1) * 100
        if abs(pct) > 0.1:
            arrow = "↑" if pct > 0 else "↓"
            price_diff = {
                "label": "BTC price", "key": "price",
                "today": today_price, "yesterday": yest_price,
                "delta_pct": pct, "arrow": arrow,
                "text": f"{arrow} BTC: ${yest_price:,.0f} → ${today_price:,.0f} ({pct:+.2f}%)",
            }

    return {
        "first_observation": False,
        "today":            today_snap,
        "yesterday":        yesterday_snap,
        "diffs":            diffs,
        "price_diff":       price_diff,
        "btc_24h_change":   _btc_24h_change(),
        "n_changes":        len(diffs) + (1 if price_diff else 0),
    }


def main():
    r = get_diff()
    print("=" * 70)
    print("BTC DASHBOARD — DAILY CHANGE LOG")
    print("=" * 70)
    if r.get("first_observation"):
        print("\n  First observation — no diff available yet")
        return
    if r.get("price_diff"):
        print(f"\n  {r['price_diff']['text']}")
    print(f"\n  {len(r['diffs'])} indicator changes vs yesterday:")
    for d in r["diffs"]:
        try: print(f"    {d['text']}")
        except UnicodeEncodeError:
            print(f"    {d['text'].encode('ascii','replace').decode()}")


if __name__ == "__main__":
    main()
