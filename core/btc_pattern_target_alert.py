"""I3: Pattern target zone alerts.

Tracks BTC price against pre-defined supply/resistance zones and
emails on entry/rejection events.

Zones (from W14C task — overhead supply mapping):
  $117,000 - $120,000   Cycle 5 peak resistance
  $108,000 - $112,000   STH realized price band
  $92,000  - $95,000    Overhead supply (pre-peak consolidation)
  $76,000  - $78,000    STH cost basis line
  $60,000  - $63,000    Major support (cycle midpoint)
  $53,000  - $55,000    LTH realized price (true floor)
  $42,000  - $45,000    Cycle 4 ATH retest
  $30,000  - $32,000    Cycle 4 analog bottom band

Events:
  ENTERED_ZONE         price entered a zone from outside
  REJECTED_AT_TOP      price hit upper zone bound then fell
  BOUNCED_AT_BOTTOM    price hit lower zone bound then rose
  BROKE_BELOW          price broke below a zone (downside)
  BROKE_ABOVE          price broke above a zone (upside)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".btc_pattern_target_state.json"


# Zones: (label, low, high, kind)
ZONES = [
    ("Cycle 5 peak",           117_000, 124_700, "resistance_top"),
    ("Pre-peak consolidation",  108_000, 112_000, "resistance"),
    ("Overhead supply",          92_000,  95_000, "resistance"),
    ("STH cost basis",           76_000,  78_000, "balance"),
    ("Major support",            60_000,  63_000, "support"),
    ("LTH realized (floor)",     53_000,  55_000, "support"),
    ("Cycle 4 ATH retest",       42_000,  45_000, "support_strong"),
    ("Cycle 4 analog bottom",    30_000,  32_000, "support_strong"),
]


def _load_state() -> dict:
    if not STATE_FILE.exists(): return {"last_zone": None, "last_price": None}
    try: return json.loads(STATE_FILE.read_text())
    except Exception: return {"last_zone": None, "last_price": None}


def _save_state(s: dict) -> None:
    try: STATE_FILE.write_text(json.dumps(s, indent=2))
    except Exception: pass


def _live_btc_price() -> float:
    try:
        import ccxt
        t = ccxt.binance().fetch_ticker("BTC/USDT")
        return float(t.get("last") or 0)
    except Exception:
        return 0.0


def _zone_for_price(price: float) -> dict:
    """Return the zone the price is in, or None if between zones."""
    for label, lo, hi, kind in ZONES:
        if lo <= price <= hi:
            return {"label": label, "low": lo, "high": hi, "kind": kind}
    # Between zones — find nearest
    return None


def _zone_label(price: float) -> str:
    z = _zone_for_price(price)
    if z: return z["label"]
    # Between zones
    above_idx = next((i for i, (l, lo, hi, k) in enumerate(ZONES) if price > hi), None)
    if above_idx is None: return "below cycle-4 analog"
    if above_idx == 0: return "above cycle 5 peak"
    return f"between {ZONES[above_idx-1][0]} and {ZONES[above_idx][0]}"


def check_pattern_targets(send_email: bool = True) -> dict:
    """Check if price entered/exited a tracked zone."""
    price = _live_btc_price()
    if price <= 0:
        return {"error": "price unavailable"}

    state = _load_state()
    last_zone_label = state.get("last_zone")
    last_price = state.get("last_price")

    current_zone = _zone_for_price(price)
    current_zone_label = current_zone["label"] if current_zone else _zone_label(price)

    state["last_zone"] = current_zone_label
    state["last_price"] = price
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    if last_zone_label is None:
        return {"price": price, "zone": current_zone_label,
                "event": "first_observation", "alert_sent": False}

    if last_zone_label == current_zone_label:
        return {"price": price, "zone": current_zone_label,
                "event": "same_zone", "alert_sent": False}

    # Zone transition!
    event = "ENTERED_ZONE" if current_zone else "EXITED_ZONE"
    if last_price and price > last_price:
        direction = "rising"
    elif last_price and price < last_price:
        direction = "falling"
    else:
        direction = "unknown"

    subject = f"!! BTC PATTERN: {last_zone_label} -> {current_zone_label} ({direction}) !!"
    body = f"""BTC PATTERN TARGET ALERT

Zone shift:   {last_zone_label} -> {current_zone_label}
Price:        ${price:,.0f}  (was ${last_price or 0:,.0f})
Direction:    {direction}
Event:        {event}

================================================================
ZONE MAP
================================================================
"""
    for label, lo, hi, kind in ZONES:
        marker = "  <==" if label == current_zone_label else "     "
        body += f"  {label:<24s} ${lo:>7,.0f} - ${hi:>7,.0f}  ({kind}){marker}\n"

    body += f"""
================================================================
INTERPRETATION
================================================================
"""
    if current_zone and current_zone["kind"] == "resistance_top":
        body += "Re-entering cycle 5 peak resistance zone. Heavy supply expected.\n"
    elif current_zone and current_zone["kind"] == "support_strong":
        body += f"Entered STRONG support {current_zone['label']}. Major bottom zone — watch for reversal.\n"
    elif current_zone and current_zone["kind"] == "support":
        body += f"Entered support {current_zone['label']}. Watch for bounce or break.\n"
    elif current_zone and current_zone["kind"] == "resistance":
        body += f"Entered resistance {current_zone['label']}. Watch for rejection or breakout.\n"
    else:
        body += "Between zones — momentum continues.\n"

    body += f"""
================================================================
NEXT
================================================================
Dashboard: http://localhost:8511
Re-check:  python -m core.btc_pattern_target_alert
"""

    if send_email:
        try:
            from ops.alerts import alert
            alert(body, level="critical", subject=subject)
            return {"price": price, "zone": current_zone_label,
                    "event": event, "alert_sent": True, "subject": subject}
        except Exception as e:
            return {"price": price, "zone": current_zone_label,
                    "event": event, "alert_sent": False, "error": str(e)}
    return {"price": price, "zone": current_zone_label,
            "event": event, "alert_sent": False, "preview": body}


def all_zones_status() -> dict:
    """Show current price relative to ALL zones — useful for dashboard."""
    price = _live_btc_price()
    rows = []
    for label, lo, hi, kind in ZONES:
        mid = (lo + hi) / 2
        if price < lo:    status = "below"
        elif price > hi:  status = "above"
        else:              status = "INSIDE"
        dist_pct = (price - mid) / mid * 100
        rows.append({
            "label": label, "low": lo, "high": hi, "kind": kind,
            "status": status, "distance_pct": dist_pct,
        })
    return {"price": price, "zones": rows,
            "current_zone": _zone_label(price)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--status", action="store_true",
                    help="show all zones status, no alert check")
    a = p.parse_args()
    if a.status:
        r = all_zones_status()
        print(f"BTC ${r['price']:,.0f}  zone: {r['current_zone']}\n")
        for z in r["zones"]:
            mark = " <==" if z["status"] == "INSIDE" else ""
            print(f"  {z['label']:<24s} ${z['low']:>7,.0f}-${z['high']:>7,.0f}  "
                  f"{z['kind']:<20s} {z['distance_pct']:+6.1f}%{mark}")
    else:
        r = check_pattern_targets(send_email=not a.no_email)
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
