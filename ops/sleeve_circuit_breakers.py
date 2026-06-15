"""Per-sleeve drawdown circuit breakers.

Each sleeve gets its own peak-equity tracking + auto-deleveraging:

    Drawdown range    Scale     Behavior
    -----------------+---------+---------------------------------
    < 10%             1.00      Full size
    10-15%            0.50      Half size
    15-20%            0.25      Quarter size
    > 20%             0.00      Paused (requires manual reset)

Compared to ops/circuit_breaker.py (portfolio-level hard kill), this fires
sleeve-by-sleeve so a single bad strategy can't drag the whole rig down,
and recovery is automatic when the sleeve climbs back toward its peak.

Usage in a sleeve runner:
    from ops.sleeve_circuit_breakers import apply_sleeve_scaling
    scale = apply_sleeve_scaling("bah_btc", current_equity)
    target_notional *= scale

State persisted to .sleeve_dd_state.json. Reset a paused sleeve with:
    python -m ops.sleeve_circuit_breakers reset <sleeve_name>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.alerts import alert


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".sleeve_dd_state.json"

# Drawdown thresholds (peak-to-trough) -> size-scale multiplier.
# 2026-05-28 tightening: trader stated 10% max portfolio DD tolerance.
# Tiers now match: deleverage starts at 5%, paused at 10%. Replaces prior
# 10/15/20 schedule which would have blown past the trader's pain threshold
# before the CB even fired.
DD_RULES = [
    (0.05, 1.00),   # dd <= 5%  -> full size
    (0.075, 0.50),  # 5-7.5%   -> half
    (0.10, 0.25),   # 7.5-10%  -> quarter
    (1.00, 0.00),   # > 10%    -> paused (manual reset required)
]

# Start equities re-baselined 2026-05-28 for $35k live target.
# Scale paper $100k -> live $35k by factor 0.35. Live deployment will use
# these tighter notionals; paper sim uses 1x but CB tiers are unchanged.
KNOWN_SLEEVES = {
    "bah_btc": {"start_equity": 3_500.0, "description": "BAH BTC long-term sleeve (10% bankroll)"},
    "xsmom": {"start_equity": 5_250.0, "description": "Cross-sectional momentum (15%)"},
    "pro_trend": {"start_equity": 7_000.0, "description": "Trend-follower with pyramiding (20%)"},
    "perp_orchestrator": {"start_equity": 17_500.0, "description": "Perp account orchestrator (50%)"},
    "spot_orchestrator": {"start_equity": 17_500.0, "description": "Spot account orchestrator (50%)"},
    "oversold_bounce": {"start_equity": 3_500.0, "description": "Tactical bounce on regime-wide RSI<25 (10%)"},
    "overbought_fade": {"start_equity": 2_625.0, "description": "Tactical short on regime-wide RSI>70 in bear (7.5%)"},
    "intraday_momentum": {"start_equity": 3_500.0, "description": "Active intraday momentum scalper (15-min cadence)"},
    "intraday_momentum_short": {"start_equity": 3_500.0, "description": "Intraday SHORT momentum (15-min, bear regime)"},
    "grid_trader": {"start_equity": 10_000.0, "description": "Continuous grid trading on BTC/ETH (5-min cadence)"},
    "consolidation_breakout": {"start_equity": 10_000.0, "description": "Livermore: trade breakouts from compressed ranges"},
}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _compute_scale(dd: float) -> float:
    for threshold, scale in DD_RULES:
        if dd <= threshold:
            return scale
    return 0.0


def get_sleeve_scale(sleeve: str) -> float:
    """Return position-size multiplier [0.0, 1.0] for a sleeve.

    Used by sleeve runners to scale their target sizes. Returns 1.0 if the
    sleeve has no state yet (first-run gets full size).
    """
    state = _load_state()
    return float(state.get(sleeve, {}).get("scale", 1.0))


def is_paused(sleeve: str) -> bool:
    """True if the sleeve is paused (drawdown > 20% and not manually reset)."""
    return get_sleeve_scale(sleeve) == 0.0


def apply_sleeve_scaling(sleeve: str, current_equity: float) -> float:
    """Update sleeve's peak/dd and return the current size-scale.

    Call this at the top of each sleeve's runner with the sleeve's current
    mark-to-market equity. Returns the multiplier to apply to position sizes.
    Side effect: fires an alert on scale transitions.
    """
    state = _load_state()
    prior = state.get(sleeve, {})

    # Initialize peak on first sighting
    if not prior:
        prior_peak = max(KNOWN_SLEEVES.get(sleeve, {}).get("start_equity", current_equity), current_equity)
    else:
        prior_peak = prior.get("peak", current_equity)

    peak = max(prior_peak, current_equity)
    dd = (peak - current_equity) / peak if peak > 0 else 0.0
    new_scale = _compute_scale(dd)
    prev_scale = prior.get("scale", 1.0)

    # Transition detection
    transition_msg = None
    if new_scale < prev_scale - 1e-9:
        transition_msg = (
            f"DELEVERAGE: {sleeve} dd={dd:.1%}, peak ${peak:,.0f} -> now ${current_equity:,.0f}. "
            f"Scale {prev_scale:.2f} -> {new_scale:.2f}"
        )
    elif new_scale > prev_scale + 1e-9 and dd < 0.05:
        transition_msg = (
            f"RESTORE: {sleeve} recovered to dd={dd:.1%}, scale {prev_scale:.2f} -> {new_scale:.2f}"
        )

    state[sleeve] = {
        "peak": peak,
        "current": current_equity,
        "drawdown": dd,
        "scale": new_scale,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_transition": transition_msg,
    }
    _save_state(state)

    if transition_msg:
        level = "critical" if new_scale == 0.0 else "warning"
        alert(f"sleeve_cb: {transition_msg}", level=level)

    return new_scale


def reset_sleeve(sleeve: str) -> dict:
    """Manually reset a paused sleeve. Resets peak to current and scale to 1.0."""
    state = _load_state()
    prev = state.get(sleeve, {})
    current = prev.get("current", KNOWN_SLEEVES.get(sleeve, {}).get("start_equity", 0))
    state[sleeve] = {
        "peak": current,
        "current": current,
        "drawdown": 0.0,
        "scale": 1.0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_transition": f"MANUAL RESET from prior scale {prev.get('scale', '?')}",
    }
    _save_state(state)
    alert(f"sleeve_cb: {sleeve} manually reset by operator", level="info")
    return state[sleeve]


def status() -> dict:
    """Return full state for dashboard / status reports."""
    state = _load_state()
    out = {}
    for sleeve_name, meta in KNOWN_SLEEVES.items():
        s = state.get(sleeve_name, {})
        out[sleeve_name] = {
            "description": meta["description"],
            "peak": s.get("peak"),
            "current": s.get("current"),
            "drawdown": s.get("drawdown", 0.0),
            "scale": s.get("scale", 1.0),
            "paused": s.get("scale", 1.0) == 0.0,
            "last_updated": s.get("last_updated"),
        }
    return out


# ===== CLI =====
def _print_status():
    print(f"{'Sleeve':<22s} {'Peak':>12s} {'Current':>12s} {'DD':>7s} {'Scale':>6s} {'Status':<12s}")
    print("-" * 80)
    for name, s in status().items():
        peak = s.get("peak")
        cur = s.get("current")
        dd = s.get("drawdown", 0)
        scale = s.get("scale", 1.0)
        if peak is None:
            print(f"{name:<22s} {'—':>12s} {'—':>12s} {'—':>7s} {scale:>5.2f}x {'unseen':<12s}")
        else:
            tag = "PAUSED" if scale == 0 else ("REDUCED" if scale < 1 else "OK")
            print(f"{name:<22s} ${peak:>10,.0f} ${cur:>10,.0f} {dd*100:>5.1f}% {scale:>5.2f}x {tag:<12s}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status")
    reset_p = sub.add_parser("reset")
    reset_p.add_argument("sleeve")
    args = parser.parse_args()

    if args.cmd == "reset":
        out = reset_sleeve(args.sleeve)
        print(json.dumps(out, indent=2, default=str))
    else:
        _print_status()


if __name__ == "__main__":
    main()
