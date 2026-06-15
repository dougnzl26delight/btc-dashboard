"""Walk-forward Sharpe gating per strategy/sleeve.

A live decay-detection layer: rolling 60-day Sharpe per sleeve, scale capital
allocation accordingly. Forces strategies to earn their capital — the single
biggest separator between professional and amateur systematic trading.

Tier table:
    Sharpe       Scale   Rationale
    >  0.5       1.00    Earning Sharpe of professional CTA — full allocation
    0.3 - 0.5    0.75    Marginal — reduce 25%
    0.0 - 0.3    0.50    Barely positive — half size
    < 0.0        0.25    Losing — cut hard but don't kill (could be noise)
    < -0.5       0.00    Persistently negative — pause and alert

Below 10 daily observations, returns 1.0 (insufficient data — let the strategy
warm up).

Combine with sleeve_circuit_breakers for full risk management:
    scale = min(get_sleeve_scale(sleeve), get_sharpe_scale(sleeve))
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pnl_db import get_sleeve_sharpe
from ops.alerts import alert


# Tier table — (sharpe_threshold_inclusive, scale)
SHARPE_TIERS = [
    (0.5, 1.00),
    (0.3, 0.75),
    (0.0, 0.50),
    (-0.5, 0.25),
    (float("-inf"), 0.00),
]

MIN_OBSERVATIONS = 10  # below this, return 1.0 (insufficient data)


def get_sharpe_scale(sleeve: str, window_days: int = 60) -> float:
    """Return scale [0.0, 1.0] based on rolling Sharpe over window_days.

    Returns 1.0 if insufficient data (under 10 daily observations).
    """
    sharpe = get_sleeve_sharpe(sleeve, days=window_days)
    if sharpe is None:
        return 1.0  # not enough data yet
    for threshold, scale in SHARPE_TIERS:
        if sharpe >= threshold:
            return scale
    return 0.0


def get_all_gates_scale(sleeve: str, *, alt_regime: bool = False,
                          btc_regime: bool = False) -> dict:
    """Compose ALL active scales: drawdown CB + Sharpe + loss-streak + correlation + event.

    Returns dict with individual scales + effective combined scale.
    Sleeve runners call this and multiply position size by 'effective'.

    Args:
        sleeve: name (must match pnl_db sleeve tags).
        alt_regime: pass True for alt-focused sleeves (xsmom, oversold_bounce,
            overbought_fade alt baskets). Multiplies by btc_dominance.alt_regime_scale,
            which is 0 in BTC_HEGEMONY, 0.5 in BTC_DOMINANT, 1.2 in ALTSEASON.
            Defaults False (no effect — BTC-aware sizing is opt-in).
        btc_regime: pass True for BTC-focused sleeves (bah_btc, pro_trend BTC-only).
            Multiplies by btc_dominance.btc_regime_scale, which is 1.2 in
            BTC_HEGEMONY, 0.8 in ALTSEASON.

    Mutual exclusion: a sleeve may set ONE of alt_regime/btc_regime — not both.
    Composition order (multiplicative): meta-confidence × dominance × min(other gates).
    """
    from ops.sleeve_circuit_breakers import get_sleeve_scale as cb_scale
    from ops.loss_streak import loss_streak_scale
    from ops.correlation_guard import correlation_guard_scale
    from core.event_calendar import is_high_vol_window

    sharpe = get_sharpe_scale(sleeve)
    cb = cb_scale(sleeve)
    streak = loss_streak_scale(sleeve)
    corr = correlation_guard_scale()
    event = is_high_vol_window()
    event_scale = 0.0 if event.get("in_window") else 1.0  # full pause in event window

    # W15.B: META-CONFIDENCE (Lopez de Prado AFML Ch 3.7)
    # Multiplier in [0.5, 1.5] based on signal STRENGTH for this sleeve right now.
    # Composed MULTIPLICATIVELY since it can scale UP (1.5x) on strong signals,
    # unlike the other gates which only scale down.
    try:
        from core.meta_confidence import get_meta_confidence
        meta = get_meta_confidence(sleeve)
    except Exception:
        meta = 1.0

    # W16.H: BTC dominance regime gating
    #   ALT sleeves: reduce in BTC_DOMINANT/HEGEMONY (capital fleeing to BTC).
    #   BTC sleeves: reduce in ALTSEASON (capital rotating out of BTC).
    dom_scale = 1.0
    dom_regime = None
    if alt_regime or btc_regime:
        try:
            from core.btc_dominance import (
                alt_regime_scale, btc_regime_scale, fetch_dominance,
                regime_classification,
            )
            if alt_regime:
                dom_scale = alt_regime_scale()
            else:
                dom_scale = btc_regime_scale()
            d = fetch_dominance()
            if d is not None:
                dom_regime = regime_classification(d["btc_dominance_pct"]).get("regime")
        except Exception:
            dom_scale = 1.0

    # W16.C: Kelly multiplier (data-driven sizing from live sleeve returns).
    # Returns 1.0 for sleeves with <14 days of P&L history — fail neutral
    # so brand-new sleeves keep their natural sizing during warm-up.
    try:
        from core.kelly_sizing import kelly_multiplier
        kelly = kelly_multiplier(sleeve)
    except Exception:
        kelly = 1.0

    # W16.E: Tail hedge urgency → portfolio-wide derisk multiplier.
    # When tail_hedge urgency = critical (4+ risk factors), reduce ALL sleeve
    # sizing to 0.3x while the trader manually buys puts on Deribit.
    # This is a portfolio-level signal, applied uniformly across all sleeves.
    tail_scale = 1.0
    tail_urgency = None
    try:
        from core.tail_hedge import compute_hedge_recommendation
        h = compute_hedge_recommendation(bankroll_usd=200_000)
        tail_urgency = h.get("urgency", "unnecessary")
        tail_scale = {
            "critical":     0.3,   # severe regime risk — cut hard
            "recommended":  0.7,   # elevated — trim
            "optional":     1.0,
            "unnecessary":  1.0,
        }.get(tail_urgency, 1.0)
    except Exception:
        tail_scale = 1.0

    # W16.F: VaR Kupiec breach → soft circuit breaker.
    # daily_report writes .var_kupiec_breach.json when 1% VaR fails Kupiec test.
    # Flag persists 3 days, scaling ALL sleeves to 0.5x while the trader
    # recalibrates VaR. After 3 days the flag auto-expires (fail open).
    var_scale = 1.0
    var_breach_active = False
    try:
        import json as _json
        from pathlib import Path as _Path
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _breach_file = _Path(__file__).resolve().parent.parent / ".var_kupiec_breach.json"
        if _breach_file.exists():
            _d = _json.loads(_breach_file.read_text())
            _ts = _dt.fromisoformat(_d["timestamp"])
            if _dt.now(tz=_tz.utc) - _ts < _td(days=3):
                var_scale = 0.5
                var_breach_active = True
    except Exception:
        var_scale = 1.0

    base = min(sharpe, cb, streak, corr, event_scale)
    effective = base * meta * dom_scale * kelly * tail_scale * var_scale
    return {
        "sleeve": sleeve,
        "sharpe_scale": sharpe,
        "cb_scale": cb,
        "loss_streak_scale": streak,
        "correlation_scale": corr,
        "event_scale": event_scale,
        "meta_confidence": meta,
        "dominance_scale": dom_scale,
        "dominance_regime": dom_regime,
        "kelly_multiplier": kelly,
        "tail_hedge_scale": tail_scale,
        "tail_hedge_urgency": tail_urgency,
        "var_breach_scale": var_scale,
        "var_breach_active": var_breach_active,
        "event_active": event.get("in_window", False),
        "event_name": event.get("event", {}).get("name") if event.get("in_window") else None,
        "effective": effective,
    }


def get_sharpe_report(sleeves: list[str]) -> list[dict]:
    """Per-sleeve Sharpe + scale for dashboard/report rendering."""
    out = []
    for s in sleeves:
        sh60 = get_sleeve_sharpe(s, days=60)
        sh30 = get_sleeve_sharpe(s, days=30)
        scale = get_sharpe_scale(s)
        out.append({
            "sleeve": s,
            "sharpe_60d": sh60,
            "sharpe_30d": sh30,
            "scale": scale,
            "status": _status_label(sh60, scale),
        })
    return out


def _status_label(sharpe, scale) -> str:
    if sharpe is None:
        return "warming-up"
    if scale == 0.0:
        return "PAUSED (Sharpe < -0.5)"
    if scale < 1.0:
        return f"REDUCED ({scale:.0%})"
    return "OK"


def main():
    """CLI status. Run `python -m ops.sharpe_gate` to see per-sleeve gates."""
    sleeves = ["spot_orchestrator", "perp_orchestrator", "bah_btc", "xsmom", "pro_trend"]
    print(f"{'Sleeve':<22s} {'Sharpe60d':>11s} {'Sharpe30d':>11s} {'Scale':>7s} {'Status':<20s}")
    print("-" * 80)
    for r in get_sharpe_report(sleeves):
        s60 = f"{r['sharpe_60d']:+.2f}" if r['sharpe_60d'] is not None else "n/a"
        s30 = f"{r['sharpe_30d']:+.2f}" if r['sharpe_30d'] is not None else "n/a"
        print(f"{r['sleeve']:<22s} {s60:>11s} {s30:>11s} {r['scale']:>6.2f}x {r['status']:<20s}")


if __name__ == "__main__":
    main()
