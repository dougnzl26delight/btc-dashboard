"""ETF-AWARE BOTTOM TRIGGER — combines hard scorecard with ETF flow context.

Traditional cycles (3, 4) had retail capitulation drive the bottom — Realized Cap
dropped 15-25%, MVRV-Z went below -1.0, Coinbase Premium flipped positive. The
scorecard's 8 hard criteria were calibrated against that pattern.

ETF era (cycle 5) is different: institutions absorb supply via ETFs at higher
prices, so the traditional capitulation signals may NEVER reach the full 6/8
threshold. A shallower bottom in the $60-70k range with only 4-5/8 criteria
firing is possible — and would still be the actionable bottom.

This module produces a 4-trigger system that combines scorecard count with
ETF flow direction:

    TRIGGER 1A (scorecard 4+/8 + ETF inflows positive):
      Shallow ETF-era bottom forming. Deploy 50% (institutions absorbing).
      Expected entry zone: $60-70k.

    TRIGGER 1B (scorecard 4+/8 + ETF outflows 5d):
      Real bottom forming — institutions also paused. Deploy 75%.
      Expected entry zone: $55-65k.

    TRIGGER 2 (scorecard 6+/8):
      Traditional bottom confirmed. Deploy 100% aggressive.
      Expected entry zone: $50-60k.

    TRIGGER 4 (ETF flows 30d outflows < -$10B):
      Bear deepening — institutions retreating. Wait longer, don't deploy.

    TRIGGER 0 / WAIT (everything else, including current state):
      Cost basis not capitulated, no actionable signal. Cash is a position.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _classify_etf_flow(last_5d: float, last_30d: float) -> str:
    """Return one of: STRONG_POSITIVE, POSITIVE, FLAT, NEGATIVE, STRONG_NEGATIVE."""
    # Strong negative = sustained outflows (30d < -$10B)
    if last_30d < -10_000:
        return "STRONG_NEGATIVE"
    # Negative = 5d outflows
    if last_5d < -1_000:
        return "NEGATIVE"
    # Strong positive = 5d >= +$5B
    if last_5d > 5_000:
        return "STRONG_POSITIVE"
    # Positive = 5d > 0
    if last_5d > 0:
        return "POSITIVE"
    return "FLAT"


def etf_aware_bottom_trigger(state: Optional[dict] = None) -> dict:
    """Compute the active deployment trigger.

    Returns:
        trigger_id:    "WAIT", "EARLY", "1A", "1B", "2", "4"
        trigger_name:  Display name
        deploy_pct:    Recommended deploy % (0-100, or "DCA")
        color:         Semantic color
        rationale:     Why this trigger fires
        scorecard:     Underlying scorecard dict
        etf_status:    ETF flow classification
        next_levels:   What would change the trigger
    """
    # Pull state if not provided
    if state is None:
        from core.btc_prediction import state_of_btc
        state = state_of_btc()

    # Pull scorecard
    from core.btc_bottom_scorecard import bottom_confirmation_scorecard
    sc = bottom_confirmation_scorecard(state)
    n_met = sc["n_met"]
    n_total = sc["n_total"]

    # ETF flows
    etf_sig = None
    for cat in ("flows", "fundamentals"):
        d = state.get("signals", {}).get(cat, {}).get("etf_flows")
        if isinstance(d, dict) and not d.get("error"):
            etf_sig = d
            break
    etf_last_5d = etf_sig.get("last_5d_M", 0) if etf_sig else 0
    etf_last_30d = etf_sig.get("last_30d_M", 0) if etf_sig else 0
    etf_last_day = etf_sig.get("last_day_M", 0) if etf_sig else 0
    etf_status = _classify_etf_flow(etf_last_5d, etf_last_30d) if etf_sig else "UNKNOWN"

    # Clemente+Alden layer average (15 signals — institutional bottom indicators)
    ca_sig_names = [
        "hashrate_drawdown", "cb_premium_streak", "aasi",
        "stablecoin_supply_ratio", "etf_pct_of_supply", "btc_dominance",
        "real_yields_10y", "difficulty_adjustment", "btc_gold_ratio",
        "multi_exch_funding", "rhodl_ratio", "reflexivity_index",
        "urpd_clusters", "hodl_waves", "fiscal_dominance",
    ]
    ca_scores = []
    for sig_name in ca_sig_names:
        for cat in ("flows", "onchain", "fundamentals", "macro", "derivatives", "regime_models"):
            d = state.get("signals", {}).get(cat, {}).get(sig_name)
            if isinstance(d, dict) and not d.get("error"):
                s = d.get("score")
                if s is not None: ca_scores.append(s)
                break
    ca_avg = sum(ca_scores) / max(1, len(ca_scores)) if ca_scores else 0
    # Clemente "strong bottom" = avg > 0.4 with 5+ bullish signals
    ca_strong_bottom = ca_avg > 0.4 and sum(1 for s in ca_scores if s > 0.3) >= 5

    # Trigger decision tree
    # NEW: Trigger 1C — Clemente+Alden strong bottom even with low scorecard
    # (institutional signals firing before traditional cohort metrics)
    if ca_strong_bottom and n_met >= 3:
        return {
            "trigger_id": "1C",
            "trigger_name": "TRIGGER 1C — CLEMENTE+ALDEN STRONG BOTTOM",
            "verdict_label": "SCALE IN 60%",
            "deploy_pct": 60,
            "color": "#26a69a",
            "rationale": (f"Clemente+Alden layer avg {ca_avg:+.2f} with "
                           f"{sum(1 for s in ca_scores if s > 0.3)} bullish signals "
                           f"(hashrate cap, CB premium streak, AASI, SSR). "
                           f"Scorecard only {n_met}/{n_total} but institutional layer "
                           f"called the 2024 bottom 6 weeks before traditional metrics."),
            "entry_zone": "$60-70k expected",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "ca_avg": ca_avg,
            "ca_bullish_count": sum(1 for s in ca_scores if s > 0.3),
            "next_levels": ("If scorecard climbs to 5+ → Trigger 1B (75%). "
                             "If scorecard hits 7+ → Trigger 2 (100%)."),
        }

    if n_met >= 7:
        return {
            "trigger_id": "2",
            "trigger_name": "TRIGGER 2 — TRADITIONAL BOTTOM CONFIRMED",
            "verdict_label": "DEPLOY 100%",
            "deploy_pct": 100,
            "color": "#1b5e20",      # deep green
            "rationale": (f"{n_met}/{n_total} hard criteria firing — traditional "
                           f"capitulation pattern. ETF flow direction doesn't matter "
                           f"at this level of confirmation."),
            "entry_zone": "$50-60k expected",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "next_levels": "Cash is fully deployed. Watch for recovery confirmation.",
        }

    if n_met >= 5 and etf_status in ("NEGATIVE", "STRONG_NEGATIVE"):
        return {
            "trigger_id": "1B",
            "trigger_name": "TRIGGER 1B — REAL BOTTOM FORMING",
            "verdict_label": "SCALE IN 75%",
            "deploy_pct": 75,
            "color": "#26a69a",      # green
            "rationale": (f"{n_met}/{n_total} hard criteria + ETF outflows "
                           f"(${etf_last_5d:+,.0f}M 5d). Both retail and institutions "
                           f"paused = real bottom forming. Deploy 75%, hold 25% reserve."),
            "entry_zone": "$55-65k expected",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "next_levels": ("Hold 25% reserve for tier-2 if price drops further "
                             "(only if scorecard reaches 6+/8)."),
        }

    if n_met >= 5 and etf_status in ("POSITIVE", "STRONG_POSITIVE", "FLAT"):
        return {
            "trigger_id": "1A",
            "trigger_name": "TRIGGER 1A — SHALLOW ETF-ERA BOTTOM",
            "verdict_label": "SCALE IN 50%",
            "deploy_pct": 50,
            "color": "#66bb6a",      # lighter green
            "rationale": (f"{n_met}/{n_total} hard criteria with ETF inflows still "
                           f"absorbing supply (${etf_last_5d:+,.0f}M 5d). Bottom "
                           f"likely shallow due to institutional wall. Deploy 50%."),
            "entry_zone": "$60-70k expected",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "next_levels": ("If scorecard reaches 6+/8 → deploy remaining 50% (Trigger 2). "
                             "If ETF flows go negative → consider 25% more (Trigger 1B mix)."),
        }

    if etf_status == "STRONG_NEGATIVE":
        return {
            "trigger_id": "4",
            "trigger_name": "TRIGGER 4 — BEAR DEEPENING",
            "verdict_label": "WAIT LONGER",
            "deploy_pct": 0,
            "color": "#b71c1c",      # deep red
            "rationale": (f"ETF outflows ${etf_last_30d:+,.0f}M over 30 days = "
                           f"institutions retreating. Cycle bear may be deepening. "
                           f"Wait another 60-90 days minimum before deploying."),
            "entry_zone": "$45-55k possible",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "next_levels": "Watch for ETF flow stabilization + scorecard 4+/8 combo.",
        }

    if n_met >= 2:
        return {
            "trigger_id": "EARLY",
            "trigger_name": "EARLY SIGNALS — PREPARE",
            "verdict_label": "WATCH",
            "deploy_pct": 0,
            "color": "#f0b90b",      # yellow
            "rationale": (f"{n_met}/{n_total} hard criteria — some signals firing but "
                           f"not enough to deploy. ETF flows ${etf_last_5d:+,.0f}M 5d. "
                           f"Capital should be liquid and ready."),
            "entry_zone": "Watching for trigger",
            "scorecard": sc,
            "etf_status": etf_status,
            "etf_5d_M": etf_last_5d,
            "etf_30d_M": etf_last_30d,
            "next_levels": ("Trigger 1A/1B fires when scorecard hits 4/8. "
                             "Trigger 2 fires at 6/8."),
        }

    # Default — no actionable signal
    return {
        "trigger_id": "WAIT",
        "trigger_name": "NO SIGNAL — WAIT",
        "verdict_label": "WAIT",
        "deploy_pct": 0,
        "color": "#ef5350",      # red
        "rationale": (f"{n_met}/{n_total} hard criteria — cost basis not capitulated. "
                       f"ETF flows ${etf_last_5d:+,.0f}M 5d "
                       f"({'institutional bull' if etf_last_5d > 0 else 'flat/negative'}). "
                       f"Cash is a position."),
        "entry_zone": "No actionable zone yet",
        "scorecard": sc,
        "etf_status": etf_status,
        "etf_5d_M": etf_last_5d,
        "etf_30d_M": etf_last_30d,
        "next_levels": ("Watch for: scorecard 2+/8 (early signals) → "
                         "scorecard 4+/8 (Trigger 1A/1B deploy 50-75%)."),
    }


def main():
    print("\n" + "=" * 78)
    print("ETF-AWARE BOTTOM TRIGGER")
    print("=" * 78)
    t = etf_aware_bottom_trigger()
    print()
    print(f"  Active trigger:  {t['trigger_id']}")
    print(f"  Verdict:         {t['verdict_label']}")
    print(f"  Deploy %:        {t['deploy_pct']}%")
    print(f"  Entry zone:      {t['entry_zone']}")
    print(f"  ETF status:      {t['etf_status']}")
    print(f"  ETF 5d:          ${t['etf_5d_M']:+,.0f}M")
    print(f"  ETF 30d:         ${t['etf_30d_M']:+,.0f}M")
    print(f"  Scorecard:       {t['scorecard']['n_met']}/{t['scorecard']['n_total']}")
    print()
    print(f"  Rationale:")
    print(f"    {t['rationale']}")
    print()
    print(f"  Next levels:")
    print(f"    {t['next_levels']}")


if __name__ == "__main__":
    main()
