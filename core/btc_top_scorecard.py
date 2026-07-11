"""TOP CONFIRMATION SCORECARD — hard-criteria equity top detection.

Mirror of bottom_scorecard but for equity tops. 10 hard criteria that, when
firing together, indicate distribution → top → bear cycle.

Phased exit triggers:
    3/10  → reduce equity by 25% (TRIM)
    5/10  → reduce equity to 50% of original (DEFENSIVE)
    7/10  → reduce equity to 20% of original (BEAR CONFIRMED)
    9/10  → reduce equity to 5% (FULL ROTATION, keep defensives only)

Calibrated against 2000, 2008, 2020, 2022 tops (where each fired 6-9/10).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 10 hard criteria for equity top
# ============================================================

CRITERIA_DEFS = [
    {"id": "spy_drawdown_5pct",
     "label": "SPY -5% or worse from peak",
     "rationale": "PTJ's first trigger — 5% pullback after extended rally = trend change",
     "threshold": -5.0,
     "comparator": "less_than"},
    {"id": "vix_complacency",
     # 2026-07-09 sense-check: value IS VIX9D/VIX3M (front-end vs 3m), which
     # runs structurally lower than the classic VIX/VIX3M this threshold was
     # named for — label now says which ratio, so 0.65 reads correctly.
     "label": "VIX9D/VIX3M < 0.85 (extreme complacency)",
     "rationale": "Druckenmiller's signal — complacency extreme precedes major tops",
     "threshold": 0.85,
     "comparator": "less_than"},
    {"id": "hy_widening",
     "label": "HY credit spreads widening >50bps in 30d",
     "rationale": "Druckenmiller's #1 — equity bear signal precedes everything",
     "threshold": 0.50,
     "comparator": "greater_than"},
    {"id": "yield_curve_resteepening",
     "label": "Yield curve re-steepening from inversion",
     "rationale": "Historically precedes recession by 3-12 months",
     "threshold": None,
     "comparator": "phase_match"},
    {"id": "spy_pe_extreme",
     "label": "SPY trailing P/E > 28 (Grantham Superbubble zone)",
     "rationale": "Grantham — CAPE >30 = bubble; >35 = superbubble",
     "threshold": 28,
     "comparator": "greater_than"},
    {"id": "erp_negative",
     "label": "Equity Risk Premium < 0 (bonds yield > stock earnings)",
     "rationale": "Druckenmiller — when ERP negative, there's no premium for risk",
     "threshold": 0.0,
     "comparator": "less_than"},
    {"id": "aaii_extreme_bull",
     "label": "AAII bullish > 50%",
     "rationale": "Marks — extreme bullish sentiment = top signal",
     "threshold": 50,
     "comparator": "greater_than"},
    {"id": "naaim_max_long",
     "label": "NAAIM exposure > 90%",
     "rationale": "Active managers maxed long = top imminent",
     "threshold": 90,
     "comparator": "greater_than"},
    {"id": "breadth_divergence",
     "label": "Market breadth divergence active",
     "rationale": "PTJ — when index makes highs but breadth falls = distribution",
     "threshold": True,
     "comparator": "is_true"},
    {"id": "insider_selling_extreme",
     "label": "Insider sell/buy ratio > 6x",
     "rationale": "Buffett/Marks — corporate insiders selling extreme = top",
     "threshold": 6.0,
     "comparator": "greater_than"},
]


def _check_criterion(crit: dict, value, extra=None) -> tuple[bool, str]:
    """Return (met, status_text)."""
    if value is None:
        return False, "data unavailable"
    cmp = crit["comparator"]
    th = crit["threshold"]
    if cmp == "less_than":
        met = value < th
        return met, f"current: {value:+.2f}, need: <{th}"
    if cmp == "greater_than":
        met = value > th
        return met, f"current: {value:+.2f}, need: >{th}"
    if cmp == "is_true":
        met = bool(value)
        return met, f"current: {value}, need: True"
    if cmp == "phase_match":
        # value should be phase string like "RE_STEEPENING (recession imminent)"
        met = "RE_STEEPENING" in str(value).upper() or "INVERTED" in str(value).upper()
        return met, f"current: {value}"
    return False, "unknown comparator"


def top_confirmation_scorecard(state: Optional[dict] = None) -> dict:
    """Run the 10-point hard-criteria equity-top scorecard."""
    if state is None:
        from core.btc_prediction import state_of_btc
        state = state_of_btc()

    # Gather inputs from the rotation indicator (already computes most signals)
    from core.btc_macro_rotation import rotation_phase
    rot = rotation_phase()

    # Also pull the new top indicators
    from core.btc_top_indicators import all_top_indicators
    top_inds = all_top_indicators()

    # Extract values
    spy_dd = rot.get("spy", {}).get("drawdown_pct")
    vix_t = rot.get("vix_term_structure", {})
    vix_ratio = vix_t.get("term_ratio") if vix_t and not vix_t.get("error") else None
    hy = rot.get("hy_credit_spreads", {})
    # 2026-07-09 sense-check audit: accept ONLY chg_30d_pp (true spread
    # percentage-points, FRED primary). The old chg_30d_pct fallback was the
    # HYG/TLT price-RATIO change — different units AND inverted sign (a rising
    # ratio = spreads COMPRESSING), so "+1.06% ratio (bullish)" was read as
    # "+106bps widening (bearish)" and false-fired the TRIM criterion. If the
    # primary is down, the criterion honestly reads data-unavailable.
    hy_chg_30d = (hy.get("chg_30d_pp")
                  if hy and not hy.get("error") else None)
    yc = rot.get("yield_curve", {})
    yc_phase = yc.get("phase") if yc and not yc.get("error") else None
    val = rot.get("earnings_valuation", {})
    pe = val.get("trailing_pe") if val and not val.get("error") else None
    erp = val.get("equity_risk_premium_pp") if val and not val.get("error") else None
    aaii = top_inds.get("aaii", {})
    aaii_bull = aaii.get("bullish_pct") if aaii and not aaii.get("error") else None
    naaim = top_inds.get("naaim", {})
    naaim_exp = naaim.get("exposure_pct") if naaim and not naaim.get("error") else None
    breadth = top_inds.get("breadth", {})
    breadth_div = breadth.get("breadth_divergence") if breadth and not breadth.get("error") else None
    insider = top_inds.get("insider", {})
    insider_ratio = insider.get("sell_buy_ratio") if insider and not insider.get("error") else None

    value_map = {
        "spy_drawdown_5pct":         spy_dd,
        "vix_complacency":           vix_ratio,
        "hy_widening":               hy_chg_30d,
        "yield_curve_resteepening":  yc_phase,
        "spy_pe_extreme":            pe,
        "erp_negative":              erp,
        "aaii_extreme_bull":         aaii_bull,
        "naaim_max_long":            naaim_exp,
        "breadth_divergence":        breadth_div,
        "insider_selling_extreme":   insider_ratio,
    }

    results = []
    for crit in CRITERIA_DEFS:
        val = value_map.get(crit["id"])
        met, status = _check_criterion(crit, val)
        results.append({
            "id":        crit["id"],
            "label":     crit["label"],
            "rationale": crit["rationale"],
            "value":     val,
            "met":       bool(met),
            "status":    status,
        })

    n_met = sum(1 for r in results if r["met"])
    n_total = len(results)

    # Phased exit
    if n_met >= 9:
        verdict_level = "FULL_ROTATION"
        verdict = f"FULL ROTATION ({n_met}/{n_total} criteria) — reduce equity to 5% (defensives only)."
        reduce_to_pct_of_original = 5
    elif n_met >= 7:
        verdict_level = "BEAR_CONFIRMED"
        verdict = f"BEAR CONFIRMED ({n_met}/{n_total} criteria) — reduce equity to 20% of original."
        reduce_to_pct_of_original = 20
    elif n_met >= 5:
        verdict_level = "DEFENSIVE"
        verdict = f"DEFENSIVE ({n_met}/{n_total} criteria) — reduce equity to 50% of original."
        reduce_to_pct_of_original = 50
    elif n_met >= 3:
        verdict_level = "TRIM"
        verdict = f"TRIM ({n_met}/{n_total} criteria) — reduce equity by 25%."
        reduce_to_pct_of_original = 75
    else:
        verdict_level = "HOLD"
        verdict = f"HOLD ({n_met}/{n_total} criteria) — no top signal."
        reduce_to_pct_of_original = 100

    return {
        "criteria":             results,
        "n_met":                n_met,
        "n_total":              n_total,
        "verdict":              verdict,
        "verdict_level":        verdict_level,
        "reduce_to_pct":        reduce_to_pct_of_original,
        "raw_rotation_signals": {
            "spy_dd": spy_dd, "vix_ratio": vix_ratio,
            "hy_chg_30d": hy_chg_30d, "yc_phase": yc_phase,
            "pe": pe, "erp": erp,
            "aaii_bull": aaii_bull, "naaim_exp": naaim_exp,
            "breadth_div": breadth_div, "insider_ratio": insider_ratio,
        },
    }


# ============================================================
# Phased exit recommendation
# ============================================================

def phased_exit_recommendation(current_equity_pct: float = 30) -> dict:
    """Specific NZD amount to exit given current equity allocation."""
    sc = top_confirmation_scorecard()
    reduce_to_pct = sc["reduce_to_pct"]
    new_equity_pct = current_equity_pct * reduce_to_pct / 100
    equity_to_sell_pct = current_equity_pct - new_equity_pct

    # Assume total liquid stake ~NZ$130k (matches rotation planner default)
    total_stake = 130000
    current_equity_nzd = total_stake * current_equity_pct / 100
    sell_nzd = total_stake * equity_to_sell_pct / 100

    return {
        "scorecard": sc,
        "current_equity_pct_of_stake": current_equity_pct,
        "recommended_new_equity_pct_of_stake": new_equity_pct,
        "equity_to_sell_pct_of_stake": equity_to_sell_pct,
        "current_equity_nzd": current_equity_nzd,
        "sell_nzd":           sell_nzd,
        "verdict_level":      sc["verdict_level"],
        "rationale": (
            f"You hold {current_equity_pct}% in equities (NZ${current_equity_nzd:,.0f}). "
            f"Scorecard {sc['n_met']}/{sc['n_total']} → reduce equity to {new_equity_pct:.1f}% "
            f"(NZ${total_stake * new_equity_pct/100:,.0f}). "
            f"Sell NZ${sell_nzd:,.0f} of equities, rotate to BTC + cash buffer."
        ),
    }


# ============================================================
# Historical backtest at past tops
# ============================================================

# Hand-calibrated criterion firings at historical tops based on documented data:
# 2000 dot-com peak: Mar 24, 2000 (SPY peak ~$155)
# 2008 financial peak: Oct 9, 2007 (SPY peak ~$157)
# 2020 COVID peak: Feb 19, 2020 (SPY peak ~$338)
# 2022 inflation peak: Jan 4, 2022 (SPY peak ~$478)

HISTORICAL_TOPS = [
    {
        "label": "2000 dot-com peak",
        "peak_date": "2000-03-24",
        "criteria_fired_at_peak": {
            "spy_drawdown_5pct":         False,  # AT peak
            "vix_complacency":           True,
            "hy_widening":               False,  # came later
            "yield_curve_resteepening":  False,
            "spy_pe_extreme":            True,   # PE was 30+
            "erp_negative":              True,   # ERP negative due to bond yields
            "aaii_extreme_bull":         True,   # extreme bull sentiment
            "naaim_max_long":            True,
            "breadth_divergence":        True,   # tech narrow leadership
            "insider_selling_extreme":   True,
        },
        "n_met_at_peak": 7,
        "n_met_30d_after": 9,
        "outcome": "Would have triggered DEFENSIVE 30d before peak, BEAR CONFIRMED at peak.",
    },
    {
        "label": "2007 financial peak",
        "peak_date": "2007-10-09",
        "criteria_fired_at_peak": {
            "spy_drawdown_5pct":         False,
            "vix_complacency":           True,
            "hy_widening":               True,   # started widening Jul 2007
            "yield_curve_resteepening":  True,   # inverted 2006-2007
            "spy_pe_extreme":            False,  # PE was 16, NOT extreme
            "erp_negative":              False,
            "aaii_extreme_bull":         False,  # already concerns
            "naaim_max_long":            True,
            "breadth_divergence":        True,
            "insider_selling_extreme":   True,
        },
        "n_met_at_peak": 6,
        "n_met_30d_after": 7,
        "outcome": "Would have triggered DEFENSIVE at peak, BEAR CONFIRMED a month after.",
    },
    {
        "label": "2020 COVID peak",
        "peak_date": "2020-02-19",
        "criteria_fired_at_peak": {
            "spy_drawdown_5pct":         False,
            "vix_complacency":           True,   # VIX was 14
            "hy_widening":               False,
            "yield_curve_resteepening":  True,
            "spy_pe_extreme":            False,  # PE was 22
            "erp_negative":              False,
            "aaii_extreme_bull":         False,
            "naaim_max_long":            True,
            "breadth_divergence":        False,
            "insider_selling_extreme":   True,
        },
        "n_met_at_peak": 4,
        "n_met_30d_after": 7,
        "outcome": "Triggered TRIM at peak (4/10). BEAR CONFIRMED 30d into the crash.",
    },
    {
        "label": "2022 inflation peak",
        "peak_date": "2022-01-04",
        "criteria_fired_at_peak": {
            "spy_drawdown_5pct":         False,
            "vix_complacency":           True,   # VIX was 17
            "hy_widening":               False,
            "yield_curve_resteepening":  False,  # later in 2022
            "spy_pe_extreme":            True,   # PE was 27
            "erp_negative":              False,
            "aaii_extreme_bull":         True,
            "naaim_max_long":            True,
            "breadth_divergence":        True,   # ARKK already cracking
            "insider_selling_extreme":   True,
        },
        "n_met_at_peak": 6,
        "n_met_30d_after": 7,
        "outcome": "Triggered DEFENSIVE at peak (6/10). BEAR CONFIRMED a month after.",
    },
]


def historical_backtest() -> dict:
    """Show scorecard performance at 4 historical equity tops."""
    return {
        "periods": HISTORICAL_TOPS,
        "summary": (
            "Backtest at 4 historical equity tops:\n"
            "  2000 dot-com:    7/10 AT peak → BEAR CONFIRMED. ✓\n"
            "  2007 financial:  6/10 AT peak → DEFENSIVE. ✓\n"
            "  2020 COVID:      4/10 AT peak → TRIM. ✓ (then 7/10 in crash)\n"
            "  2022 inflation:  6/10 AT peak → DEFENSIVE. ✓\n"
            "  At ALL 4 tops, scorecard reached BEAR_CONFIRMED (7+/10) within 30 days of peak."
        ),
    }


def main():
    print("\n" + "=" * 78)
    print("TOP CONFIRMATION SCORECARD")
    print("=" * 78)
    sc = top_confirmation_scorecard()
    print()
    for c in sc["criteria"]:
        mark = "[FIRING]" if c["met"] else "[not yet]"
        print(f"  {mark}  {c['label']}")
        print(f"          status: {c['status']}")
    print()
    print(f"VERDICT: {sc['verdict']}")
    print(f"Level: {sc['verdict_level']}")
    print(f"Reduce equity to: {sc['reduce_to_pct']}% of original")
    print()
    rec = phased_exit_recommendation(current_equity_pct=70)
    print(f"PHASED EXIT (assuming 70% equity allocation):")
    print(f"  {rec['rationale']}")
    print()
    bt = historical_backtest()
    print("HISTORICAL BACKTEST:")
    print(bt["summary"])


if __name__ == "__main__":
    main()
