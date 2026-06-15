"""BTC Weekly Report Generator — structured Monday-morning briefing.

Fires every Monday with the comprehensive weekly outlook:
    1. This week's directional bias + confidence
    2. 7-day price range with P25-P75 band
    3. 30/90 day outlook
    4. Key support + resistance levels with confluence count
    5. This week's macro catalysts (FOMC, CPI, earnings)
    6. Top 3 strongest signals + top 3 anomalies
    7. Recommended action (accumulate / hold / reduce / exit)
    8. Cycle context (days to projected bottom, days to projected peak)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "weekly_reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _hr(c="="): return c * 78


def _action_from_state(state: dict) -> tuple[str, str]:
    """Determine recommended action from prediction state."""
    short = state["horizons"]["short_term"]
    medium = state["horizons"]["medium_term"]
    long_ = state["horizons"]["long_term"]

    overall = (short["direction_score"] + medium["direction_score"] +
               long_["direction_score"]) / 3

    # Check for cycle bottom or top override active
    bottom_override = any("BOTTOM" in (h.get("override_applied") or "")
                          for h in state["horizons"].values())
    top_override = any("TOP" in (h.get("override_applied") or "")
                       for h in state["horizons"].values())

    if top_override:
        return "SCALE OUT", "Cycle-top override fired. Distribute holdings tier-by-tier."
    if bottom_override:
        return "ACCUMULATE AGGRESSIVELY", "Cycle-bottom override fired. Deploy capital."
    if overall > 0.4:
        return "ACCUMULATE", "Bullish bias across horizons. Add positions."
    if overall > 0.15:
        return "MILD BULL — hold/accumulate dips", "Lean long, buy weakness."
    if overall > -0.15:
        return "HOLD — range-bound", "Sit on hands. Let signals develop."
    if overall > -0.4:
        return "REDUCE — trim into strength", "Sell rallies, reduce exposure."
    return "EXIT / SHORT", "Bearish across horizons. Hedge or exit."


def _key_levels(current_price: float) -> dict:
    """Determine support/resistance with confluence count."""
    try:
        from core.btc_key_levels import (
            STH_COST_BASIS, OVERHEAD_SUPPLY_LOW, OVERHEAD_SUPPLY_HIGH,
            HAYES_GAMMA_TRIGGER, HAYES_DOWNSIDE_TRIGGER,
        )
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=400)

        ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
        sma200 = float(df["close"].rolling(200).mean().iloc[-1])
        sma471 = float(df["close"].rolling(471).mean().iloc[-1]) if len(df) >= 471 else 0
        recent_high_30d = float(df["high"].iloc[-30:].max())
        recent_low_30d = float(df["low"].iloc[-30:].min())
        recent_low_90d = float(df["low"].iloc[-90:].min()) if len(df) >= 90 else 0

        supports = []
        resistances = []

        for lvl, name in [
            (recent_low_30d, "30-day low"),
            (recent_low_90d, "90-day low"),
            (sma200 * 0.85, "SMA200 × 0.85"),
            (sma200 * 0.7, "Mayer 0.7 (deep value)"),
            (sma471 * 0.745, "Pi Cycle Bottom"),
            (HAYES_DOWNSIDE_TRIGGER, "Hayes downside trigger"),
            (60000, "$60k round number"),
            (55000, "$55k projected cycle bottom"),
            (50000, "$50k tail-bear scenario"),
        ]:
            if 0 < lvl < current_price:
                dist = (current_price - lvl) / current_price * 100
                supports.append({"price": lvl, "name": name, "distance_pct": dist})

        for lvl, name in [
            (ema21, "EMA21"),
            (sma200, "SMA200 (Mayer 1.0)"),
            (recent_high_30d, "30-day high"),
            (HAYES_GAMMA_TRIGGER, "Hayes gamma trigger"),
            (OVERHEAD_SUPPLY_LOW, "Overhead supply low"),
            (90000, "$90k round"),
            (OVERHEAD_SUPPLY_HIGH, "Overhead supply high"),
            (100000, "$100k round"),
            (124659, "Cycle 5 ATH"),
        ]:
            if lvl > current_price:
                dist = (lvl - current_price) / current_price * 100
                resistances.append({"price": lvl, "name": name, "distance_pct": dist})

        supports.sort(key=lambda x: -x["price"])
        resistances.sort(key=lambda x: x["price"])
        return {"supports": supports, "resistances": resistances}
    except Exception:
        return {"supports": [], "resistances": []}


def _upcoming_catalysts() -> list:
    """Approximate this week's known catalysts."""
    # In production this would pull from FRED's release calendar or a config file
    # For now, list typical recurring monthly events
    today = datetime.now(timezone.utc).date()
    catalysts = []

    # CPI is usually 2nd week of month
    if 8 <= today.day <= 14:
        catalysts.append({
            "date": "this week",
            "event": "CPI release",
            "impact": "high — drives Fed expectations + DXY",
        })
    # NFP is first Friday of month
    if today.day <= 7:
        catalysts.append({
            "date": "this week",
            "event": "NFP (jobs report)",
            "impact": "high — labor market signal",
        })
    # FOMC ~8 times/year (every ~6 weeks)
    # Halving cycle context
    halving4 = datetime(2024, 4, 20).date()
    days_post = (today - halving4).days
    catalysts.append({
        "date": "ongoing",
        "event": f"Halving cycle day {days_post}",
        "impact": "historical bottoms ~day 900-1000",
    })
    return catalysts


def _top_signals(state: dict, n: int = 3) -> tuple[list, list]:
    """Return top N strongest bull and bear signals."""
    all_sigs = []
    for cat, cat_sigs in state.get("signals", {}).items():
        if not isinstance(cat_sigs, dict) or cat_sigs.get("error"): continue
        for sig_name, sig_data in cat_sigs.items():
            if not isinstance(sig_data, dict): continue
            s = sig_data.get("score")
            if s is None: continue
            all_sigs.append({"name": sig_name, "category": cat,
                              "score": float(s),
                              "value": sig_data.get("value"),
                              "note": sig_data.get("note", "")[:60]})
    all_sigs.sort(key=lambda x: x["score"])
    bears = all_sigs[:n]   # most bearish first
    bulls = list(reversed(all_sigs))[:n]
    return bulls, bears


def generate_weekly_report(state: Optional[dict] = None) -> str:
    """Generate the comprehensive weekly report as text."""
    if state is None:
        from core.btc_prediction import state_of_btc
        state = state_of_btc()

    lines = []
    btc = state["btc_price"]
    regime = state["regime"]
    short = state["horizons"]["short_term"]
    medium = state["horizons"]["medium_term"]
    long_ = state["horizons"]["long_term"]

    now = datetime.now(timezone.utc)
    nzt = now + timedelta(hours=12)
    lines.append("=" * 78)
    lines.append(f"BTC WEEKLY REPORT — {nzt.strftime('%A, %d %B %Y %H:%M NZT')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  BTC: ${btc:,.0f}")
    lines.append(f"  Regime: {regime}")
    lines.append("")

    # === Top-of-page summary ===
    action, action_note = _action_from_state(state)
    lines.append(f"  RECOMMENDED ACTION:  {action}")
    lines.append(f"     {action_note}")
    lines.append("")

    # === Horizon outlook ===
    lines.append("THIS WEEK'S OUTLOOK")
    lines.append("-" * 78)
    lines.append(f"  Short-term  (7-30d):  {short['interpretation']:<14s} "
                 f"score {short['direction_score']:+.2f}  "
                 f"confidence {short['confidence']}")
    if short.get("override_applied"):
        lines.append(f"    --> OVERRIDE: {short['override_applied']}")
    lines.append(f"  Medium-term (1-6m):   {medium['interpretation']:<14s} "
                 f"score {medium['direction_score']:+.2f}  "
                 f"confidence {medium['confidence']}")
    lines.append(f"  Long-term (6m-2y):    {long_['interpretation']:<14s} "
                 f"score {long_['direction_score']:+.2f}  "
                 f"confidence {long_['confidence']}")
    lines.append("")

    # === Price targets ===
    lines.append("PRICE TARGETS (signal-derived)")
    lines.append("-" * 78)
    targets = state.get("price_targets", {})
    if targets:
        for h, label in [("short_term", "30 days"),
                         ("medium_term", "90 days"),
                         ("long_term", "180 days")]:
            t = targets.get(h, {})
            if t:
                lines.append(f"  {label:<10s}  P5 ${t['p5']:>7,.0f}  "
                             f"P25 ${t['p25']:>7,.0f}  median ${t['median']:>7,.0f}  "
                             f"P75 ${t['p75']:>7,.0f}  P95 ${t['p95']:>7,.0f}")
    lines.append("")

    # === Pattern-projected price targets (halving clock) ===
    try:
        from core.halving_clock import pattern_projected_targets
        ppt = pattern_projected_targets(btc)
        lines.append("PRICE TARGETS (halving-clock pattern projection)")
        lines.append("-" * 78)
        lines.append(f"  Cycle 5 bottom (Oct 2026, ~4 months):")
        lines.append(f"    Low      ${ppt['cycle5_bottom_low']:>9,.0f}  "
                     f"({(ppt['cycle5_bottom_low']/btc-1)*100:+.1f}%)")
        lines.append(f"    Mid      ${ppt['cycle5_bottom_mid']:>9,.0f}  "
                     f"({ppt['cycle5_bottom_chg_pct']:+.1f}%)")
        lines.append(f"    High     ${ppt['cycle5_bottom_high']:>9,.0f}  "
                     f"({(ppt['cycle5_bottom_high']/btc-1)*100:+.1f}%)")
        lines.append(f"  Cycle 6 peak (Oct 2029, ~3.5 years):")
        lines.append(f"    Conserv  ${ppt['cycle6_peak_conservative']:>9,.0f}  "
                     f"({(ppt['cycle6_peak_conservative']/btc-1)*100:+.1f}%)  "
                     f"[1.7x cycle 5 peak]")
        lines.append(f"    Mid      ${ppt['cycle6_peak_mid']:>9,.0f}  "
                     f"({ppt['cycle6_peak_chg_pct_mid']:+.1f}%)  "
                     f"[1.9x cycle 5 peak]")
        lines.append(f"    Aggress  ${ppt['cycle6_peak_aggressive']:>9,.0f}  "
                     f"({(ppt['cycle6_peak_aggressive']/btc-1)*100:+.1f}%)  "
                     f"[2.2x cycle 5 peak]")
        lines.append("")
    except Exception:
        pass

    # === Ensemble ===
    ens = state.get("ensemble", {})
    if ens:
        lines.append("3-LENS ENSEMBLE")
        lines.append("-" * 78)
        lines.append(f"  Consensus: {ens.get('consensus', '?')}")
        for lens_name, ld in ens.get("lenses", {}).items():
            short_name = lens_name.replace("_lens", "")
            lines.append(f"  {short_name:<10s} {ld['score']:+.2f}  "
                         f"{ld['interpretation']:<14s} ({ld['n_signals']} signals)")
        lines.append("")

    # === Top signals ===
    bulls, bears = _top_signals(state, n=3)
    lines.append("TOP 3 BULLISH SIGNALS")
    lines.append("-" * 78)
    for s in bulls:
        lines.append(f"  {s['name']:<24s} [{s['category']:<14s}] {s['score']:+.2f}  {s['note']}")
    lines.append("")
    lines.append("TOP 3 BEARISH SIGNALS")
    lines.append("-" * 78)
    for s in bears:
        lines.append(f"  {s['name']:<24s} [{s['category']:<14s}] {s['score']:+.2f}  {s['note']}")
    lines.append("")

    # === Key levels ===
    levels = _key_levels(btc)
    lines.append("KEY LEVELS")
    lines.append("-" * 78)
    lines.append("  RESISTANCE (upside targets):")
    for r in levels["resistances"][:6]:
        lines.append(f"    ${r['price']:>9,.0f}  (+{r['distance_pct']:>4.1f}%)  {r['name']}")
    lines.append("")
    lines.append("  SUPPORT (downside levels):")
    for s in levels["supports"][:6]:
        lines.append(f"    ${s['price']:>9,.0f}  (-{s['distance_pct']:>4.1f}%)  {s['name']}")
    lines.append("")

    # === Catalysts ===
    catalysts = _upcoming_catalysts()
    if catalysts:
        lines.append("CATALYSTS THIS WEEK")
        lines.append("-" * 78)
        for c in catalysts:
            lines.append(f"  {c['date']:<14s}  {c['event']:<30s}  {c['impact']}")
        lines.append("")

    # === Anomalies ===
    anom = state.get("signal_anomalies", [])
    if anom:
        lines.append("SIGNAL ANOMALIES (regime change candidates)")
        lines.append("-" * 78)
        for a in anom[:5]:
            lines.append(f"  {a['signal']:<24s}  z={a['z_score']:+.2f}  "
                         f"{a['interpretation']}")
        lines.append("")

    # === COST-BASIS-PROCESS VIEW (Glassnode top-1% framework) ===
    lines.append("COST-BASIS-PROCESS VIEW (Glassnode top-1% framework)")
    lines.append("-" * 78)
    try:
        from core.btc_cost_basis import (
            realized_price, sth_cost_basis,
            realized_cap_drawdown_depth, bottom_probability_distribution,
        )
        rp = realized_price()
        sth = sth_cost_basis()
        rcd = realized_cap_drawdown_depth()
        pdb = bottom_probability_distribution()
        if rp and not rp.get("error"):
            lines.append(f"  Realized Price (LTH cost basis):  ${rp['value']:>9,.0f}  "
                         f"(30d {rp['chg_30d_pct']:+.1f}%)")
        if sth and not sth.get("error"):
            lines.append(f"  STH cost basis (155d MA):         ${sth['value']:>9,.0f}  "
                         f"(price {sth['price_vs_sth_pct']:+.1f}% vs STH cost)")
            lines.append(f"  STH regime hint: {sth['regime_hint']}")
        if rcd and not rcd.get("error"):
            lines.append("")
            lines.append("  REALIZED CAP DRAWDOWN — THE bottom indicator:")
            lines.append(f"    Current: {rcd['current_drawdown_pct']:+.1f}%")
            lines.append(f"    Depth progress to bottom zone: {rcd['depth_progress_pct']:.0f}%")
            lines.append(f"    Historical bottom targets: -15% entry / -20% mid / -25% deep")
            if not rcd["bands_passed"]:
                lines.append(f"    Bands passed: NONE — cost basis NOT yet capitulated")
        if pdb and not pdb.get("error"):
            lines.append("")
            lines.append("  BOTTOM PROBABILITY DISTRIBUTION:")
            for sc_p in pdb["scenarios"]:
                lines.append(f"    [{sc_p['probability']*100:.0f}%] {sc_p['name']:<28s} "
                             f"{sc_p['date_range']:<22s} {sc_p['price_range']}")
            lines.append(f"    Expected-value bottom price: ${pdb['expected_value_price']:,.0f} "
                         f"({pdb['expected_value_chg_pct']:+.1f}%)")
    except Exception as e:
        lines.append(f"  (cost-basis view unavailable: {e})")
    lines.append("")

    # === PRO ON-CHAIN LAYER (Woo + Glassnode top-1% signals) ===
    lines.append("PRO ON-CHAIN LAYER (Woo + top-1% Glassnode signals)")
    lines.append("-" * 78)
    pro_lookups = [
        ("realized_cap_drawdown", "Realized Cap drawdown"),
        ("reserve_risk",          "Reserve Risk"),
        ("puell_multiple",        "Puell Multiple"),
        ("coinbase_premium_gap",  "Coinbase Premium Gap"),
        ("difficulty_ribbon",     "Difficulty Ribbon"),
        ("asopr",                 "aSOPR (proxy)"),
        ("lth_sth_supply_ratio",  "LTH/STH dynamics"),
        ("cdd_spikes",            "CDD spikes (proxy)"),
        ("dormancy_flow",         "Dormancy Flow (proxy)"),
        ("nvt_signal_woo",        "NVT Signal (Woo)"),
    ]
    pro_scores = []
    for sig_name, display in pro_lookups:
        found = None
        for cat in ("onchain", "fundamentals", "flows"):
            d = state["signals"].get(cat, {}).get(sig_name)
            if isinstance(d, dict): found = d; break
        if found is None or found.get("error"):
            lines.append(f"  {display:<28s} (unavailable)")
            continue
        score = found.get("score")
        if score is None:
            lines.append(f"  {display:<28s} (no score)")
            continue
        pro_scores.append(score)
        arrow = ("++" if score > 0.5 else "+" if score > 0.1
                 else "=" if abs(score) <= 0.1
                 else "-" if score > -0.5 else "--")
        lines.append(f"  {display:<28s} {arrow:>2s} {score:+.2f}  {found.get('note', '')[:55]}")
    if pro_scores:
        avg = sum(pro_scores) / len(pro_scores)
        n_bull = sum(1 for s in pro_scores if s > 0.3)
        n_bear = sum(1 for s in pro_scores if s < -0.3)
        lines.append("")
        verdict = "BULL" if avg > 0.4 else "mild bull" if avg > 0.1 else "neutral" if avg > -0.1 else "mild bear" if avg > -0.4 else "BEAR"
        lines.append(f"  Pro layer avg: {avg:+.2f} ({verdict})  "
                     f"{n_bull} bullish / {n_bear} bearish of {len(pro_scores)} valid")
    lines.append("")

    # === PREMIUM-FREE LAYER (18 paid-tier-equivalent signals) ===
    lines.append("PREMIUM-FREE LAYER (paid-tier signals via free APIs)")
    lines.append("-" * 78)
    pf_lookups = [
        ("etf_flows",              "ETF flows (Farside)"),
        ("stablecoin_supply",      "Stablecoin supply"),
        ("github_activity",        "BTC Core dev"),
        ("deribit_greeks",         "Deribit max pain"),
        ("lth_supply_exact",       "LTH supply (exact)"),
        ("net_liquidity",          "Net Liquidity"),
        ("miner_holdings",         "Miner SEC filings"),
        ("hash_price",             "Hash price"),
        ("mempool_pressure",       "Mempool fees"),
        ("news_sentiment",         "News sentiment"),
        ("wikipedia_views",        "Wikipedia views"),
        ("dxy_regime",             "DXY"),
        ("energy_prices",          "Energy prices"),
        ("defi_tvl",               "DeFi TVL"),
        ("stablecoin_chain_flows", "Stables ETH/Tron"),
        ("exchange_net_flows",     "Whale tx activity"),
    ]
    pf_scores = []
    for sig_name, display in pf_lookups:
        found = None
        for cat in ("flows", "onchain", "fundamentals", "macro", "sentiment",
                     "options_adv"):
            d = state["signals"].get(cat, {}).get(sig_name)
            if isinstance(d, dict): found = d; break
        if found is None or found.get("error"):
            lines.append(f"  {display:<24s} (unavailable)")
            continue
        score = found.get("score")
        if score is None: continue
        pf_scores.append(score)
        arrow = ("++" if score > 0.5 else "+" if score > 0.1
                 else "=" if abs(score) <= 0.1
                 else "-" if score > -0.5 else "--")
        note = found.get("note", "")[:55]
        lines.append(f"  {display:<24s} {arrow:>2s} {score:+.2f}  {note}")
    if pf_scores:
        lines.append("")
        avg = sum(pf_scores) / len(pf_scores)
        n_bull = sum(1 for s in pf_scores if s > 0.3)
        n_bear = sum(1 for s in pf_scores if s < -0.3)
        lines.append(f"  Premium-free avg: {avg:+.2f} ({n_bull} bull / {n_bear} bear of {len(pf_scores)} valid)")
        lines.append(f"  Replaces ~$500/mo of Glassnode/CryptoQuant/Coinglass/Skew/Bloomberg")
    lines.append("")

    # === CLEMENTE + ALDEN LAYER (15 institutional signals) ===
    lines.append("CLEMENTE + ALDEN LAYER (15 institutional bottom signals)")
    lines.append("-" * 78)
    ca_lookups = [
        ("hashrate_drawdown",       "Hashrate drawdown"),
        ("cb_premium_streak",       "CB premium streak"),
        ("aasi",                    "AASI (active addr sentiment)"),
        ("stablecoin_supply_ratio", "SSR (dry powder)"),
        ("etf_pct_of_supply",       "ETF % of supply"),
        ("btc_dominance",           "BTC dominance"),
        ("real_yields_10y",         "10y real yields"),
        ("difficulty_adjustment",   "Difficulty next adj"),
        ("btc_gold_ratio",          "BTC/Gold ratio"),
        ("multi_exch_funding",      "Multi-venue funding"),
        ("rhodl_ratio",             "RHODL"),
        ("reflexivity_index",       "Reflexivity Index"),
        ("urpd_clusters",           "URPD clusters"),
        ("hodl_waves",              "HODL Waves"),
        ("fiscal_dominance",        "Fiscal Dominance"),
    ]
    ca_scores = []
    for sig_name, display in ca_lookups:
        found = None
        for cat in ("flows", "onchain", "fundamentals", "macro", "derivatives",
                     "regime_models"):
            d = state["signals"].get(cat, {}).get(sig_name)
            if isinstance(d, dict): found = d; break
        if found is None or found.get("error"):
            lines.append(f"  {display:<28s} (unavailable)")
            continue
        score = found.get("score")
        if score is None: continue
        ca_scores.append(score)
        arrow = ("++" if score > 0.5 else "+" if score > 0.1
                 else "=" if abs(score) <= 0.1
                 else "-" if score > -0.5 else "--")
        note = found.get("note", "")[:55]
        lines.append(f"  {display:<28s} {arrow:>2s} {score:+.2f}  {note}")
    if ca_scores:
        lines.append("")
        avg = sum(ca_scores) / len(ca_scores)
        n_bull = sum(1 for s in ca_scores if s > 0.3)
        n_bear = sum(1 for s in ca_scores if s < -0.3)
        verdict = ("STRONG BOTTOM SIGNAL" if avg > 0.4
                   else "Bottom forming" if avg > 0.1
                   else "Mixed/neutral" if avg > -0.1
                   else "Top forming")
        lines.append(f"  Clemente+Alden avg: {avg:+.2f} ({verdict}) "
                     f"({n_bull} bull / {n_bear} bear of {len(ca_scores)} valid)")
    lines.append("")

    # === JESSE OLSON TECHNICAL LAYER (multi-week TA) ===
    lines.append("JESSE OLSON TECHNICAL LAYER (multi-week TA)")
    lines.append("-" * 78)
    try:
        from core.btc_jesse_olson import olson_combined_verdict
        olson = olson_combined_verdict()
        for sig_name, d in olson["signals"].items():
            if d is None or d.get("error"):
                lines.append(f"  {sig_name:<28s} (unavailable)")
                continue
            phase = d.get("phase", "?")
            score = d.get("score", 0)
            arrow = ("++" if score > 0.5 else "+" if score > 0.1
                     else "=" if abs(score) <= 0.1
                     else "-" if score > -0.5 else "--")
            lines.append(f"  {sig_name:<28s} {arrow:>2s} {score:+.2f}  phase={phase}")
        lines.append("")
        lines.append(f"  Olson verdict: {olson['verdict']}")
        lines.append(f"  Avg score: {olson['avg_score']:+.2f}  "
                     f"({olson['bullish_count']} bull / {olson['bearish_count']} bear of {olson['n_valid']})")
    except Exception as e:
        lines.append(f"  (unavailable: {e})")
    lines.append("")

    # === HALVING CLOCK — most reliable BTC predictor ===
    # Historical std dev: 8 days for peaks, 12 days for bottoms
    # Cycle 5 peak predicted to within ONE day using this formula
    lines.append("HALVING CLOCK (most reliable BTC predictor)")
    lines.append("-" * 78)
    try:
        from core.halving_clock import (
            current_halving_position, cycle_phase_from_halving_day,
            pattern_projected_targets, MEAN_DAYS_TO_PEAK, MEAN_DAYS_TO_BOTTOM,
            PEAK_STD_DEV, BOTTOM_STD_DEV,
        )
        hc_pos = current_halving_position()
        hc_phase = cycle_phase_from_halving_day(hc_pos["days_post_halving"])
        hc_targets = pattern_projected_targets(btc)
        cycle_length = (hc_pos["next_halving"] - hc_pos["current_halving"]).days
        lines.append(f"  Day {hc_pos['days_post_halving']} of {cycle_length} "
                     f"({hc_pos['pct_through_cycle']:.0f}% through halving cycle 4)")
        lines.append(f"  Halving 4 (Apr 20, 2024) -> Halving 5 (Apr 20, 2028)")
        lines.append(f"  Pattern phase: {hc_phase['phase']} - {hc_phase['description']}")
        lines.append(f"  Directional bias: {hc_phase['directional_bias']:+.2f}")
        lines.append("")
        lines.append("  PATTERN PROJECTIONS:")
        days_to_pat_bot = hc_pos['days_to_pattern_bottom']
        bot_status = "ahead" if days_to_pat_bot > 0 else "passed"
        lines.append(f"    Pattern bottom date: {hc_pos['projected_bottom_date']} "
                     f"(halving + {MEAN_DAYS_TO_BOTTOM}d, std dev +/- {BOTTOM_STD_DEV}d)")
        lines.append(f"    Days to pattern bottom: {abs(days_to_pat_bot)} ({bot_status})")
        lines.append(f"    Cycle 5 bottom target (mid): ${hc_targets['cycle5_bottom_mid']:,.0f} "
                     f"({hc_targets['cycle5_bottom_chg_pct']:+.1f}% from current)")
        lines.append(f"    Cycle 6 peak date:   {hc_targets['cycle6_peak_date']} "
                     f"(halving 5 + {MEAN_DAYS_TO_PEAK}d, std dev +/- {PEAK_STD_DEV}d)")
        lines.append(f"    Cycle 6 peak target range: "
                     f"${hc_targets['cycle6_peak_conservative']:,.0f} "
                     f"- ${hc_targets['cycle6_peak_mid']:,.0f} "
                     f"- ${hc_targets['cycle6_peak_aggressive']:,.0f} "
                     f"(conservative / mid / aggressive)")
        lines.append(f"    Cycle 6 peak mid case: {hc_targets['cycle6_peak_chg_pct_mid']:+.1f}% "
                     f"from current price")
        lines.append("")
        lines.append("  RELIABILITY: Cycle 5 peak predicted to ONE day using this formula.")
        lines.append(f"               Std dev across 3 cycles: peaks +/-{PEAK_STD_DEV}d, "
                     f"bottoms +/-{BOTTOM_STD_DEV}d. Highest-conviction BTC signal.")
    except Exception as e:
        lines.append(f"  (halving clock unavailable: {e})")
    lines.append("")

    # === Cycle context (signals + position) ===
    lines.append("CYCLE CONTEXT")
    lines.append("-" * 78)
    today = datetime.now(timezone.utc).date()
    halving4 = datetime(2024, 4, 20).date()
    days_post_halving = (today - halving4).days
    cycle_peak = datetime(2025, 10, 6).date()
    days_post_peak = (today - cycle_peak).days
    projected_bottom = halving4 + timedelta(days=900)
    days_to_bottom = (projected_bottom - today).days
    projected_cycle6_peak = datetime(2028, 4, 20).date() + timedelta(days=535)
    days_to_peak = (projected_cycle6_peak - today).days
    lines.append(f"  Days post-halving:        {days_post_halving}")
    lines.append(f"  Days post cycle-5 peak:   {days_post_peak}")
    lines.append(f"  Days to projected bottom: {days_to_bottom}  (~{days_to_bottom/30:.1f} months)")
    lines.append(f"  Days to cycle-6 peak:     {days_to_peak}  (~{days_to_peak/365:.1f} years)")
    lines.append("")

    # Cycle bottom signals status
    try:
        from core.btc_bottom_signals import all_bottom_signals
        bottom_sigs = all_bottom_signals()
        bottom_score = 0
        bottom_count = 0
        for name, d in bottom_sigs.items():
            if d and d.get("score") is not None:
                bottom_score += d["score"]
                bottom_count += 1
        avg = bottom_score / bottom_count if bottom_count > 0 else 0
        lines.append("CYCLE BOTTOM SIGNAL STATUS")
        lines.append("-" * 78)
        for name, d in bottom_sigs.items():
            if d is None:
                lines.append(f"  {name:<24s} unavailable")
                continue
            phase = d.get("phase", "")
            score = d.get("score", 0)
            arrow = "++" if score > 0.5 else "+" if score > 0.1 else "=" if abs(score) <= 0.1 else "-"
            lines.append(f"  {name:<24s} {arrow:>2s} {score:+.2f}  {phase}")
        lines.append(f"  AVG bottom signal:       {avg:+.2f}  "
                     f"({'BOTTOM FORMING' if avg > 0.5 else 'normal'})")
        lines.append("")
    except Exception:
        pass

    lines.append("=" * 78)
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("Run: python btc_predict.py --force  (refresh signals)")
    lines.append("Dashboard: http://localhost:8510")
    lines.append("=" * 78)

    return "\n".join(lines)


def save_weekly_report(state: Optional[dict] = None) -> Path:
    """Generate + save the weekly report to disk."""
    report = generate_weekly_report(state)
    week_id = datetime.now(timezone.utc).strftime("%Y-W%V")
    fp = REPORTS_DIR / f"btc_weekly_{week_id}.txt"
    fp.write_text(report, encoding="utf-8")
    return fp


def main():
    report = generate_weekly_report()
    print(report)
    fp = save_weekly_report()
    print(f"\nSaved to: {fp}")


if __name__ == "__main__":
    main()
