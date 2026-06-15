"""BTC directional forecast — synthesis of every signal in the rig.

Combines:
    Short-term (1-30d):  multi_timeframe confluence, EMA21, MACD, RSI, BB
    Medium-term (30-180d): F&G + MVRV joint signal, cycle position, BTC.D
    Long-term (1-2y):    cycle composite, Glassnode key levels, Hayes triggers
    Macro overlay:       NQ/SPY/VIX/DXY/TNX correlation regime
    Tail risk:           Deribit DVOL + skew, volatility regime

Output: probabilistic forecast across 4 horizons with:
    - Base case + bull case + bear case price targets
    - Key support / resistance levels to watch
    - Confidence level per horizon
    - Specific catalysts that would invalidate the view

This is a SYNTHESIS, not a model. It tells you what the rig's signals are
collectively saying. Use it for sizing decisions, not for false certainty.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd


def _hr(char="="):
    print(char * 76)


def _section(title):
    print()
    _hr("=")
    print(title)
    _hr("=")


def _subsection(title):
    print()
    print(f"  {title}")
    print(f"  {'-' * (len(title) + 2)}")


def get_current_btc_price() -> float:
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0


def short_term_signals(current_px: float) -> dict:
    """1-30 day technical signals."""
    out = {"horizon": "1-30 days", "votes": [], "score": 0}

    # Multi-TF confluence
    try:
        from core.multi_timeframe import confluence
        c = confluence("BTC/USDT")
        out["mtf_confluence"] = c["confluence_score"]
        out["mtf_direction"] = c["net_direction"]
        out["mtf_verdict"] = c["verdict"]
        if c["net_direction"] > 0:
            out["votes"].append(("MTF confluence", "BULL", c["confluence_score"]))
            out["score"] += c["confluence_score"]
        elif c["net_direction"] < 0:
            out["votes"].append(("MTF confluence", "BEAR", c["confluence_score"]))
            out["score"] -= c["confluence_score"]
        else:
            out["votes"].append(("MTF confluence", "NEUTRAL", c["confluence_score"]))
    except Exception:
        out["votes"].append(("MTF confluence", "unavailable", 0))

    # Exit signal state (EMA21, MACD, RSI, BB)
    try:
        import json
        es_file = Path(__file__).resolve().parent / "btc_exit_signal_state.json"
        if es_file.exists():
            es = json.loads(es_file.read_text())
            btc = es.get("BTC/USDT", {})
            if btc:
                rsi = btc.get("rsi", 50)
                macd_h = btc.get("macd_hist", 0)
                ema21 = btc.get("ema21_price", current_px)
                above_ema = btc.get("above_ema", True)
                bb_pct = btc.get("bb_pct", 0.5)
                out["rsi"] = rsi
                out["macd_hist"] = macd_h
                out["ema21"] = ema21
                out["bb_pct"] = bb_pct
                if rsi < 30:
                    out["votes"].append(("RSI oversold", "BULL contrarian", 0.6))
                    out["score"] += 0.4
                elif rsi > 70:
                    out["votes"].append(("RSI overbought", "BEAR contrarian", 0.6))
                    out["score"] -= 0.4
                else:
                    out["votes"].append(("RSI", "NEUTRAL", rsi/100))
                if macd_h > 0:
                    out["votes"].append(("MACD histogram", "BULL", 0.5))
                    out["score"] += 0.3
                else:
                    out["votes"].append(("MACD histogram", "BEAR", 0.5))
                    out["score"] -= 0.3
                if above_ema:
                    out["votes"].append(("Above EMA21", "BULL", 0.4))
                    out["score"] += 0.2
                else:
                    out["votes"].append(("Below EMA21", "BEAR", 0.4))
                    out["score"] -= 0.2
                if bb_pct < 0.2:
                    out["votes"].append(("BB lower band", "BULL contrarian", 0.4))
                    out["score"] += 0.2
                elif bb_pct > 0.8:
                    out["votes"].append(("BB upper band", "BEAR contrarian", 0.4))
                    out["score"] -= 0.2
    except Exception:
        pass

    return out


def medium_term_signals(current_px: float) -> dict:
    """30-180 day cycle + sentiment signals."""
    out = {"horizon": "30-180 days", "votes": [], "score": 0}

    # F&G
    try:
        from core.fear_greed import latest, cycle_composite_score
        fg = latest()
        comp = cycle_composite_score()
        out["fg_value"] = fg.get("value")
        out["fg_regime"] = fg.get("regime")
        out["composite_score"] = comp.get("composite_score")
        v = fg.get("value", 50)
        if v <= 25:
            out["votes"].append(("F&G extreme fear", "BULL contrarian", 0.8))
            out["score"] += 0.6
        elif v <= 45:
            out["votes"].append(("F&G fear", "BULL mild", 0.5))
            out["score"] += 0.3
        elif v >= 75:
            out["votes"].append(("F&G extreme greed", "BEAR contrarian", 0.8))
            out["score"] -= 0.6
        elif v >= 55:
            out["votes"].append(("F&G greed", "BEAR mild", 0.5))
            out["score"] -= 0.3
        else:
            out["votes"].append(("F&G neutral", "NEUTRAL", 0.3))
    except Exception:
        pass

    # MVRV cycle position
    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        out["mvrv"] = cp.get("mvrv")
        out["cycle_score"] = cp.get("score")
        out["cycle_phase"] = cp.get("phase")
        score = cp.get("score", 50)
        if score < 25:
            out["votes"].append(("MVRV deep bear", "STRONG BULL", 0.9))
            out["score"] += 0.7
        elif score < 50:
            out["votes"].append(("MVRV early bull", "BULL", 0.6))
            out["score"] += 0.4
        elif score < 75:
            out["votes"].append(("MVRV mid bull", "NEUTRAL", 0.3))
        elif score < 90:
            out["votes"].append(("MVRV late bull", "BEAR", 0.6))
            out["score"] -= 0.4
        else:
            out["votes"].append(("MVRV euphoria", "STRONG BEAR", 0.9))
            out["score"] -= 0.7
    except Exception:
        pass

    # BTC dominance regime
    try:
        from core.btc_dominance import status
        dom = status()
        out["btc_dom_pct"] = dom.get("btc_dominance_pct")
        out["dom_regime"] = dom.get("regime")
        # BTC dominance neutral wrt BTC direction (it's a flow signal between BTC and alts)
        out["votes"].append((f"BTC.D {dom.get('regime', '?')}", "NEUTRAL (flow signal)", 0.3))
    except Exception:
        pass

    # F&G + MVRV historical forward returns (from backtest)
    # Hard-coded from _bt_followups.py result for transparency
    out["historical_180d_when_signal_fires"] = {
        "n_observations": 317,
        "mean": 0.203,
        "median": 0.226,
        "hit_rate": 0.62,
        "p25": -0.185,
        "p75": 0.581,
        "p5": -0.334,
        "p95": 0.660,
    }

    return out


def long_term_signals(current_px: float) -> dict:
    """1-2 year cycle + Glassnode key levels."""
    out = {"horizon": "1-2 years", "votes": [], "score": 0}

    try:
        from core.btc_key_levels import get_status
        bs = get_status()
        out["regime"] = bs.get("regime")
        out["sth_mvrv_proxy"] = bs.get("sth_mvrv_proxy")
        out["sth_mvrv_regime"] = bs.get("sth_mvrv_regime")
    except Exception:
        pass

    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        out["cycle_phase"] = cp.get("phase")
        score = cp.get("score", 50)
        if score < 40:
            out["votes"].append(("Cycle accumulation phase", "STRONG BULL 2yr", 0.85))
            out["score"] += 0.7
        elif score < 60:
            out["votes"].append(("Cycle mid", "BULL 2yr", 0.5))
            out["score"] += 0.3
        elif score < 80:
            out["votes"].append(("Cycle late", "BEAR 2yr", 0.4))
            out["score"] -= 0.3
        else:
            out["votes"].append(("Cycle peak", "STRONG BEAR 2yr", 0.85))
            out["score"] -= 0.7
    except Exception:
        pass

    # Halving cycle context (BTC halved 2024-04, ~4 year cycle pattern)
    out["halving_cycle_day"] = (datetime.now(timezone.utc) -
                                  datetime(2024, 4, 20, tzinfo=timezone.utc)).days
    out["votes"].append(
        (f"Halving cycle day {out['halving_cycle_day']}",
         "BULL pattern (historical peaks 12-18 months post-halving)", 0.5)
    )
    if 400 < out["halving_cycle_day"] < 600:
        out["score"] += 0.3

    return out


def macro_overlay() -> dict:
    """Macro regime per Hayes thesis."""
    out = {"votes": []}
    try:
        from core.macro_correlation import regime_status, latest_metrics
        r = regime_status()
        m = latest_metrics()
        out["regime"] = r.get("regime")
        out["de_risk_level"] = r.get("de_risk_level")
        out["VIX"] = m.get("VIX", {}).get("value")
        out["TNX"] = m.get("TNX", {}).get("value")
        out["DXY"] = m.get("DXY", {}).get("value")
        if r.get("regime") == "normal":
            out["votes"].append(("Macro NORMAL", "NEUTRAL", 0.3))
        elif r.get("regime") in ("caution", "de-risk"):
            out["votes"].append((f"Macro {r['regime'].upper()}", "BEAR", 0.6))
        elif r.get("regime") == "full_kill":
            out["votes"].append(("Macro FULL_KILL", "STRONG BEAR", 0.95))
    except Exception:
        pass
    return out


def tail_risk_overlay() -> dict:
    """Tail risk signal."""
    out = {"votes": []}
    try:
        from core.tail_hedge import compute_hedge_recommendation
        h = compute_hedge_recommendation(bankroll_usd=200_000)
        out["urgency"] = h.get("urgency")
        out["risk_factors"] = h.get("risk_factor_count")
        if h.get("urgency") == "critical":
            out["votes"].append(("Tail hedge CRITICAL", "BEAR regime change risk", 0.85))
        elif h.get("urgency") == "recommended":
            out["votes"].append(("Tail hedge recommended", "Mild BEAR risk", 0.5))
        else:
            out["votes"].append(("Tail hedge unnecessary", "no near-term tail event", 0.3))
    except Exception:
        pass
    return out


def key_levels(current_px: float) -> dict:
    """Critical price levels — supports + resistances from on-chain + technicals."""
    out = {"current": current_px, "supports": [], "resistances": []}

    # On-chain / Glassnode / Hayes levels
    try:
        from core.btc_key_levels import (
            STH_COST_BASIS, OVERHEAD_SUPPLY_LOW, OVERHEAD_SUPPLY_HIGH,
            HAYES_GAMMA_TRIGGER, HAYES_DOWNSIDE_TRIGGER, STH_MVRV_PROFITABILITY,
        )
        # Supports below current
        for lvl, name in [
            (HAYES_DOWNSIDE_TRIGGER, "Hayes downside trigger (sell BTC if breached)"),
            (STH_COST_BASIS, "STH cost basis (psychological support)"),
            (current_px * 0.9, "10% drawdown floor"),
        ]:
            if lvl < current_px:
                out["supports"].append((lvl, name, (current_px / lvl - 1) * 100))

        # Resistances above current
        for lvl, name in [
            (OVERHEAD_SUPPLY_LOW, "Overhead supply low ($92k)"),
            (HAYES_GAMMA_TRIGGER, "Hayes gamma squeeze trigger ($90k)"),
            (OVERHEAD_SUPPLY_HIGH, "Overhead supply high ($117k)"),
        ]:
            if lvl > current_px:
                out["resistances"].append((lvl, name, (lvl / current_px - 1) * 100))
    except Exception:
        pass

    # Technical levels — EMA21, SMA200
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=220)
        if not df.empty and len(df) >= 200:
            ema21 = float(df["close"].ewm(span=21).mean().iloc[-1])
            sma200 = float(df["close"].rolling(200).mean().iloc[-1])
            mayer = current_px / sma200 if sma200 > 0 else 1.0
            recent_low_30d = float(df["low"].iloc[-30:].min())
            recent_high_30d = float(df["high"].iloc[-30:].max())

            for lvl, name in [
                (recent_low_30d, "30-day low"),
                (sma200 * 0.7, "Mayer 0.7x (deep value)"),
            ]:
                if lvl < current_px:
                    out["supports"].append((lvl, name, (current_px / lvl - 1) * 100))

            for lvl, name in [
                (ema21, "EMA21 (intraday resistance)"),
                (sma200, f"SMA200 (Mayer {mayer:.2f}, bull/bear divider)"),
                (recent_high_30d, "30-day high"),
            ]:
                if lvl > current_px:
                    out["resistances"].append((lvl, name, (lvl / current_px - 1) * 100))
    except Exception:
        pass

    # Sort
    out["supports"].sort(key=lambda x: -x[0])  # closest first
    out["resistances"].sort(key=lambda x: x[0])  # closest first
    return out


def scenarios(current_px: float, st: dict, mt: dict, lt: dict, macro: dict, tail: dict) -> dict:
    """Synthesize 4 scenarios with rough probabilities."""
    # Sum scores (each in -1 to +1 range, weighted by confidence)
    overall_score = st["score"] + mt["score"] * 1.5 + lt["score"] * 0.8
    # Add macro/tail tilts
    if macro.get("regime") == "full_kill": overall_score -= 1.0
    elif macro.get("regime") == "de-risk": overall_score -= 0.5
    if tail.get("urgency") == "critical": overall_score -= 0.8

    return {
        "overall_directional_score": overall_score,
        "interpretation": (
            "STRONG BULL" if overall_score > 2.0 else
            "BULL" if overall_score > 0.8 else
            "NEUTRAL" if overall_score > -0.8 else
            "BEAR" if overall_score > -2.0 else
            "STRONG BEAR"
        ),
    }


def main():
    print()
    _hr("=")
    print(f"BTC FORECAST — synthesis of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    _hr("=")

    current_px = get_current_btc_price()
    if current_px <= 0:
        print("ERROR: could not fetch current BTC price")
        return
    print(f"\nCurrent BTC price: ${current_px:,.2f}")

    # Gather all signals
    st = short_term_signals(current_px)
    mt = medium_term_signals(current_px)
    lt = long_term_signals(current_px)
    macro = macro_overlay()
    tail = tail_risk_overlay()
    levels = key_levels(current_px)
    syn = scenarios(current_px, st, mt, lt, macro, tail)

    # SHORT TERM
    _section(f"SHORT TERM ({st['horizon']}) — TECHNICAL READ")
    print(f"  MTF confluence:  {st.get('mtf_confluence', '?'):.2f}   "
          f"direction {st.get('mtf_direction','?')}   verdict {st.get('mtf_verdict','?')}")
    print(f"  RSI(14):         {st.get('rsi', '?'):.0f}")
    print(f"  MACD histogram:  {st.get('macd_hist', '?'):+.2f}")
    print(f"  EMA21:           ${st.get('ema21', 0):,.0f}  (current ${'above' if current_px > st.get('ema21', current_px) else 'below'})")
    print(f"  BB position:     {st.get('bb_pct', 0)*100:.0f}%")
    print(f"  Net votes: {len(st['votes'])} signals, score {st['score']:+.2f}")

    # MEDIUM TERM
    _section(f"MEDIUM TERM ({mt['horizon']}) — CYCLE + SENTIMENT")
    print(f"  F&G Index:        {mt.get('fg_value','?')}  ({mt.get('fg_regime','?')})")
    print(f"  MVRV:             {mt.get('mvrv', 0):.2f}  cycle {mt.get('cycle_score',0):.0f}/100 ({mt.get('cycle_phase','?')})")
    print(f"  Composite score:  {mt.get('composite_score',0):.0f}/100")
    print(f"  BTC.D:            {mt.get('btc_dom_pct',0):.2f}% ({mt.get('dom_regime','?')})")
    print(f"  Net votes: {len(mt['votes'])} signals, score {mt['score']:+.2f}")
    print()
    print(f"  HISTORICAL BASE RATE (n={mt['historical_180d_when_signal_fires']['n_observations']} firings of joint F&G+MVRV):")
    h = mt['historical_180d_when_signal_fires']
    print(f"    180d hit rate:     {h['hit_rate']*100:.0f}%")
    print(f"    180d median:       {h['median']*100:+.0f}%  -> ${current_px * (1+h['median']):,.0f}")
    print(f"    180d P25 (worst quartile): {h['p25']*100:+.0f}%  -> ${current_px * (1+h['p25']):,.0f}")
    print(f"    180d P75 (best quartile):  {h['p75']*100:+.0f}%  -> ${current_px * (1+h['p75']):,.0f}")

    # LONG TERM
    _section(f"LONG TERM ({lt['horizon']}) — CYCLE POSITION")
    print(f"  Halving cycle day:    {lt.get('halving_cycle_day','?')} (BTC last halved 2024-04-20)")
    print(f"  Cycle phase:          {lt.get('cycle_phase','?')}")
    print(f"  STH-MVRV proxy:       {lt.get('sth_mvrv_proxy',0):.2f} ({lt.get('sth_mvrv_regime','?')})")
    print(f"  Net votes: {len(lt['votes'])} signals, score {lt['score']:+.2f}")

    # MACRO
    _section("MACRO OVERLAY (Hayes thesis)")
    print(f"  Regime:               {macro.get('regime','?').upper()}")
    print(f"  De-risk level:        {macro.get('de_risk_level',0)}")
    print(f"  VIX:  {macro.get('VIX',0):.1f}   TNX (10Y): {macro.get('TNX',0):.2f}   DXY: {macro.get('DXY',0):.1f}")

    # TAIL
    _section("TAIL RISK OVERLAY")
    print(f"  Urgency:              {tail.get('urgency','?').upper()}")
    print(f"  Risk factors:         {tail.get('risk_factors',0)}/6")

    # KEY LEVELS
    _section("KEY PRICE LEVELS")
    print(f"  Current: ${current_px:,.2f}")
    _subsection("RESISTANCE (upside targets, closest first)")
    for lvl, name, dist in levels["resistances"][:6]:
        print(f"    ${lvl:>10,.0f}  (+{dist:>5.1f}%)  {name}")
    _subsection("SUPPORT (downside levels, closest first)")
    for lvl, name, dist in levels["supports"][:6]:
        print(f"    ${lvl:>10,.0f}  (-{dist:>5.1f}%)  {name}")

    # SYNTHESIS
    _section("SYNTHESIS — DIRECTIONAL OUTLOOK")
    print(f"\n  Overall directional score: {syn['overall_directional_score']:+.2f}")
    print(f"  Interpretation: {syn['interpretation']}")
    print()

    # Scenarios
    print("  4-SCENARIO PROBABILISTIC FORECAST (90 days)")
    print("  " + "-" * 70)
    h = mt['historical_180d_when_signal_fires']
    # Scale 180d -> 90d via sqrt(time) for vol; assume symmetric
    scale = (90/180) ** 0.5
    print(f"    BEAR case  (P25):   ${current_px * (1 + h['p25']*scale):,.0f}  ({h['p25']*scale*100:+.0f}%) — 25% chance")
    print(f"    BASE case  (median):${current_px * (1 + h['median']*scale):,.0f}  ({h['median']*scale*100:+.0f}%) — 50% chance")
    print(f"    BULL case  (P75):   ${current_px * (1 + h['p75']*scale):,.0f}  ({h['p75']*scale*100:+.0f}%) — 25% chance")
    print(f"    TAIL bear  (P5):    ${current_px * (1 + h['p5']*scale):,.0f}  ({h['p5']*scale*100:+.0f}%) — 5% chance")
    print(f"    TAIL bull  (P95):   ${current_px * (1 + h['p95']*scale):,.0f}  ({h['p95']*scale*100:+.0f}%) — 5% chance")
    print()
    print("  180-DAY FORECAST")
    print("  " + "-" * 70)
    print(f"    BEAR case  (P25):   ${current_px * (1 + h['p25']):,.0f}  ({h['p25']*100:+.0f}%)")
    print(f"    BASE case  (median):${current_px * (1 + h['median']):,.0f}  ({h['median']*100:+.0f}%)")
    print(f"    BULL case  (P75):   ${current_px * (1 + h['p75']):,.0f}  ({h['p75']*100:+.0f}%)")

    # ALL SIGNALS VOTE TALLY
    _section("ALL DIRECTIONAL VOTES (every signal in the rig)")
    all_votes = st["votes"] + mt["votes"] + lt["votes"] + macro["votes"] + tail["votes"]
    print(f"  {'Signal':<32s} {'Vote':<28s} {'Strength':>10s}")
    print("  " + "-" * 70)
    for sig, vote, strength in all_votes:
        print(f"  {sig:<32s} {vote:<28s} {strength:>10.2f}")

    # INVALIDATION
    _section("WHAT WOULD INVALIDATE THIS VIEW")
    print(f"\n  This BULL bias would FLIP to bear if:")
    print(f"    • BTC breaks below ${current_px * 0.93:,.0f} (-7%) — invalidates short-term hold")
    print(f"    • F&G crosses above 75 without price progress — sentiment exhaustion")
    print(f"    • MVRV cycle score crosses above 75 — late-cycle warning")
    print(f"    • Macro regime shifts to DE_RISK — Hayes thesis broken")
    print(f"    • Tail hedge urgency triggers CRITICAL — regime-change risk")
    print(f"    • VaR Kupiec test FAILS for >7 days — model breakdown")
    print()

    _hr("=")
    print("REMEMBER: this is a SIGNAL SYNTHESIS, not a prediction.")
    print("Edge is positive over 180 days (62% hit, +20% mean).")
    print("Near-term (30-90d) outcomes have wide variance.")
    _hr("=")


if __name__ == "__main__":
    main()
