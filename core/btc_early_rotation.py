"""Early Rotation Signal — the pro move.

Standard rotation logic: "wait for equities to top, then rotate to BTC."
Fatal flaw: in liquidity crunches BTC has 1.5x beta to SPY (2022:
SPY -25%, BTC -77%; 2020 Mar: SPY -34% in 1 month, BTC -50% same time).

Pro move (Druckenmiller, PTJ, Zulauf, Howell):
  1. Watch LEADING indicators of equity weakness (not lagging)
  2. When they crack, rotate equity → CASH (not BTC)
  3. Wait for BTC bottom scorecard to fire → deploy cash → BTC

Their wisdom:
  - Druckenmiller: "I'm 60% cash before recession starts. I cannot afford
    to be wrong on this. I always sell too early."
  - Tudor Jones: "If the 200d breaks, I'm out of stocks. Period."
  - Zulauf: "Defensive sectors outperforming = professionals already
    rotating. You're 3-6 months late if you wait for the headlines."
  - Howell: "Global liquidity peaks 9 months before SPY peak."

The DESTINATION matters:
  - If equity top scoring rises FAST but BTC bottom still distant ->
    rotate to CASH (T-bills, BIL ETF). Capital preservation first.
  - If equity top fires AND BTC bottom near (BTC scorecard ≥ 6) ->
    rotate equity directly to BTC. Maximum compounding.
  - Don't rotate equity to BTC during a liquidity crunch — BTC will
    drop MORE than equities. You'd lose twice.

This module outputs ONE of three actions:
  - HOLD          — neither flashing, stay 30% equity
  - ROTATE_TO_CASH — equity warning + BTC NOT bottomed (preserve capital)
  - ROTATE_TO_BTC  — equity warning + BTC bottomed (deploy at the low)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# LEADING EQUITY INDICATORS
# ============================================================
# These fire BEFORE the equity top scorecard does. The point is to
# get ahead of the standard signal by 3-9 months.

def _yf(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fast yfinance wrapper with cache."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty: return None
        return df
    except Exception:
        return None


def small_cap_leading_large() -> dict:
    """Russell 2000 vs SPY ratio falling = small caps weakening first.

    Small caps top 3-6 months before large caps historically (1999, 2007, 2021).
    Russell-2000 / SPY ratio in downtrend over 90 days = leading signal.
    """
    iwm = _yf("IWM", period="6mo")  # Russell 2000 ETF
    spy = _yf("SPY", period="6mo")
    if iwm is None or spy is None or iwm.empty or spy.empty:
        return {"firing": False, "status": "data unavailable"}
    df = pd.DataFrame({
        "iwm": iwm["Close"].reset_index(drop=True),
        "spy": spy["Close"].reset_index(drop=True),
    }).dropna()
    if len(df) < 90:
        return {"firing": False, "status": "insufficient history"}
    df["ratio"] = df["iwm"] / df["spy"]
    ratio_90d_ago = float(df["ratio"].iloc[-90])
    ratio_now = float(df["ratio"].iloc[-1])
    change_pct = (ratio_now / ratio_90d_ago - 1) * 100
    # Firing if Russell underperforming SPY by >3% over 90d
    firing = change_pct < -3.0
    return {
        "firing": firing,
        "ratio_now": ratio_now,
        "ratio_90d_ago": ratio_90d_ago,
        "change_pct_90d": change_pct,
        "status": (f"IWM/SPY ratio: {change_pct:+.1f}% over 90d "
                   f"(<-3% = small caps leading down)"),
        "rationale": ("Small caps top 3-6 months before large caps. "
                       "Russell 2000 weakness = early warning."),
    }


def defensive_rotation() -> dict:
    """Defensive sectors (XLP, XLU, XLV) outperforming offensive (XLK, XLY, QQQ).

    Professional money rotates to staples/utilities/healthcare 3-6 months
    before public sees the top. Ratio of defensive/offensive rising = pros
    already exiting risk.
    """
    # Defensive
    xlp = _yf("XLP", period="6mo")  # Consumer staples
    xlu = _yf("XLU", period="6mo")  # Utilities
    xlv = _yf("XLV", period="6mo")  # Healthcare
    # Offensive
    xlk = _yf("XLK", period="6mo")  # Tech
    xly = _yf("XLY", period="6mo")  # Consumer discretionary

    if any(x is None or x.empty for x in [xlp, xlu, xlv, xlk, xly]):
        return {"firing": False, "status": "sector data unavailable"}

    n = min(len(xlp), len(xlu), len(xlv), len(xlk), len(xly))
    if n < 90:
        return {"firing": False, "status": "insufficient sector history"}

    def_now = (float(xlp["Close"].iloc[-1]) + float(xlu["Close"].iloc[-1]) +
                float(xlv["Close"].iloc[-1])) / 3
    off_now = (float(xlk["Close"].iloc[-1]) + float(xly["Close"].iloc[-1])) / 2
    def_90 = (float(xlp["Close"].iloc[-90]) + float(xlu["Close"].iloc[-90]) +
                float(xlv["Close"].iloc[-90])) / 3
    off_90 = (float(xlk["Close"].iloc[-90]) + float(xly["Close"].iloc[-90])) / 2
    ratio_now = def_now / off_now
    ratio_90 = def_90 / off_90
    change_pct = (ratio_now / ratio_90 - 1) * 100
    # Firing if defensive outperforming offensive by >5% over 90d
    firing = change_pct > 5.0
    return {
        "firing": firing,
        "ratio_now": ratio_now,
        "change_pct_90d": change_pct,
        "status": (f"Defensive/Offensive sector ratio: {change_pct:+.1f}% over 90d "
                   f"(>+5% = pros already rotating defensive)"),
        "rationale": ("Staples/utilities/healthcare outperforming tech/discretionary "
                       "= institutional rotation to safety, ahead of headlines."),
    }


def yield_curve_resteepening_velocity() -> dict:
    """Yield curve UN-inverting from inversion = recession start signal.

    Historically: yield curve inverts ~12-18 months before recession,
    then UN-inverts 0-6 months before recession officially begins.
    Speed of re-steepening matters: fast = imminent.
    """
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("T10Y2Y", days=730)
        if df is None or df.empty:
            return {"firing": False, "status": "FRED data unavailable"}
        df = df.sort_values("date").reset_index(drop=True)
        current = float(df["value"].iloc[-1])
        min_180d = float(df["value"].tail(180).min())
        delta = current - min_180d
        # Firing: was inverted (<0), now positive or fast moving up
        was_inverted = min_180d < -0.2
        re_steep_fast = delta > 0.5  # >50bp move off the lows
        firing = was_inverted and re_steep_fast
        return {
            "firing": firing,
            "current": current,
            "min_180d": min_180d,
            "delta_bps": delta * 100,
            "status": (f"10y-2y spread: {current:+.2f}% (was {min_180d:+.2f}% bottom, "
                       f"+{delta*100:.0f}bps move = "
                       f"{'RE-STEEPENING FAST' if firing else 'not yet'})"),
            "rationale": ("Yield curve un-inverting = recession begins within 6 months. "
                           "Every recession since 1970s started this way."),
        }
    except Exception as e:
        return {"firing": False, "status": f"error: {e!r}"[:80]}


def hy_credit_spread_widening_velocity() -> dict:
    """HY credit spread WIDENING accelerating = credit cracking.

    Spreads widen BEFORE equity tops because bond market sees stress first.
    Speed of widening matters: 50bp+ in 30 days = credit crisis incoming.
    """
    try:
        from core.btc_clemente_alden import _fred_csv
        df = _fred_csv("BAMLH0A0HYM2", days=180)
        if df is None or df.empty or len(df) < 30:
            # Fallback: use HYG/TLT ratio as proxy
            hyg = _yf("HYG", period="3mo")
            tlt = _yf("TLT", period="3mo")
            if hyg is None or tlt is None:
                return {"firing": False, "status": "HY data unavailable"}
            ratio_now = float(hyg["Close"].iloc[-1]) / float(tlt["Close"].iloc[-1])
            ratio_30 = float(hyg["Close"].iloc[-min(30, len(hyg))]) / float(
                tlt["Close"].iloc[-min(30, len(tlt))])
            change_pct = (ratio_now / ratio_30 - 1) * 100
            # HYG/TLT FALLING = credit risk rising
            firing = change_pct < -2.0
            return {
                "firing": firing,
                "proxy": True,
                "change_pct_30d": change_pct,
                "status": (f"HYG/TLT proxy: {change_pct:+.1f}% over 30d "
                           f"(<-2% = credit risk rising fast)"),
                "rationale": "HY proxy via ETF ratio (FRED unavailable). Credit > equity.",
            }
        df = df.sort_values("date").reset_index(drop=True)
        current = float(df["value"].iloc[-1])
        spread_30d_ago = float(df["value"].iloc[-30])
        delta_bps = (current - spread_30d_ago) * 100
        firing = delta_bps > 50  # >50bp widening in 30 days
        return {
            "firing": firing,
            "current_bps": current * 100,
            "delta_30d_bps": delta_bps,
            "status": (f"HY spread: {current*100:.0f}bps now, "
                       f"{delta_bps:+.0f}bps over 30d "
                       f"(>+50bps = stress accelerating)"),
            "rationale": ("Credit market sees stress first. Spreads widen 1-3 months "
                           "before equity tops historically."),
        }
    except Exception as e:
        return {"firing": False, "status": f"error: {e!r}"[:80]}


def spy_below_200dma_breach() -> dict:
    """SPY closes >1.5% below 200d moving average = PTJ's trend break.

    'If the 200d breaks, I'm out of stocks. Period.' — Paul Tudor Jones
    The 200d is the line where bull becomes bear. Decisive break (>1.5%
    below) means trend has flipped.
    """
    spy = _yf("SPY", period="1y")
    if spy is None or spy.empty or len(spy) < 200:
        return {"firing": False, "status": "SPY data unavailable"}
    close = float(spy["Close"].iloc[-1])
    ma200 = float(spy["Close"].tail(200).mean())
    pct_below = (close / ma200 - 1) * 100
    firing = pct_below < -1.5
    return {
        "firing": firing,
        "close": close,
        "ma200": ma200,
        "pct_vs_200d": pct_below,
        "status": (f"SPY {close:.0f} vs 200d {ma200:.0f} = {pct_below:+.2f}% "
                   f"({'BROKEN' if firing else 'holding'})"),
        "rationale": "PTJ rule: 200d break = bull trend over. Sell all stocks.",
    }


def vix_term_backwardation_persistent() -> dict:
    """VIX9D > VIX3M (front-month > back-month) = stress now > expected later.

    Persistent backwardation (>3 days) means the market is in stress mode,
    not just a one-day spike. Pros watch this for confirmation of regime change.
    """
    try:
        v9 = _yf("^VIX9D", period="1mo")
        v3m = _yf("^VIX3M", period="1mo")
        if v9 is None or v3m is None or v9.empty or v3m.empty:
            return {"firing": False, "status": "VIX data unavailable"}
        n = min(len(v9), len(v3m), 14)
        ratios = (v9["Close"].tail(n).reset_index(drop=True) /
                   v3m["Close"].tail(n).reset_index(drop=True))
        ratio_now = float(ratios.iloc[-1])
        days_backwardation = int((ratios > 1.0).sum())
        firing = days_backwardation >= 3
        return {
            "firing": firing,
            "ratio_now": ratio_now,
            "days_backwardation_2w": days_backwardation,
            "status": (f"VIX9D/VIX3M = {ratio_now:.2f}, "
                       f"{days_backwardation}/14d backwardation "
                       f"(>=3d = persistent stress)"),
            "rationale": ("Front-month > back-month for 3+ days = market accepts the "
                           "stress is structural, not transient. Regime has shifted."),
        }
    except Exception as e:
        return {"firing": False, "status": f"error: {e!r}"[:80]}


def btc_spy_correlation_high() -> dict:
    """30d BTC-SPY correlation > 0.7 = BTC will follow equities down.

    When correlation is high, BTC is NOT a safe haven — it's leveraged SPY.
    Rotating equity to BTC in this regime = doubling down on the same trade.
    """
    spy = _yf("SPY", period="3mo")
    btc = _yf("BTC-USD", period="3mo")
    if spy is None or btc is None or spy.empty or btc.empty:
        return {"firing": False, "status": "BTC/SPY data unavailable"}
    df = pd.DataFrame({
        "spy": spy["Close"].pct_change().reset_index(drop=True),
        "btc": btc["Close"].pct_change().reset_index(drop=True),
    }).dropna()
    if len(df) < 30:
        return {"firing": False, "status": "insufficient history"}
    corr = float(df.tail(30).corr().iloc[0, 1])
    firing = corr > 0.7
    return {
        "firing": firing,
        "correlation_30d": corr,
        "status": (f"BTC-SPY 30d correlation: {corr:.2f} "
                   f"({'HIGH — BTC will follow equities' if firing else 'low/mid'})"),
        "rationale": ("When BTC-SPY corr > 0.7, rotating equity to BTC is doubling "
                       "down on the same trade. Wait for decoupling or BTC bottom."),
    }


# ============================================================
# ACCELERATION SCORE (how fast is the equity top scorecard rising?)
# ============================================================

def equity_top_acceleration() -> dict:
    """Speed at which equity-top scorecard is rising.

    Standard scorecard: 3 → 5 → 7 → 9 over weeks/months.
    Acceleration: how many points added in the last 7 / 30 days?
    Fast acceleration = pre-empt the standard threshold.
    """
    try:
        # Use the stored history of the top scorecard via the state file
        state_file = Path(__file__).resolve().parent.parent / ".btc_top_phase_state.json"
        if not state_file.exists():
            # No history yet, just compute current
            from core.btc_top_scorecard import top_confirmation_scorecard
            sc = top_confirmation_scorecard()
            return {
                "current_n_met": sc["n_met"],
                "delta_30d": None,
                "accelerating": False,
                "status": (f"Current {sc['n_met']}/{sc['n_total']} firing. "
                           "No history file yet — acceleration unknown."),
                "rationale": "Need 7+ days of history to compute acceleration.",
            }
        import json
        st = json.loads(state_file.read_text())
        current_n = st.get("last_n_met", 0)
        # Without a richer history, treat current as snapshot
        from core.btc_top_scorecard import top_confirmation_scorecard
        sc = top_confirmation_scorecard()
        current_n_met = sc["n_met"]
        # Acceleration: did we jump by 2+ since last recorded?
        delta = current_n_met - current_n
        accelerating = delta >= 2
        return {
            "current_n_met": current_n_met,
            "previous_n_met": current_n,
            "delta_since_last": delta,
            "accelerating": accelerating,
            "status": (f"Top scorecard: {current_n} -> {current_n_met} ({delta:+d}). "
                       f"{'ACCELERATING' if accelerating else 'steady'}"),
            "rationale": ("Acceleration = market deteriorating faster than baseline. "
                           "Pre-empt the 5/10 or 7/10 threshold."),
        }
    except Exception as e:
        return {"firing": False, "status": f"error: {e!r}"[:80], "accelerating": False}


# ============================================================
# BTC BOTTOM PROXIMITY
# ============================================================

def btc_bottom_proximity() -> dict:
    """How close is BTC to its bottom? Bottom scorecard threshold = 6/8.

    If <3 firing: BTC FAR from bottom (rotate to CASH not BTC)
    If 3-5 firing: BTC midway (start scaling in carefully)
    If 6+ firing:   BTC AT/NEAR bottom (rotate equity to BTC directly)
    """
    try:
        from core.btc_bottom_scorecard import bottom_confirmation_scorecard
        sc = bottom_confirmation_scorecard()
        n_met = sc.get("n_met", 0)
        n_total = sc.get("n_total", 10)   # confirmation scorecard is 10 (was 8)
        if n_met >= 6:
            zone = "AT_BOTTOM"; destination = "BTC"
        elif n_met >= 3:
            zone = "APPROACHING"; destination = "BTC_DCA"
        else:
            zone = "FAR_FROM_BOTTOM"; destination = "CASH"
        return {
            "n_met": n_met,
            "n_total": n_total,
            "zone": zone,
            "rotation_destination": destination,
            "status": (f"BTC bottom scorecard: {n_met}/{n_total} firing "
                       f"-> {zone} -> rotate equity to {destination}"),
            "rationale": ("BTC bottom must fire to justify BTC as rotation target. "
                           "Otherwise route equity to cash (BIL/SGOV)."),
        }
    except Exception:
        # Fallback: read from disk cache
        try:
            from core.dashboard_cache import get_cached
            bs = get_cached("bottom_signals")
            if bs is None:
                return {"n_met": 0, "n_total": 8, "zone": "UNKNOWN",
                        "rotation_destination": "CASH", "status": "bottom scorecard unavailable"}
            # crude proxy: count signals firing
            return {"n_met": 0, "n_total": 8, "zone": "UNKNOWN",
                    "rotation_destination": "CASH",
                    "status": "bottom signals available but scorecard not computed"}
        except Exception:
            return {"n_met": 0, "n_total": 8, "zone": "UNKNOWN",
                    "rotation_destination": "CASH",
                    "status": "fallback failed"}


# ============================================================
# THE DECISION ENGINE
# ============================================================

def early_rotation_signal(current_equity_pct: float = 30,
                            total_stake_nzd: float = 130_000,
                            current_btc_pct: float = 0) -> dict:
    """current_btc_pct accepted for API compatibility; not currently used
    by the early-rotation math (which only sizes equity rotations)."""
    """The pro decision: HOLD / ROTATE_TO_CASH / ROTATE_TO_BTC.

    Logic:
      Step 1: Count leading-indicator warnings (max 7).
      Step 2: Determine velocity (accelerating?).
      Step 3: Determine BTC bottom proximity.
      Step 4: Combine into action.

    The crucial twist: when leading indicators fire BUT BTC isn't at
    bottom, route to CASH (BIL/SGOV). Don't rotate equity to BTC during
    a liquidity crunch.
    """
    # Gather leading indicators
    inds = {
        "small_caps_leading_down": small_cap_leading_large(),
        "defensive_sector_rotation": defensive_rotation(),
        "yield_curve_resteepening": yield_curve_resteepening_velocity(),
        "hy_spread_widening": hy_credit_spread_widening_velocity(),
        "spy_below_200d": spy_below_200dma_breach(),
        "vix_backwardation": vix_term_backwardation_persistent(),
        "btc_spy_correlation_high": btc_spy_correlation_high(),
    }
    n_firing = sum(1 for v in inds.values() if v.get("firing"))
    n_total = len(inds)

    # Acceleration
    accel = equity_top_acceleration()
    accelerating = accel.get("accelerating", False)

    # BTC bottom proximity
    btc_prox = btc_bottom_proximity()
    btc_zone = btc_prox.get("zone", "UNKNOWN")
    btc_n_met = btc_prox.get("n_met", 0)
    btc_n_total = btc_prox.get("n_total", 10)   # live total (confirmation scorecard = 10)

    # Correlation gate — if BTC-SPY corr is high, rotating to BTC is risky
    high_corr = inds["btc_spy_correlation_high"].get("firing", False)

    # ---- Decision matrix ----
    action = "HOLD"
    target_pct = current_equity_pct  # default no change
    destination = "NONE"
    urgency = "LOW"
    rationale_lines = []

    if n_firing >= 5 or (n_firing >= 3 and accelerating):
        # Major warning — definite rotation needed
        if btc_n_met >= 6 and not high_corr:
            # BTC is near bottom AND decoupled — go straight to BTC
            action = "ROTATE_TO_BTC"
            target_pct = 5  # mostly out of equities, into BTC
            destination = "BTC"
            urgency = "IMMEDIATE"
            rationale_lines.append(
                f"{n_firing}/{n_total} leading indicators firing AND BTC bottom near "
                f"({btc_n_met}/{btc_n_total}). Rotate equity directly to BTC."
            )
        else:
            # Equity dropping but BTC NOT ready — go to CASH first
            action = "ROTATE_TO_CASH"
            target_pct = 10
            destination = "CASH (BIL/SGOV T-bills)"
            urgency = "HIGH" if accelerating else "MEDIUM"
            rationale_lines.append(
                f"{n_firing}/{n_total} leading indicators firing — equity weakness "
                f"detected. But BTC scorecard only {btc_n_met}/{btc_n_total} "
                f"({btc_zone}). Route to CASH first, wait for BTC bottom."
            )
            if high_corr:
                rationale_lines.append(
                    "BTC-SPY correlation > 0.7 = BTC will follow equities down. "
                    "Cash now, BTC later."
                )
    elif n_firing >= 3:
        # Yellow zone — partial rotation
        action = "REDUCE_TO_CASH"
        target_pct = 20
        destination = "CASH"
        urgency = "MEDIUM"
        rationale_lines.append(
            f"{n_firing}/{n_total} leading indicators firing. Reduce equity by 1/3, "
            f"park in cash. Watch for acceleration."
        )
    elif n_firing >= 2:
        action = "WATCH"
        urgency = "LOW"
        rationale_lines.append(
            f"{n_firing}/{n_total} leading indicators firing — early warning. "
            f"No action yet, but tighten stops on equity."
        )
    else:
        rationale_lines.append(
            f"Only {n_firing}/{n_total} leading indicators firing. Stay allocated."
        )

    # Compute dollar amounts
    current_equity_nzd = total_stake_nzd * (current_equity_pct / 100)
    target_equity_nzd = total_stake_nzd * (target_pct / 100)
    rotation_nzd = max(0, current_equity_nzd - target_equity_nzd)

    return {
        "action": action,
        "destination": destination,
        "urgency": urgency,
        "n_firing": n_firing,
        "n_total": n_total,
        "accelerating": accelerating,
        "current_equity_pct": current_equity_pct,
        "target_equity_pct": target_pct,
        "current_equity_nzd": round(current_equity_nzd),
        "target_equity_nzd": round(target_equity_nzd),
        "rotation_nzd": round(rotation_nzd),
        "btc_bottom_zone": btc_zone,
        "btc_bottom_n_met": btc_n_met,
        "btc_bottom_n_total": btc_n_total,
        "rationale": " ".join(rationale_lines),
        "indicators": inds,
        "acceleration": accel,
        "btc_proximity": btc_prox,
        "asof": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CLI for testing
# ============================================================

def main():
    import json
    r = early_rotation_signal()
    print("=" * 70)
    print("EARLY ROTATION SIGNAL")
    print("=" * 70)
    print(f"  Action:       {r['action']}")
    print(f"  Destination:  {r['destination']}")
    print(f"  Urgency:      {r['urgency']}")
    print(f"  Firing:       {r['n_firing']}/{r['n_total']}")
    print(f"  Equity now:   ${r['current_equity_nzd']:,} ({r['current_equity_pct']}%)")
    print(f"  Equity tgt:   ${r['target_equity_nzd']:,} ({r['target_equity_pct']}%)")
    print(f"  Rotate now:   ${r['rotation_nzd']:,}")
    print(f"  BTC bottom:   {r['btc_bottom_n_met']}/8 ({r['btc_bottom_zone']})")
    print()
    print(f"  Rationale:    {r['rationale']}")
    print()
    print("INDICATORS:")
    for k, v in r["indicators"].items():
        mark = "[FIRING]" if v.get("firing") else "[ok    ]"
        try: print(f"  {mark} {k:30s} {v.get('status', '')[:80]}")
        except UnicodeEncodeError:
            s = v.get('status', '').encode('ascii', 'replace').decode()
            print(f"  {mark} {k:30s} {s[:80]}")


if __name__ == "__main__":
    main()
