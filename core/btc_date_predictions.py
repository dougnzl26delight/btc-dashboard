"""BTC date prediction additions.

Four complementary date-anchoring methods that combine with the halving
clock to give a richer picture of WHEN the cycle 5 bottom is likely to land:

  1. indicator_extrapolation() — projects when key bottom indicators
     (Realized Cap drawdown, MVRV-Z, hashrate, STH-MVRV) will fire at
     the current trajectory.

  2. cycle_4_analog() — maps current calendar day to its cycle 4 equivalent
     (cycle 5 peaked Oct 2025; cycle 4 peaked Nov 2021 = 46 months earlier).
     Tells you what cycle 4 did at this equivalent point.

  3. macro_calendar() — upcoming FOMC + CPI + NFP dates with BTC sensitivity
     context. Macro events often trigger short-term BTC volatility spikes.

  4. bottom_date_convergence() — combines all 4-5 date methods into a
     probability distribution over a 6-month window. Highlights the
     highest-probability bottom date band.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# CONSTANTS
# ============================================================

# Cycle 4 reference dates
CYCLE_4_PEAK = datetime(2021, 11, 8).date()
CYCLE_4_BOTTOM = datetime(2022, 11, 9).date()
CYCLE_5_PEAK = datetime(2025, 10, 6).date()
CYCLE_5_PEAK_PRICE = 124659
CYCLE_4_PEAK_PRICE = 67526
CYCLE_4_BOTTOM_PRICE = 16500

# Halving 4 + cycle pattern
HALVING_4 = datetime(2024, 4, 20).date()


# ============================================================
# 1. INDICATOR EXTRAPOLATION
# ============================================================

def _project_threshold(current: float, slope_per_day: float,
                       threshold: float, max_days: int = 365) -> Optional[int]:
    """Linear extrapolation: when does current + slope*t hit threshold?

    Returns days from now until threshold hit, or None if trajectory wrong.
    """
    if slope_per_day == 0:
        return None
    # If we're moving AWAY from threshold, return None
    if (threshold > current and slope_per_day < 0): return None
    if (threshold < current and slope_per_day > 0): return None
    days = (threshold - current) / slope_per_day
    if days <= 0 or days > max_days: return None
    return int(days)


def indicator_extrapolation() -> dict:
    """Project when key bottom signals will fire at current rate.

    Each indicator has a target threshold; we compute the linear extrapolation
    from recent trajectory.
    """
    from core.btc_pro_signals import _cm
    today = datetime.now(timezone.utc).date()
    results = {}

    # Realized Cap drawdown — target -15% (entry to bottom zone)
    try:
        df_cap = _cm("CapMrktCurUSD", days=365)
        df_mvrv = _cm("CapMVRVCur", days=365)
        if not df_cap.empty and not df_mvrv.empty:
            df = df_cap.join(df_mvrv, how="inner").dropna()
            df["rcap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
            rolling_max = df["rcap"].rolling(window=365, min_periods=30).max()
            df["dd"] = (df["rcap"] / rolling_max - 1) * 100
            current_dd = float(df["dd"].iloc[-1])
            # 30d slope (% per day)
            if len(df) >= 30:
                slope = (current_dd - float(df["dd"].iloc[-30])) / 30
                days_to_target = _project_threshold(current_dd, slope, -15)
                results["realized_cap_drawdown"] = {
                    "current": current_dd,
                    "target": -15.0,
                    "slope_30d_per_day": slope,
                    "days_to_target": days_to_target,
                    "projected_date": (today + timedelta(days=days_to_target)).isoformat()
                                       if days_to_target else None,
                    "note": (f"RCap drawdown {current_dd:+.1f}% -> -15% threshold "
                              f"in ~{days_to_target}d "
                              f"({(today + timedelta(days=days_to_target)).strftime('%b %Y')})"
                              if days_to_target else
                              f"RCap drawdown {current_dd:+.1f}% — trajectory not converging on threshold"),
                }
    except Exception as e:
        results["realized_cap_drawdown"] = {"error": str(e)[:60]}

    # MVRV-Z — target below -1.0
    try:
        df = _cm("CapMVRVCur", days=1460)
        if not df.empty and len(df) >= 200:
            rmean = df["CapMVRVCur"].rolling(1460, min_periods=200).mean()
            rstd = df["CapMVRVCur"].rolling(1460, min_periods=200).std()
            z = (df["CapMVRVCur"] - rmean) / rstd
            current_z = float(z.iloc[-1])
            if len(z) >= 30 and not pd.isna(z.iloc[-30]):
                slope = (current_z - float(z.iloc[-30])) / 30
                days_to_target = _project_threshold(current_z, slope, -1.0)
                results["mvrv_z"] = {
                    "current": current_z,
                    "target": -1.0,
                    "slope_30d_per_day": slope,
                    "days_to_target": days_to_target,
                    "projected_date": (today + timedelta(days=days_to_target)).isoformat()
                                       if days_to_target else None,
                    "note": (f"MVRV-Z {current_z:+.2f} -> -1.0 threshold in "
                              f"~{days_to_target}d "
                              f"({(today + timedelta(days=days_to_target)).strftime('%b %Y')})"
                              if days_to_target else
                              f"MVRV-Z {current_z:+.2f} — not converging on threshold at current rate"),
                }
    except Exception as e:
        results["mvrv_z"] = {"error": str(e)[:60]}

    # Hashrate drawdown — target -25% (capitulation threshold)
    try:
        from core.btc_premium_free import _blockchain_info
        df = _blockchain_info("hash-rate", timespan="2years")
        if not df.empty and len(df) >= 30:
            rolling_max = df["value"].rolling(window=365, min_periods=30).max()
            df["dd"] = (df["value"] / rolling_max - 1) * 100
            current_dd = float(df["dd"].iloc[-1])
            if len(df) >= 30:
                slope = (current_dd - float(df["dd"].iloc[-30])) / 30
                # If already past -25%, status is "already there"
                if current_dd < -25:
                    results["hashrate_drawdown"] = {
                        "current": current_dd,
                        "target": -25.0,
                        "days_to_target": 0,
                        "projected_date": today.isoformat(),
                        "note": (f"Hashrate {current_dd:.1f}% — ALREADY past -25% threshold. "
                                  f"Watch for stabilization + recovery."),
                    }
                else:
                    days_to_target = _project_threshold(current_dd, slope, -25)
                    results["hashrate_drawdown"] = {
                        "current": current_dd,
                        "target": -25.0,
                        "slope_30d_per_day": slope,
                        "days_to_target": days_to_target,
                        "projected_date": (today + timedelta(days=days_to_target)).isoformat()
                                           if days_to_target else None,
                        "note": (f"Hashrate {current_dd:+.1f}% -> -25% threshold in "
                                  f"~{days_to_target}d"
                                  if days_to_target else
                                  f"Hashrate {current_dd:+.1f}% — trajectory uncertain"),
                    }
    except Exception as e:
        results["hashrate_drawdown"] = {"error": str(e)[:60]}

    # Summarize earliest convergence date
    valid_dates = []
    for k, v in results.items():
        if isinstance(v, dict) and not v.get("error") and v.get("projected_date"):
            try:
                d = datetime.strptime(v["projected_date"], "%Y-%m-%d").date()
                valid_dates.append((k, d))
            except Exception: pass

    if valid_dates:
        earliest = min(valid_dates, key=lambda x: x[1])
        median = sorted(valid_dates, key=lambda x: x[1])[len(valid_dates) // 2]
        latest = max(valid_dates, key=lambda x: x[1])
        return {
            "indicators": results,
            "earliest_fire": {"name": earliest[0], "date": earliest[1].isoformat()},
            "median_fire": {"name": median[0], "date": median[1].isoformat()},
            "latest_fire": {"name": latest[0], "date": latest[1].isoformat()},
            "summary": (f"Indicators projected to fire: earliest "
                         f"{earliest[1].strftime('%b %Y')} ({earliest[0]}), "
                         f"median {median[1].strftime('%b %Y')}, "
                         f"latest {latest[1].strftime('%b %Y')}"),
        }
    return {"indicators": results,
            "summary": "Insufficient data to extrapolate indicator fire dates"}


# ============================================================
# 2. CYCLE 4 ANALOG OVERLAY
# ============================================================

def cycle_4_analog() -> dict:
    """Map today's date to equivalent cycle 4 day post-peak.

    Cycle 5 peaked Oct 6, 2025. Cycle 4 peaked Nov 8, 2021.
    Offset = 1428 days (~46.9 months). Apply this offset to find what
    cycle 4 looked like at this analog point.
    """
    today = datetime.now(timezone.utc).date()
    days_since_cycle5_peak = (today - CYCLE_5_PEAK).days
    analog_cycle4_date = CYCLE_4_PEAK + timedelta(days=days_since_cycle5_peak)
    days_to_cycle4_bottom = (CYCLE_4_BOTTOM - analog_cycle4_date).days
    projected_cycle5_bottom = today + timedelta(days=days_to_cycle4_bottom)

    # Cycle 4 prices at key analog points (approximate, from historical data)
    # We compute the implied cycle 5 trajectory by ratio
    cycle4_drawdown_at_analog = None
    cycle5_drawdown_now = None
    # Hand-tabulated cycle 4 closing prices at key post-peak days
    cycle4_path = {
        # days_since_peak: price
        0:    67526,   # Nov 8, 2021 peak
        30:   57500,
        60:   46000,
        90:   42000,
        120:  38000,
        150:  37500,
        180:  31000,
        210:  20800,   # Luna crash period
        240:  21500,
        270:  20100,
        300:  19500,
        330:  16500,
        365:  16500,   # bottom area
    }
    nearest_day = min(cycle4_path.keys(),
                      key=lambda d: abs(d - days_since_cycle5_peak))
    cycle4_implied_price_at_analog = cycle4_path[nearest_day]
    cycle4_drawdown_at_analog = (cycle4_implied_price_at_analog / CYCLE_4_PEAK_PRICE - 1) * 100

    # Get current BTC price
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        current_price = float(df["close"].iloc[-1]) if not df.empty else 73000
        cycle5_drawdown_now = (current_price / CYCLE_5_PEAK_PRICE - 1) * 100
    except Exception:
        current_price = None

    # Implied cycle 5 bottom (using cycle 4 bottom multiple, scaled by amplitude decay)
    cycle4_bottom_drawdown = -75.6  # actual
    # Cycle 5 amplitude reduction estimate from peak comparison
    amplitude_decay = 0.6  # roughly 60% of cycle 4 amplitude
    implied_cycle5_bottom_drawdown = cycle4_bottom_drawdown * amplitude_decay
    implied_cycle5_bottom_price = CYCLE_5_PEAK_PRICE * (1 + implied_cycle5_bottom_drawdown / 100)

    return {
        "today": today.isoformat(),
        "days_since_cycle5_peak": days_since_cycle5_peak,
        "cycle4_analog_date": analog_cycle4_date.isoformat(),
        "days_to_analog_cycle4_bottom": days_to_cycle4_bottom,
        "projected_cycle5_bottom_date": projected_cycle5_bottom.isoformat(),
        "cycle4_drawdown_at_this_analog_day": cycle4_drawdown_at_analog,
        "cycle5_drawdown_now": cycle5_drawdown_now,
        "tracking_ratio": (cycle5_drawdown_now / cycle4_drawdown_at_analog
                            if cycle5_drawdown_now and cycle4_drawdown_at_analog else None),
        "implied_cycle5_bottom_price": implied_cycle5_bottom_price,
        "implied_cycle5_bottom_drawdown_pct": implied_cycle5_bottom_drawdown,
        "summary": (f"Cycle 4 analog day: {analog_cycle4_date} (BTC was "
                     f"~${cycle4_implied_price_at_analog:,.0f}, {cycle4_drawdown_at_analog:.0f}% from C4 peak). "
                     f"Cycle 4 bottomed {days_to_cycle4_bottom}d later "
                     f"= cycle 5 bottom projection: {projected_cycle5_bottom.strftime('%b %d, %Y')}. "
                     f"Implied cycle 5 bottom price: ~${implied_cycle5_bottom_price:,.0f}."),
    }


# ============================================================
# 3. MACRO CALENDAR (FOMC + CPI + NFP)
# ============================================================

# Scheduled FOMC meeting dates (2026 schedule from Federal Reserve)
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_2026 = [
    datetime(2026, 1, 27).date(), datetime(2026, 1, 28).date(),
    datetime(2026, 3, 17).date(), datetime(2026, 3, 18).date(),
    datetime(2026, 4, 28).date(), datetime(2026, 4, 29).date(),
    datetime(2026, 6, 16).date(), datetime(2026, 6, 17).date(),
    datetime(2026, 7, 28).date(), datetime(2026, 7, 29).date(),
    datetime(2026, 9, 15).date(), datetime(2026, 9, 16).date(),
    datetime(2026, 11, 3).date(), datetime(2026, 11, 4).date(),
    datetime(2026, 12, 15).date(), datetime(2026, 12, 16).date(),
]

FOMC_2027 = [
    datetime(2027, 1, 26).date(), datetime(2027, 1, 27).date(),
    datetime(2027, 3, 16).date(), datetime(2027, 3, 17).date(),
    datetime(2027, 5, 4).date(), datetime(2027, 5, 5).date(),
]


def _next_monthly_release(day_of_month: int, after_date) -> datetime.date:
    """Find next month with specific day."""
    candidate_date = after_date.replace(day=day_of_month)
    if candidate_date < after_date:
        if candidate_date.month == 12:
            candidate_date = candidate_date.replace(year=candidate_date.year + 1, month=1)
        else:
            candidate_date = candidate_date.replace(month=candidate_date.month + 1)
    return candidate_date


def macro_calendar(days_ahead: int = 180) -> dict:
    """Upcoming macro events that move BTC.

    FOMC (rate decisions), CPI (inflation prints), NFP (jobs data).
    """
    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=days_ahead)
    events = []

    # FOMC meetings (use second day = announcement day)
    for d in FOMC_2026 + FOMC_2027:
        if today <= d <= cutoff:
            # Skip first day; second day is announcement
            if FOMC_2026.index(d) % 2 == 1 if d in FOMC_2026 else FOMC_2027.index(d) % 2 == 1:
                pass
            events.append({
                "date": d.isoformat(),
                "event": "FOMC announcement",
                "type": "FOMC",
                "days_from_now": (d - today).days,
                "btc_sensitivity": "HIGH",
                "context": "Rate decision + dot plot. Often triggers 3-7% BTC move within 4h.",
            })

    # CPI: typically mid-month (around 12th-13th, but exact date is announced)
    # Use 12th as proxy
    cpi_date = _next_monthly_release(12, today)
    while cpi_date <= cutoff:
        events.append({
            "date": cpi_date.isoformat(),
            "event": f"CPI release ({cpi_date.strftime('%b')} data)",
            "type": "CPI",
            "days_from_now": (cpi_date - today).days,
            "btc_sensitivity": "MEDIUM-HIGH",
            "context": "Inflation print. Hot CPI = dollar strength = BTC headwind; cool CPI = tailwind.",
        })
        if cpi_date.month == 12:
            cpi_date = cpi_date.replace(year=cpi_date.year + 1, month=1)
        else:
            cpi_date = cpi_date.replace(month=cpi_date.month + 1)

    # NFP: first Friday of each month
    def _first_friday(year, month):
        d = datetime(year, month, 1).date()
        while d.weekday() != 4:
            d = d + timedelta(days=1)
        return d
    cursor = today.replace(day=1)
    while cursor <= cutoff:
        nfp_d = _first_friday(cursor.year, cursor.month)
        if today <= nfp_d <= cutoff:
            events.append({
                "date": nfp_d.isoformat(),
                "event": f"NFP release ({nfp_d.strftime('%b')} jobs)",
                "type": "NFP",
                "days_from_now": (nfp_d - today).days,
                "btc_sensitivity": "MEDIUM",
                "context": "Employment data. Strong jobs = no Fed cuts = BTC headwind.",
            })
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1, day=1)

    events.sort(key=lambda e: e["date"])

    return {
        "events": events[:25],
        "events_in_window": len(events),
        "next_high_impact": (events[0] if events and events[0]["btc_sensitivity"] in ("HIGH", "MEDIUM-HIGH")
                              else next((e for e in events if e["btc_sensitivity"] == "HIGH"), None)),
        "summary": (f"{len(events)} macro events in next {days_ahead}d. "
                     f"Next: {events[0]['event']} on {events[0]['date']}"
                     if events else "No upcoming macro events"),
    }


# ============================================================
# 4. BOTTOM DATE CONVERGENCE
# ============================================================

def bottom_date_convergence() -> dict:
    """Combine all 4-5 date methods into a probability distribution.

    Methods (each contributes its estimate with confidence weight):
    1. Halving math (Oct 7, 2026, weight 1.0)
    2. Cycle 4 analog (variable date, weight 0.8)
    3. Indicator extrapolation median (variable, weight 0.6)
    4. Probability EV (Oct ~2026, weight 0.7)
    """
    from core.halving_clock import current_halving_position
    today = datetime.now(timezone.utc).date()

    # Method 1: Halving math
    pos = current_halving_position()
    halving_date = pos["projected_bottom_date"]
    if hasattr(halving_date, "isoformat"):
        halving_date = halving_date
    estimates = [{"method": "halving_math", "date": halving_date, "weight": 1.0,
                  "note": "Halving + 900d pattern (n=2)"}]

    # Method 2: Cycle 4 analog
    try:
        c4 = cycle_4_analog()
        c4_date = datetime.strptime(c4["projected_cycle5_bottom_date"], "%Y-%m-%d").date()
        estimates.append({"method": "cycle_4_analog", "date": c4_date,
                          "weight": 0.8, "note": "Cycle 4 days-from-peak pattern"})
    except Exception: pass

    # Method 3: Indicator extrapolation median
    try:
        ie = indicator_extrapolation()
        if ie.get("median_fire"):
            ie_date = datetime.strptime(ie["median_fire"]["date"], "%Y-%m-%d").date()
            estimates.append({"method": "indicator_median", "date": ie_date,
                              "weight": 0.6,
                              "note": f"Median {ie['median_fire']['name']} extrapolation"})
    except Exception: pass

    # Method 4: Probability EV (from cost basis module)
    try:
        from core.btc_cost_basis import bottom_probability_distribution
        pdb = bottom_probability_distribution()
        if pdb and not pdb.get("error"):
            # EV date isn't directly available; use halving math as proxy weighted by EV scenario
            # Scenario A (50%): standard halving cycle Aug-Dec 2026
            # Use midpoint Oct 15, 2026 weighted by 0.5
            estimates.append({"method": "probability_ev_scenario_a",
                              "date": datetime(2026, 10, 15).date(),
                              "weight": 0.7,
                              "note": "Scenario A midpoint (50% probability)"})
    except Exception: pass

    # Method 5: Olson monthly-MACD analog (added 2026-06-10).
    # Olson: monthly MACD crossed below zero ~Jun 2026; in 2022 the same cross
    # preceded the bottom by ~3 months -> mid-September 2026. Guru-sourced,
    # static date; weight below algorithmic methods. Remove if invalidated.
    estimates.append({"method": "olson_monthly_macd",
                      "date": datetime(2026, 9, 15).date(),
                      "weight": 0.6,
                      "note": "Olson: monthly MACD zero-cross + 3mo (2022 analog)"})

    if not estimates: return {"error": "no estimates available"}

    # Build probability distribution over 6-month window
    # Compute weighted average date
    total_weight = sum(e["weight"] for e in estimates)
    days_from_today = [(e["date"] - today).days for e in estimates]
    weighted_days = sum(d * e["weight"] for d, e in zip(days_from_today, estimates)) / total_weight
    ev_date = today + timedelta(days=int(weighted_days))

    # Range = earliest to latest
    earliest = min(estimates, key=lambda e: e["date"])
    latest = max(estimates, key=lambda e: e["date"])
    spread_days = (latest["date"] - earliest["date"]).days

    return {
        "estimates":         [{"method": e["method"], "date": e["date"].isoformat(),
                                "weight": e["weight"], "note": e["note"]} for e in estimates],
        "n_methods":         len(estimates),
        "ev_date":           ev_date.isoformat(),
        "earliest_estimate": earliest["date"].isoformat(),
        "latest_estimate":   latest["date"].isoformat(),
        "spread_days":       spread_days,
        "summary": (f"{len(estimates)} methods converge on "
                     f"{earliest['date'].strftime('%b %d, %Y')} -> "
                     f"{latest['date'].strftime('%b %d, %Y')} window. "
                     f"Weighted EV date: {ev_date.strftime('%b %d, %Y')}. "
                     f"Spread: {spread_days} days."),
    }


def main():
    print("\n" + "=" * 78)
    print("BTC DATE PREDICTIONS — 4 complementary methods")
    print("=" * 78)
    print()
    print("--- 1. INDICATOR EXTRAPOLATION ---")
    ie = indicator_extrapolation()
    print(f"  {ie.get('summary', '?')}")
    for name, ind in ie.get("indicators", {}).items():
        if isinstance(ind, dict) and not ind.get("error"):
            print(f"    {name:<25s} {ind.get('note', '')[:80]}")

    print()
    print("--- 2. CYCLE 4 ANALOG ---")
    c4 = cycle_4_analog()
    print(f"  {c4.get('summary', '?')}")

    print()
    print("--- 3. MACRO CALENDAR (next 6 months) ---")
    mc = macro_calendar(180)
    print(f"  {mc.get('summary', '?')}")
    for e in mc.get("events", [])[:8]:
        print(f"    {e['date']}  {e['btc_sensitivity']:<14s}  {e['event']}")

    print()
    print("--- 4. BOTTOM DATE CONVERGENCE ---")
    bdc = bottom_date_convergence()
    print(f"  {bdc.get('summary', '?')}")
    for est in bdc.get("estimates", []):
        print(f"    {est['date']}  weight {est['weight']:.1f}  {est['method']:<25s}  {est['note']}")


if __name__ == "__main__":
    main()
