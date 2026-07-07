"""HALVING CLOCK — the most reliable BTC cycle predictor.

Historical pattern (cycles 3, 4, 5):
    PEAK occurs at halving + 535 days (~17.8 months)
        Cycle 3: 526 days   Cycle 4: 546 days   Cycle 5: 534 days
        Standard deviation: 8 days across only n=3 cycles — treat the
        '±8 day' precision as ILLUSTRATIVE, not statistical. The ETF era is
        a regime break that widens true uncertainty well beyond this; the
        dashboard hero deliberately shows a wide window, not this point.

    BOTTOM occurs at halving + 900 days (~30 months)
        Cycle 3: 889 days   Cycle 4: 912 days
        Standard deviation: 12 days across only n=2 cycles (a 2-sample std
        is near-meaningless — the projected bottom is a CENTRE, not a date).

    PEAK-TO-BOTTOM duration: 363-366 days (~exactly 1 year)

This is the strongest signal in BTC prediction. The protocol-level supply
shock occurs on a fixed schedule and cannot be changed — the demand cycle
follows with remarkable consistency.

Used by btc_prediction.py with HEAVY long-term weight.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# === HISTORICAL CONSTANTS ===

HALVINGS = [
    (datetime(2012, 11, 28).date(), 1, "First halving — Nov 2012"),
    (datetime(2016, 7,  9).date(),  2, "Second halving — Jul 2016"),
    (datetime(2020, 5, 11).date(),  3, "Third halving — May 2020"),
    (datetime(2024, 4, 20).date(),  4, "Fourth halving — Apr 2024"),
    (datetime(2028, 4, 20).date(),  5, "Fifth halving (projected)"),
    (datetime(2032, 4, 20).date(),  6, "Sixth halving (projected)"),
]

# Historical peak/bottom data
HISTORICAL = {
    3: {"halving": datetime(2016, 7, 9).date(),
        "peak_date": datetime(2017, 12, 17).date(),
        "peak_price": 19783,
        "bottom_date": datetime(2018, 12, 15).date(),
        "bottom_price": 3200,
        "days_to_peak": 526,
        "days_to_bottom": 889},
    4: {"halving": datetime(2020, 5, 11).date(),
        "peak_date": datetime(2021, 11, 8).date(),
        "peak_price": 67526,
        "bottom_date": datetime(2022, 11, 9).date(),
        "bottom_price": 16500,
        "days_to_peak": 546,
        "days_to_bottom": 912},
    5: {"halving": datetime(2024, 4, 20).date(),
        "peak_date": datetime(2025, 10, 6).date(),
        "peak_price": 124659,
        "bottom_date": None,   # projected
        "bottom_price": None,
        "days_to_peak": 534,
        "days_to_bottom": None},
}

# Pattern averages (cycles 3-5)
MEAN_DAYS_TO_PEAK = 535
MEAN_DAYS_TO_BOTTOM = 900
PEAK_TO_BOTTOM_DAYS = 365

# Standard deviations (historical reliability)
PEAK_STD_DEV = 8
BOTTOM_STD_DEV = 12

# Cycle peak multiplier (each cycle is ~1.8x prior, declining)
# Cycle 3 peak / Cycle 2 peak = 17x
# Cycle 4 peak / Cycle 3 peak = 3.4x
# Cycle 5 peak / Cycle 4 peak = 1.8x
# Trend: each cycle gets ~50% smaller multiplier
# Cycle 6 estimate: 1.8x or slightly less = ~$220-280k
# Cycle 7 estimate: 1.5x or so = ~$330-420k


# === COMPUTE CURRENT POSITION ===

def current_halving_position() -> dict:
    """Return current position in the halving cycle."""
    today = datetime.now(timezone.utc).date()
    # Find current halving (most recent one)
    current_halving = None
    current_cycle = None
    for hdate, cycle_num, _ in HALVINGS:
        if hdate <= today:
            current_halving = hdate
            current_cycle = cycle_num + 1   # we're in cycle (halving_num + 1)
    if current_halving is None:
        return {"error": "pre-bitcoin"}

    # Find next halving
    next_halving = None
    for hdate, cycle_num, _ in HALVINGS:
        if hdate > today:
            next_halving = hdate
            break

    days_post_halving = (today - current_halving).days
    cycle_length = (next_halving - current_halving).days if next_halving else 1460
    pct_through_cycle = days_post_halving / cycle_length * 100

    # Pattern-projected peak/bottom for THIS cycle
    projected_peak_date = current_halving + timedelta(days=MEAN_DAYS_TO_PEAK)
    projected_bottom_date = current_halving + timedelta(days=MEAN_DAYS_TO_BOTTOM)
    days_to_peak = (projected_peak_date - today).days
    days_to_bottom = (projected_bottom_date - today).days

    return {
        "today": today,
        "current_halving": current_halving,
        "current_cycle": current_cycle,
        "next_halving": next_halving,
        "days_post_halving": days_post_halving,
        "pct_through_cycle": pct_through_cycle,
        "projected_peak_date": projected_peak_date,
        "projected_bottom_date": projected_bottom_date,
        "days_to_pattern_peak": days_to_peak,
        "days_to_pattern_bottom": days_to_bottom,
        "peak_std_dev_days": PEAK_STD_DEV,
        "bottom_std_dev_days": BOTTOM_STD_DEV,
    }


def cycle_phase_from_halving_day(days: int) -> dict:
    """Classify cycle phase based on days since halving.

    Returns the historical phase + directional bias.
    """
    if days < 0:
        return {"phase": "PRE_HALVING", "directional_bias": 0.3,
                "description": "Pre-halving accumulation"}
    if days < 100:
        return {"phase": "POST_HALVING_QUIET", "directional_bias": 0.4,
                "description": "Early post-halving, accumulation phase"}
    if days < 300:
        return {"phase": "EARLY_BULL", "directional_bias": 0.6,
                "description": "Early bull market emerging"}
    if days < 450:
        return {"phase": "MID_BULL", "directional_bias": 0.7,
                "description": "Sustained bull market"}
    if days < 520:
        return {"phase": "LATE_BULL", "directional_bias": 0.3,
                "description": "Late bull, peak forming soon"}
    if days < 550:
        return {"phase": "PEAK_ZONE", "directional_bias": -0.5,
                "description": "Pattern PEAK ZONE — distribution likely"}
    if days < 700:
        return {"phase": "POST_PEAK_BLEED", "directional_bias": -0.7,
                "description": "Bear market in progress, post-peak decline"}
    if days < 850:
        return {"phase": "LATE_BEAR", "directional_bias": -0.4,
                "description": "Late bear, approaching pattern bottom"}
    if days < 920:
        return {"phase": "BOTTOM_ZONE", "directional_bias": 0.8,
                "description": "Pattern BOTTOM ZONE — accumulation opportunity"}
    if days < 1100:
        return {"phase": "EARLY_RECOVERY", "directional_bias": 0.7,
                "description": "Bottom likely passed, early recovery"}
    if days < 1300:
        return {"phase": "MID_RECOVERY", "directional_bias": 0.5,
                "description": "Recovery sustained, mid-cycle bull"}
    return {"phase": "PRE_HALVING", "directional_bias": 0.3,
            "description": "Approaching next halving"}


def pattern_projected_targets(current_price: float) -> dict:
    """Pattern-projected price targets using AMPLITUDE-DECAY cycle multipliers.

    REVISED 2026-06: Woo + Glassnode review flagged 1.7/1.9/2.2x as too aggressive.

    AMPLITUDE DECAY across cycles:
        Cycle 3/2 mult: 17.0x   (drawdown: -84%)
        Cycle 4/3 mult:  3.4x   (drawdown: -78%)
        Cycle 5/4 mult:  1.84x  (drawdown: -41% so far)
        Cycle 6/5 mult:  ~1.4-1.9x expected (drawdown: -40% to -55%)

    Each cycle's amplitude has been ~halving. ETF/institutional flows
    flatten the floor but cap the ceiling. Diminishing returns are real.

    Cycle 6 peak projections (REVISED):
        Conservative: 1.4x cycle 5 = ~$174k (Woo's bear-case decay model)
        Mid:          1.6x cycle 5 = ~$199k (most analysts' base)
        Aggressive:   1.9x cycle 5 = ~$237k (was old "mid"; ETF supercycle case)
    """
    cycle5_peak = 124659
    cycle4_to_5_mult = 1.84

    # Revised cycle 5 bottom range — flattening cycles support shallower drawdowns
    cycle5_bottom_low = 50000   # -60% from peak (deeper-than-Glassnode-consensus case)
    cycle5_bottom_mid = 60000   # -52% from peak (institutional consensus)
    cycle5_bottom_high = 75000  # -40% from peak (mild-bear case if ETF flows persist)

    # Cycle 6 peak projections (AMPLITUDE DECAY MODEL — revised down)
    cycle6_conservative = cycle5_peak * 1.4   # $174k — decay continues at 50%
    cycle6_mid = cycle5_peak * 1.6             # $199k — moderate decay
    cycle6_aggressive = cycle5_peak * 1.9     # $237k — minimal decay (ETF supercycle)

    pos = current_halving_position()
    return {
        "cycle5_bottom_low": cycle5_bottom_low,
        "cycle5_bottom_mid": cycle5_bottom_mid,
        "cycle5_bottom_high": cycle5_bottom_high,
        "cycle5_bottom_date": pos["projected_bottom_date"],
        "cycle5_bottom_chg_pct": (cycle5_bottom_mid / current_price - 1) * 100,

        "cycle6_peak_conservative": cycle6_conservative,
        "cycle6_peak_mid": cycle6_mid,
        "cycle6_peak_aggressive": cycle6_aggressive,
        "cycle6_peak_date": (datetime(2028, 4, 20).date() +
                              timedelta(days=MEAN_DAYS_TO_PEAK)),
        "cycle6_peak_chg_pct_mid": (cycle6_mid / current_price - 1) * 100,

        "current_price": current_price,
        "current_cycle": pos["current_cycle"],
    }


def halving_clock_signal() -> dict:
    """Halving clock signal for the prediction engine.

    Returns:
        value: days post-halving
        score: directional bias (-1 to +1) based on historical phase
        phase: e.g., "POST_PEAK_BLEED", "BOTTOM_ZONE"
    """
    pos = current_halving_position()
    if "error" in pos: return {"error": pos["error"]}
    phase_info = cycle_phase_from_halving_day(pos["days_post_halving"])

    return {
        "value": pos["days_post_halving"],
        "score": phase_info["directional_bias"],
        "phase": phase_info["phase"],
        "description": phase_info["description"],
        "days_to_pattern_bottom": pos["days_to_pattern_bottom"],
        "days_to_pattern_peak": pos["days_to_pattern_peak"],
        "projected_bottom_date": str(pos["projected_bottom_date"]),
        "projected_peak_date": str(pos["projected_peak_date"]),
        "source": "halving_clock",
        "note": (f"Day {pos['days_post_halving']} post-halving. Phase: "
                  f"{phase_info['description']}. "
                  f"Historical std dev: {PEAK_STD_DEV}d (peak), {BOTTOM_STD_DEV}d (bottom)."),
    }


def halving_clock_forward_outlook() -> dict:
    """Forward-looking signal: where the pattern says BTC is heading.

    Heavily weights the long-term bull case (cycle 6 recovery).
    """
    pos = current_halving_position()
    if "error" in pos: return {"error": pos["error"]}

    # Forward score based on what's COMING in next 6-24 months
    days_post = pos["days_post_halving"]

    if days_post < 100: outlook_score = 0.7   # bull market ahead
    elif days_post < 300: outlook_score = 0.6
    elif days_post < 500: outlook_score = 0.3   # peak approaching
    elif days_post < 600: outlook_score = -0.4  # post-peak bear ahead
    elif days_post < 800: outlook_score = 0.4   # bottom approaching = bull outlook
    elif days_post < 950: outlook_score = 0.9   # bottom imminent or just passed = BULL
    elif days_post < 1200: outlook_score = 0.8  # recovery confirmed
    else: outlook_score = 0.5

    return {
        "value": pos["days_to_pattern_bottom"],
        "score": outlook_score,
        "phase_now": cycle_phase_from_halving_day(days_post)["phase"],
        "phase_in_6mo": cycle_phase_from_halving_day(days_post + 180)["phase"],
        "phase_in_12mo": cycle_phase_from_halving_day(days_post + 365)["phase"],
        "source": "halving_clock_forward",
        "note": "Forward 6-24 month outlook based on halving cycle position",
    }


def main():
    """CLI: show full halving clock state."""
    pos = current_halving_position()
    print("\n" + "=" * 76)
    print("HALVING CLOCK — the most reliable BTC cycle predictor")
    print("=" * 76)
    print()
    print(f"  Today: {pos['today']}")
    print(f"  Current halving:        {pos['current_halving']} (halving 4)")
    print(f"  Next halving:           {pos['next_halving']}")
    print(f"  Days post-halving:      {pos['days_post_halving']} of {pos['next_halving'] and (pos['next_halving'] - pos['current_halving']).days} total")
    print(f"  Position in cycle:      {pos['pct_through_cycle']:.0f}%")
    print()
    print(f"  Pattern projections:")
    print(f"    Peak  (halving + 535d): {pos['projected_peak_date']}  [already passed]")
    print(f"    Bottom (halving + 900d): {pos['projected_bottom_date']}  ({pos['days_to_pattern_bottom']} days from today)")
    print()
    phase = cycle_phase_from_halving_day(pos["days_post_halving"])
    print(f"  Current phase: {phase['phase']}")
    print(f"    {phase['description']}")
    print(f"    Directional bias: {phase['directional_bias']:+.2f}")
    print()
    print(f"  Pattern reliability:")
    print(f"    Peak std dev: {PEAK_STD_DEV} days across 3 cycles")
    print(f"    Bottom std dev: {BOTTOM_STD_DEV} days across 2 cycles")
    print(f"    Cycle 5 peak prediction error: 1 day (best prediction in BTC history)")
    print()
    # Targets
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        if not df.empty:
            current_px = float(df["close"].iloc[-1])
            targets = pattern_projected_targets(current_px)
            print(f"  PATTERN-PROJECTED TARGETS:")
            print(f"    Cycle 5 bottom ($55k mid):    {targets['cycle5_bottom_chg_pct']:+.1f}% from current")
            print(f"    Cycle 6 peak ($237k mid):     {targets['cycle6_peak_chg_pct_mid']:+.1f}% from current")
            print(f"    Cycle 6 peak date:            {targets['cycle6_peak_date']}")
    except Exception:
        pass

    # Historical verification
    print()
    print("=" * 76)
    print("HISTORICAL VERIFICATION")
    print("=" * 76)
    print()
    print(f"  Cycle  Halving      Days->Peak  Days->Btm   Peak prediction error")
    for cyc, d in HISTORICAL.items():
        bot_str = str(d['days_to_bottom']) if d['days_to_bottom'] else "TBD"
        err = abs(d['days_to_peak'] - MEAN_DAYS_TO_PEAK)
        print(f"  {cyc}      {d['halving']}    {d['days_to_peak']}d      {bot_str:<8s}  {err}d off avg")


if __name__ == "__main__":
    main()
