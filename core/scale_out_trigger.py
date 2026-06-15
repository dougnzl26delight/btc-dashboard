"""BTC TOP scale-out trigger — the exit-side twin of the rotation trigger.

Rotation trigger (equity->BTC) fires ONCE near the bottom. This trigger
manages the OTHER end: scaling OUT of BTC near the next cycle top
(projected window ~2029, halving Apr 2028 + ~535d pattern).

Design differences vs the bottom (deliberate):
  - Bottoms are processes (months long) -> single-shot rotation works.
  - Tops are events (blow-off weeks)    -> PHASED exits: 25% / 50% / 75%.
    Round-tripping a top costs more than imperfect tranche timing.

State machine:
  DORMANT       BTC > 15% below its 365d high (bear/recovery — never trim here)
  ARMED         within 15% of the 365d high, no tier conditions met
  TRIM_25       top scorecard >= scaled(4)/16  OR  ATH-stagnation criterion fires
  SCALE_OUT_50  top scorecard >= scaled(6)/16  AND Olson technical layer bearish
  EXIT_75       top scorecard >= scaled(8)/16  OR
                (scaled(6) AND Olson bearish AND ATH stagnation)

Cycle-aware: thresholds scale by the same cycle-6 ETF modifier as the
bottom (Oct 2025 proved classic top signals under-fire in muted cycles;
the ATH-stagnation detector is the cycle-agnostic backstop — it fired
60 days before the cycle-5 peak in backtest when everything else slept).

Each tier escalation is surfaced via the hourly alert email with ** flags.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Baseline tier thresholds (out of 16 criteria), before cycle scaling
TRIM_THRESHOLD = 4
SCALE_OUT_THRESHOLD = 6
EXIT_THRESHOLD = 8
# Arming proximity: within this % below the rolling 365d high
ARM_PROXIMITY_PCT = 15

TIER_RANK = {"DORMANT": 0, "ARMED": 1, "TRIM_25": 2,
             "SCALE_OUT_50": 3, "EXIT_75": 4}

# Portfolio context for NZ$ action lines
TOTAL_STAKE_NZD = 130_000
POST_ROTATION_BTC_PCT = 0.93   # after the equity->BTC rotation executes
CURRENT_BTC_PCT = 0.30          # pre-rotation


def _btc_vs_365d_high() -> dict:
    """BTC price vs its rolling 365d high (the arming condition)."""
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=400)
        if df is None or df.empty:
            return {"error": "no data"}
        closes = df["close"].dropna()
        high_365 = float(closes.max())
        px = float(closes.iloc[-1])
        try:
            import ccxt
            live = float(ccxt.binance().fetch_ticker("BTC/USDT").get("last") or 0)
            if live > 0: px = live
        except Exception:
            pass
        pct_below = (px / high_365 - 1) * 100
        return {"price": px, "high_365d": high_365,
                "pct_below_high": round(pct_below, 1)}
    except Exception as e:
        return {"error": str(e)}


def _olson_bearish() -> tuple[bool, str]:
    """Is Olson's BTC technical layer (3wk MACD / weekly HA / RSI div) bearish?"""
    try:
        from core.dashboard_cache import get_cached
        ol = get_cached("olson") or {}
        vl = str(ol.get("verdict_level") or ol.get("verdict") or "").upper()
        bear_n = ol.get("bearish_count", 0) or 0
        bull_n = ol.get("bullish_count", 0) or 0
        bearish = ("BEAR" in vl) or (bear_n > bull_n)
        return bearish, f"{vl or '?'} (bear {bear_n} vs bull {bull_n})"
    except Exception:
        return False, "olson cache unavailable"


def evaluate_scale_out_trigger() -> dict:
    """Evaluate the BTC top scale-out state machine."""
    from core.dashboard_cache import get_cached

    # Cycle scaling — same modifier as the bottom trigger
    try:
        from core.rotation_validation import cycle6_modifier
        cyc = cycle6_modifier()
        scale = cyc.get("suggested_scale", 1.0) or 1.0
        era = cyc.get("era", "?")
    except Exception:
        scale, era = 1.0, "?"
    t_trim = max(2, round(TRIM_THRESHOLD * scale))
    t_scale = max(3, round(SCALE_OUT_THRESHOLD * scale))
    t_exit = max(4, round(EXIT_THRESHOLD * scale))

    # Inputs
    prox = _btc_vs_365d_high()
    nt = get_cached("btc_native_top_scorecard") or {}
    n_met = nt.get("n_met") or 0
    n_total = nt.get("n_total") or 16
    ath_stagnation = any(
        c.get("met") and "stagnation" in (c.get("label") or "").lower()
        for c in nt.get("criteria", [])
    )
    olson_bear, olson_detail = _olson_bearish()

    pct_below = prox.get("pct_below_high")
    near_high = (pct_below is not None and pct_below > -ARM_PROXIMITY_PCT)

    # State machine — most severe first, gated by arming proximity
    if not near_high:
        tier = "DORMANT"
        color = "#888"
        action = (f"BTC {pct_below if pct_below is not None else '?'}% below its 365d high — "
                  f"scale-out sleeps through the bear/recovery. Arms within "
                  f"{ARM_PROXIMITY_PCT}% of the high (next bull, ~2029 window).")
    elif n_met >= t_exit or (n_met >= t_scale and olson_bear and ath_stagnation):
        tier = "EXIT_75"
        color = "#ef4444"
        action = "SELL 75% of BTC holdings. Cycle-top evidence overwhelming."
    elif n_met >= t_scale and olson_bear:
        tier = "SCALE_OUT_50"
        color = "#ef4444"
        action = "SELL 50% of BTC holdings. Scorecard + Olson technicals both confirm."
    elif n_met >= t_trim or ath_stagnation:
        tier = "TRIM_25"
        color = "#f0b90b"
        action = ("SELL 25% of BTC holdings. " +
                  ("ATH stagnation fired (the cycle-5-proven signal). "
                    if ath_stagnation else "") +
                  "First tranche off the table.")
    else:
        tier = "ARMED"
        color = "#22c55e"
        action = (f"Near the highs ({pct_below:+.1f}% vs 365d high) with only "
                  f"{n_met}/{n_total} top criteria firing. Ride. Watching for "
                  f"{t_trim}+ (trim), {t_scale}+ & Olson bear (halve), {t_exit}+ (exit).")

    # NZ$ context for the action tiers
    pct_map = {"TRIM_25": 0.25, "SCALE_OUT_50": 0.50, "EXIT_75": 0.75}
    sell_pct = pct_map.get(tier, 0)
    btc_now_nzd = int(TOTAL_STAKE_NZD * CURRENT_BTC_PCT)
    btc_post_nzd = int(TOTAL_STAKE_NZD * POST_ROTATION_BTC_PCT)

    return {
        "ts":               datetime.now(timezone.utc).isoformat(),
        "tier":             tier,
        "tier_rank":        TIER_RANK.get(tier, 0),
        "color":            color,
        "action":           action,
        "sell_pct_of_btc":  int(sell_pct * 100),
        "sell_nzd_if_pre_rotation":  int(btc_now_nzd * sell_pct),
        "sell_nzd_if_post_rotation": int(btc_post_nzd * sell_pct),
        # live inputs
        "btc_price":        prox.get("price"),
        "high_365d":        prox.get("high_365d"),
        "pct_below_high":   pct_below,
        "arm_proximity":    ARM_PROXIMITY_PCT,
        "top_n_met":        n_met,
        "top_n_total":      n_total,
        "ath_stagnation":   ath_stagnation,
        "olson_bearish":    olson_bear,
        "olson_detail":     olson_detail,
        "cycle_era":        era,
        "cycle_scale":      scale,
        "thresholds":       {"trim": t_trim, "scale_out": t_scale, "exit": t_exit},
    }


def main():
    r = evaluate_scale_out_trigger()
    print(f"TIER: {r['tier']}")
    print(f"  {r['action'].encode('ascii', 'replace').decode()}")
    print(f"  BTC ${r.get('btc_price') or 0:,.0f} vs 365d high ${r.get('high_365d') or 0:,.0f} "
          f"({r.get('pct_below_high')}%)")
    print(f"  Top scorecard {r['top_n_met']}/{r['top_n_total']} | "
          f"ATH stagnation: {r['ath_stagnation']} | Olson bearish: {r['olson_bearish']}")
    print(f"  Thresholds (era {r['cycle_era']}, x{r['cycle_scale']}): {r['thresholds']}")


if __name__ == "__main__":
    main()
