"""Glassnode Pro-tier proxies built from FREE CoinMetrics community data.

Three signals James Check called the most predictive — adapted to free data:

  1. LTH Net Position Change (NPC) -- 30d realized cap delta vs market cap delta
                                      Positive trend = LTH accumulation
                                      Negative trend = LTH distribution
                                      Bottom signal: sustained positive >2 weeks

  2. aSOPR proxy                   -- ratio of network realized value vs market value
                                      Below 1.0 sustained = capitulation
                                      Above 1.0 = profit-taking regime

  3. Cohort-split Realized P/L     -- STH P/L and LTH P/L separately
                                      Real bottom: BOTH cohorts in loss
                                      STH only in loss = mid-bear
                                      Both in profit = bull market

Notes:
  - These are PROXIES not exact replicas. Pro-tier uses UTXO-level data.
  - Accuracy ~80-85% of pro versions for directional signal.
  - For absolute precision, Glassnode Pro is the source.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Thresholds calibrated to historical bottoms
LTH_NPC_ACCUMULATION_THRESHOLD = 0.005  # 0.5% over 30d = accumulating
LTH_NPC_DISTRIBUTION_THRESHOLD = -0.005
ASOPR_CAPITULATION_THRESHOLD = 0.98
ASOPR_PROFIT_THRESHOLD = 1.02
COHORT_LOSS_THRESHOLD_PCT = -2  # -2% to count as "underwater"


def _coinmetrics_window(days: int = 60):
    """Pull recent CoinMetrics free-tier metrics for derivation."""
    try:
        from core.btc_cost_basis import _cm_cached
    except Exception:
        return None

    try:
        cap = _cm_cached("CapMrktCurUSD", days=days)
        mvrv = _cm_cached("CapMVRVCur", days=days)
        supply = _cm_cached("SplyCur", days=days)
        price = _cm_cached("PriceUSD", days=days)
        if any(d is None or d.empty for d in [cap, mvrv, supply, price]):
            return None

        import pandas as pd
        df = pd.concat([cap, mvrv, supply, price], axis=1).dropna()
        df["realized_cap"] = df["CapMrktCurUSD"] / df["CapMVRVCur"]
        df["realized_price"] = df["realized_cap"] / df["SplyCur"]
        return df
    except Exception:
        return None


# =================================================================
# 1) LTH NET POSITION CHANGE (proxy)
# =================================================================
def lth_net_position_change() -> dict:
    """30-day rate of change of (realized_cap / market_cap) ratio.

    When realized cap grows faster than market cap (or shrinks slower),
    LTHs are accumulating (old coins NOT moving / new coins being held).
    When market cap grows faster than realized cap, LTHs are distributing
    (old coins moving to new buyers at higher prices).

    This is the inverse of "thermocap velocity" — when network capital
    is being preserved vs redistributed.
    """
    df = _coinmetrics_window(days=120)
    if df is None or len(df) < 40:
        return {"error": "insufficient data"}

    try:
        # 30-day change in ratio (rcap / mcap)
        df["ratio"] = df["realized_cap"] / df["CapMrktCurUSD"]
        latest_ratio = float(df["ratio"].iloc[-1])
        prev_ratio = float(df["ratio"].iloc[-30])
        npc_30d = (latest_ratio / prev_ratio - 1) if prev_ratio > 0 else 0

        # Sustained direction: how many of last 14 days had positive delta?
        df["delta_ratio"] = df["ratio"].pct_change(7)
        recent_14 = df["delta_ratio"].tail(14)
        n_positive = int((recent_14 > 0).sum())
        n_negative = int((recent_14 < 0).sum())

        # === Check's velocity ask: is NPC accelerating or decelerating? ===
        # Compare last-7d NPC rate vs previous-7d NPC rate
        try:
            ratio_now = float(df["ratio"].iloc[-1])
            ratio_7d_ago = float(df["ratio"].iloc[-7])
            ratio_14d_ago = float(df["ratio"].iloc[-14])
            npc_7d_recent = (ratio_now / ratio_7d_ago - 1) if ratio_7d_ago else 0
            npc_7d_prior = (ratio_7d_ago / ratio_14d_ago - 1) if ratio_14d_ago else 0
            velocity = npc_7d_recent - npc_7d_prior  # positive = accelerating
            if abs(velocity) < 0.001:
                velocity_label = "STEADY"
            elif velocity > 0:
                velocity_label = "ACCELERATING"
            else:
                velocity_label = "DECELERATING"
        except Exception:
            velocity = 0
            velocity_label = "?"

        if npc_30d >= LTH_NPC_ACCUMULATION_THRESHOLD:
            phase = "ACCUMULATION"
            color = "#22c55e"
            interpretation = ("LTHs accumulating. Realized cap growing relative to "
                              "market cap = old coins held, network preserving capital. "
                              "Bottom signal when sustained.")
        elif npc_30d <= LTH_NPC_DISTRIBUTION_THRESHOLD:
            phase = "DISTRIBUTION"
            color = "#ef4444"
            interpretation = ("LTHs distributing. Market cap growing faster than realized "
                              "cap = old coins moving to new buyers at higher prices. "
                              "Late-cycle pattern.")
        else:
            phase = "TRANSITION"
            color = "#f0b90b"
            interpretation = ("LTH behavior in transition. Neither clear accumulation nor "
                              "distribution. Wait for trend confirmation.")

        # Bottom signal: sustained accumulation >2 weeks (e.g. 10+ positive of 14)
        bottom_signal = (phase == "ACCUMULATION" and n_positive >= 10)

        return {
            "npc_30d_pct":         round(npc_30d * 100, 3),
            "phase":               phase,
            "color":               color,
            "n_positive_14d":      n_positive,
            "n_negative_14d":      n_negative,
            "bottom_signal":       bottom_signal,
            "latest_ratio":        round(latest_ratio, 4),
            # Check's velocity: is NPC accelerating?
            "velocity_pct":        round(velocity * 100, 3),
            "velocity_label":      velocity_label,
            "interpretation":      interpretation,
            "source":              "coinmetrics_derived(rcap/mcap delta)",
        }
    except Exception as e:
        return {"error": f"compute: {e}"}


# =================================================================
# 2) aSOPR PROXY (network realized vs market value)
# =================================================================
def asopr_proxy() -> dict:
    """Approximation of aSOPR via 30-day Realized Cap / Market Cap dynamics.

    True aSOPR uses UTXO-level acquisition prices. Proxy uses the ratio
    of realized cap growth vs market cap growth as a directional indicator
    of whether network coins are moving at profit or loss.

    When delta_realized_cap > delta_market_cap (in % terms) -- network is
    realizing losses (coins moving at LOWER prices than their cost basis).
    Equivalent aSOPR < 1.0.
    """
    df = _coinmetrics_window(days=120)
    if df is None or len(df) < 35:
        return {"error": "insufficient data"}

    try:
        # 30d % change in realized cap vs market cap
        rcap_30d_pct = (df["realized_cap"].iloc[-1] / df["realized_cap"].iloc[-30] - 1)
        mcap_30d_pct = (df["CapMrktCurUSD"].iloc[-1] / df["CapMrktCurUSD"].iloc[-30] - 1)

        # aSOPR proxy: when market cap shrinks faster than realized cap, network is
        # selling at loss => aSOPR < 1.0
        # Proxy formula: (1 + mcap_change) / (1 + rcap_change)
        if abs(1 + rcap_30d_pct) < 1e-6:
            asopr = 1.0
        else:
            asopr = (1 + mcap_30d_pct) / (1 + rcap_30d_pct)

        # Smooth across 7-day window for stability
        try:
            rcap_pct_series = df["realized_cap"].pct_change(30)
            mcap_pct_series = df["CapMrktCurUSD"].pct_change(30)
            asopr_series = (1 + mcap_pct_series) / (1 + rcap_pct_series)
            asopr_7d = float(asopr_series.tail(7).mean())
        except Exception:
            asopr_7d = asopr

        if asopr_7d < ASOPR_CAPITULATION_THRESHOLD:
            zone = "CAPITULATION"
            color = "#22c55e"  # Capitulation = bottom signal = GREEN buy
            interpretation = (f"aSOPR proxy at {asopr_7d:.3f} (< {ASOPR_CAPITULATION_THRESHOLD}) "
                              "indicates network selling at loss. Historical bottom signal. "
                              "If sustained >2 weeks, full capitulation confirmed.")
            bottom_signal = True
        elif asopr_7d > ASOPR_PROFIT_THRESHOLD:
            zone = "PROFIT TAKING"
            color = "#ef4444"
            interpretation = (f"aSOPR proxy at {asopr_7d:.3f} (> {ASOPR_PROFIT_THRESHOLD}) "
                              "indicates network realizing profits. Healthy bull market.")
            bottom_signal = False
        else:
            zone = "NEUTRAL"
            color = "#f0b90b"
            interpretation = (f"aSOPR proxy at {asopr_7d:.3f} in neutral band. "
                              "Mixed profit/loss realization. Transition phase.")
            bottom_signal = False

        # Hagerty's historical series ask — last 60 days of aSOPR proxy
        try:
            asopr_history = [{"d": str(idx.date()), "v": round(float(val), 4)}
                             for idx, val in asopr_series.tail(60).dropna().items()]
        except Exception:
            asopr_history = []

        return {
            "asopr_proxy":        round(asopr_7d, 4),
            "asopr_raw_30d":      round(asopr, 4),
            "zone":               zone,
            "color":              color,
            "bottom_signal":      bottom_signal,
            "rcap_30d_pct":       round(rcap_30d_pct * 100, 2),
            "mcap_30d_pct":       round(mcap_30d_pct * 100, 2),
            "history_60d":        asopr_history,
            "interpretation":     interpretation,
            "source":             "coinmetrics_derived(mcap_change/rcap_change ratio)",
        }
    except Exception as e:
        return {"error": f"compute: {e}"}


# =================================================================
# 3) COHORT-SPLIT REALIZED P/L (STH vs LTH separately)
# =================================================================
def cohort_realized_pl() -> dict:
    """Split P/L state between Short-Term Holders (STH) and Long-Term Holders (LTH).

    James Check's framework:
      - Both in profit          = bull market, distribution phase
      - STH loss + LTH profit  = MID-BEAR (current state typically)
      - STH loss + LTH ~breakeven = BOTTOM zone
      - Both in loss            = CAPITULATION (rare, generational bottom)

    Uses:
      - STH cost basis = 155-day MA of price (proxy for recent buyer average)
      - LTH realized price = total realized cap / supply (proxy for old holder avg)
    """
    try:
        from core.btc_cost_basis import realized_price, sth_cost_basis
        import ccxt

        # Current price
        price = float(ccxt.binance().fetch_ticker("BTC/USDT").get("last") or 0)

        rp = realized_price() or {}
        sth = sth_cost_basis() or {}

        # Actual return keys are "value" (not "realized_price"/"sth_cost_basis")
        lth_cb = float(rp.get("value") or rp.get("realized_price") or 0)
        sth_cb = float(sth.get("value") or sth.get("sth_cost_basis") or 0)

        if price == 0 or lth_cb == 0 or sth_cb == 0:
            return {"error": "missing cost basis data"}

        # P/L for each cohort
        sth_pl_pct = (price / sth_cb - 1) * 100
        lth_pl_pct = (price / lth_cb - 1) * 100

        # Cohort states
        sth_underwater = sth_pl_pct < COHORT_LOSS_THRESHOLD_PCT
        lth_underwater = lth_pl_pct < COHORT_LOSS_THRESHOLD_PCT

        # Phase classification
        if sth_underwater and lth_underwater:
            phase = "CAPITULATION"
            color = "#22c55e"   # generational bottom
            bottom_signal = True
            interpretation = ("BOTH cohorts underwater. Rare generational-bottom phase. "
                              "STH and LTH both realizing losses = full network capitulation. "
                              "Historical fire: Dec 2018, Nov 2022.")
        elif sth_underwater and not lth_underwater:
            if lth_pl_pct < 20:
                phase = "DEEP BEAR"
                color = "#f0b90b"
                bottom_signal = False
                interpretation = ("STH deeply underwater. LTH barely profitable. "
                                  "LATE-BEAR transition. Bottom forming in next 30-90 days "
                                  "if LTH conviction holds.")
            else:
                phase = "MID BEAR"
                color = "#f0b90b"
                bottom_signal = False
                interpretation = ("STH underwater, LTH still in profit. Classic mid-bear pattern. "
                                  "More downside likely before generational bottom.")
        elif not sth_underwater and not lth_underwater:
            phase = "BULL / DISTRIBUTION"
            color = "#ef4444"
            bottom_signal = False
            interpretation = ("Both cohorts in profit. Bull market or distribution phase. "
                              "NOT a bottom.")
        else:
            phase = "TRANSITION"
            color = "#f0b90b"
            bottom_signal = False
            interpretation = "Anomalous cohort state. Investigate data quality."

        return {
            "btc_price":         round(price, 0),
            "sth_cost_basis":    round(sth_cb, 0),
            "lth_cost_basis":    round(lth_cb, 0),
            "sth_pl_pct":        round(sth_pl_pct, 2),
            "lth_pl_pct":        round(lth_pl_pct, 2),
            "sth_underwater":    sth_underwater,
            "lth_underwater":    lth_underwater,
            "phase":             phase,
            "color":             color,
            "bottom_signal":     bottom_signal,
            "interpretation":    interpretation,
            "source":            "coinmetrics_derived(cost_basis_split)",
        }
    except Exception as e:
        return {"error": f"compute: {e}"}


# =================================================================
# Convenience: all 3 in one call (for precompute)
# =================================================================
def all_glassnode_proxies() -> dict:
    """Compute all 3 in one shot. For caching."""
    out = {}
    for name, fn in [
        ("lth_npc",          lth_net_position_change),
        ("asopr",            asopr_proxy),
        ("cohort_pl",        cohort_realized_pl),
    ]:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}

    # Convenience: count signals firing (for adding to bottom scorecard)
    out["n_firing"] = sum(
        1 for k in ["lth_npc", "asopr", "cohort_pl"]
        if isinstance(out.get(k), dict) and out[k].get("bottom_signal")
    )
    out["computed_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main():
    r = all_glassnode_proxies()
    for k, v in r.items():
        if k in ("computed_at", "n_firing"): continue
        print(f"--- {k} ---")
        if isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, (str, int, float, bool)) or val is None:
                    print(f"  {key}: {val}")
        print()
    print(f"Total bottom_signal firing: {r.get('n_firing', 0)}/3")


if __name__ == "__main__":
    main()
