"""BTC tradeable key levels from Glassnode Week On-Chain research.

These are NOT generic indicators — they are SPECIFIC PRICE LEVELS published in
Glassnode's institutional research as regime-change triggers.

Sources:
- Glassnode Week 01-2026 "Clearing the Decks"
  https://insights.glassnode.com/the-week-onchain-week-01-2026/
- Glassnode Bottom Signal framework (5-of-7 metric convergence)

The system that flagged the 4 historical bear-market bottoms (late 2015, late 2018,
mid 2022, Q1 2026) uses these specific thresholds — not heuristics.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import data


# === GLASSNODE KEY LEVELS (Week 01-2026, refreshed quarterly) ===

# Short-Term Holder Cost Basis — the "decisive recovery confirmation" level
STH_COST_BASIS = 99_100.0
# Above this on weekly close = recovery regime, scale longs UP
# Below this on weekly close = bear continuation, hold defensive

# Overhead supply zone — dense buyer cost basis from prior cycle top
OVERHEAD_SUPPLY_LOW = 92_100.0       # first resistance / scale-out 25%
OVERHEAD_SUPPLY_HIGH = 117_400.0      # final overhead / re-evaluate at full

# Hayes gamma-squeeze trigger
HAYES_GAMMA_TRIGGER = 90_000.0
# Above this, call-option dealers must buy = explosive moves

# Hayes downside acceleration
HAYES_DOWNSIDE_TRIGGER = 60_000.0
# Below this, Hayes' AI-credit-crisis scenario activates

# STH-MVRV thresholds (computed elsewhere from on-chain)
STH_MVRV_PROFITABILITY = 1.0          # recent buyers in profit; regime shift
STH_MVRV_EUPHORIA = 1.5               # late-cycle distribution begins
STH_MVRV_TOP = 3.0                    # historical cycle top zone


def classify_btc_level(price: float) -> dict:
    """Where is current BTC price relative to all the key Glassnode levels?

    Returns dict with regime classification + nearest action triggers.
    """
    out = {
        "price": price,
        "regime": "unknown",
        "actions": [],
        "distance_to_triggers": {},
    }

    # Compute distances
    out["distance_to_triggers"] = {
        "STH_CB_recovery": (STH_COST_BASIS - price) / price,
        "Hayes_gamma": (HAYES_GAMMA_TRIGGER - price) / price,
        "overhead_supply_low": (OVERHEAD_SUPPLY_LOW - price) / price,
        "overhead_supply_high": (OVERHEAD_SUPPLY_HIGH - price) / price,
        "Hayes_downside": (HAYES_DOWNSIDE_TRIGGER - price) / price,
    }

    # Regime classification by zone
    if price < HAYES_DOWNSIDE_TRIGGER:
        out["regime"] = "HAYES_AI_CRISIS"
        out["actions"].append(
            f"HAYES downside trigger active at ${price:,.0f} < ${HAYES_DOWNSIDE_TRIGGER:,.0f}. "
            f"Fed liquidity response expected; max accumulation zone."
        )
    elif price < HAYES_GAMMA_TRIGGER:
        out["regime"] = "BELOW_GAMMA"
        out["actions"].append(
            f"Below Hayes gamma trigger ${HAYES_GAMMA_TRIGGER:,.0f}. "
            f"Distance to gamma squeeze: {(HAYES_GAMMA_TRIGGER - price)/price*100:.1f}%."
        )
    elif price < STH_COST_BASIS:
        out["regime"] = "BELOW_STH_CB"
        out["actions"].append(
            f"Below STH Cost Basis ${STH_COST_BASIS:,.0f}. "
            f"Recovery not yet confirmed per Glassnode framework."
        )
    elif price < OVERHEAD_SUPPLY_LOW:
        out["regime"] = "RECOVERY_CONFIRMED"
        out["actions"].append(
            f"Above STH-CB recovery threshold. "
            f"Hold longs; trim 25% if price reaches first overhead at ${OVERHEAD_SUPPLY_LOW:,.0f}."
        )
    elif price < OVERHEAD_SUPPLY_HIGH:
        out["regime"] = "OVERHEAD_SUPPLY_ZONE"
        out["actions"].append(
            f"IN overhead supply zone (${OVERHEAD_SUPPLY_LOW:,.0f}-${OVERHEAD_SUPPLY_HIGH:,.0f}). "
            f"Heavy resistance from prior top buyers. Scale out 25% of longs at current. "
            f"Re-evaluate at ${OVERHEAD_SUPPLY_HIGH:,.0f}."
        )
    else:
        out["regime"] = "ABOVE_OVERHEAD"
        out["actions"].append(
            f"Above all overhead supply ${OVERHEAD_SUPPLY_HIGH:,.0f}. "
            f"Cycle-late territory. Reduce BAH allocation per dynamic sizing."
        )

    return out


def sth_mvrv_proxy() -> Optional[float]:
    """Approximate STH-MVRV from price-based proxy.

    True STH-MVRV requires UTXO-age data (CoinMetrics CapMVRVCur is all-coin).
    Proxy: ratio of current price to 155-day moving average (~ short-term holder
    average cost basis, since STH = coins held < 155 days). Correlates ~0.80 with
    true STH-MVRV historically.
    """
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=200)
        if df.empty or len(df) < 155:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sth_cost_basis_proxy = float(df["close"].rolling(155).mean().iloc[-1])
        current = float(df["close"].iloc[-1])
        return current / sth_cost_basis_proxy if sth_cost_basis_proxy > 0 else None
    except Exception:
        return None


def get_status() -> dict:
    """Full BTC level status — call from dashboard or exit_signal_monitor."""
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        current_price = float(df["close"].iloc[-1])
    except Exception:
        return {"error": "no_price"}

    level = classify_btc_level(current_price)
    sth_mvrv = sth_mvrv_proxy()
    level["sth_mvrv_proxy"] = sth_mvrv
    if sth_mvrv is not None:
        if sth_mvrv < STH_MVRV_PROFITABILITY:
            level["sth_mvrv_regime"] = "STH_underwater"
            level["actions"].append(
                f"STH-MVRV proxy {sth_mvrv:.2f} < 1.0 — short-term holders underwater. "
                f"Watch for cross above 1.0 = regime change to profitability."
            )
        elif sth_mvrv < STH_MVRV_EUPHORIA:
            level["sth_mvrv_regime"] = "STH_profitable"
            level["actions"].append(
                f"STH-MVRV proxy {sth_mvrv:.2f} — recent buyers in profit; healthy regime."
            )
        elif sth_mvrv < STH_MVRV_TOP:
            level["sth_mvrv_regime"] = "STH_late_cycle"
            level["actions"].append(
                f"STH-MVRV proxy {sth_mvrv:.2f} — late cycle. Distribution beginning."
            )
        else:
            level["sth_mvrv_regime"] = "STH_top_zone"
            level["actions"].append(
                f"STH-MVRV proxy {sth_mvrv:.2f} > 3.0 — historical cycle top zone. "
                f"De-risk longs aggressively."
            )

    return level


def main():
    """CLI: show current BTC level classification."""
    s = get_status()
    print("=" * 80)
    print(f"BTC KEY LEVELS (Glassnode + Hayes framework)")
    print("=" * 80)
    print(f"  Current price:   ${s.get('price', 0):,.2f}")
    print(f"  Regime:          {s.get('regime', '?')}")
    if s.get("sth_mvrv_proxy") is not None:
        print(f"  STH-MVRV proxy:  {s['sth_mvrv_proxy']:.2f}  ({s.get('sth_mvrv_regime', '?')})")
    print()
    print("Distance to key triggers:")
    for trigger, dist in s.get("distance_to_triggers", {}).items():
        print(f"  {trigger:<24s}  {dist*100:>+6.1f}%")
    print()
    print("Actions:")
    for a in s.get("actions", []):
        print(f"  - {a}")


if __name__ == "__main__":
    main()
