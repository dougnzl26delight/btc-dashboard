"""Fear & Greed Index — alternative.me public API.

Daily sentiment gauge composited from:
    - Volatility (25%)
    - Market momentum/volume (25%)
    - Social media (15%)
    - Surveys (15%)
    - BTC dominance (10%)
    - Trends (10%)

Scale 0-100:
    0-25   = EXTREME FEAR  — contrarian BUY signal (3-of-3 extremes preceded all major bottoms)
    25-50  = FEAR          — accumulation zone
    50-75  = GREED         — caution; trim some longs
    75-100 = EXTREME GREED — contrarian SELL signal

Composite usage in rig:
    - F&G < 25 + cycle_score < 20 = MAX BAH allocation
    - F&G > 75 + cycle_score > 60 = trim aggressively
    - F&G whipsaws ±20 in a week = regime change
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / ".fear_greed_cache.json"
CACHE_TTL = 4 * 3600  # 4 hours


def _read_cache() -> Optional[dict]:
    if not CACHE_FILE.exists():
        return None
    try:
        d = json.loads(CACHE_FILE.read_text())
        if time.time() - d.get("fetched_at", 0) < CACHE_TTL:
            return d.get("data")
    except Exception:
        pass
    return None


def _write_cache(data: dict) -> None:
    CACHE_FILE.write_text(json.dumps({
        "fetched_at": time.time(),
        "data": data,
    }))


def fetch_history(days: int = 30) -> Optional[list[dict]]:
    """Fetch F&G index for last N days from alternative.me."""
    cached = _read_cache()
    if cached and len(cached) >= days:
        return cached[:days]

    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            payload = json.loads(r.read())
    except Exception:
        return None

    items = payload.get("data", [])
    if not items:
        return None

    parsed = []
    for d in items:
        try:
            parsed.append({
                "ts_unix": int(d["timestamp"]),
                "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).date().isoformat(),
                "value": int(d["value"]),
                "classification": d.get("value_classification", ""),
                "next_update_seconds": int(d.get("time_until_update", 0)) if d.get("time_until_update") else 0,
            })
        except Exception:
            continue
    _write_cache(parsed)
    return parsed


def latest() -> dict:
    """Current F&G reading with composite analysis."""
    history = fetch_history(30)
    if not history:
        return {"value": None, "classification": "unavailable", "source": "fetch_failed"}
    current = history[0]
    value = current["value"]

    # 2026-06-01 RECALIBRATED for muted institutional cycles:
    # Cycle-5 peak (2025-10-06) saw F&G top at 71 (GREED, not EXTREME_GREED).
    # The old >75 EXTREME threshold never fired at the cycle high.
    # Each cycle's peak F&G is decaying:  cycle 3 ~90, cycle 4 ~84, cycle 5 ~71.
    # Projected cycle 6 peak F&G: ~60-65. Thresholds adjusted accordingly.
    if value <= 25:
        regime = "EXTREME_FEAR"
        action = "Contrarian buy zone. Historical bottoms cluster here. Load BAH BTC."
    elif value <= 45:
        regime = "FEAR"
        action = "Accumulation zone. Normal entry size."
    elif value <= 55:
        regime = "NEUTRAL"
        action = "No regime signal."
    elif value <= 65:
        regime = "GREED"
        action = "Caution. Reduce new long entries."
    else:
        regime = "EXTREME_GREED"
        action = "Contrarian sell zone. Trim longs aggressively."

    # 7-day trend
    if len(history) >= 7:
        seven_day_change = history[0]["value"] - history[6]["value"]
    else:
        seven_day_change = 0

    return {
        "value": value,
        "classification": current["classification"],
        "regime": regime,
        "action": action,
        "7d_change": seven_day_change,
        "history_7d": [h["value"] for h in history[:7]],
        "source": "alternative.me",
    }


def cycle_composite_score() -> dict:
    """Composite F&G + cycle_position score for a unified signal."""
    fg = latest()
    fg_val = fg.get("value")
    cycle_data = None
    try:
        from core.onchain import cycle_position
        cycle_data = cycle_position()
    except Exception:
        pass

    if fg_val is None or cycle_data is None:
        return {"composite": None, "reason": "missing_inputs"}

    cycle_score = cycle_data.get("score", 50)
    # Average (0-100 scale, both directions: deep bear = buy, euphoria = sell)
    # F&G: 0=fear/buy, 100=greed/sell  — already aligned with cycle 0=bear/buy
    composite = (fg_val + cycle_score) / 2

    if composite < 25:
        composite_action = "MAX CONVICTION BUY — fear + cycle bottom convergence"
    elif composite < 50:
        composite_action = "Accumulation — buy dips"
    elif composite < 60:
        composite_action = "Neutral — no edge signal"
    elif composite < 75:
        composite_action = "Reduce risk — late cycle"
    else:
        composite_action = "MAX CONVICTION SELL — euphoria + late cycle convergence"

    return {
        "composite_score": composite,
        "fg_value": fg_val,
        "cycle_score": cycle_score,
        "composite_action": composite_action,
    }


def main():
    print("=" * 70)
    print("FEAR & GREED INDEX (alternative.me)")
    print("=" * 70)
    fg = latest()
    if fg.get("value") is None:
        print("Unavailable")
        return
    print()
    print(f"Current value:     {fg['value']:>3d} / 100")
    print(f"Classification:    {fg['classification']}")
    print(f"Regime:            {fg['regime']}")
    print(f"7-day change:      {fg['7d_change']:+d}")
    print(f"Last 7 days:       {fg['history_7d']}")
    print()
    print(f"Action: {fg['action']}")
    print()
    print("=" * 70)
    print("F&G x Cycle Composite")
    print("=" * 70)
    comp = cycle_composite_score()
    if comp.get("composite_score") is not None:
        print(f"  F&G: {comp['fg_value']}  +  Cycle: {comp['cycle_score']:.0f}  ->  Composite: {comp['composite_score']:.0f}")
        print(f"  Action: {comp['composite_action']}")


if __name__ == "__main__":
    main()
