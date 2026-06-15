"""BTC dominance regime monitor.

BTC dominance (BTC market cap / total crypto market cap) is the cleanest
regime indicator for cross-sectional crypto strategies:

    BTC.D rising  -> capital fleeing alts to BTC ("flight to quality")
                     XSMOM long-alt / short-alt edges DIE
                     BAH BTC over-performs alts
    BTC.D falling -> "altseason" — alts outperforming BTC
                     XSMOM long-alt edges work
                     Mean reversion on alts works
    BTC.D ranging -> regime-neutral; standard sleeve operation

Composite signal logic:
    BTC.D > 60% AND rising  -> reduce alt exposure; favor BAH BTC
    BTC.D < 50% AND falling -> activate altcoin sleeves
    BTC.D 50-60% sideways   -> neutral; normal operation

Free data via CoinGecko global endpoint (no auth, rate-limited 10-30/min).
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
CACHE_FILE = REPO_ROOT / ".btc_dominance_cache.json"
CACHE_TTL = 3600  # 1 hour


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
    CACHE_FILE.write_text(json.dumps({"fetched_at": time.time(), "data": data}))


def fetch_dominance() -> Optional[dict]:
    """Fetch current BTC dominance from CoinGecko."""
    cached = _read_cache()
    if cached:
        return cached
    try:
        url = "https://api.coingecko.com/api/v3/global"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read())
    except Exception:
        return None
    g = payload.get("data", {})
    btc_dominance = g.get("market_cap_percentage", {}).get("btc")
    eth_dominance = g.get("market_cap_percentage", {}).get("eth")
    if btc_dominance is None:
        return None
    data = {
        "btc_dominance_pct": float(btc_dominance),
        "eth_dominance_pct": float(eth_dominance) if eth_dominance else None,
        "total_mcap_usd": float(g.get("total_market_cap", {}).get("usd", 0)),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(data)
    return data


def regime_classification(btc_dominance: float) -> dict:
    """Classify regime + trading implications based on BTC.D level."""
    if btc_dominance > 65:
        regime = "BTC_HEGEMONY"
        action = "Capital concentrated in BTC. Alts severely underperforming. Favor BAH BTC; reduce alt sleeves."
        favor_btc = True
        favor_alts = False
        alt_scale = 0.0
        btc_scale = 1.2
    elif btc_dominance > 55:
        regime = "BTC_DOMINANT"
        action = "BTC leading. Alts weak. Normal BAH; reduce alt mean-reversion + momentum sleeves."
        favor_btc = True
        favor_alts = False
        alt_scale = 0.5
        btc_scale = 1.1
    elif btc_dominance > 45:
        regime = "BALANCED"
        action = "No dominance regime. Standard sleeve operation."
        favor_btc = None
        favor_alts = None
        alt_scale = 1.0
        btc_scale = 1.0
    elif btc_dominance > 40:
        regime = "ALT_RECOVERY"
        action = "Alts starting to outperform. Increase alt sleeves; XSMOM long-alt edges activate."
        favor_btc = False
        favor_alts = True
        alt_scale = 1.1
        btc_scale = 0.9
    else:
        regime = "ALTSEASON"
        action = "Full altseason. Maximum alt exposure; minimum BTC concentration."
        favor_btc = False
        favor_alts = True
        alt_scale = 1.2
        btc_scale = 0.8

    return {
        "btc_dominance_pct": btc_dominance,
        "regime": regime,
        "action": action,
        "favor_btc": favor_btc,
        "favor_alts": favor_alts,
        "alt_scale": alt_scale,
        "btc_scale": btc_scale,
    }


def alt_regime_scale() -> float:
    """Public API for alt-sleeve gating.

    Returns multiplier in [0.0, 1.2] for sleeves that primarily trade ALTS
    (xsmom long-alt, oversold_bounce alt basket, intraday_momentum alt-leaning).

    BTC.D regime    -> alt_scale
    HEGEMONY  >65%  -> 0.0   (zero alt exposure — capital flees to BTC)
    DOMINANT  55-65 -> 0.5   (half size — alts underperform)
    BALANCED  45-55 -> 1.0   (normal)
    RECOVERY  40-45 -> 1.1   (lean in on alts)
    ALTSEASON <40%  -> 1.2   (overweight alts)

    Returns 1.0 (no effect) on data fetch failure — fail open, not closed.
    """
    try:
        d = fetch_dominance()
        if d is None:
            return 1.0
        c = regime_classification(d["btc_dominance_pct"])
        return float(c.get("alt_scale", 1.0))
    except Exception:
        return 1.0


def btc_regime_scale() -> float:
    """Counterpart to alt_regime_scale — for BTC-focused sleeves (BAH BTC, pro_trend BTC).

    HEGEMONY  -> 1.2 (lean in on BTC)
    BALANCED  -> 1.0
    ALTSEASON -> 0.8 (lighten BTC; capital rotating out)
    """
    try:
        d = fetch_dominance()
        if d is None:
            return 1.0
        c = regime_classification(d["btc_dominance_pct"])
        return float(c.get("btc_scale", 1.0))
    except Exception:
        return 1.0


def status() -> dict:
    """Current dominance regime."""
    d = fetch_dominance()
    if d is None:
        return {"error": "fetch_failed"}
    classification = regime_classification(d["btc_dominance_pct"])
    return {
        **d,
        **classification,
    }


def main():
    print("=" * 70)
    print("BTC DOMINANCE REGIME (CoinGecko)")
    print("=" * 70)
    s = status()
    if s.get("error"):
        print(f"\n{s['error']}")
        return
    print()
    print(f"BTC dominance:    {s['btc_dominance_pct']:.2f}%")
    if s.get("eth_dominance_pct"):
        print(f"ETH dominance:    {s['eth_dominance_pct']:.2f}%")
    print(f"Total mcap:       ${s.get('total_mcap_usd', 0)/1e9:.0f}B")
    print()
    print(f"Regime:           {s['regime']}")
    print(f"Favor BTC:        {s['favor_btc']}")
    print(f"Favor alts:       {s['favor_alts']}")
    print()
    print(f"Action: {s['action']}")


if __name__ == "__main__":
    main()
