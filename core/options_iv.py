"""Deribit BTC implied volatility — DVOL index + ATM IV + skew.

Deribit BTC DVOL is the crypto equivalent of VIX: market-priced 30-day
forward expected vol. Reading it complements realized vol:

    Realized vol  : backward-looking, what already happened
    Implied vol   : forward-looking, what the market expects

When IV >> realized = market is hedging, fear premium
When IV << realized = market is complacent before a move
When skew is steep (puts much pricier than calls) = downside protection demand

Free public Deribit REST API — no auth needed for market data.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

DERIBIT_API = "https://www.deribit.com/api/v2"
CACHE_DIR = REPO_ROOT / ".deribit_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 600  # 10 min


def _cached_get(url: str) -> Optional[dict]:
    cache_key = url.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
    cache_path = CACHE_DIR / f"{cache_key[:200]}.json"
    if cache_path.exists():
        try:
            d = json.loads(cache_path.read_text())
            if time.time() - d["t"] < CACHE_TTL:
                return d["data"]
        except Exception:
            pass
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        cache_path.write_text(json.dumps({"t": time.time(), "data": data}))
        return data
    except Exception:
        return None


def get_dvol(currency: str = "BTC") -> dict:
    """DVOL index — Deribit's implied volatility index. 30d forward."""
    url = f"{DERIBIT_API}/public/get_volatility_index_data?currency={currency}&start_timestamp={int((time.time() - 30*86400) * 1000)}&end_timestamp={int(time.time() * 1000)}&resolution=86400"
    payload = _cached_get(url)
    if payload is None or "result" not in payload:
        return {"dvol": None, "source": "unavailable"}
    rows = payload["result"].get("data", [])
    if not rows:
        return {"dvol": None, "source": "unavailable"}
    # rows: [[ts, open, high, low, close]]
    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else last
    return {
        "dvol": last[4],                   # close
        "dvol_open": last[1],
        "dvol_high": last[2],
        "dvol_low": last[3],
        "change_1d": (last[4] - prev[4]) / prev[4] if prev[4] > 0 else 0,
        "history_30d_mean": sum(r[4] for r in rows) / len(rows),
        "history_30d_max": max(r[4] for r in rows),
        "history_30d_min": min(r[4] for r in rows),
        "source": "deribit",
    }


def get_atm_iv(currency: str = "BTC") -> dict:
    """ATM (at-the-money) IV for nearest weekly expiry."""
    # Get index price
    idx_url = f"{DERIBIT_API}/public/get_index?currency={currency}"
    idx_data = _cached_get(idx_url)
    if idx_data is None:
        return {"atm_iv": None, "source": "unavailable"}
    spot = idx_data["result"][f"{currency}"]

    # Get all near-the-money options
    instruments_url = f"{DERIBIT_API}/public/get_instruments?currency={currency}&kind=option&expired=false"
    inst_data = _cached_get(instruments_url)
    if inst_data is None:
        return {"atm_iv": None, "source": "unavailable"}

    # Find soonest expiry
    instruments = inst_data["result"]
    if not instruments:
        return {"atm_iv": None, "source": "unavailable"}
    soonest_exp = min(i["expiration_timestamp"] for i in instruments)
    near = [i for i in instruments if i["expiration_timestamp"] == soonest_exp]

    # Find ATM call and put (closest strike to spot)
    near_call = min((i for i in near if i["option_type"] == "call"),
                    key=lambda x: abs(x["strike"] - spot), default=None)
    near_put = min((i for i in near if i["option_type"] == "put"),
                    key=lambda x: abs(x["strike"] - spot), default=None)
    if near_call is None or near_put is None:
        return {"atm_iv": None, "source": "unavailable"}

    # Fetch tickers for IV
    def _iv(instrument_name):
        url = f"{DERIBIT_API}/public/ticker?instrument_name={instrument_name}"
        d = _cached_get(url)
        if d is None or "result" not in d:
            return None
        return d["result"].get("mark_iv")

    call_iv = _iv(near_call["instrument_name"])
    put_iv = _iv(near_put["instrument_name"])
    atm_iv = (call_iv + put_iv) / 2 if (call_iv and put_iv) else (call_iv or put_iv)

    return {
        "atm_iv_pct": atm_iv,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "skew_call_minus_put": (call_iv - put_iv) if (call_iv and put_iv) else None,
        "spot": spot,
        "strike": near_call["strike"],
        "expiry": near_call["expiration_timestamp"],
        "source": "deribit",
    }


def get_iv_regime() -> dict:
    """Compose DVOL + ATM IV into a regime signal.

    DVOL > 80   : high-vol regime (fear); typically follows large moves
    DVOL 50-80  : normal
    DVOL 30-50  : compressed (squeezes coming?)
    DVOL < 30   : extreme low vol — historically precedes large moves
    """
    dvol_d = get_dvol()
    iv_d = get_atm_iv()
    dvol = dvol_d.get("dvol")
    atm_iv = iv_d.get("atm_iv_pct")

    if dvol is None:
        return {"regime": "unknown", "reason": "no_dvol"}

    if dvol > 80:
        regime = "high_vol_fear"
    elif dvol > 50:
        regime = "normal"
    elif dvol > 30:
        regime = "compressed"
    else:
        regime = "extreme_low"

    skew = iv_d.get("skew_call_minus_put")
    skew_label = None
    if skew is not None:
        if skew < -5:
            skew_label = "puts_expensive (bearish_hedging)"
        elif skew > 5:
            skew_label = "calls_expensive (bullish_speculation)"
        else:
            skew_label = "neutral"

    return {
        "regime": regime,
        "dvol": dvol,
        "atm_iv": atm_iv,
        "dvol_vs_30d_mean": (dvol - dvol_d["history_30d_mean"]) / dvol_d["history_30d_mean"] if dvol_d.get("history_30d_mean") else 0,
        "skew": skew,
        "skew_label": skew_label,
    }


def main():
    print("=" * 70)
    print("DERIBIT BTC IMPLIED VOLATILITY")
    print("=" * 70)
    print()
    dvol = get_dvol()
    if dvol.get("dvol") is not None:
        print(f"DVOL (BTC 30d forward IV): {dvol['dvol']:.2f}")
        print(f"  1d change: {dvol['change_1d']*100:+.2f}%")
        print(f"  30d range: {dvol['history_30d_min']:.1f} - {dvol['history_30d_max']:.1f}")
        print(f"  30d mean:  {dvol['history_30d_mean']:.2f}")
    else:
        print("DVOL: unavailable")
    print()
    atm = get_atm_iv()
    if atm.get("atm_iv_pct") is not None:
        print(f"ATM IV (nearest weekly): {atm['atm_iv_pct']:.2f}%  strike ${atm['strike']:,.0f}")
        print(f"  Call IV: {atm['call_iv']:.2f}%   Put IV: {atm['put_iv']:.2f}%")
        if atm.get("skew_call_minus_put") is not None:
            print(f"  Skew (call - put): {atm['skew_call_minus_put']:+.2f}pp")
    print()
    regime = get_iv_regime()
    print(f"REGIME: {regime.get('regime', 'unknown')}  [{regime.get('skew_label', '?')}]")


if __name__ == "__main__":
    main()
