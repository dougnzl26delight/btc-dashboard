"""Multi-exchange data source with divergence detection.

Pulls the same pair from Binance + Kraken (and optionally Coinbase) and:
  - Reports per-exchange price
  - Detects divergence > THRESHOLD (default 1%)
  - If divergent, writes a halt-marker for the orchestrator to skip cycle

Why: single-exchange dependence is the #1 retail-rig blow-up risk. Glitches
on one venue (stale tickers, frozen books, hacks) can fire trades on bad
data. Cross-venue check costs nothing and catches it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ccxt

HALT_FILE = REPO_ROOT / ".multi_exchange_halt.json"
DIVERGENCE_THRESHOLD = 0.01   # 1% disagreement = halt

# Single shared instance per exchange (lazy-init)
_EXCHANGES: dict[str, "ccxt.Exchange"] = {}


def _ex(name: str):
    if name not in _EXCHANGES:
        cls = getattr(ccxt, name)
        _EXCHANGES[name] = cls({"enableRateLimit": True, "timeout": 8000})
    return _EXCHANGES[name]


def fetch_prices(pair: str, exchanges: Optional[list[str]] = None) -> dict:
    """Fetch the same pair across multiple exchanges. Returns {exchange: price}."""
    if exchanges is None:
        exchanges = ["binance", "kraken"]
    # Map common pair format to per-exchange format
    pair_map = {
        "binance": pair,                                           # BTC/USDT
        "kraken": pair.replace("USDT", "USD"),                     # BTC/USD on Kraken
        "coinbase": pair.replace("USDT", "USD"),
    }
    out = {}
    for name in exchanges:
        try:
            ex = _ex(name)
            t = ex.fetch_ticker(pair_map.get(name, pair))
            last = float(t.get("last") or t.get("close") or 0)
            if last > 0:
                out[name] = last
        except Exception as e:
            out[name] = None
    return out


def check_divergence(pair: str, threshold: float = DIVERGENCE_THRESHOLD) -> dict:
    """Returns {ok: bool, max_div: float, prices: dict, reason: str}."""
    prices = fetch_prices(pair)
    valid = {k: v for k, v in prices.items() if v and v > 0}
    if len(valid) < 2:
        return {"ok": True, "max_div": 0.0, "prices": prices,
                "reason": f"only {len(valid)} venues responding — can't compare"}
    vmin, vmax = min(valid.values()), max(valid.values())
    div = (vmax - vmin) / vmin if vmin > 0 else 0
    ok = div < threshold
    return {
        "ok": ok,
        "max_div": div,
        "prices": valid,
        "reason": ("aligned" if ok else
                   f"divergence {div*100:.2f}% across {len(valid)} venues exceeds {threshold*100:.1f}%"),
    }


def is_halted() -> bool:
    """True if a prior cycle wrote a halt marker."""
    return HALT_FILE.exists()


def set_halt(reason: str, details: dict) -> None:
    HALT_FILE.write_text(json.dumps({
        "halted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "details": details,
    }, indent=2, default=str))


def clear_halt() -> None:
    if HALT_FILE.exists():
        HALT_FILE.unlink()


def run_cross_check(pairs: list[str], threshold: float = DIVERGENCE_THRESHOLD) -> dict:
    """Run divergence check across multiple pairs. Halt if any diverge.

    Returns summary dict. Callers (orchestrator) should check is_halted()
    before trading.
    """
    results = {}
    halted = False
    for pair in pairs:
        r = check_divergence(pair, threshold)
        results[pair] = r
        if not r["ok"]:
            halted = True

    if halted:
        set_halt("cross-venue divergence", {"pairs": results})
        from ops.alerts import alert
        bad = [p for p, r in results.items() if not r["ok"]]
        alert(f"multi_exchange: HALT — divergence on {bad}", level="critical")
    else:
        # Clear any prior halt if all pairs realign
        if is_halted():
            clear_halt()

    return {"halted": halted, "results": results}


def main():
    """CLI: print current cross-venue price snapshot."""
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    out = run_cross_check(pairs)
    print(f"Halted: {out['halted']}")
    for pair, r in out["results"].items():
        prices_str = ", ".join(f"{k}=${v:,.2f}" if v else f"{k}=N/A"
                                for k, v in r["prices"].items())
        flag = "OK" if r["ok"] else "DIVERGENT"
        print(f"  {pair:<10s} {flag:<10s} div={r['max_div']*100:.2f}%   {prices_str}")


if __name__ == "__main__":
    main()
