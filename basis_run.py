"""Funding-rate basis arbitrage cycle.

Runs alongside the main signal-cycle orchestrator, not entangled with it.
Uses a separate perp broker + basis executor to manage basis-trade legs.

Why separate from run.py?
  - Spot leg of basis trade conflicts with signal cycle's spot positioning
  - Easier reasoning: signal-cycle = directional, basis-cycle = market-neutral
  - Can be scheduled independently (basis arb checks more frequently)

Usage:
  python basis_run.py            # one cycle
  Schedule via setup_crypto_scheduler.ps1 (every 4 hours recommended)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import basis_executor
from core.broker import Broker
from core.perp_broker import PerpBroker
from ops import alerts, watchdog
from strategies.funding_basis_arb import latest_funding_bps_8h, latest_signal, rank_universe_by_funding


# Same universe as main orchestrator
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
]
# 2026-05-31 W10: continuous-capture sizing. Sub-account has $20k allocated;
# spread across top-N funding-paying pairs at $5k each.
NOTIONAL_PER_PAIR = 5_000.0
MAX_CONCURRENT_BASIS = 4            # cap to $20k total (matches sub-account)
CONTINUOUS_CAPTURE = True           # NEW: hold top-N funding pairs even if marginal


def basis_cycle(mode: str = "paper") -> dict:
    """One basis-arb cycle: settle funding, open new positions, close stale ones."""
    # W9: isolated sub-account for basis arb
    spot = Broker(mode=mode, long_only=False, sleeve="basis_arb")
    perp = PerpBroker(mode=mode, sleeve="basis_arb")

    # 1. Settle funding on every open position
    funding_payments = basis_executor.settle_all_funding(perp_broker=perp)
    for pair, payment in funding_payments.items():
        if abs(payment) > 0.01:
            alerts.alert(
                f"Funding settled on {pair} basis: ${payment:+,.4f}",
                level="info",
            )

    # 2. W14.A: only enter pairs with 5-day stable funding ABOVE 2bp/8h threshold.
    # Single-spike entries removed — academic finding shows they net negative.
    actions: list[dict] = []
    ranked = rank_universe_by_funding(UNIVERSE)
    eligible = [r for r in ranked if r.get("qualifies_for_entry")]
    targets = set(r["pair"] for r in eligible[:MAX_CONCURRENT_BASIS])
    print(f"basis_run W14.A: ranked={len(ranked)}, qualified={len(eligible)}, "
          f"opening_target={len(targets)}")

    # Close any open positions not in targets (funding turned bad)
    for pair in UNIVERSE:
        if basis_executor.is_basis_open(pair) and pair not in targets:
            try:
                result = basis_executor.close_basis_position(
                    pair, spot_broker=spot, perp_broker=perp,
                )
                funding_collected = result.get("funding_collected", 0)
                actions.append({"action": "close", "pair": pair,
                                "funding_collected": funding_collected,
                                "reason": "no_longer_top_funding"})
                alerts.alert(
                    f"BASIS CLOSE {pair}: funding ${funding_collected:+,.2f} collected, "
                    f"current funding fell below threshold",
                    level="trade",
                )
            except Exception as e:
                alerts.alert(f"Basis close failed for {pair}: {e}", level="warning")

    # Open new positions for any targets not yet open
    for pair in targets:
        if basis_executor.is_basis_open(pair):
            continue
        funding_bps = latest_funding_bps_8h(pair)
        try:
            result = basis_executor.open_basis_position(
                pair, NOTIONAL_PER_PAIR, spot_broker=spot, perp_broker=perp,
            )
            actions.append({"action": "open", "pair": pair,
                            "funding_bps_8h": funding_bps,
                            "annualized": funding_bps * 3 * 365 / 100})
            alerts.alert(
                f"BASIS OPEN {pair}: ${NOTIONAL_PER_PAIR:,.0f} notional, "
                f"funding={funding_bps:+.2f} bps/8h ({funding_bps*3*365/100:+.1f}% ann)",
                level="trade",
            )
        except Exception as e:
            alerts.alert(f"Basis open failed for {pair}: {e}", level="warning")

    watchdog.beat()

    summary = basis_executor.basis_summary()
    return {
        "n_open_positions": len(summary),
        "n_funding_settled": len(funding_payments),
        "n_actions": len(actions),
        "actions": actions,
        "open_positions": summary,
    }


if __name__ == "__main__":
    import json
    result = basis_cycle()
    print(json.dumps(result, indent=2, default=str))
