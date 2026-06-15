"""Funding-rate basis arbitrage executor.

Coordinates the two legs of the basis trade:
  - Long spot via core/broker.py
  - Short perp via core/perp_broker.py

State tracking:
  Each pair has either an OPEN basis position (both legs active) or no position.
  Position is delta-neutral by construction (equal notional, opposite sides).

Trade lifecycle:
  1. Signal fires (funding > entry_threshold) and not already in basis trade for pair
     → open_basis_position(): buy spot + short perp, equal notional
  2. Each cycle:
     → settle_funding(): credit/debit funding on perp leg
  3. Signal exits (funding < exit_threshold) or stop-loss on basis spread:
     → close_basis_position(): sell spot + cover perp short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import Broker
from core.perp_broker import PerpBroker


REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS_STATE = REPO_ROOT / ".basis_positions.json"


def _load_basis_state() -> dict:
    if BASIS_STATE.exists():
        return json.loads(BASIS_STATE.read_text())
    return {}


def _save_basis_state(state: dict) -> None:
    BASIS_STATE.write_text(json.dumps(state, indent=2))


def is_basis_open(pair: str) -> bool:
    return pair in _load_basis_state()


def open_basis_position(
    pair: str,
    quote_amount: float,
    spot_broker: Broker | None = None,
    perp_broker: PerpBroker | None = None,
) -> dict:
    """Open both legs simultaneously (long spot, short perp) at equal notional."""
    if is_basis_open(pair):
        return {"status": "already_open", "pair": pair}

    spot = spot_broker or Broker(mode="paper", long_only=False)
    perp = perp_broker or PerpBroker(mode="paper")

    spot_fill = spot.place_market_order(pair, "buy", quote_amount)
    perp_fill = perp.open_position(pair, "short", quote_amount)

    state = _load_basis_state()
    state[pair] = {
        "opened_at": str(spot_fill.get("price")),
        "spot_qty": float(spot_fill.get("qty", 0)),
        "spot_entry_price": float(spot_fill.get("price", 0)),
        "perp_contracts": float(perp_fill.get("contracts", 0)),
        "perp_entry_price": float(perp_fill.get("price", 0)),
        "notional_usdt": quote_amount,
    }
    _save_basis_state(state)
    return {
        "status": "opened",
        "pair": pair,
        "spot_fill": spot_fill,
        "perp_fill": perp_fill,
    }


def close_basis_position(
    pair: str,
    spot_broker: Broker | None = None,
    perp_broker: PerpBroker | None = None,
) -> dict:
    state = _load_basis_state()
    if pair not in state:
        return {"status": "not_open", "pair": pair}

    spot = spot_broker or Broker(mode="paper", long_only=False)
    perp = perp_broker or PerpBroker(mode="paper")

    pos = state[pair]
    spot_qty = pos["spot_qty"]
    perp_contracts = pos["perp_contracts"]

    # Sell spot at market
    try:
        spot_ticker = spot.get_ticker(pair)
        sell_value = spot_qty * float(spot_ticker["bid"])
    except Exception:
        sell_value = 0
    if spot_qty > 0:
        spot.place_market_order(pair, "sell", sell_value)

    # Close perp short (buy back contracts)
    perp_close = perp.close_position(pair)

    # Funding accumulated lives on perp broker side
    bal = perp.get_balance()
    base = pair.split("/")[0]
    funding_collected = bal.get("accumulated_funding", {}).get(base, 0.0)

    del state[pair]
    _save_basis_state(state)
    return {
        "status": "closed",
        "pair": pair,
        "spot_sold_value": sell_value,
        "perp_close": perp_close,
        "funding_collected": funding_collected,
    }


def settle_all_funding(perp_broker: PerpBroker | None = None) -> dict:
    """Settle funding on every open basis position. Call each orchestrator cycle."""
    perp = perp_broker or PerpBroker(mode="paper")
    state = _load_basis_state()
    payments = {}
    for pair in state:
        payment = perp.settle_funding(pair)
        if payment != 0:
            payments[pair] = payment
    return payments


def basis_summary() -> list[dict]:
    """Return list of currently-open basis positions with their P&L."""
    state = _load_basis_state()
    perp = PerpBroker(mode="paper")
    out = []
    for pair, pos in state.items():
        base = pair.split("/")[0]
        try:
            current_price = float(perp.ticker(pair)["last"])
        except Exception:
            current_price = pos["perp_entry_price"]
        spot_pnl = pos["spot_qty"] * (current_price - pos["spot_entry_price"])
        perp_pnl = -pos["perp_contracts"] * (current_price - pos["perp_entry_price"])  # short
        bal = perp.get_balance()
        accum_funding = bal.get("accumulated_funding", {}).get(base, 0.0)
        out.append({
            "pair": pair,
            "notional_usdt": pos["notional_usdt"],
            "spot_qty": pos["spot_qty"],
            "spot_entry": pos["spot_entry_price"],
            "perp_contracts": pos["perp_contracts"],
            "perp_entry": pos["perp_entry_price"],
            "current_price": current_price,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "delta_pnl": spot_pnl + perp_pnl,  # should be ~0 (delta-neutral)
            "accumulated_funding": accum_funding,
            "total_pnl": spot_pnl + perp_pnl + accum_funding,
        })
    return out


if __name__ == "__main__":
    print("Open basis positions:", basis_summary())
