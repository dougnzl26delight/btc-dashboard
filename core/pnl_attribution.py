"""P&L attribution: separate systematic, discretionary, basis arb sleeves.

Discipline requirement: when comparing live performance to backtest, the
backtest only models the systematic strategy. Discretionary force-entries
contaminate the comparison. This module tags every position with its
origin sleeve and tracks per-sleeve P&L.

Sleeves:
  - systematic_pro_trend  : entries via strategies.pro_trend.cycle()
  - discretionary         : entries via force_entry.py
  - basis_arb             : entries via basis_executor

Each pro_trend state file gains an "origin_sleeve" field on the unit
records. The dashboard reads this and shows per-sleeve P&L.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
ATTRIB_FILE = REPO_ROOT / ".pnl_attribution.json"


def load_attribution() -> dict:
    """Returns {pair: {origin_sleeve, opened_at, entry_price, qty, side, ...}}."""
    if ATTRIB_FILE.exists():
        return json.loads(ATTRIB_FILE.read_text())
    return {}


def save_attribution(state: dict) -> None:
    ATTRIB_FILE.write_text(json.dumps(state, indent=2, default=str))


def tag_entry(pair: str, sleeve: str, side: str, entry_price: float,
              qty: float, opened_at: str | None = None) -> None:
    """Tag a new position with its origin sleeve."""
    valid_sleeves = {"systematic_pro_trend", "discretionary", "basis_arb",
                     "xsmom", "bah_btc", "oversold_bounce", "overbought_fade",
                     "intraday_momentum", "intraday_momentum_short", "grid_trader",
                     "consolidation_breakout"}
    if sleeve not in valid_sleeves:
        raise ValueError(f"unknown sleeve: {sleeve}")
    state = load_attribution()
    state[pair] = {
        "sleeve": sleeve, "side": side,
        "entry_price": float(entry_price), "qty": float(qty),
        "opened_at": opened_at or "now",
    }
    save_attribution(state)


def untag(pair: str) -> dict | None:
    """Remove a closed position. Returns the prior tag for record-keeping."""
    state = load_attribution()
    prior = state.pop(pair, None)
    save_attribution(state)
    return prior


def get_sleeve(pair: str) -> str:
    """Return the sleeve tag for a pair, or 'unknown' if not tagged."""
    return load_attribution().get(pair, {}).get("sleeve", "unknown")


def per_sleeve_pnl(current_prices: dict[str, float]) -> dict:
    """Compute unrealized P&L grouped by sleeve, given current prices."""
    state = load_attribution()
    totals: dict[str, float] = {
        "systematic_pro_trend": 0.0,
        "discretionary": 0.0,
        "basis_arb": 0.0,
        "xsmom": 0.0,
        "bah_btc": 0.0,
        "unknown": 0.0,
    }
    counts: dict[str, int] = {k: 0 for k in totals}
    for pair, tag in state.items():
        last = current_prices.get(pair)
        if last is None:
            continue
        side = tag.get("side", "long")
        sign = 1 if side == "long" else -1
        pnl = sign * tag["qty"] * (last - tag["entry_price"])
        sleeve = tag.get("sleeve", "unknown")
        totals[sleeve] += pnl
        counts[sleeve] += 1
    return {"per_sleeve_pnl": totals, "per_sleeve_n_positions": counts}


def initialize_from_existing_state() -> dict:
    """One-time: scan existing state files and tag pre-existing positions.

    Heuristic: NEAR/BTC/ETH that were force-entered get 'discretionary';
    everything else found in PRO_TREND_PAIRS state files gets 'systematic_pro_trend'.
    Basis arb has its own state file.
    """
    from strategies import pro_trend  # late import to avoid circular

    state = load_attribution()
    actions = []

    # Discretionary force-entries (known from session log)
    discretionary_pairs = {"NEAR/USDT", "BTC/USDT", "ETH/USDT"}

    for pair_file in REPO_ROOT.glob(".pro_trend_state_*.json"):
        try:
            data = json.loads(pair_file.read_text())
        except Exception:
            continue
        if not data.get("units"):
            continue
        base = pair_file.stem.removeprefix(".pro_trend_state_")
        pair = f"{base}/USDT"
        sleeve = "discretionary" if pair in discretionary_pairs else "systematic_pro_trend"

        for u in data["units"]:
            qty = float(u["qty"])
            entry = float(u["entry_price"])
            side = data.get("side", "long")
            existing = state.get(pair)
            if existing and existing.get("entry_price") == entry:
                continue
            state[pair] = {
                "sleeve": sleeve, "side": side,
                "entry_price": entry, "qty": qty,
                "opened_at": "pre-existing",
            }
            actions.append({"pair": pair, "sleeve": sleeve, "qty": qty})

    # Basis arb positions
    basis_state_file = REPO_ROOT / ".basis_positions.json"
    if basis_state_file.exists():
        try:
            basis_state = json.loads(basis_state_file.read_text())
            for pair, pos in basis_state.items():
                state[f"basis:{pair}"] = {
                    "sleeve": "basis_arb", "side": "neutral",
                    "entry_price": float(pos.get("spot_entry_price", 0)),
                    "qty": float(pos.get("spot_qty", 0)),
                    "opened_at": "pre-existing",
                }
                actions.append({"pair": f"basis:{pair}", "sleeve": "basis_arb"})
        except Exception:
            pass

    save_attribution(state)
    return {"tagged": actions, "total": len(actions)}


if __name__ == "__main__":
    print("Initializing P&L attribution from existing state files...")
    result = initialize_from_existing_state()
    print(f"Tagged {result['total']} positions:")
    for r in result["tagged"]:
        print(f"  {r['pair']:<20s} -> {r['sleeve']}")
    print()
    print("Current attribution state:")
    state = load_attribution()
    print(json.dumps(state, indent=2, default=str))
