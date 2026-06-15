"""Align paper accounts to the new pro_trend production strategy.

Closes residuals, resets state files, restarts with a clean $100k bankroll
on both spot and perp paper accounts. Run once after major strategy change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent

from core.broker import Broker
from core.data import ticker
from core.perp_broker import PerpBroker


START_USDT = 100_000.0


def close_residual_spot_positions():
    """Close any non-zero positions in the spot paper account."""
    state_file = REPO_ROOT / ".paper_state.json"
    if not state_file.exists():
        return {"action": "skipped", "reason": "no spot state"}
    state = json.loads(state_file.read_text())
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")

    spot = Broker(mode="paper", long_only=False)
    actions = []
    for asset, qty in list(positions.items()):
        if abs(qty) < 1e-9:
            continue
        pair = f"{asset}/{quote}"
        try:
            price = float(ticker(pair)["last"])
            if qty > 0:
                spot.place_market_order(pair, "sell", qty * price)
            else:
                spot.place_market_order(pair, "buy", abs(qty) * price)
            actions.append({"asset": asset, "qty": qty, "closed_at": price})
        except Exception as e:
            actions.append({"asset": asset, "qty": qty, "error": str(e)})
    return {"action": "closed_residuals", "n": len(actions), "details": actions}


def reset_spot_paper_state(bankroll: float = START_USDT):
    f = REPO_ROOT / ".paper_state.json"
    f.write_text(json.dumps({
        "cash_quote": bankroll,
        "quote_currency": "USDT",
        "positions": {},
    }, indent=2))
    return {"action": "reset_spot_paper", "bankroll": bankroll}


def reset_perp_paper_state(bankroll: float = START_USDT):
    f = REPO_ROOT / ".paper_perp_state.json"
    f.write_text(json.dumps({
        "cash_quote": bankroll,
        "quote_currency": "USDT",
        "positions": {},
        "entry_prices": {},
        "accumulated_funding": {},
        "last_funding_ts": {},
    }, indent=2))
    return {"action": "reset_perp_paper", "bankroll": bankroll}


def reset_pro_trend_state():
    cleared = []
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        f.unlink()
        cleared.append(f.name)
    # Also remove old single-pair state file
    old = REPO_ROOT / ".pro_trend_state.json"
    if old.exists():
        old.unlink()
        cleared.append(old.name)
    return {"action": "reset_pro_trend", "cleared": cleared}


def reset_basis_state():
    f = REPO_ROOT / ".basis_positions.json"
    if f.exists():
        f.unlink()
        return {"action": "reset_basis", "cleared": True}
    return {"action": "reset_basis", "cleared": False}


def reset_aux_state():
    """Clean up auxiliary state files."""
    cleared = []
    for name in [".position_hwm.json", ".kill_switch.json", ".peak_equity.json"]:
        f = REPO_ROOT / name
        if f.exists():
            f.unlink()
            cleared.append(name)
    return {"action": "reset_aux", "cleared": cleared}


if __name__ == "__main__":
    print("=== Aligning paper accounts ===")
    print()
    print("Step 1: closing residual spot positions...")
    r = close_residual_spot_positions()
    print(f"  {r}")
    print()
    print("Step 2: resetting spot paper to $100k...")
    r = reset_spot_paper_state()
    print(f"  {r}")
    print()
    print("Step 3: resetting perp paper to $100k...")
    r = reset_perp_paper_state()
    print(f"  {r}")
    print()
    print("Step 4: clearing pro_trend per-pair state...")
    r = reset_pro_trend_state()
    print(f"  cleared: {r['cleared']}")
    print()
    print("Step 5: clearing basis position state...")
    r = reset_basis_state()
    print(f"  {r}")
    print()
    print("Step 6: clearing auxiliary state files...")
    r = reset_aux_state()
    print(f"  cleared: {r['cleared']}")
    print()
    print("=== Done. System aligned to clean $100k paper bankroll ===")
    print("Next pro_trend cycle will start fresh.")
