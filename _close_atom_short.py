"""Manually close ATOM short on the paper perp account.

ATOM short has been losing daily during the relief rally; closing pre-rebalance
to stop the bleed. Will release capital for the next XSMOM cycle on May 24.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.perp_broker import PerpBroker


def main():
    pb = PerpBroker(mode="paper")
    bal_before = pb._state.cash_quote
    atom_qty = pb._state.positions.get("ATOM", 0.0)
    atom_entry = pb._state.entry_prices.get("ATOM", 0.0)
    print(f"Before close:")
    print(f"  Cash:        ${bal_before:,.2f}")
    print(f"  ATOM qty:    {atom_qty:.4f}  (short)")
    print(f"  ATOM entry:  ${atom_entry:.4f}")
    print()

    result = pb.close_position("ATOM/USDT")
    print(f"Close result: {result}")
    print()

    bal_after = pb._state.cash_quote
    print(f"After close:")
    print(f"  Cash:           ${bal_after:,.2f}")
    print(f"  Realized P&L:   ${result.get('realized_pnl', 0):,.2f}")
    print(f"  Cash delta:     ${bal_after - bal_before:+,.2f}")
    print()
    remaining = {a: q for a, q in pb._state.positions.items() if q != 0}
    print(f"Remaining open positions: {remaining}")


if __name__ == "__main__":
    main()
