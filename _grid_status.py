"""Grid trader status — config + activity."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# State
print("=" * 70)
print("GRID TRADER STATUS")
print("=" * 70)
state = json.loads((ROOT / ".paper_state_grid_trader.json").read_text())
print(f"Cash: ${state['cash_quote']:,.2f}")
print(f"Positions: {state.get('positions', {})}")
print()

# Grid config
gs = json.loads((ROOT / ".grid_trader_state.json").read_text())
print("Grid configuration:")
for pair, g in gs.get("grids", {}).items():
    n_filled = len(g.get("filled_levels", []))
    print(f"  {pair}:")
    print(f"    Center: ${g['center']:,.2f}")
    print(f"    Last check: ${g['last_check_price']:,.2f}")
    print(f"    Filled levels: {n_filled}/10")
    print(f"    Recentered at: {g.get('recentered_at', '?')}")
    print(f"    Levels:")
    for level_name, level_price in g["levels"].items():
        status = "FILLED" if level_name in g.get("filled_levels", []) else "open"
        print(f"      {level_name:<10s}  ${level_price:>10,.4f}  [{status}]")

# Trades
print()
print("Grid trades in pnl_db:")
c = sqlite3.connect(str(ROOT / ".pnl.db"))
c.row_factory = sqlite3.Row
rows = c.execute("SELECT ts, pair, side, qty, price, realized_pnl, note FROM trades WHERE sleeve='grid_trader' ORDER BY ts DESC LIMIT 20").fetchall()
print(f"Total grid trades: {len(rows)}")
for r in rows:
    print(f"  {r['ts'][:19]}  {r['pair']:<10s} {r['side']:<6s} qty={r['qty']:.4f} @ {r['price']:.2f}  {r['note']}")
