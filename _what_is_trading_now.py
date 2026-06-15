"""What's the rig doing RIGHT NOW — trades + open positions snapshot."""
import sys
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent

# === Recent trades (last 24h) ===
print("=" * 90)
print("TRADES IN LAST 24 HOURS")
print("=" * 90)
c = sqlite3.connect(str(ROOT / ".pnl.db"))
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT ts, sleeve, pair, side, qty, price, realized_pnl, note "
    "FROM trades WHERE ts > datetime('now', '-1 day') ORDER BY ts DESC LIMIT 30"
).fetchall()
print(f"Trade count last 24h: {len(rows)}")
if rows:
    print(f"\n{'When (UTC)':<20s} {'Sleeve':<22s} {'Pair':<10s} {'Side':<16s} {'Qty':>12s} {'Price':>10s} {'PnL':>10s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['ts'][:19]:<20s} {r['sleeve']:<22s} {r['pair']:<10s} "
              f"{r['side']:<16s} {r['qty']:>12.4f} {r['price']:>10.4f} {r['realized_pnl']:>+9.2f}")
else:
    print("  No trades in last 24h — strategies are sitting tight")

# === Open positions across all sub-accounts ===
print()
print("=" * 90)
print("OPEN POSITIONS RIGHT NOW")
print("=" * 90)

# Spot sub-accounts
print("\nSPOT:")
for f in sorted(ROOT.glob(".paper_state_*.json")):
    if "legacy" in f.stem:
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    name = f.stem.replace(".paper_state_", "")
    open_pos = {k: v for k, v in d.get("positions", {}).items() if abs(v) > 1e-6}
    if open_pos:
        for asset, qty in open_pos.items():
            print(f"  {name:<20s}  {asset:<6s}  qty={qty:>14.6f}  (LONG)")
    else:
        cash = d.get("cash_quote", 0)
        print(f"  {name:<20s}  CASH ONLY  ${cash:>10,.0f}")

# Perp sub-accounts
print("\nPERP:")
for f in sorted(ROOT.glob(".paper_perp_state_*.json")):
    if "legacy" in f.stem:
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    name = f.stem.replace(".paper_perp_state_", "")
    open_pos = {k: v for k, v in d.get("positions", {}).items() if abs(v) > 1e-9}
    if open_pos:
        for asset, qty in open_pos.items():
            side = "LONG " if qty > 0 else "SHORT"
            print(f"  {name:<20s}  {asset:<6s}  {side}  qty={abs(qty):>10.4f}")
    else:
        cash = d.get("cash_quote", 0)
        print(f"  {name:<20s}  CASH ONLY  ${cash:>10,.0f}")

# === Last 5 alerts/events ===
print()
print("=" * 90)
print("RECENT ACTIVITY (watchdog/log)")
print("=" * 90)
ws_log = ROOT / ".ws_feed.log"
if ws_log.exists():
    lines = ws_log.read_text(encoding="utf-8").strip().split("\n")
    for line in lines[-3:]:
        print(f"  {line[:120]}")
else:
    print("  No WS log yet")
