"""Show recent trades from pnl_db.sqlite."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / ".pnl.db"
c = sqlite3.connect(str(DB))
c.row_factory = sqlite3.Row

rows = c.execute(
    "SELECT ts, sleeve, pair, side, qty, price, realized_pnl, note FROM trades ORDER BY ts DESC LIMIT 30"
).fetchall()

print(f"TRADES LOGGED IN PNL DB ({len(rows)} most recent)")
print()
print(f"{'When (UTC)':<20s} {'Sleeve':<22s} {'Pair':<10s} {'Side':<12s} {'Qty':>14s} {'Price':>12s} {'Realized':>10s}  Note")
print("-" * 130)
for r in rows:
    print(f"{r['ts'][:19]:<20s} {r['sleeve']:<22s} {r['pair']:<10s} {r['side']:<12s} "
          f"{r['qty']:>14.4f} {r['price']:>11.4f}  {r['realized_pnl']:>+9.2f}  {r['note'] or ''}")

n = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
print()
print(f"Total trades in DB: {n}")

# Per sleeve summary
print()
print("REALIZED P&L BY SLEEVE:")
sleeves = c.execute(
    "SELECT sleeve, COUNT(*) AS n, SUM(realized_pnl) AS rpnl "
    "FROM trades GROUP BY sleeve ORDER BY rpnl DESC"
).fetchall()
for s in sleeves:
    print(f"  {s['sleeve']:<22s}  trades={s['n']:>4d}  realized={s['rpnl']:>+9.2f}")
