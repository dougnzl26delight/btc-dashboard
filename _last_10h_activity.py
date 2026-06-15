"""What happened over the last 10 hours."""
import sys
import sqlite3
import json
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent

cutoff = datetime.now(timezone.utc) - timedelta(hours=10)
print("=" * 90)
print(f"ACTIVITY: last 10 hours (since {cutoff.strftime('%Y-%m-%d %H:%M UTC')})")
print(f"NOW:                            {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 90)

# === 1. Trades from pnl_db ===
print()
print("--- TRADES (from pnl_db.trades) ---")
c = sqlite3.connect(str(ROOT / ".pnl.db"))
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT ts, sleeve, pair, side, qty, price, realized_pnl, note "
    "FROM trades WHERE ts > ? ORDER BY ts DESC",
    (cutoff.isoformat(),)
).fetchall()
print(f"Trade count: {len(rows)}")
if rows:
    print(f"{'When (UTC)':<20s} {'Sleeve':<22s} {'Pair':<10s} {'Side':<14s} {'Qty':>14s} {'Price':>10s} {'Realized':>10s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['ts'][:19]:<20s} {r['sleeve']:<22s} {r['pair']:<10s} "
              f"{r['side']:<14s} {r['qty']:>14.4f} {r['price']:>10.4f} {r['realized_pnl']:>+9.2f}")

# === 2. Paper trades CSV (covers spot orchestrator + direct broker calls) ===
print()
print("--- PAPER TRADES CSV (broker-level) ---")
csv_file = ROOT / "paper_trades.csv"
if csv_file.exists():
    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        recent_csv = [r for r in reader if r.get("ts_utc", "") > cutoff.isoformat()]
    print(f"Trade rows: {len(recent_csv)}")
    for r in recent_csv[-20:]:  # last 20
        sleeve = r.get("sleeve", "?")
        print(f"  {r['ts_utc'][:19]}  {sleeve:<20s} {r['pair']:<10s} {r['side']:<5s} qty={r['qty']} @ {r['price']}")

# === 3. Watchdog beat — when was the rig last alive? ===
print()
print("--- WATCHDOG ---")
wd = ROOT / ".watchdog_beat"
if wd.exists():
    mtime = datetime.fromtimestamp(wd.stat().st_mtime, tz=timezone.utc)
    age = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
    print(f"Last beat: {mtime.isoformat()}  ({age:.1f} min ago)")
else:
    print("No watchdog file — cron tasks may not have fired")

# === 4. WS feed log ===
print()
print("--- WS FEED ACTIVITY ---")
ws_log = ROOT / ".ws_feed.log"
if ws_log.exists():
    lines = ws_log.read_text(encoding="utf-8").strip().split("\n")
    recent_ws = [l for l in lines if l[:19] > cutoff.strftime("%Y-%m-%d %H:%M:%S")]
    print(f"Log lines in window: {len(recent_ws)}")
    # Count reconnects + opens
    n_open = sum(1 for l in recent_ws if "WS opened" in l)
    n_close = sum(1 for l in recent_ws if "WS closed" in l)
    n_recon = sum(1 for l in recent_ws if "Reconnecting" in l)
    print(f"  Opens: {n_open}  Closes: {n_close}  Reconnects: {n_recon}")
    print("Last 3 log entries:")
    for l in lines[-3:]:
        print(f"  {l[:120]}")
else:
    print("No WS log")

# === 5. Daily equity snapshot ===
print()
print("--- DAILY EQUITY SNAPSHOTS (from pnl_db.daily_equity) ---")
eq_rows = c.execute(
    "SELECT date, sleeve, equity FROM daily_equity WHERE date >= ? ORDER BY date DESC, sleeve",
    (cutoff.date().isoformat(),)
).fetchall()
print(f"Snapshot rows: {len(eq_rows)}")
for r in eq_rows[:20]:
    print(f"  {r['date']}  {r['sleeve']:<22s}  ${r['equity']:>10,.2f}")

# === 6. Recent alerts/signals ===
print()
print("--- SIGNALS LOGGED (last 10h) ---")
sig_rows = c.execute(
    "SELECT ts, strategy, pair, value, note "
    "FROM signals WHERE ts > ? ORDER BY ts DESC LIMIT 20",
    (cutoff.isoformat(),)
).fetchall()
print(f"Signal count: {len(sig_rows)}")
for r in sig_rows[:10]:
    note = (r["note"] or "")[:50]
    print(f"  {r['ts'][:19]}  {r['strategy']:<22s}  val={r['value']:.2f}  {note}")

c.close()
