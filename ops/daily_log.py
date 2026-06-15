"""Daily snapshot logger. Run once a day from a scheduled task to build
the audit trail proving the wiring stayed alive across the paper period.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python ops/daily_log.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import Broker
from core.data import ticker
from ops.alerts import alert


LOG = Path(__file__).resolve().parent.parent / "daily_status.csv"


def snapshot(pairs: list[str] | None = None) -> dict:
    pairs = pairs or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    b = Broker(mode="paper")
    bal = b.get_balance()
    prices = {p: ticker(p)["last"] for p in pairs}
    equity = bal.get("USDT", 0.0) + sum(
        bal.get(p.split("/")[0], 0.0) * prices[p] for p in pairs
    )
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "equity_usdt": round(equity, 2),
        "cash_usdt": round(bal.get("USDT", 0.0), 2),
    }
    for p in pairs:
        base = p.split("/")[0]
        row[f"price_{base}"] = prices[p]
        row[f"qty_{base}"] = bal.get(base, 0.0)

    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == "__main__":
    row = snapshot()
    print(row)
    alert(f"Daily snapshot: equity ${row['equity_usdt']:,.2f}", level="info")
