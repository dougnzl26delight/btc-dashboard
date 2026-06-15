"""Realized P&L computation via FIFO lot matching.

Walk through the trade log chronologically, maintain a queue of open lots
per asset. When a trade reduces position (opposite side), close the oldest
lot first (FIFO), realizing the P&L on that closed quantity.

Output:
  - Per-trade realized P&L
  - Per-asset cumulative realized P&L
  - Summary stats
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REPO_ROOT = Path(__file__).resolve().parent.parent
TRADES_FILE = REPO_ROOT / "paper_trades.csv"


def compute_realized_pnl() -> pd.DataFrame:
    """Walk trades chronologically, FIFO-match opposite-side trades to realize P&L.
    Returns DataFrame of close events with realized P&L per closed lot."""
    if not TRADES_FILE.exists():
        return pd.DataFrame()

    trades = pd.read_csv(TRADES_FILE)
    if trades.empty:
        return pd.DataFrame()

    trades = trades.sort_values("ts_utc").reset_index(drop=True)
    open_lots: dict[str, deque] = {}  # asset -> deque of (ts, qty_signed, price)
    closes = []

    for _, row in trades.iterrows():
        asset = str(row["pair"]).split("/")[0]
        side = str(row["side"])
        qty = float(row["qty"])
        price = float(row["price"])
        signed_qty = qty if side == "buy" else -qty

        lots = open_lots.setdefault(asset, deque())

        if not lots or (lots[0][1] * signed_qty > 0):
            # Same direction or empty — open new lot
            lots.append((row["ts_utc"], signed_qty, price))
            continue

        # Opposite direction — close lots FIFO
        remaining = abs(signed_qty)
        while remaining > 1e-12 and lots and (lots[0][1] * signed_qty < 0):
            lot_ts, lot_qty, lot_price = lots[0]
            lot_abs = abs(lot_qty)
            close_qty = min(lot_abs, remaining)
            # P&L on this closed slice
            direction = 1 if lot_qty > 0 else -1
            pnl = direction * (price - lot_price) * close_qty
            closes.append({
                "ts_close": row["ts_utc"],
                "ts_open": lot_ts,
                "asset": asset,
                "direction": "LONG" if direction > 0 else "SHORT",
                "qty": close_qty,
                "open_price": lot_price,
                "close_price": price,
                "realized_pnl": pnl,
            })
            remaining -= close_qty
            new_lot_qty = lot_qty - direction * close_qty
            if abs(new_lot_qty) < 1e-12:
                lots.popleft()
            else:
                lots[0] = (lot_ts, new_lot_qty, lot_price)

        # If signed_qty extends past available offsetting lots, the rest opens new lot
        if remaining > 1e-12:
            new_qty = (signed_qty / abs(signed_qty)) * remaining
            lots.append((row["ts_utc"], new_qty, price))

    return pd.DataFrame(closes)


def realized_summary() -> dict:
    df = compute_realized_pnl()
    if df.empty:
        return {"n_closes": 0, "total_realized": 0.0}
    by_asset = df.groupby("asset")["realized_pnl"].agg(["sum", "count", "mean"])
    return {
        "n_closes": int(len(df)),
        "total_realized": float(df["realized_pnl"].sum()),
        "n_winners": int((df["realized_pnl"] > 0).sum()),
        "n_losers": int((df["realized_pnl"] < 0).sum()),
        "win_rate": float((df["realized_pnl"] > 0).mean()),
        "avg_pnl": float(df["realized_pnl"].mean()),
        "best": float(df["realized_pnl"].max()),
        "worst": float(df["realized_pnl"].min()),
        "by_asset": by_asset.to_dict(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(realized_summary(), indent=2, default=str))
