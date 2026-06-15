"""NZ tax export from pnl_db.

Generates a CSV report for IRD-style crypto-trade reporting.

NZ rules (as of 2026):
  - Crypto trades are taxable disposals — every sell event creates a gain/loss
  - Tax year: 1 Apr - 31 Mar
  - FIFO accounting required for cost basis (most common method)
  - NZD conversion at trade date is required

This script outputs two files:
  1. tax_exports/trades_<tax_year>.csv    — every trade with realized gain/loss
  2. tax_exports/summary_<tax_year>.csv   — annual totals per sleeve/pair

NOTE: NZD/USDT conversion is currently STUBBED at 1.65. For real filings,
plug in RBNZ daily rate or use Wise/IRD-acceptable source. The accountant
will care about this.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DB_FILE = REPO_ROOT / ".pnl.db"
EXPORT_DIR = REPO_ROOT / "tax_exports"
EXPORT_DIR.mkdir(exist_ok=True)

# NZ Financial Year runs Apr 1 to Mar 31
# Year 2026 FY = "1 Apr 2025 - 31 Mar 2026"
# Stub conversion rate. Replace with daily RBNZ rate for real filing.
NZD_PER_USD = 1.65


def _nz_fy_bounds(fy_end_year: int) -> tuple[str, str]:
    """For FY ending March of `fy_end_year`, return (start_iso, end_iso)."""
    start = f"{fy_end_year - 1}-04-01"
    end = f"{fy_end_year}-03-31"
    return start, end


def export_trades(fy_end_year: int) -> Path:
    """Export every trade in the NZ tax year to CSV."""
    if not DB_FILE.exists():
        print(f"No pnl.db at {DB_FILE} — nothing to export")
        return None
    start, end = _nz_fy_bounds(fy_end_year)
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT ts, sleeve, pair, side, qty, price, notional, realized_pnl, note "
        "FROM trades WHERE ts >= ? AND ts <= ? ORDER BY ts",
        (start, end),
    ).fetchall()
    c.close()

    out_path = EXPORT_DIR / f"trades_FY{fy_end_year}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Trade Date (UTC)",
            "Sleeve",
            "Pair",
            "Side",
            "Qty",
            "Price (USDT)",
            "Notional (USDT)",
            "Notional (NZD)",
            "Realized PnL (USDT)",
            "Realized PnL (NZD)",
            "Notes",
        ])
        total_pnl_usdt = 0.0
        total_pnl_nzd = 0.0
        for r in rows:
            pnl = r["realized_pnl"] or 0
            notional = r["notional"] or 0
            total_pnl_usdt += pnl
            total_pnl_nzd += pnl * NZD_PER_USD
            w.writerow([
                r["ts"],
                r["sleeve"],
                r["pair"],
                r["side"],
                f"{r['qty']:.8f}",
                f"{r['price']:.4f}",
                f"{notional:.2f}",
                f"{notional * NZD_PER_USD:.2f}",
                f"{pnl:.2f}",
                f"{pnl * NZD_PER_USD:.2f}",
                r["note"] or "",
            ])
        # Totals row
        w.writerow([])
        w.writerow(["TOTAL", "", "", "", "", "", "", "",
                    f"{total_pnl_usdt:.2f}", f"{total_pnl_nzd:.2f}", ""])
    print(f"Wrote {len(rows)} trade rows to {out_path}")
    print(f"  Net realized: ${total_pnl_usdt:,.2f} USDT  =  NZ${total_pnl_nzd:,.2f}")
    return out_path


def export_summary(fy_end_year: int) -> Path:
    """Per-sleeve and per-pair annual summary."""
    if not DB_FILE.exists():
        return None
    start, end = _nz_fy_bounds(fy_end_year)
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    by_sleeve = c.execute(
        "SELECT sleeve, COUNT(*) AS n_trades, "
        "SUM(notional) AS gross_volume, "
        "SUM(realized_pnl) AS realized_pnl "
        "FROM trades WHERE ts >= ? AND ts <= ? "
        "GROUP BY sleeve ORDER BY realized_pnl DESC",
        (start, end),
    ).fetchall()
    by_pair = c.execute(
        "SELECT pair, COUNT(*) AS n_trades, "
        "SUM(notional) AS gross_volume, "
        "SUM(realized_pnl) AS realized_pnl "
        "FROM trades WHERE ts >= ? AND ts <= ? "
        "GROUP BY pair ORDER BY realized_pnl DESC",
        (start, end),
    ).fetchall()
    c.close()

    out_path = EXPORT_DIR / f"summary_FY{fy_end_year}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"NZ FY {fy_end_year} Summary (1 Apr {fy_end_year-1} - 31 Mar {fy_end_year})"])
        w.writerow([f"NZD conversion rate (stub): {NZD_PER_USD}"])
        w.writerow([])
        w.writerow(["By Sleeve"])
        w.writerow(["Sleeve", "N Trades", "Gross Volume USDT", "Realized PnL USDT", "Realized PnL NZD"])
        for r in by_sleeve:
            w.writerow([r["sleeve"], r["n_trades"], f"{r['gross_volume']:.2f}",
                        f"{r['realized_pnl']:.2f}", f"{r['realized_pnl'] * NZD_PER_USD:.2f}"])
        w.writerow([])
        w.writerow(["By Pair"])
        w.writerow(["Pair", "N Trades", "Gross Volume USDT", "Realized PnL USDT", "Realized PnL NZD"])
        for r in by_pair:
            w.writerow([r["pair"], r["n_trades"], f"{r['gross_volume']:.2f}",
                        f"{r['realized_pnl']:.2f}", f"{r['realized_pnl'] * NZD_PER_USD:.2f}"])
    print(f"Wrote summary to {out_path}")
    return out_path


def main():
    """Default: export current NZ FY (ending March of current calendar year + 0 or +1)."""
    today = datetime.now(timezone.utc).date()
    fy_end_year = today.year if today.month <= 3 else today.year + 1
    print(f"Exporting NZ FY {fy_end_year} (covers {_nz_fy_bounds(fy_end_year)[0]} to {_nz_fy_bounds(fy_end_year)[1]})")
    export_trades(fy_end_year)
    export_summary(fy_end_year)
    print()
    print("Hand both CSVs to your NZ crypto-aware accountant.")
    print("Stub NZD rate is 1.65 — replace with RBNZ daily rates for real filing.")


if __name__ == "__main__":
    main()
