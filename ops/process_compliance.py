"""Process compliance tracker — Mark Douglas, *Trading in the Zone*.

The book's central insight: "Instead of tracking wins and losses trade by trade,
you start tracking whether you executed your process." This module IS that.

What it does:
    1. Reads signals (pnl_db.signals table) and trades (pnl_db.trades table)
    2. For each signal that should have produced a trade, checks if it did
    3. Computes a daily compliance score: % of signals executed without override
    4. Logs every MANUAL_OVERRIDE entry (signal generated but no trade, OR
       trade taken without signal) with an explanation field

A profitable rig with 90% compliance is healthier than a profitable rig with
60% compliance, even if the latter shows higher P&L today. The discipline
gap is what kills traders over the 2-5 year horizon.

Surfaces in daily_report at 19:00 NZT alongside P&L.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DB_FILE = Path(__file__).resolve().parent.parent / ".pnl.db"
COMPLIANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS compliance_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    event_type   TEXT NOT NULL,    -- signal_with_trade / signal_no_trade / trade_no_signal / manual_override
    sleeve       TEXT NOT NULL,
    pair         TEXT,
    detail       TEXT,
    operator_reason TEXT             -- required for manual_override
);
CREATE INDEX IF NOT EXISTS idx_compliance_ts ON compliance_events(ts);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    c.executescript(COMPLIANCE_SCHEMA)
    return c


def log_compliance_event(event_type: str, sleeve: str, pair: str | None = None,
                          detail: str = "", operator_reason: str = "") -> int:
    """Record a process-compliance event."""
    valid = {"signal_with_trade", "signal_no_trade", "trade_no_signal",
             "manual_override", "manual_position_modify"}
    if event_type not in valid:
        raise ValueError(f"event_type must be one of {valid}")
    if event_type in {"manual_override", "manual_position_modify"} and not operator_reason:
        raise ValueError(f"{event_type} requires operator_reason")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO compliance_events"
            "(ts, event_type, sleeve, pair, detail, operator_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(),
             event_type, sleeve, pair, detail, operator_reason),
        )
        return cur.lastrowid


def compute_daily_score(date_iso: str | None = None) -> dict:
    """Compute today's (or specified date's) compliance score.

    Score = signals_with_trade / (signals_with_trade + signals_no_trade)
    Override count + reason audit included.
    """
    if date_iso is None:
        date_iso = datetime.now(timezone.utc).date().isoformat()
    start = f"{date_iso}T00:00:00"
    end = f"{date_iso}T23:59:59"

    with _conn() as c:
        # Count signal events today (from signals table)
        signals_count = c.execute(
            "SELECT strategy, COUNT(*) AS n FROM signals "
            "WHERE ts >= ? AND ts <= ? GROUP BY strategy",
            (start, end),
        ).fetchall()
        # Count trades today (from trades table)
        trades_count = c.execute(
            "SELECT sleeve, COUNT(*) AS n FROM trades "
            "WHERE ts >= ? AND ts <= ? GROUP BY sleeve",
            (start, end),
        ).fetchall()
        # Manual overrides today
        overrides = c.execute(
            "SELECT * FROM compliance_events "
            "WHERE ts >= ? AND ts <= ? "
            "AND event_type IN ('manual_override', 'manual_position_modify') "
            "ORDER BY ts",
            (start, end),
        ).fetchall()

    sig_map = {r["strategy"]: r["n"] for r in signals_count}
    trade_map = {r["sleeve"]: r["n"] for r in trades_count}

    # Heuristic: any sleeve that generated signals should have produced
    # roughly matching trade volume. Compliance per sleeve = min(trades/signals, 1.0).
    per_sleeve = {}
    all_sleeves = set(sig_map) | set(trade_map)
    for s in all_sleeves:
        n_sig = sig_map.get(s, 0)
        n_trd = trade_map.get(s, 0)
        if n_sig == 0:
            # Trades without signals — possible discretionary
            compliance = None
            note = "trades_without_signals_in_log"
        else:
            ratio = min(n_trd / n_sig, 1.0) if n_sig > 0 else 0
            compliance = ratio
            note = ""
        per_sleeve[s] = {
            "signals": n_sig,
            "trades": n_trd,
            "compliance": compliance,
            "note": note,
        }

    valid_sleeves = [v for v in per_sleeve.values() if v["compliance"] is not None]
    overall_compliance = (
        sum(v["compliance"] for v in valid_sleeves) / len(valid_sleeves)
        if valid_sleeves else None
    )

    return {
        "date": date_iso,
        "overall_compliance": overall_compliance,
        "per_sleeve": per_sleeve,
        "manual_overrides": [dict(r) for r in overrides],
        "n_manual_overrides": len(overrides),
        "verdict": _verdict(overall_compliance, len(overrides)),
    }


def _verdict(compliance: float | None, n_overrides: int) -> str:
    if compliance is None:
        return "NO DATA"
    if n_overrides > 3:
        return "EXCESSIVE OVERRIDES — review discipline"
    if compliance > 0.95:
        return "DISCIPLINED — executing process as designed"
    if compliance > 0.85:
        return "GOOD — minor signal-trade misalignment"
    if compliance > 0.70:
        return "OK — review missed signals"
    return "POOR — significant process drift"


def recent_overrides(days: int = 7) -> list[dict]:
    """Audit log of recent operator overrides."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM compliance_events "
            "WHERE ts > ? AND event_type IN ('manual_override', 'manual_position_modify') "
            "ORDER BY ts DESC",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def main():
    """CLI: today's compliance scorecard."""
    today = compute_daily_score()
    print("=" * 90)
    print(f"PROCESS COMPLIANCE — {today['date']}")
    print(f"Mark Douglas: 'Track whether you executed your process, not whether you won.'")
    print("=" * 90)
    print()
    oc = today["overall_compliance"]
    print(f"Overall compliance: {oc*100:.0f}%" if oc is not None else "Overall: NO DATA")
    print(f"Verdict: {today['verdict']}")
    print(f"Manual overrides today: {today['n_manual_overrides']}")
    print()
    print(f"{'Sleeve':<24s} {'Signals':>8s} {'Trades':>7s} {'Compliance':>11s} Notes")
    print("-" * 70)
    for s, d in today["per_sleeve"].items():
        compl_str = f"{d['compliance']*100:.0f}%" if d["compliance"] is not None else "n/a"
        print(f"  {s:<22s} {d['signals']:>8d} {d['trades']:>7d} {compl_str:>11s}  {d['note']}")
    if today["manual_overrides"]:
        print()
        print("Override audit:")
        for o in today["manual_overrides"]:
            print(f"  [{o['ts'][:19]}] {o['event_type']:<24s} {o['sleeve']}/{o['pair'] or '?'}: {o['operator_reason']}")


if __name__ == "__main__":
    main()
