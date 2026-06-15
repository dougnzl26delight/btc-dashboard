"""SQLite-backed P&L attribution + telemetry.

Persists trades, signals, and daily sleeve-equity snapshots. Foundation for:
  - Walk-forward rolling Sharpe per sleeve (live decay detection)
  - Per-signal IC tracking (which strategies actually earn)
  - Audit replay (every decision is logged)

Schema:
    trades        — fill-level execution log
    signals       — per-cycle signal value per (strategy, pair)
    daily_equity  — sleeve equity at end-of-day
    daily_pnl     — derived daily P&L per sleeve

Designed to be append-only — never DELETE or UPDATE rows in production.
The DB file lives at .pnl.db at repo root.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = REPO_ROOT / ".pnl.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    sleeve      TEXT NOT NULL,
    pair        TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    notional    REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    pair        TEXT NOT NULL,
    value       REAL NOT NULL,
    regime      TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS daily_equity (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    sleeve      TEXT NOT NULL,
    equity      REAL NOT NULL,
    cash        REAL,
    open_mtm    REAL,
    note        TEXT,
    UNIQUE(date, sleeve)
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    sleeve      TEXT NOT NULL,
    pnl_usd     REAL NOT NULL,
    pnl_pct     REAL NOT NULL,
    realized    REAL DEFAULT 0,
    unrealized  REAL DEFAULT 0,
    UNIQUE(date, sleeve)
);

CREATE INDEX IF NOT EXISTS idx_trades_sleeve_ts ON trades(sleeve, ts);
CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals(strategy, ts);
CREATE INDEX IF NOT EXISTS idx_daily_equity_sleeve_date ON daily_equity(sleeve, date);
CREATE INDEX IF NOT EXISTS idx_daily_pnl_sleeve_date ON daily_pnl(sleeve, date);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create tables and indices if missing. Idempotent."""
    with _conn() as c:
        c.executescript(SCHEMA)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ===== writers =====
def log_trade(sleeve: str, pair: str, side: str, qty: float, price: float,
              realized_pnl: float = 0.0, note: str = "") -> int:
    """Append a trade row. Returns inserted id."""
    init_db()
    notional = abs(qty) * price
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO trades(ts, sleeve, pair, side, qty, price, notional, realized_pnl, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now_iso(), sleeve, pair, side, qty, price, notional, realized_pnl, note),
        )
        return cur.lastrowid


def log_signal(strategy: str, pair: str, value: float, regime: Optional[str] = None,
               note: str = "") -> int:
    """Append a signal observation."""
    init_db()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO signals(ts, strategy, pair, value, regime, note) VALUES (?, ?, ?, ?, ?, ?)",
            (_now_iso(), strategy, pair, value, regime, note),
        )
        return cur.lastrowid


def snapshot_daily_equity(sleeve: str, equity: float,
                          cash: Optional[float] = None,
                          open_mtm: Optional[float] = None,
                          note: str = "") -> None:
    """Record end-of-day equity for one sleeve. Idempotent per (date, sleeve)."""
    init_db()
    date = _today_iso()
    with _conn() as c:
        # Compute daily P&L vs yesterday
        prev = c.execute(
            "SELECT equity FROM daily_equity WHERE sleeve=? AND date<? ORDER BY date DESC LIMIT 1",
            (sleeve, date),
        ).fetchone()
        c.execute(
            "INSERT INTO daily_equity(date, sleeve, equity, cash, open_mtm, note) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date, sleeve) DO UPDATE SET equity=excluded.equity, "
            "cash=excluded.cash, open_mtm=excluded.open_mtm, note=excluded.note",
            (date, sleeve, equity, cash, open_mtm, note),
        )
        if prev:
            prev_eq = prev["equity"]
            pnl_usd = equity - prev_eq
            pnl_pct = pnl_usd / prev_eq if prev_eq > 0 else 0
            c.execute(
                "INSERT INTO daily_pnl(date, sleeve, pnl_usd, pnl_pct) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(date, sleeve) DO UPDATE SET pnl_usd=excluded.pnl_usd, "
                "pnl_pct=excluded.pnl_pct",
                (date, sleeve, pnl_usd, pnl_pct),
            )


# ===== queries =====
def get_sleeve_returns(sleeve: str, days: int = 60) -> list[float]:
    """Daily return series (most recent first) for sleeve, last N days."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT pnl_pct FROM daily_pnl WHERE sleeve=? ORDER BY date DESC LIMIT ?",
            (sleeve, days),
        ).fetchall()
    return [r["pnl_pct"] for r in rows]


def get_sleeve_sharpe(sleeve: str, days: int = 60, ann_factor: int = 365) -> Optional[float]:
    """Annualized Sharpe ratio for sleeve over last N days. None if insufficient data."""
    rets = get_sleeve_returns(sleeve, days)
    if len(rets) < 10:
        return None
    import statistics
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    if sd == 0:
        return None
    return (mu / sd) * (ann_factor ** 0.5)


def get_recent_trades(sleeve: Optional[str] = None, limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as c:
        if sleeve:
            rows = c.execute(
                "SELECT * FROM trades WHERE sleeve=? ORDER BY ts DESC LIMIT ?",
                (sleeve, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM trades ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_sleeve_summary() -> list[dict]:
    """One-row-per-sleeve aggregate: trade count, total realized PnL, sharpe."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT sleeve, COUNT(*) AS n_trades, SUM(realized_pnl) AS total_realized "
            "FROM trades GROUP BY sleeve"
        ).fetchall()
        sleeves = [dict(r) for r in rows]
    for s in sleeves:
        s["sharpe_60d"] = get_sleeve_sharpe(s["sleeve"], 60)
        s["sharpe_30d"] = get_sleeve_sharpe(s["sleeve"], 30)
    return sleeves


def main():
    """CLI: show DB summary INCLUDING Deflated Sharpe Ratio per sleeve.

    W14.D: raw Sharpe is misleading after testing many strategies. DSR adjusts
    for selection bias + skew + kurtosis. Use DSR > 0.95 as live-deploy gate.
    """
    init_db()
    print(f"DB: {DB_FILE}")
    print()
    print("=" * 110)
    print("SLEEVE SUMMARY  with DEFLATED SHARPE RATIO (Bailey/Lopez de Prado, n=30 trials assumed)")
    print("=" * 110)
    print(f"{'Sleeve':<22s} {'Trades':>7s} {'Realized':>12s} "
          f"{'Raw SR60':>9s} {'DSR':>7s} {'Hurdle SR':>10s} {'Verdict':<22s}")
    print("-" * 110)
    try:
        from core.deflated_sharpe import deflated_sharpe
        DSR_AVAILABLE = True
    except Exception:
        DSR_AVAILABLE = False

    for s in get_sleeve_summary():
        sharpe60 = s.get("sharpe_60d")
        sharpe60_s = f"{sharpe60:+.2f}" if sharpe60 is not None else "n/a"
        dsr_str = "n/a"
        hurdle_str = "n/a"
        verdict_str = ""
        if DSR_AVAILABLE:
            returns = get_sleeve_returns(s["sleeve"], days=90)
            if len(returns) >= 30:
                try:
                    d = deflated_sharpe(returns, num_trials=30, periods_per_year=365)
                    dsr_str = f"{d['dsr']:.3f}"
                    hurdle_str = f"{d['sr_threshold_annualized']:+.2f}"
                    if d["passes"]:
                        verdict_str = "ROBUST (deploy)"
                    elif d["dsr"] > 0.5:
                        verdict_str = "WEAK (test small)"
                    else:
                        verdict_str = "REJECT (overfit)"
                except Exception:
                    pass
        print(f"  {s['sleeve']:<20s} {s['n_trades']:>7d}  ${s['total_realized']:>+10,.2f}  "
              f"{sharpe60_s:>9s} {dsr_str:>7s} {hurdle_str:>10s} {verdict_str:<22s}")
    with _conn() as c:
        n_eq = c.execute("SELECT COUNT(*) FROM daily_equity").fetchone()[0]
        n_sig = c.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        n_tr = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    print()
    print(f"daily_equity rows: {n_eq}    signals rows: {n_sig}    trades rows: {n_tr}")
    print()
    print("Interpretation: DSR is P(true SR > selection-adjusted threshold).")
    print("  > 0.95 = robust edge (deploy live)")
    print("  0.50-0.95 = weak; small live test only")
    print("  < 0.50 = likely backtest-overfit; do NOT deploy")


if __name__ == "__main__":
    main()
