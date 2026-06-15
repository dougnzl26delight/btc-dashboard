"""Prediction outcome tracking + adaptive weight learning.

Logs every prediction to SQLite. After the horizon elapses, the actual
outcome is scored vs the prediction. Hit rate is computed per signal and
per category. Weights auto-adapt over time so signals that work get more
influence and signals that don't get less.

This is what separates retail-quant from institutional-grade: the system
LEARNS from its own outcomes.

Tables:
    predictions       (id, ts, btc_price, horizon, direction_score,
                        interpretation, signal_scores_json, status)
    signal_outcomes   (prediction_id, signal_name, category, score_at_time,
                        actual_direction, was_correct, scored_at)

Hit rate per signal: rolling 90-day percentage of correct directional calls.
Adaptive weight per signal: base_weight × (hit_rate / 0.5) → high hit rate
amplifies weight, low hit rate suppresses.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DB_FILE = REPO / ".prediction_outcomes.sqlite"

# Horizon → days (matches btc_prediction.py)
HORIZON_DAYS = {
    "intraday": 1,
    "short_term": 30,
    "medium_term": 90,
    "long_term": 180,
}


def _conn():
    """Get connection with row factory."""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if not exist."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                btc_price REAL NOT NULL,
                horizon TEXT NOT NULL,
                direction_score REAL NOT NULL,
                interpretation TEXT NOT NULL,
                confidence TEXT,
                signal_scores_json TEXT,
                regime TEXT,
                status TEXT DEFAULT 'pending',
                resolved_at TEXT,
                actual_price REAL,
                actual_return REAL,
                was_correct INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
            CREATE INDEX IF NOT EXISTS idx_pred_horizon ON predictions(horizon);
            CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);

            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                signal_name TEXT NOT NULL,
                category TEXT NOT NULL,
                score_at_time REAL NOT NULL,
                actual_direction INTEGER,
                was_correct INTEGER,
                scored_at TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_sig_name ON signal_outcomes(signal_name);
            CREATE INDEX IF NOT EXISTS idx_sig_correct ON signal_outcomes(was_correct);
        """)


def log_prediction(state: dict) -> dict:
    """Log a prediction to the outcomes DB. Returns dict of inserted ids per horizon."""
    init_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    btc_price = state.get("btc_price", 0)
    if btc_price <= 0:
        return {"error": "no_price"}

    ids = {}
    with _conn() as c:
        for h_name, hd in state.get("horizons", {}).items():
            cur = c.execute(
                "INSERT INTO predictions "
                "(ts, btc_price, horizon, direction_score, interpretation, "
                "confidence, signal_scores_json, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now_iso, btc_price, h_name,
                 hd.get("direction_score", 0),
                 hd.get("interpretation", "?"),
                 hd.get("confidence", "?"),
                 json.dumps(hd.get("breakdown", {})),
                 state.get("regime"))
            )
            pred_id = cur.lastrowid
            ids[h_name] = pred_id

            # Log each contributing signal
            signals = state.get("signals", {})
            for cat, cat_sigs in signals.items():
                if not isinstance(cat_sigs, dict) or cat_sigs.get("error"): continue
                for sig_name, sig_data in cat_sigs.items():
                    if not isinstance(sig_data, dict): continue
                    score = sig_data.get("score")
                    if score is None: continue
                    c.execute(
                        "INSERT INTO signal_outcomes "
                        "(prediction_id, signal_name, category, score_at_time) "
                        "VALUES (?, ?, ?, ?)",
                        (pred_id, sig_name, cat, float(score))
                    )
    return ids


def resolve_due_predictions() -> dict:
    """Mark predictions whose horizon has elapsed and score their outcomes.

    Returns summary of how many were resolved.
    """
    init_db()
    now = datetime.now(timezone.utc)
    resolved_count = 0
    correct_count = 0

    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=2)
        current_price = float(df["close"].iloc[-1]) if not df.empty else None
    except Exception:
        current_price = None

    if current_price is None:
        return {"error": "no_current_price"}

    with _conn() as c:
        pending = c.execute(
            "SELECT * FROM predictions WHERE status='pending'"
        ).fetchall()

        for p in pending:
            ts = datetime.fromisoformat(p["ts"])
            horizon = p["horizon"]
            days_required = HORIZON_DAYS.get(horizon, 30)
            elapsed = (now - ts).days
            if elapsed < days_required: continue

            actual_return = current_price / p["btc_price"] - 1
            pred_score = p["direction_score"]
            # Was the directional call correct?
            #   pred_score > 0 (bull) and actual_return > 0 → correct
            #   pred_score < 0 (bear) and actual_return < 0 → correct
            was_correct = (
                (pred_score > 0.1 and actual_return > 0) or
                (pred_score < -0.1 and actual_return < 0) or
                (-0.1 <= pred_score <= 0.1 and abs(actual_return) < 0.05)  # neutral hit
            )

            c.execute(
                "UPDATE predictions SET status='resolved', resolved_at=?, "
                "actual_price=?, actual_return=?, was_correct=? WHERE id=?",
                (now.isoformat(), current_price, actual_return,
                 1 if was_correct else 0, p["id"])
            )

            # Also score each contributing signal
            sigs = c.execute(
                "SELECT * FROM signal_outcomes WHERE prediction_id=?",
                (p["id"],)
            ).fetchall()
            for s in sigs:
                signal_score = s["score_at_time"]
                signal_correct = (
                    (signal_score > 0.1 and actual_return > 0) or
                    (signal_score < -0.1 and actual_return < 0) or
                    (-0.1 <= signal_score <= 0.1 and abs(actual_return) < 0.05)
                )
                actual_dir = 1 if actual_return > 0.02 else (-1 if actual_return < -0.02 else 0)
                c.execute(
                    "UPDATE signal_outcomes SET actual_direction=?, was_correct=?, "
                    "scored_at=? WHERE id=?",
                    (actual_dir, 1 if signal_correct else 0, now.isoformat(), s["id"])
                )

            resolved_count += 1
            if was_correct: correct_count += 1

    return {
        "resolved": resolved_count,
        "correct": correct_count,
        "hit_rate": correct_count / max(1, resolved_count),
    }


def hit_rates_by_signal(min_observations: int = 10, lookback_days: int = 180) -> dict:
    """Compute hit rate per signal over recent lookback window."""
    init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    with _conn() as c:
        rows = c.execute(
            "SELECT signal_name, category, "
            "  COUNT(*) AS n, "
            "  SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) AS n_correct "
            "FROM signal_outcomes "
            "WHERE scored_at IS NOT NULL AND scored_at > ? "
            "GROUP BY signal_name, category "
            "HAVING n >= ? "
            "ORDER BY n_correct * 1.0 / n DESC",
            (cutoff, min_observations)
        ).fetchall()

    return {
        r["signal_name"]: {
            "category": r["category"],
            "n_observations": r["n"],
            "n_correct": r["n_correct"],
            "hit_rate": r["n_correct"] / r["n"] if r["n"] > 0 else None,
        }
        for r in rows
    }


def hit_rates_by_horizon() -> dict:
    """Hit rate per horizon."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT horizon, "
            "  COUNT(*) AS n, "
            "  SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) AS n_correct "
            "FROM predictions "
            "WHERE status='resolved' "
            "GROUP BY horizon"
        ).fetchall()
    return {
        r["horizon"]: {
            "n_observations": r["n"],
            "n_correct": r["n_correct"],
            "hit_rate": r["n_correct"] / r["n"] if r["n"] > 0 else None,
        }
        for r in rows
    }


def adaptive_signal_weight(signal_name: str) -> float:
    """Return signal weight scaled by recent hit rate.

    Hit rate 0.5 = baseline (weight 1.0)
    Hit rate 0.7 = 1.4x weight (rewards good signals)
    Hit rate 0.3 = 0.6x weight (suppresses bad signals)
    Insufficient data = 1.0 (no change)
    """
    hr = hit_rates_by_signal(min_observations=10).get(signal_name, {})
    if not hr or hr.get("hit_rate") is None: return 1.0
    return max(0.3, min(1.7, 0.4 + hr["hit_rate"] * 1.2))


def detect_signal_anomalies() -> list:
    """Detect signals that have suddenly diverged from their typical range.

    Compares current score to historical mean +/- 2 std. Useful for flagging
    "regime change" or "model uncertainty" events.
    """
    init_db()
    anomalies = []
    with _conn() as c:
        # Get recent (last 30) per-signal scores
        rows = c.execute(
            "SELECT signal_name, category, score_at_time, scored_at "
            "FROM signal_outcomes "
            "WHERE scored_at IS NOT NULL "
            "ORDER BY scored_at DESC"
        ).fetchall()

    if not rows: return []

    # Group by signal
    from collections import defaultdict
    by_signal = defaultdict(list)
    for r in rows:
        by_signal[r["signal_name"]].append(r["score_at_time"])

    for sig_name, scores in by_signal.items():
        if len(scores) < 20: continue
        recent_5 = scores[:5]
        historical = scores[5:]
        if not historical: continue
        mean = sum(historical) / len(historical)
        std = (sum((s - mean) ** 2 for s in historical) / len(historical)) ** 0.5
        if std == 0: continue
        recent_mean = sum(recent_5) / len(recent_5)
        z = (recent_mean - mean) / std
        if abs(z) > 2.0:
            anomalies.append({
                "signal": sig_name,
                "recent_mean": recent_mean,
                "historical_mean": mean,
                "z_score": z,
                "interpretation": (
                    "abnormally bearish vs history" if z < -2
                    else "abnormally bullish vs history"
                ),
            })
    return anomalies


def main():
    """CLI: show current outcome stats."""
    init_db()
    print("\n" + "=" * 72)
    print("BTC PREDICTION OUTCOMES — hit rates + adaptive weights")
    print("=" * 72)

    # Resolve any due predictions
    res = resolve_due_predictions()
    if "error" not in res:
        print(f"\nResolved {res['resolved']} predictions this run "
              f"({res['correct']} correct, {res['hit_rate']*100:.0f}% hit rate)")

    # Hit rates by horizon
    print("\n[HIT RATES BY HORIZON]")
    hr = hit_rates_by_horizon()
    if hr:
        for h, d in hr.items():
            print(f"  {h:<14s}: {d['n_correct']}/{d['n_observations']} "
                  f"({d['hit_rate']*100 if d['hit_rate'] else 0:.0f}% hit)")
    else:
        print("  No resolved predictions yet (need to wait for horizons to elapse)")

    # Top + bottom signals by hit rate
    print("\n[TOP SIGNALS BY HIT RATE]")
    sigs = hit_rates_by_signal(min_observations=5)
    if sigs:
        for name, d in list(sigs.items())[:10]:
            adaptive_w = adaptive_signal_weight(name)
            print(f"  {name:<28s}  {d['category']:<14s}  "
                  f"hit {d['hit_rate']*100:>4.0f}%  n={d['n_observations']:>3d}  "
                  f"weight {adaptive_w:.2f}x")
    else:
        print("  Need at least 10 scored observations per signal for ranking")

    # Anomalies
    anom = detect_signal_anomalies()
    if anom:
        print("\n[SIGNAL ANOMALIES — flagged regime change candidates]")
        for a in anom:
            print(f"  {a['signal']:<28s}  z={a['z_score']:+.2f}  "
                  f"recent {a['recent_mean']:+.2f} vs hist {a['historical_mean']:+.2f}")


if __name__ == "__main__":
    main()
