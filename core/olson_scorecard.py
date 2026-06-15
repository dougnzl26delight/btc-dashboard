"""Olson paid-signal scorecard — auto-log his calls and grade them honestly.

Thin wrapper around the generic engine in core.guru_scorecard. Olson was the
first guru graded this way; the engine was later generalized so Cowen (and any
other monitored handle) grade identically. This module preserves Olson's public
API + log path so the dashboard panel and the daily scorecard task are unchanged:

  1. capture each NEW directional call from his tweets (asset + bull/bear + the
     price when he said it),
  2. auto-grade it by the forward return over a fixed horizon (default 30d),
  3. report HIT-RATE + PAYOFF RATIO (avg win / avg loss = his "R") + EXPECTANCY.

Persisted to .olson_scorecard_log.json so the record GROWS over months. Small
samples mean little — the value is after 20-30 graded calls. NOT advice.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from core.guru_scorecard import (
    HORIZON_DAYS,
    update_and_grade as _engine_update,
    scorecard as _engine_scorecard,
)

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / ".olson_scorecard_log.json"

OLSON_CFG = {
    "handle":      "JesseOlson",
    "name":        "Jesse Olson",
    "seed_handle": "JesseOlson",
    "log":         LOG,
    "horizon_days": HORIZON_DAYS,
}


def update_and_grade(now: datetime | None = None) -> dict:
    """Capture new Olson calls, grade matured ones, persist. Returns scorecard."""
    return _engine_update(OLSON_CFG, now=now)


def olson_scorecard() -> dict:
    """Read-only aggregate of the current log (no capture/grade — for the dashboard)."""
    return _engine_scorecard(OLSON_CFG)


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    r = update_and_grade()
    print(f"VERDICT: {r['verdict']}")
    print(f"  logged={r['n_logged']} scored={r['n_scored']} pending={r['n_pending']} "
          f"auto-scored={r['n_auto_scored']}")
    print(f"  hit-rate={r['hit_rate_pct']}%  payoff(R)={r['payoff_R']}  "
          f"avg win={r['avg_win_pct']}%  avg loss={r['avg_loss_pct']}%  "
          f"expectancy={r['expectancy_pct']}%/call")
