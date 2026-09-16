"""Real-time execution locks — shared between realtime_monitor and the cron sleeves.

The realtime_monitor service closes a position within ~1s of a stop breach.
The cron sleeve runners are the 30-minute safety net over the same positions.
Without coordination both sides can close the same position (double exit).

Whoever executes first writes a lock; the other side skips that pair until the
lock expires. Locks are TTL-bounded so a crashed writer cannot wedge a pair
permanently — the safety net must always come back on its own.

Before this module the locks were written by realtime_monitor and read by
nobody, and the TTL was never enforced (stale locks survived for months).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LOCK_PREFIX = ".rt_exec_lock_"
DEFAULT_TTL_SEC = 300


def _slug(pair: str) -> str:
    return pair.replace("/", "_")


def lock_path(sleeve: str, pair: str) -> Path:
    return REPO_ROOT / f"{LOCK_PREFIX}{sleeve}_{_slug(pair)}.json"


def _expired(path: Path, now: float | None = None) -> bool:
    """True if the lock is missing, unreadable, or past its TTL.

    An unreadable lock is treated as expired: a corrupt file must not be able
    to block the cron safety net forever.
    """
    now = now if now is not None else time.time()
    try:
        data = json.loads(path.read_text())
        locked_at = datetime.fromisoformat(data["locked_at"]).timestamp()
        ttl = float(data.get("ttl_seconds", DEFAULT_TTL_SEC))
    except Exception:
        return True
    return (now - locked_at) > ttl


def acquire(sleeve: str, pair: str, reason: str = "", ttl: int = DEFAULT_TTL_SEC) -> Path:
    """Claim the pair for this executor. Overwrites an existing (possibly stale) lock."""
    path = lock_path(sleeve, pair)
    path.write_text(json.dumps({
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "sleeve": sleeve,
        "pair": pair,
        "reason": reason,
        "ttl_seconds": ttl,
    }))
    return path


def release(sleeve: str, pair: str) -> None:
    """Drop the lock so the other side can take the pair immediately."""
    try:
        lock_path(sleeve, pair).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def is_active(sleeve: str, pair: str) -> bool:
    """True if a live (non-expired) lock exists. Expired locks are cleaned up here."""
    path = lock_path(sleeve, pair)
    if not path.exists():
        return False
    if _expired(path):
        try:
            path.unlink()
        except Exception:
            pass
        return False
    return True


def any_active(pair: str) -> bool:
    """True if ANY sleeve holds a live lock on this pair."""
    slug = _slug(pair)
    for path in REPO_ROOT.glob(f"{LOCK_PREFIX}*_{slug}.json"):
        if _expired(path):
            try:
                path.unlink()
            except Exception:
                pass
            continue
        return True
    return False


def sweep() -> int:
    """Delete every expired lock. Returns how many were removed."""
    removed = 0
    now = time.time()
    for path in REPO_ROOT.glob(f"{LOCK_PREFIX}*.json"):
        if _expired(path, now):
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed
