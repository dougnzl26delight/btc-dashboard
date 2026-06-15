"""Loss acceptance protocol — Douglas + Livermore.

Douglas: 'The loss only damages you when you don't take it. Once taken, accept it.'
Livermore: 'A loss never bothers me after I take it. I forget it overnight.'

The hardest psychological skill is NOT making structural changes during pain.
Most retail traders disable a sleeve after a bad week — then watch it print
money without them.

This module: when a sleeve realizes a loss > LOSS_THRESHOLD of allocation,
write a 48-hour cooldown lock. During cooldown:
    1. Sleeve continues to operate per its rules
    2. Any code/operator attempting to MODIFY the sleeve's parameters is
       blocked unless an operator_reason is provided + logged
    3. Manual position closes/adds get logged as compliance overrides

The lock is your psychological scaffolding. The CODE refuses to let you
override during pain. After 48 hours, the emotional charge has faded.

This is the Mark Douglas mechanical-stage discipline made literal.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LOCK_DIR = Path(__file__).resolve().parent.parent / ".cooldown_locks"
LOCK_DIR.mkdir(exist_ok=True)

LOSS_THRESHOLD_PCT = 0.01  # 1% loss triggers cooldown
COOLDOWN_HOURS = 48


def lock_path(sleeve: str) -> Path:
    return LOCK_DIR / f"{sleeve}.json"


def trigger_cooldown(sleeve: str, loss_pct: float, loss_usd: float, trigger_trade_id: int | None = None):
    """Write a cooldown lock for a sleeve after a significant loss."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
    lock_data = {
        "sleeve": sleeve,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expiry.isoformat(),
        "trigger_loss_pct": loss_pct,
        "trigger_loss_usd": loss_usd,
        "trigger_trade_id": trigger_trade_id,
        "douglas_reminder": (
            "Mark Douglas, Trading in the Zone: "
            "'The loss only damages you when you don't take it. "
            "Once taken, forget it. Continue executing your process.'"
        ),
        "livermore_reminder": (
            "Jesse Livermore: 'A loss never bothers me after I take it. "
            "I forget it overnight. Being wrong - not taking the loss - is what does damage.'"
        ),
    }
    lock_path(sleeve).write_text(json.dumps(lock_data, indent=2))
    try:
        from ops.alerts import alert
        alert(
            f"LOSS COOLDOWN: {sleeve} locked for {COOLDOWN_HOURS}h after "
            f"{loss_pct*100:.2f}% loss (${loss_usd:.2f}). Wait before adjusting.",
            level="warning",
        )
    except Exception:
        pass
    return lock_data


def is_locked(sleeve: str) -> tuple[bool, dict | None]:
    """Returns (is_locked, lock_data_or_None)."""
    p = lock_path(sleeve)
    if not p.exists():
        return False, None
    try:
        data = json.loads(p.read_text())
        expiry = datetime.fromisoformat(data["expires_at"])
        if datetime.now(timezone.utc) > expiry:
            return False, data  # expired
        return True, data
    except Exception:
        return False, None


def require_override(sleeve: str, action: str, operator_reason: str) -> bool:
    """Called when operator attempts to modify a locked sleeve.

    Returns True if override accepted (must include reason).
    Logs the override to compliance events.
    """
    locked, lock_data = is_locked(sleeve)
    if not locked:
        return True  # not locked, free to modify
    if not operator_reason or len(operator_reason) < 20:
        from ops.alerts import alert
        alert(
            f"BLOCKED: attempted to modify locked sleeve {sleeve} without reason. "
            f"Cooldown ends {lock_data['expires_at']}. "
            f"Operator must provide reason ≥20 chars to override.",
            level="critical",
        )
        return False
    # Override accepted — log it
    try:
        from ops.process_compliance import log_compliance_event
        log_compliance_event(
            event_type="manual_override",
            sleeve=sleeve,
            detail=f"Override on locked sleeve. Action: {action}",
            operator_reason=operator_reason,
        )
    except Exception:
        pass
    return True


def check_recent_trades_and_lock():
    """Daily check: scan recent pnl_db trades. Trigger cooldown for any sleeve
    with realized_pnl loss > 1% of allocated capital in last 24h."""
    try:
        import sqlite3
        DB = Path(__file__).resolve().parent.parent / ".pnl.db"
        c = sqlite3.connect(str(DB))
        c.row_factory = sqlite3.Row
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = c.execute(
            "SELECT sleeve, SUM(realized_pnl) AS total_pnl "
            "FROM trades WHERE ts > ? GROUP BY sleeve "
            "HAVING total_pnl < 0",
            (cutoff,),
        ).fetchall()
        # Per-sleeve allocations (approx, for threshold computation)
        from ops.sleeve_circuit_breakers import KNOWN_SLEEVES
        locks_triggered = []
        for r in rows:
            sleeve = r["sleeve"]
            pnl_usd = r["total_pnl"]
            allocation = KNOWN_SLEEVES.get(sleeve, {}).get("start_equity", 10_000)
            loss_pct = abs(pnl_usd) / allocation
            if loss_pct >= LOSS_THRESHOLD_PCT:
                already_locked, _ = is_locked(sleeve)
                if not already_locked:
                    trigger_cooldown(sleeve, loss_pct, pnl_usd)
                    locks_triggered.append(sleeve)
        c.close()
        return locks_triggered
    except Exception as e:
        return []


def status() -> dict:
    """All sleeves: locked / unlocked + expiry."""
    out = {}
    for p in LOCK_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            sleeve = p.stem
            locked, lock_data = is_locked(sleeve)
            out[sleeve] = {
                "locked": locked,
                "expires_at": data.get("expires_at"),
                "locked_at": data.get("locked_at"),
                "trigger_loss_pct": data.get("trigger_loss_pct"),
            }
        except Exception:
            pass
    return out


def main():
    print("=" * 80)
    print("LOSS ACCEPTANCE LOCKS — Douglas/Livermore discipline")
    print("=" * 80)
    print(f"Threshold: {LOSS_THRESHOLD_PCT*100}% sleeve allocation loss triggers cooldown")
    print(f"Cooldown duration: {COOLDOWN_HOURS} hours")
    print()
    locks = status()
    if not locks:
        print("No active cooldowns. All sleeves free to be modified.")
    else:
        for s, d in locks.items():
            state = "LOCKED" if d["locked"] else "expired"
            print(f"  {s:<22s} [{state:<7s}]  expires {d['expires_at'][:19]}  loss {d['trigger_loss_pct']*100:.2f}%")
    print()
    print("--- Daily auto-check for new cooldowns ---")
    triggered = check_recent_trades_and_lock()
    if triggered:
        print(f"Triggered new cooldowns: {triggered}")
    else:
        print("No new losses exceeding threshold in last 24h.")


if __name__ == "__main__":
    main()
