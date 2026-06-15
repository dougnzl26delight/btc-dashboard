"""Real-time kill switch — flash-crash protection.

The daily DD-kill in pro_trend.cycle() is end-of-day. A flash crash
during Asian session (e.g., May 2021 BTC -30% in 2h, COVID March 2020
-50% in 24h, FTX collapse Nov 2022) could blow through stops with no
intervention until next daily cycle.

This monitor runs every 5 minutes and:

  1. Logs current portfolio MTM to .equity_realtime_log.jsonl
  2. Computes velocity over rolling windows (10min, 60min, 24h)
  3. If ANY of the kill triggers fire, FLATTEN EVERYTHING and lock out
     re-entry for 24h via .kill_switch_lock.json

Kill triggers (any one fires):
  RT1: -5% in 10 min     (extreme intraday move)
  RT2: -8% in 60 min     (sustained crash)
  RT3: -15% in 24h       (catastrophic; backstop for missed intraday)

Lockout: writes timestamp to .kill_switch_lock.json. Daily cycle reads
this file at start; if lockout active, returns immediately without
making new entries. Manual unlock: delete the file.

Scheduled as Crypto_realtime_kill_switch_5min.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.broker import Broker
from core.perp_broker import PerpBroker
from core.pnl_attribution import load_attribution, untag
from ops.alerts import alert


REALTIME_LOG = REPO_ROOT / ".equity_realtime_log.jsonl"
LOCK_FILE = REPO_ROOT / ".kill_switch_lock.json"

# Kill triggers — drop magnitudes (positive = drop)
TRIGGER_10MIN_PCT = 0.05    # 5% drop in 10 min
TRIGGER_60MIN_PCT = 0.08    # 8% drop in 60 min
TRIGGER_24H_PCT = 0.15      # 15% drop in 24h

LOCKOUT_HOURS = 24


def is_locked() -> tuple[bool, str | None]:
    if not LOCK_FILE.exists():
        return False, None
    try:
        data = json.loads(LOCK_FILE.read_text())
        locked_until = datetime.fromisoformat(data["locked_until"])
        if datetime.now(timezone.utc) < locked_until:
            return True, data.get("reason", "unspecified")
        return False, None
    except Exception:
        return False, None


def write_lock(reason: str) -> None:
    locked_until = (datetime.now(timezone.utc)
                    + timedelta(hours=LOCKOUT_HOURS)).isoformat()
    LOCK_FILE.write_text(json.dumps({
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "locked_until": locked_until,
        "reason": reason,
        "lockout_hours": LOCKOUT_HOURS,
    }, indent=2))


def current_mtm(mode: str = "paper") -> dict:
    """Snapshot total equity across spot + perp + open positions."""
    spot = Broker(mode=mode, long_only=False)
    perp = PerpBroker(mode=mode)

    spot_cash = float(spot.get_balance().get("USDT", 0))
    perp_balance = perp.get_balance()
    perp_cash = float(perp_balance.get("USDT", 0))

    attrib = load_attribution()
    position_value = 0.0
    for pair, tag in attrib.items():
        if pair.startswith("basis:"):
            continue
        try:
            ticker = (perp.ticker(pair) if tag.get("side") == "short"
                      else spot.get_ticker(pair))
            last = float(ticker.get("last") or ticker.get("close") or 0)
        except Exception:
            last = float(tag["entry_price"])
        if last <= 0:
            last = float(tag["entry_price"])
        sign = 1 if tag["side"] == "long" else -1
        pnl = sign * tag["qty"] * (last - tag["entry_price"])
        position_value += pnl
        if tag["side"] == "long":
            position_value += tag["qty"] * tag["entry_price"]  # add base notional

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_mtm": spot_cash + perp_cash + position_value,
    }


def append_realtime_log(snapshot: dict) -> None:
    """Append; trim file if it gets too big (keep last 24h ≈ 288 entries)."""
    with REALTIME_LOG.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    # Trim if oversized — keep last 1000 lines (~3.5 days at 5min cadence)
    try:
        lines = REALTIME_LOG.read_text().strip().split("\n")
        if len(lines) > 1000:
            REALTIME_LOG.write_text("\n".join(lines[-1000:]) + "\n")
    except Exception:
        pass


def load_history() -> list[dict]:
    if not REALTIME_LOG.exists():
        return []
    try:
        return [json.loads(line)
                for line in REALTIME_LOG.read_text().strip().split("\n") if line]
    except Exception:
        return []


def check_velocity(history: list[dict], current_mtm: float) -> list[dict]:
    """Return list of triggered kill criteria."""
    if not history:
        return []
    triggers = []
    now = datetime.now(timezone.utc)

    def pct_drop_since(minutes: int, threshold: float, label: str):
        cutoff = now - timedelta(minutes=minutes)
        old = None
        for snap in reversed(history):
            ts = datetime.fromisoformat(snap["ts"])
            if ts <= cutoff:
                old = float(snap["total_mtm"])
                break
        if old is None:
            return None
        drop = (old - current_mtm) / old if old > 0 else 0
        if drop >= threshold:
            return {
                "id": label, "drop_pct": drop, "window_min": minutes,
                "old_mtm": old, "current_mtm": current_mtm,
                "reason": f"{drop:.1%} drop in {minutes}min ({label}: threshold {threshold:.0%})",
            }
        return None

    for trig in [
        pct_drop_since(10, TRIGGER_10MIN_PCT, "RT1"),
        pct_drop_since(60, TRIGGER_60MIN_PCT, "RT2"),
        pct_drop_since(60 * 24, TRIGGER_24H_PCT, "RT3"),
    ]:
        if trig:
            triggers.append(trig)

    return triggers


def flatten_all(mode: str = "paper") -> dict:
    """Close every open position via correct broker."""
    spot = Broker(mode=mode, long_only=False)
    perp = PerpBroker(mode=mode)
    attrib = load_attribution()

    closed = []
    for pair, tag in list(attrib.items()):
        if pair.startswith("basis:"):
            continue
        try:
            if tag["side"] == "long":
                # Try perp first (1.5x routing), fall back to spot
                try:
                    perp.close_position(pair)
                    closed.append({"pair": pair, "side": "long", "broker": "perp"})
                except Exception:
                    last = float(spot.get_ticker(pair).get("last", 0))
                    if last > 0:
                        spot.place_market_order(pair, "sell", tag["qty"] * last)
                        closed.append({"pair": pair, "side": "long", "broker": "spot"})
            else:
                perp.close_position(pair)
                closed.append({"pair": pair, "side": "short", "broker": "perp"})
            untag(pair)

            # Wipe pro_trend state file too
            f = REPO_ROOT / f".pro_trend_state_{pair.split('/')[0]}.json"
            if f.exists():
                f.write_text(json.dumps({
                    "side": None, "units": [], "extreme": 0, "trail_stop": 0,
                    "peak_equity": 100_000,
                }))
        except Exception as e:
            alert(f"FLATTEN FAILED {pair}: {e}", level="critical")
    return {"closed": closed, "n_closed": len(closed)}


def main(mode: str = "paper") -> dict:
    # Lockout already active? Just log and skip.
    locked, lock_reason = is_locked()
    if locked:
        return {"status": "locked", "lock_reason": lock_reason}

    snapshot = current_mtm(mode=mode)
    append_realtime_log(snapshot)

    history = load_history()
    triggers = check_velocity(history, snapshot["total_mtm"])

    if triggers:
        worst = max(triggers, key=lambda t: t["drop_pct"])
        alert(
            f"FLASH CRASH KILL SWITCH FIRED — {worst['reason']}. "
            f"Flattening all positions. Lockout {LOCKOUT_HOURS}h.",
            level="critical",
        )
        flatten = flatten_all(mode=mode)
        write_lock(f"{worst['id']}: {worst['reason']}")
        return {
            "status": "FIRED",
            "triggers": triggers,
            "flattened": flatten,
            "lockout_until": (datetime.now(timezone.utc)
                              + timedelta(hours=LOCKOUT_HOURS)).isoformat(),
        }

    return {
        "status": "ok",
        "current_mtm": snapshot["total_mtm"],
        "n_history_points": len(history),
    }


if __name__ == "__main__":
    print(f"Realtime kill switch — {datetime.now(timezone.utc).isoformat()}")
    result = main()
    print(json.dumps(result, indent=2, default=str))
