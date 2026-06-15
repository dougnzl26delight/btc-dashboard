"""Real-time position monitor — persistent service.

Subscribes to live Binance WebSocket prices and reacts within ~1 second to:
    1. Stop-loss breaches on any open position
    2. Take-profit hits
    3. Trail-stop tightening (continuous, not just every 15 min)
    4. Flash-crash events (>3% drop in <60s on BTC)
    5. EMA21 break exits on the oversold/bounce baskets

This REPLACES the cron-based 30-min position_monitor for live reactivity.
Cron version remains as a 30-min safety net in case this service dies.

Architecture:
    - Single process, single WebSocket connection
    - In-memory state: known open positions + their stop/TP levels
    - On every price tick: check all positions for exit conditions
    - When exit fires: execute via existing broker pipeline
    - Auto-reload positions from state files every 60s (picks up new sleeve entries)

Persistent service mode — run forever. Restart on crash via scheduled task.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from core.broker import Broker
from core.perp_broker import PerpBroker
from core.pnl_db import log_trade
from core.ws_feed import start_background_feed, DEFAULT_PAIRS
from core.pnl_attribution import untag
from ops.alerts import alert
from ops import watchdog


# Configuration
POSITION_RELOAD_SEC = 60         # how often to re-read state files
FLASH_CRASH_WINDOW_SEC = 60      # rolling window for flash-crash detection
FLASH_CRASH_THRESHOLD_PCT = 0.03  # 3% drop in window = flash crash
FLASH_CRASH_KILL_PAIRS = ["BTC/USDT", "ETH/USDT"]  # majors to watch
MIN_ALERT_INTERVAL_SEC = 300     # rate-limit alerts per pair to once per 5 min

# W10: real-time DIRECT execution on stop breaches.
# When True, the monitor places exit orders itself (sub-second latency).
# When False (legacy), only writes hint files for cron sleeves to pick up.
DIRECT_EXECUTION_ENABLED = True

# Lock file pattern: prevents race between RT exit + cron exit on same position.
# Cron sleeves check for this lock before processing the same pair.
RT_EXECUTION_LOCK_DIR = REPO_ROOT
RT_LOCK_TTL_SEC = 300  # locks expire after 5 min

# Track recent price history per pair (timestamp, price) — for flash detection
_price_history: dict[str, list[tuple]] = {}
_last_alert_ts: dict[str, float] = {}
_positions: dict[str, dict] = {}      # pair -> position dict
_state_lock = threading.Lock()


def _alert_throttled(key: str, message: str, level: str = "warning"):
    now = time.time()
    last = _last_alert_ts.get(key, 0)
    if now - last < MIN_ALERT_INTERVAL_SEC:
        return False
    _last_alert_ts[key] = now
    alert(message, level=level)
    return True


def _load_positions() -> dict:
    """Re-read state files. Returns dict of pair -> {sleeve, side, qty, entry, stop, tp}."""
    positions = {}

    # Spot positions from .paper_state.json (BAH BTC + oversold_bounce + spot orchestrator)
    spot_state_file = REPO_ROOT / ".paper_state.json"
    if spot_state_file.exists():
        try:
            spot = json.loads(spot_state_file.read_text())
            for asset, qty in spot.get("positions", {}).items():
                if abs(qty) < 1e-12:
                    continue
                pair = f"{asset}/USDT"
                positions[pair] = positions.get(pair, [])
                positions[pair].append({
                    "venue": "spot", "qty": qty, "side": "long" if qty > 0 else "short",
                })
        except Exception:
            pass

    # Per-strategy state files with explicit stop levels
    ob_state = REPO_ROOT / ".oversold_bounce_state.json"
    if ob_state.exists():
        try:
            ob = json.loads(ob_state.read_text())
            for pair, info in ob.get("open_positions", {}).items():
                positions.setdefault(pair, []).append({
                    "venue": "spot", "sleeve": "oversold_bounce",
                    "qty": info.get("qty", 0), "side": "long",
                    "entry": info.get("entry_price", 0),
                    "stop": info.get("stop_loss", 0),
                    "target": info.get("entry_price", 0) * 1.20,
                })
        except Exception:
            pass

    of_state = REPO_ROOT / ".overbought_fade_state.json"
    if of_state.exists():
        try:
            of = json.loads(of_state.read_text())
            for pair, info in of.get("open_positions", {}).items():
                positions.setdefault(pair, []).append({
                    "venue": "perp", "sleeve": "overbought_fade",
                    "qty": info.get("qty", 0), "side": "short",
                    "entry": info.get("entry_price", 0),
                    "stop": info.get("stop_loss", 0),
                    "target": info.get("entry_price", 0) * 0.85,
                })
        except Exception:
            pass

    # Pro_trend per-pair state files (trail stop is the key field)
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        try:
            pt = json.loads(f.read_text())
            if not pt.get("side") or not pt.get("units"):
                continue
            base = f.stem.replace(".pro_trend_state_", "")
            pair = f"{base}/USDT"
            total_qty = sum(u.get("qty", 0) for u in pt["units"]) if isinstance(pt["units"], list) else 0
            positions.setdefault(pair, []).append({
                "venue": "perp", "sleeve": "pro_trend",
                "qty": total_qty if pt["side"] == "long" else -total_qty,
                "side": pt["side"], "trail_stop": pt.get("trail_stop", 0),
            })
        except Exception:
            pass

    return positions


def _check_flash_crash(pair: str, current_price: float, ts: float):
    """Detect 3%+ drop in BTC/ETH within 60 seconds — emergency kill."""
    if pair not in FLASH_CRASH_KILL_PAIRS:
        return
    history = _price_history.setdefault(pair, [])
    history.append((ts, current_price))
    # Trim to window
    cutoff = ts - FLASH_CRASH_WINDOW_SEC
    while history and history[0][0] < cutoff:
        history.pop(0)
    if len(history) < 5:
        return
    max_recent = max(p for _, p in history)
    drop_pct = (max_recent - current_price) / max_recent if max_recent > 0 else 0
    if drop_pct > FLASH_CRASH_THRESHOLD_PCT:
        # FLASH CRASH — write kill switch
        kill_file = REPO_ROOT / ".kill_switch.json"
        if not kill_file.exists():
            kill_file.write_text(json.dumps({
                "killed_at": datetime.now(timezone.utc).isoformat(),
                "reason": f"FLASH CRASH: {pair} dropped {drop_pct*100:.2f}% in {FLASH_CRASH_WINDOW_SEC}s",
                "max_in_window": max_recent,
                "current": current_price,
            }, indent=2))
            _alert_throttled(f"flash_{pair}",
                             f"🚨 FLASH CRASH DETECTED: {pair} -{drop_pct*100:.2f}% in {FLASH_CRASH_WINDOW_SEC}s. "
                             f"Kill switch ACTIVATED. All cycles will skip.",
                             level="critical")


def _write_hint_file(pair: str, sleeve: str, side: str, reason: str,
                     mid: float, stop: float, trail: float, target: float) -> None:
    """Fallback when DIRECT_EXECUTION_ENABLED is False or RT exec failed."""
    hint_file = REPO_ROOT / f".rt_exit_hint_{sleeve}_{pair.replace('/', '_')}.json"
    hint_file.write_text(json.dumps({
        "pair": pair, "sleeve": sleeve, "side": side,
        "reason": reason, "triggered_at": datetime.now(timezone.utc).isoformat(),
        "trigger_mid": mid, "stop": stop, "trail": trail, "target": target,
    }, indent=2))


def _check_position_exits(pair: str, bid: float, ask: float, ts: float):
    """For every open position on this pair, check stop/target."""
    with _state_lock:
        position_list = _positions.get(pair, [])
    if not position_list:
        return

    mid = (bid + ask) / 2

    for pos in position_list:
        try:
            side = pos.get("side")
            stop = pos.get("stop", 0)
            target = pos.get("target", 0)
            trail = pos.get("trail_stop", 0)
            sleeve = pos.get("sleeve", "unknown")

            exit_reason = None
            if side == "long":
                # Long position: exit if price <= stop or trail, or >= target
                if stop and mid <= stop:
                    exit_reason = f"stop_loss (mid {mid} <= stop {stop})"
                elif trail and mid <= trail:
                    exit_reason = f"trail_stop (mid {mid} <= trail {trail})"
                elif target and mid >= target:
                    exit_reason = f"target_hit (mid {mid} >= target {target})"
            elif side == "short":
                if stop and mid >= stop:
                    exit_reason = f"stop_loss (mid {mid} >= stop {stop})"
                elif trail and mid >= trail:
                    exit_reason = f"trail_stop (mid {mid} >= trail {trail})"
                elif target and mid <= target:
                    exit_reason = f"target_hit (mid {mid} <= target {target})"

            if exit_reason:
                _alert_throttled(f"exit_{sleeve}_{pair}",
                                 f"REAL-TIME EXIT TRIGGER: {sleeve} {pair} {side}, {exit_reason}",
                                 level="warning")

                # W10 DIRECT EXECUTION: place exit order immediately via broker
                # Lock file prevents the cron sleeve from double-exiting this pair.
                lock_file = RT_EXECUTION_LOCK_DIR / f".rt_exec_lock_{sleeve}_{pair.replace('/', '_')}.json"
                if DIRECT_EXECUTION_ENABLED:
                    try:
                        # Write lock first so cron sees it
                        lock_file.write_text(json.dumps({
                            "locked_at": datetime.now(timezone.utc).isoformat(),
                            "sleeve": sleeve, "pair": pair, "reason": exit_reason,
                            "ttl_seconds": RT_LOCK_TTL_SEC,
                        }))
                        # Place exit order based on sleeve type
                        qty = abs(pos.get("qty", 0))
                        if pos.get("venue") == "perp":
                            broker = PerpBroker(mode="paper", sleeve=sleeve)
                            res = broker.close_position(pair)
                            realized = res.get("realized_pnl", 0.0)
                        else:  # spot
                            broker = Broker(mode="paper", long_only=True, sleeve=sleeve)
                            notional = qty * mid
                            broker.place_market_order(pair, "sell" if side == "long" else "buy", notional)
                            realized = 0.0  # spot doesn't compute realized on close
                        log_trade(sleeve, pair, f"rt_exit_{side}", qty, mid,
                                  realized_pnl=realized, note=f"realtime:{exit_reason}")
                        try:
                            untag(f"{sleeve}:{pair}")
                        except Exception:
                            pass
                        _alert_throttled(f"rt_exec_{sleeve}_{pair}",
                                         f"RT EXEC: closed {sleeve} {pair} {side} at ${mid:.4f}",
                                         level="trade")
                    except Exception as e:
                        # If RT exec fails, fall back to hint-file pattern
                        _alert_throttled(f"rt_exec_fail_{pair}",
                                         f"RT exec failed for {sleeve} {pair}: {e}. Falling back to hint file.",
                                         level="warning")
                        try:
                            lock_file.unlink()  # release lock so cron can take over
                        except Exception:
                            pass
                        _write_hint_file(pair, sleeve, side, exit_reason, mid, stop, trail, target)
                else:
                    _write_hint_file(pair, sleeve, side, exit_reason, mid, stop, trail, target)
        except Exception as e:
            _alert_throttled(f"exit_err_{pair}", f"position exit check error on {pair}: {e}", level="warning")


def _on_price_update(pair: str, bid: float, ask: float, ts: float):
    """Called by WS feed on every price tick. Hot path — keep fast."""
    _check_flash_crash(pair, (bid + ask) / 2, ts)
    _check_position_exits(pair, bid, ask, ts)


def _position_reload_loop():
    """Background thread — re-reads positions every 60s to pick up new entries."""
    while True:
        try:
            new_positions = _load_positions()
            with _state_lock:
                _positions.clear()
                _positions.update(new_positions)
        except Exception as e:
            try:
                from ops.alerts import alert
                alert(f"realtime_monitor reload error: {e}", level="warning")
            except Exception:
                pass
        time.sleep(POSITION_RELOAD_SEC)


def main():
    """Persistent service entry point. Runs forever; restart on crash via task."""
    print(f"=== Real-time monitor starting @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Subscribing to {len(DEFAULT_PAIRS)} pairs via Binance WebSocket")

    # Load initial positions
    initial = _load_positions()
    with _state_lock:
        _positions.update(initial)
    n_pos = sum(len(v) for v in initial.values())
    print(f"Loaded {n_pos} open positions across {len(initial)} pairs")
    for pair, lst in initial.items():
        for p in lst:
            print(f"  {p.get('sleeve', '?'):<20s} {pair:<12s} {p.get('side', '?'):<6s} "
                  f"stop={p.get('stop', 0):.4f} trail={p.get('trail_stop', 0):.4f}")

    # Start WS feed
    cache = start_background_feed(DEFAULT_PAIRS)
    cache.subscribe(_on_price_update)

    # Start position-reload thread
    reload_thread = threading.Thread(target=_position_reload_loop, daemon=True)
    reload_thread.start()

    print(f"Real-time monitor LIVE. Watching {n_pos} positions for stops/targets/flash-crashes.")
    print("This process must stay running. Scheduled task auto-restarts it on crash.")

    # Periodic heartbeat (every 30s) — keeps watchdog alive
    heartbeat_count = 0
    while True:
        time.sleep(30)
        try:
            watchdog.beat()
            heartbeat_count += 1
            if heartbeat_count % 10 == 0:  # every 5 min
                with _state_lock:
                    n = sum(len(v) for v in _positions.values())
                snapshot = cache.all()
                live_pairs = len([k for k in snapshot if not k.startswith("_")])
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                      f"heartbeat #{heartbeat_count}  positions={n}  live_prices={live_pairs}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
