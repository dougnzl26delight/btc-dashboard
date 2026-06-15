"""Intraday trail stop monitor for pro_trend positions.

The daily pro_trend cycle only runs once per day at 14:10 NZ. Between
cycles, prices can move 20%+ in crypto (especially during Asian-session
flash moves). This monitor runs every 15 minutes and:

  1. Reads each .pro_trend_state_{BASE}.json with open units
  2. Fetches current price + 24h high/low via REST
  3. Updates extreme + trail_stop in state file (ratchets up for longs)
  4. If price has BREACHED the trail_stop, closes the position immediately
     via the correct broker (perp for shorts and 1.5x longs, spot for 1x)

This mirrors the daily cycle's trail/exit logic but at higher cadence.
It does NOT make new entries — only manages existing positions.

Scheduled as Crypto_pro_trend_intraday_15min.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.broker import Broker
from core.perp_broker import PerpBroker
from core.pnl_attribution import untag
from ops.alerts import alert
from strategies import pro_trend


def _load_state(pair: str) -> dict | None:
    """Load state file for a pair; return None if missing/empty/corrupt."""
    f = REPO_ROOT / f".pro_trend_state_{pair.split('/')[0]}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _save_state(pair: str, state: dict) -> None:
    f = REPO_ROOT / f".pro_trend_state_{pair.split('/')[0]}.json"
    f.write_text(json.dumps(state, indent=2, default=str))


def _all_managed_pairs() -> list[str]:
    """Universe pairs + orphans (any state file with units)."""
    pairs = list(pro_trend.PRO_TREND_PAIRS)
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        base = f.stem.removeprefix(".pro_trend_state_")
        pair = f"{base}/USDT"
        if pair not in pairs:
            try:
                st = json.loads(f.read_text())
                if st.get("units"):
                    pairs.append(pair)
            except Exception:
                continue
    return pairs


def _get_intraday_high_low(pair: str) -> dict:
    """Best-effort fetch of current ticker + intraday range.

    ccxt fetch_ticker returns 24h rolling high/low which is good enough for
    detecting flash moves. For 5min granularity we'd need fetch_ohlcv with
    timeframe='5m' but that's heavier. 24h works for stop-monitoring.
    """
    t = data._EX.fetch_ticker(pair)
    return {
        "last": float(t.get("last") or t.get("close") or 0),
        "high24": float(t.get("high") or 0),
        "low24": float(t.get("low") or 0),
        "bid": float(t.get("bid") or t.get("last") or 0),
        "ask": float(t.get("ask") or t.get("last") or 0),
    }


FAST_MOVE_DROP_THRESHOLD = 0.15  # 15% drop in 24h = fast-move shutoff trigger


def check_pair(pair: str, mode: str = "paper") -> dict:
    """Manage one pair's stops intraday. No entries; only updates + exits."""
    state = _load_state(pair)
    if not state or not state.get("units"):
        return {"pair": pair, "status": "no_open_position"}

    side = state.get("side")
    units = state["units"]
    extreme = float(state.get("extreme", 0))
    trail_stop = float(state.get("trail_stop", 0))

    try:
        tk = _get_intraday_high_low(pair)
    except Exception as e:
        return {"pair": pair, "status": "fetch_failed", "error": str(e)}

    price = tk["last"]
    high = tk["high24"]
    low = tk["low24"]
    if price <= 0:
        return {"pair": pair, "status": "no_price"}

    # ATR-based stop distance — read from the units' entry_atr
    # Use the LAST unit's entry_atr as the reference (most recent regime)
    last_unit_atr = float(units[-1].get("entry_atr", 0))
    if last_unit_atr <= 0:
        return {"pair": pair, "status": "no_atr_in_state"}

    stop_dist = pro_trend.ATR_STOP_MULT * last_unit_atr
    actions = []

    # === MAY 2021 FAST-MOVE SHUTOFF ===
    # If a long position sees a >15% drop within 24h while still above its
    # trail stop, tighten the trail to current price. This protects against
    # flash crashes where the SMA200 hasn't caught up yet but the move is
    # clearly a regime break. Bidirectional analog applies to shorts.
    if side == "long" and high > 0:
        intraday_drop = (high - price) / high if high > price else 0
        if intraday_drop > FAST_MOVE_DROP_THRESHOLD and price > trail_stop:
            new_trail = price * 0.99  # 1% buffer below current to allow exit on bounce
            old_trail = trail_stop
            trail_stop = max(trail_stop, new_trail)
            actions.append({
                "action": "fast_move_tighten_long",
                "intraday_drop_pct": intraday_drop,
                "old_trail": old_trail, "new_trail": trail_stop,
            })
            alert(
                f"FAST-MOVE SHUTOFF (LONG) {pair}: 24h drop {intraday_drop:.1%}, "
                f"trail tightened from ${old_trail:,.4f} to ${trail_stop:,.4f}",
                level="warning",
            )
    elif side == "short" and low > 0:
        intraday_pump = (price - low) / low if price > low else 0
        if intraday_pump > FAST_MOVE_DROP_THRESHOLD and price < trail_stop:
            new_trail = price * 1.01
            old_trail = trail_stop
            trail_stop = min(trail_stop, new_trail) if trail_stop > 0 else new_trail
            actions.append({
                "action": "fast_move_tighten_short",
                "intraday_pump_pct": intraday_pump,
                "old_trail": old_trail, "new_trail": trail_stop,
            })
            alert(
                f"FAST-MOVE SHUTOFF (SHORT) {pair}: 24h pump {intraday_pump:.1%}, "
                f"trail tightened from ${old_trail:,.4f} to ${trail_stop:,.4f}",
                level="warning",
            )

    # === LONG ===
    if side == "long":
        # Update extreme + trail upward
        if high > extreme:
            extreme = high
            new_trail = high - stop_dist
            if new_trail > trail_stop:
                trail_stop = new_trail
                actions.append({"action": "trail_updated", "new_trail": trail_stop})

        # Check breach — use intraday LOW vs trail
        if low <= trail_stop:
            close_qty = sum(u["qty"] for u in units)
            close_price = trail_stop  # assume filled at trail level
            try:
                # Long routes through perp if 1.5x, else spot
                if pro_trend.LEVERAGE_MULTIPLIER > 1.0:
                    perp = PerpBroker(mode=mode)
                    perp.close_position(pair)
                else:
                    spot = Broker(mode=mode, long_only=False)
                    spot.place_market_order(pair, "sell", close_qty * close_price)
                actions.append({
                    "action": "intraday_exit_long", "reason": "trail_hit_intraday",
                    "trail_stop": trail_stop, "close_price": close_price,
                    "n_units": len(units),
                })
                alert(
                    f"INTRADAY TRAIL HIT (LONG) {pair}: stop ${trail_stop:,.4f}, "
                    f"low ${low:,.4f}, closed {len(units)} units",
                    level="trade",
                )
                untag(pair)
                _save_state(pair, {
                    "side": None, "units": [], "extreme": 0, "trail_stop": 0,
                    "peak_equity": state.get("peak_equity", 100_000),
                })
                return {"pair": pair, "status": "exited_intraday", "actions": actions}
            except Exception as e:
                alert(f"INTRADAY EXIT FAILED {pair}: {e}", level="critical")
                actions.append({"action": "exit_failed", "error": str(e)})

    # === SHORT ===
    elif side == "short":
        # Update extreme + trail downward
        if extreme == 0 or low < extreme:
            extreme = low
            new_trail = low + stop_dist
            if trail_stop == 0 or new_trail < trail_stop:
                trail_stop = new_trail
                actions.append({"action": "trail_updated", "new_trail": trail_stop})

        # Check breach — intraday HIGH vs trail
        if high >= trail_stop:
            try:
                perp = PerpBroker(mode=mode)
                close_result = perp.close_position(pair)
                actions.append({
                    "action": "intraday_exit_short", "reason": "trail_hit_intraday",
                    "trail_stop": trail_stop, "high": high,
                    "n_units": len(units),
                    "realized": close_result.get("realized_pnl", 0),
                })
                alert(
                    f"INTRADAY TRAIL HIT (SHORT) {pair}: stop ${trail_stop:,.4f}, "
                    f"high ${high:,.4f}, closed {len(units)} units, "
                    f"realized ${close_result.get('realized_pnl', 0):+,.2f}",
                    level="trade",
                )
                untag(pair)
                _save_state(pair, {
                    "side": None, "units": [], "extreme": 0, "trail_stop": 0,
                    "peak_equity": state.get("peak_equity", 100_000),
                })
                return {"pair": pair, "status": "exited_intraday", "actions": actions}
            except Exception as e:
                alert(f"INTRADAY EXIT FAILED {pair}: {e}", level="critical")
                actions.append({"action": "exit_failed", "error": str(e)})

    # No exit fired — save updated extreme/trail
    state["extreme"] = extreme
    state["trail_stop"] = trail_stop
    _save_state(pair, state)

    return {
        "pair": pair, "status": "managed",
        "side": side, "price": price,
        "trail_stop": trail_stop, "extreme": extreme,
        "actions": actions,
    }


if __name__ == "__main__":
    import json as _j
    print(f"Pro-trend intraday monitor — {datetime.now(timezone.utc).isoformat()}")
    pairs = _all_managed_pairs()
    print(f"Pairs to check: {pairs}")
    print()
    for pair in pairs:
        r = check_pair(pair)
        print(f"  {pair:<12s} {r['status']:<20s}  "
              f"trail={r.get('trail_stop','-')}  "
              f"actions={len(r.get('actions',[]))}")
        for a in r.get("actions", []):
            print(f"      -> {a}")
