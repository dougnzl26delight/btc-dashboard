"""Pro trend follower for BTC — state machine for live execution.

Holds state across cycles in .pro_trend_state.json:
  - List of currently-open units (each: qty, entry_price, entry_atr)
  - Current trailing stop level
  - High-water mark (highest high since entry)

Each cycle:
  1. Load state + fetch fresh OHLCV
  2. Compute SMA200, ATR(14), Donchian-20-high
  3. Decide: entry / pyramid / exit / hold
  4. Execute via spot Broker
  5. Save state

This is a real-time wrapper around the backtest logic in core/pro_trend_backtest.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.broker import Broker
from core.swing_backtest import compute_atr


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".pro_trend_state.json"

PAIR = "BTC/USDT"

# Production parameters (best from walk-forward: 4 ATR / 2 pyramid / 2% risk)
SMA_FILTER = 200
DONCHIAN_WINDOW = 20
ATR_PERIOD = 14
ATR_STOP_MULT = 4.0
RISK_PCT_PER_UNIT = 0.02
PYRAMID_ATR_STEP = 2.0
MAX_PYRAMID_UNITS = 2
DRAWDOWN_KILL_PCT = 0.25
ROUND_TRIP_BPS = 30


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"units": [], "high_water": 0.0, "trail_stop": 0.0, "peak_equity": 100_000.0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def reset_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def cycle(mode: str = "paper") -> dict:
    """One pro-trend cycle. Idempotent — call from a scheduled task."""
    df = data.ohlcv_extended(PAIR, days_back=400)
    if df.empty or len(df) < SMA_FILTER + 10:
        return {"status": "insufficient_data"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(DONCHIAN_WINDOW).max().shift(1)
    df["sma_filter"] = df["close"].rolling(SMA_FILTER).mean()
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df = df.dropna()

    last = df.iloc[-1]
    price = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    atr = float(last["atr"])
    sma = float(last["sma_filter"])
    donchian = float(last["donchian_high"])

    state = load_state()
    units = state.get("units", [])
    high_water = float(state.get("high_water", 0))
    trail_stop = float(state.get("trail_stop", 0))
    peak_equity = float(state.get("peak_equity", 100_000))

    broker = Broker(mode=mode, long_only=False)
    cash = float(broker.get_balance().get("USDT", 0))

    # Mark-to-market
    pos_qty = sum(u["qty"] for u in units)
    unrealized = pos_qty * price - sum(u["qty"] * u["entry_price"] for u in units)
    mtm_eq = cash + pos_qty * price
    if mtm_eq > peak_equity:
        peak_equity = mtm_eq

    actions = []
    equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

    # === KILL SWITCH ===
    if equity_dd > DRAWDOWN_KILL_PCT and units:
        for u in units:
            broker.place_market_order(PAIR, "sell", u["qty"] * price)
        actions.append({"action": "dd_kill", "n_units_closed": len(units)})
        units = []
        high_water = trail_stop = 0

    # === EXIT CHECKS ===
    if units:
        if high > high_water:
            high_water = high
            new_trail = high - ATR_STOP_MULT * atr
            if new_trail > trail_stop:
                trail_stop = new_trail

        stop_hit = low <= trail_stop
        sma_break = price < sma

        if stop_hit or sma_break:
            for u in units:
                broker.place_market_order(PAIR, "sell", u["qty"] * price)
            actions.append({
                "action": "exit",
                "reason": "trail_stop" if stop_hit else "sma_break",
                "n_units_closed": len(units),
                "exit_price": price,
            })
            units = []
            high_water = trail_stop = 0

    # === PYRAMID CHECK ===
    if units and len(units) < MAX_PYRAMID_UNITS:
        last_unit = units[-1]
        if high >= last_unit["entry_price"] + PYRAMID_ATR_STEP * last_unit["entry_atr"]:
            stop_dist = ATR_STOP_MULT * atr
            if stop_dist > 0:
                qty = (mtm_eq * RISK_PCT_PER_UNIT) / stop_dist
                qty = min(qty, mtm_eq * 0.25 / price)
                quote_amount = qty * price
                broker.place_market_order(PAIR, "buy", quote_amount)
                units.append({
                    "qty": qty,
                    "entry_price": price,
                    "entry_atr": atr,
                })
                actions.append({"action": "pyramid", "qty": qty, "entry": price})

    # === ENTRY CHECK ===
    if not units and price > sma and high >= donchian and atr > 0:
        stop_dist = ATR_STOP_MULT * atr
        qty = (mtm_eq * RISK_PCT_PER_UNIT) / stop_dist
        qty = min(qty, mtm_eq * 0.25 / price)
        quote_amount = qty * price
        broker.place_market_order(PAIR, "buy", quote_amount)
        units = [{"qty": qty, "entry_price": price, "entry_atr": atr}]
        high_water = high
        trail_stop = price - stop_dist
        actions.append({
            "action": "entry",
            "qty": qty, "entry": price,
            "stop": trail_stop, "stop_distance_pct": stop_dist / price,
        })

    save_state({
        "units": units,
        "high_water": high_water,
        "trail_stop": trail_stop,
        "peak_equity": peak_equity,
    })

    return {
        "status": "ok",
        "price": price,
        "sma": sma,
        "donchian": donchian,
        "atr": atr,
        "n_units_open": len(units),
        "trail_stop": trail_stop,
        "high_water": high_water,
        "equity_dd": equity_dd,
        "actions": actions,
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(cycle(), indent=2, default=str))
