"""Grid trader — continuous oscillation capture on BTC and ETH.

The most active sleeve in the rig. Doesn't care about trend or regime —
profits from price oscillation within a range. Fires multiple trades per day.

Mechanics:
    - Define grid range around current price (e.g., +/- 5%)
    - Place 5 buy levels below current, 5 sell levels above
    - When price hits a level, the trade fills
    - Buy fills get matched with sell orders at next level UP (profit)
    - Sell fills get matched with buy orders at next level DOWN (re-entry)
    - Each "grid step" earns the spread between levels

Edge source:
    Crypto volatility creates constant oscillation. A 5% range with 10
    levels = ~0.5% per step. At 10 fills/day = ~5% raw return (before costs).
    After 30bps round-trip costs per fill: ~0.2% per fill = ~2%/day.

Risk:
    - Strong trend breakout = grid moves entirely to one side, large loss
    - Mitigation: re-center grid every 24h if price moves out of range
    - Mitigation: regime gate — pause if BTC volatility expands beyond threshold

Allocation: $5k per pair, max 2 pairs (BTC, ETH) = $10k sleeve total.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.broker import Broker
from core.pnl_db import log_trade, log_signal
from core.pnl_attribution import tag_entry, untag
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_all_gates_scale


NAME = "grid_trader"
STATE_FILE = REPO_ROOT / ".grid_trader_state.json"

PAIRS = ["BTC/USDT", "ETH/USDT"]
SLEEVE_BASELINE = 10_000.0
CAPITAL_PER_PAIR = 5_000.0

GRID_RANGE_PCT = 0.05               # +/- 5% around center
N_LEVELS = 10                       # 5 buy + 5 sell levels
ORDER_SIZE_USDT = 500               # per grid level

RECENTER_DAYS = 1                   # re-center grid every 24h
VOL_PAUSE_THRESHOLD = 0.05          # daily vol > 5% = too volatile, pause


def _realized_vol(pair: str, days: int = 7) -> float:
    try:
        df = data.ohlcv_extended(pair, days_back=days + 5)
        if df.empty:
            return 0.05
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(rets.iloc[-days:].std()) if len(rets) >= days else 0.05
    except Exception:
        return 0.05


def _current_price(pair: str) -> float:
    """W13 fix: prefer LIVE WebSocket price from cache. Fall back to ticker.

    The grid needs to react to intraday moves, but ohlcv_extended returns the
    DAILY CLOSE which doesn't update until end-of-day = stale grid that never
    fills. Live WS cache updates every 1s.
    """
    # Try live WS cache first
    try:
        live_file = REPO_ROOT / ".live_prices.json"
        if live_file.exists():
            cache = json.loads(live_file.read_text())
            entry = cache.get(pair)
            if entry and isinstance(entry, dict):
                mid = entry.get("mid") or ((entry.get("bid", 0) + entry.get("ask", 0)) / 2)
                if mid and mid > 0:
                    return float(mid)
    except Exception:
        pass
    # Fall back to ticker via ccxt
    try:
        ticker = data._EX.fetch_ticker(pair)
        return float(ticker.get("last") or ticker.get("close") or 0)
    except Exception:
        pass
    # Last resort: stale daily close
    try:
        df = data.ohlcv_extended(pair, days_back=2)
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0


def _build_grid(center: float) -> dict:
    """Generate buy/sell levels around center price."""
    levels = {}
    step = GRID_RANGE_PCT / (N_LEVELS / 2)
    for i in range(1, N_LEVELS // 2 + 1):
        buy_price = center * (1 - i * step)
        sell_price = center * (1 + i * step)
        levels[f"buy_{i}"] = buy_price
        levels[f"sell_{i}"] = sell_price
    return {"center": center, "levels": levels}


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"grids": {}, "history": []}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def cycle(mode: str = "paper") -> dict:
    """One grid cycle. Checks each grid for fills, re-centers if needed."""
    state = load_state()
    grids = state.get("grids", {})
    actions = []

    spot = Broker(mode=mode, long_only=True, sleeve="grid_trader")
    cash = float(spot.get_balance().get("USDT", 0))

    # W16+: Sleeve gating. Grid trader runs continuously but should respect
    # drawdown CB / loss-streak / tail-hedge / VaR-breach gates uniformly.
    sleeve_scale = apply_sleeve_scaling(NAME, SLEEVE_BASELINE)
    if is_paused(NAME):
        return {"status": "sleeve_paused"}
    gates = get_all_gates_scale(NAME)
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused", "event": gates["event_name"]}
    # Effective per-grid order size — every order multiplied by gates
    grid_order_size = ORDER_SIZE_USDT * effective_scale
    if grid_order_size < 50:  # min trade
        return {"status": "gates_below_floor", "effective_scale": effective_scale}

    for pair in PAIRS:
        # Skip pair if vol too high (regime pause)
        vol = _realized_vol(pair)
        if vol > VOL_PAUSE_THRESHOLD:
            log_signal(NAME, pair, 0.0, note=f"vol_pause_{vol*100:.1f}pct")
            continue

        current_price = _current_price(pair)
        if current_price <= 0:
            continue

        grid_data = grids.get(pair)

        # Initialize or re-center
        needs_recenter = False
        if grid_data is None:
            needs_recenter = True
            reason = "initial"
        else:
            last_recenter = datetime.fromisoformat(grid_data.get("recentered_at", "2020-01-01"))
            age_days = (datetime.now(timezone.utc) - last_recenter).days
            center = grid_data.get("center", current_price)
            distance_from_center = abs(current_price / center - 1)
            if distance_from_center > GRID_RANGE_PCT * 0.8:
                needs_recenter = True
                reason = f"out_of_range_{distance_from_center*100:.1f}pct"
            elif age_days >= RECENTER_DAYS:
                needs_recenter = True
                reason = f"age_{age_days}d"

        if needs_recenter:
            # Close any existing position first (matters in paper)
            existing_holdings = spot.get_balance().get(pair.split("/")[0], 0)
            if existing_holdings > 1e-6:
                try:
                    notional = existing_holdings * current_price
                    spot.place_market_order(pair, "sell", notional)
                    realized = 0  # paper doesn't compute
                    log_trade(NAME, pair, "sell", existing_holdings, current_price,
                              realized_pnl=realized, note=f"grid_recenter_{reason}")
                    actions.append({"action": "recenter_close", "pair": pair,
                                    "qty": existing_holdings, "reason": reason})
                except Exception as e:
                    actions.append({"action": "recenter_close_failed", "pair": pair, "error": str(e)})

            # Build new grid
            new_grid = _build_grid(current_price)
            new_grid["recentered_at"] = datetime.now(timezone.utc).isoformat()
            new_grid["last_check_price"] = current_price
            new_grid["filled_levels"] = []
            grids[pair] = new_grid
            actions.append({"action": "recenter", "pair": pair, "center": current_price, "reason": reason})

            # Initial buy at center - 1 grid step (start with 1 unit holding)
            try:
                spot.place_market_order(pair, "buy", grid_order_size)
                qty = grid_order_size / current_price
                log_trade(NAME, pair, "buy", qty, current_price,
                          note=f"grid_initial_at_center")
                tag_entry(f"grid:{pair}", sleeve=NAME, side="long",
                          entry_price=current_price, qty=qty)
                actions.append({"action": "grid_initial_buy", "pair": pair,
                                "price": current_price, "notional": grid_order_size})
            except Exception as e:
                actions.append({"action": "grid_initial_buy_failed", "pair": pair, "error": str(e)})

            continue

        # === Check for grid fills ===
        last_check = grid_data.get("last_check_price", current_price)
        filled = set(grid_data.get("filled_levels", []))
        center = grid_data["center"]

        # Direction: did price MOVE down (buy fills) or UP (sell fills)?
        for level_name, level_price in grid_data["levels"].items():
            if level_name in filled:
                continue
            if level_name.startswith("buy_"):
                # Buy level: fills if price went FROM above TO below this level
                if last_check >= level_price >= current_price:
                    try:
                        spot.place_market_order(pair, "buy", grid_order_size)
                        qty = grid_order_size / level_price
                        log_trade(NAME, pair, "buy", qty, level_price,
                                  note=f"grid_fill_{level_name}")
                        filled.add(level_name)
                        actions.append({"action": "grid_buy_fill", "pair": pair,
                                        "level": level_name, "price": level_price})
                    except Exception as e:
                        actions.append({"action": "grid_buy_failed", "pair": pair,
                                        "level": level_name, "error": str(e)})
            elif level_name.startswith("sell_"):
                # Sell level: fills if price went FROM below TO above this level
                if last_check <= level_price <= current_price:
                    try:
                        spot.place_market_order(pair, "sell", grid_order_size)
                        qty = grid_order_size / level_price
                        log_trade(NAME, pair, "sell", qty, level_price,
                                  note=f"grid_fill_{level_name}")
                        filled.add(level_name)
                        actions.append({"action": "grid_sell_fill", "pair": pair,
                                        "level": level_name, "price": level_price})
                    except Exception as e:
                        actions.append({"action": "grid_sell_failed", "pair": pair,
                                        "level": level_name, "error": str(e)})

        grid_data["last_check_price"] = current_price
        grid_data["filled_levels"] = list(filled)
        grids[pair] = grid_data

    state["grids"] = grids
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return {
        "status": "ok",
        "n_pairs_active": len(grids),
        "n_actions": len(actions),
        "actions": actions,
    }
