"""Livermore consolidation breakout — *Reminiscences of a Stock Operator*.

"The longer the market stays in a range, the more significant the breakout."
                                                            — Jesse Livermore

Most retail Donchian breakout systems fire on every 20-day high. Livermore's
edge was specifically: identify pairs that have been UNUSUALLY tight for
30-90 days, THEN trade the breakout. Setup quality matters more than the
trigger itself.

Mechanism:
    1. Compute current 30-day Bollinger Band width
    2. Compute average BB width over 90 days
    3. Compression score = current_BB / avg_BB
    4. Watch only pairs where compression_score < 0.6 (tight range)
    5. Enter on first daily close outside that compressed range
    6. Stop at the OPPOSITE side of the compressed range
    7. Target 1× range expansion at T1 (close 50%)
    8. Target 2× range expansion at T2 (close remainder)

Long-only currently; short variant trivially derivable.
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


NAME = "consolidation_breakout"
STATE_FILE = REPO_ROOT / ".consolidation_breakout_state.json"

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT",
            "DOGE/USDT", "AVAX/USDT", "BNB/USDT", "DOT/USDT", "ATOM/USDT"]

# Allocation
SLEEVE_BASELINE = 10_000.0
PER_TRADE_USDT = 1_500              # 15% of sleeve per breakout
MAX_CONCURRENT = 3

# Detection
RECENT_BB_WINDOW = 30
LONG_BB_WINDOW = 90
COMPRESSION_THRESHOLD = 0.6          # tight range = <60% of normal width

# Exit
T1_RANGE_MULTIPLE = 1.0              # close 50% at 1x range expansion
T2_RANGE_MULTIPLE = 2.0              # close rest at 2x
STOP_TO_OPPOSITE_SIDE = True         # stop at opposite end of compressed range
TIME_CAP_DAYS = 21

# W16.D: Multi-timeframe confluence floor.
# Compression breakouts are tactical entries; require higher-TF directional
# alignment to filter false breakouts that fade back into the range.
# Regime-gated (2026-06-01 _bt_mtf_bull_regime.py): skip MTF when BTC is in
# clear bear (30d < -8%). MTF rejects winning bounces in downtrends.
MIN_MTF_CONFLUENCE = 0.5
MTF_DOWNTREND_THRESHOLD = -0.08


def _bb_width(close: pd.Series, window: int) -> float:
    """Bollinger Band width (2 sigma, in % of price)."""
    if len(close) < window:
        return 0.0
    rolling = close.rolling(window)
    upper = rolling.mean() + 2 * rolling.std()
    lower = rolling.mean() - 2 * rolling.std()
    return float((upper.iloc[-1] - lower.iloc[-1]) / close.iloc[-1])


def _detect_compression(pair: str) -> dict | None:
    """For one pair: is it currently in a compressed range?"""
    try:
        df = data.ohlcv_extended(pair, days_back=LONG_BB_WINDOW + 30)
        if df.empty or len(df) < LONG_BB_WINDOW:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        recent_bb = _bb_width(df["close"], RECENT_BB_WINDOW)
        # Avg of past LONG_BB_WINDOW BB widths
        rolling_bb = df["close"].rolling(RECENT_BB_WINDOW).std().rolling(LONG_BB_WINDOW).mean()
        avg_bb = float(rolling_bb.iloc[-1] / df["close"].iloc[-1] * 4)  # 2 sigma × 2 sides
        if avg_bb <= 0:
            return None
        compression = recent_bb / avg_bb
        # Range bounds (range high/low over recent_bb_window)
        range_high = float(df["high"].iloc[-RECENT_BB_WINDOW:].max())
        range_low = float(df["low"].iloc[-RECENT_BB_WINDOW:].min())
        current = float(df["close"].iloc[-1])
        return {
            "pair": pair,
            "compression_score": compression,
            "is_compressed": compression < COMPRESSION_THRESHOLD,
            "range_high": range_high,
            "range_low": range_low,
            "range_size": range_high - range_low,
            "current_price": current,
            "above_range": current > range_high,
            "below_range": current < range_low,
        }
    except Exception:
        return None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"watching": {}, "open_positions": {}, "history": []}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def cycle(mode: str = "paper") -> dict:
    """One cycle: scan for compressions, fire breakouts, manage exits."""
    state = load_state()
    open_pos = state.get("open_positions", {})
    watching = state.get("watching", {})

    # 2026-06-01 — empirical regime pause (ops.regime_gate).
    # Breakouts fade in clear bear; pause new entries prospectively.
    bear_pause = False
    try:
        from ops.regime_gate import should_pause_sleeve
        p = should_pause_sleeve(NAME)
        bear_pause = p["should_pause"]
    except Exception:
        pass

    sleeve_scale = apply_sleeve_scaling(NAME, SLEEVE_BASELINE)
    if is_paused(NAME):
        return {"status": "sleeve_paused"}
    gates = get_all_gates_scale(NAME)
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused"}

    spot = Broker(mode=mode, long_only=True, sleeve=NAME)
    cash = float(spot.get_balance().get("USDT", 0))
    actions = []

    # === Exit logic ===
    for pair, info in list(open_pos.items()):
        try:
            df = data.ohlcv_extended(pair, days_back=5)
            if df.empty:
                continue
            current = float(df["close"].iloc[-1])
        except Exception:
            continue

        entry = info["entry_price"]
        qty_total = info["qty"]
        target_1 = info["target_1"]
        target_2 = info["target_2"]
        stop = info["stop_loss"]
        opened_at = datetime.fromisoformat(info["opened_at"])
        age_days = (datetime.now(timezone.utc) - opened_at).days
        already_t1_hit = info.get("t1_hit", False)

        exit_reason = None
        partial = False
        if current <= stop:
            exit_reason = "stop_loss"
        elif age_days >= TIME_CAP_DAYS:
            exit_reason = "time_cap"
        elif not already_t1_hit and current >= target_1:
            # 50% partial at T1
            partial = True
            qty_to_sell = qty_total * 0.5
            notional = qty_to_sell * current
            realized = (current - entry) * qty_to_sell
            try:
                spot.place_market_order(pair, "sell", notional)
                log_trade(NAME, pair, "sell", qty_to_sell, current, realized_pnl=realized,
                          note="T1_partial_50pct")
                info["qty"] = qty_total - qty_to_sell
                info["t1_hit"] = True
                actions.append({"action": "T1_partial", "pair": pair,
                                "realized_pnl": realized, "qty_sold": qty_to_sell})
            except Exception:
                pass
        elif already_t1_hit and current >= target_2:
            exit_reason = "T2_target"

        if exit_reason and not partial:
            qty = info["qty"]
            notional = qty * current
            realized = (current - entry) * qty
            try:
                spot.place_market_order(pair, "sell", notional)
                log_trade(NAME, pair, "sell", qty, current, realized_pnl=realized,
                          note=f"exit:{exit_reason}")
                untag(f"conso:{pair}")
                actions.append({"action": "exit", "pair": pair, "reason": exit_reason,
                                "entry": entry, "exit": current, "realized_pnl": realized})
                state.setdefault("history", []).append({
                    **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_price": current, "exit_reason": exit_reason,
                    "realized_pnl": realized,
                })
                del open_pos[pair]
            except Exception:
                pass

    # === Entry logic — scan for compressed ranges breaking ===
    # Skip entirely if regime-paused (bear). Exit logic above still ran.
    if bear_pause:
        state["watching"] = watching
        state["open_positions"] = open_pos
        state["last_scan"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        return {
            "status": "bear_regime_paused",
            "n_open_positions": len(open_pos),
            "actions": actions,
            "note": "consolidation_breakout paused — BTC 30d < -8% (clear bear). "
                    "Breakouts fade back into range in downtrends. Resume on regime change.",
        }

    candidates = []
    rejected_for_mtf = 0
    for pair in UNIVERSE:
        d = _detect_compression(pair)
        if d is None or not d["is_compressed"]:
            continue
        watching[pair] = {
            "compression_score": d["compression_score"],
            "range_high": d["range_high"],
            "range_low": d["range_low"],
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        # Long entry: price breaks above compressed range
        if d["above_range"]:
            # W16.D regime-gated: require MTF confluence only in non-bear regimes.
            try:
                # Check regime via BTC 30d return
                btc_df = data.ohlcv_extended("BTC/USDT", days_back=35)
                apply_mtf = True
                if not btc_df.empty and len(btc_df) >= 31:
                    btc_30d = float(btc_df["close"].iloc[-1] / btc_df["close"].iloc[-31] - 1)
                    apply_mtf = btc_30d > MTF_DOWNTREND_THRESHOLD
                if apply_mtf:
                    from core.multi_timeframe import confluence
                    mtf = confluence(pair)
                    mtf_score = mtf.get("confluence_score", 0.0)
                    mtf_dir = mtf.get("net_direction", 0)
                    if mtf is not None and (mtf_score < MIN_MTF_CONFLUENCE or mtf_dir <= 0):
                        rejected_for_mtf += 1
                        continue
                    d["mtf_confluence"] = mtf_score
                    d["mtf_verdict"] = mtf.get("verdict", "no_data")
                else:
                    d["mtf_confluence"] = None
                    d["mtf_verdict"] = "skipped_bear_regime"
            except Exception:
                # MTF unavailable — let base detection through
                pass
            candidates.append(d)

    log_signal(NAME, "_universe", float(len(candidates)),
               note=f"n_breakouts={len(candidates)}, rejected_mtf={rejected_for_mtf}")

    available_slots = MAX_CONCURRENT - len(open_pos)
    for c in candidates[:available_slots]:
        pair = c["pair"]
        if pair in open_pos:
            continue
        notional = PER_TRADE_USDT * effective_scale
        if notional < 100 or notional > cash:
            continue
        try:
            entry_price = c["current_price"]
            qty = notional / entry_price
            stop_price = c["range_low"] * 0.98 if STOP_TO_OPPOSITE_SIDE else entry_price * 0.95
            target_1 = entry_price + c["range_size"] * T1_RANGE_MULTIPLE
            target_2 = entry_price + c["range_size"] * T2_RANGE_MULTIPLE
            spot.place_market_order(pair, "buy", notional)
            entry_record = {
                "entry_price": entry_price, "qty": qty,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "compression_score": c["compression_score"],
                "range_size": c["range_size"],
                "stop_loss": stop_price,
                "target_1": target_1, "target_2": target_2,
                "t1_hit": False,
            }
            open_pos[pair] = entry_record
            log_trade(NAME, pair, "buy", qty, entry_price,
                      note=f"breakout:compr{c['compression_score']:.2f}")
            tag_entry(f"conso:{pair}", sleeve=NAME, side="long",
                      entry_price=entry_price, qty=qty)
            actions.append({
                "action": "entry", "pair": pair,
                "price": entry_price, "qty": qty,
                "stop": stop_price, "target_1": target_1, "target_2": target_2,
                "compression_score": c["compression_score"],
            })
        except Exception:
            pass

    state["watching"] = watching
    state["open_positions"] = open_pos
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return {
        "status": "ok",
        "n_watching": len(watching),
        "n_compressed_breaking": len(candidates),
        "n_open_positions": len(open_pos),
        "actions": actions,
    }
