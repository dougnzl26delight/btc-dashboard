"""Overbought-fade tactical short sleeve — mirror of oversold_bounce.

Active ONLY in BEAR regime. When 3+ pairs simultaneously hit RSI > 70,
short the 3 most overbought via perp account. Captures failed relief-rally
bounces during bear-market countertrend pops.

Regime gate (hard): aborts if BTC > SMA200 (no shorts in bull markets).
                    Crypto's structural bull bias means shorts in bulls
                    are a one-way ticket to the cleaners.

Entry params:
    RSI_OVERBOUGHT_THRESHOLD = 70
    MIN_CONFIRMING_PAIRS = 3       (cross-sectional confirmation)
    TOP_N_OVERBOUGHT = 3            (smaller basket than longs — shorts squeeze)
    BASKET_ALLOCATION_PCT = 0.10    (smaller size — shorts are scarier)

Exit (any of):
    - RSI back to <= 50              (recovered from overbought)
    - +15% PROFIT (price dropped 15%) (target hit; shorts have asymmetric ceiling)
    - Recent 10-day HIGH + 2% buffer (stop loss; squeezes can run fast)
    - 14 days elapsed                (time cap; shorter than longs)

Risk note: shorts can lose more than 100% if a squeeze unwinds violently.
Per-position stops are mandatory. Sleeve circuit breaker + Sharpe gate apply.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.perp_broker import PerpBroker
from core.pnl_attribution import tag_entry, untag
from core.pnl_db import log_trade, log_signal
from core.vol_sizing import vol_weighted_allocation
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_sharpe_scale, get_all_gates_scale


NAME = "overbought_fade"
STATE_FILE = REPO_ROOT / ".overbought_fade_state.json"

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "BNB/USDT",
    "DOT/USDT", "ATOM/USDT",
]

# Entry params
RSI_OVERBOUGHT_THRESHOLD = 70
MIN_CONFIRMING_PAIRS = 3
TOP_N_OVERBOUGHT = 3
BASKET_ALLOCATION_PCT = 0.10
PER_PAIR_PCT = BASKET_ALLOCATION_PCT / TOP_N_OVERBOUGHT  # ~3.3% per short

# Exit params
RSI_EXIT_THRESHOLD = 50
TARGET_PROFIT_PCT = 0.15           # shorts: 15% price drop = 15% gain
TIME_CAP_DAYS = 14
STOP_LOSS_BUFFER = 0.02            # stop 2% above recent 10-day HIGH

# Regime gate
REGIME_PAIR = "BTC/USDT"
SMA_LOOKBACK = 200
REGIME_DROP_THRESHOLD = -0.10      # BTC 14d return must be < -10% (firmly bear)


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _check_bear_regime() -> tuple[bool, str]:
    """True if BTC firmly in bear regime — required for shorts to fire.

    Criteria: BTC closing below 200-day SMA AND 14-day return < -10%.
    Returns (is_bear, reason).
    """
    try:
        df = data.ohlcv_extended(REGIME_PAIR, days_back=SMA_LOOKBACK + 20)
        if df.empty or len(df) < SMA_LOOKBACK:
            return False, "insufficient BTC data"
        sma = float(df["close"].rolling(SMA_LOOKBACK).mean().iloc[-1])
        price = float(df["close"].iloc[-1])
        ret_14d = float(df["close"].iloc[-1] / df["close"].iloc[-15] - 1)
        if price > sma:
            return False, f"BTC above SMA200 (${price:,.0f} > ${sma:,.0f}) — not bear"
        if ret_14d > REGIME_DROP_THRESHOLD:
            return False, f"BTC 14d return {ret_14d*100:+.1f}% (need < {REGIME_DROP_THRESHOLD*100:.0f}%)"
        return True, f"bear confirmed: BTC ${price:,.0f} < SMA200 ${sma:,.0f}, 14d {ret_14d*100:+.1f}%"
    except Exception as e:
        return False, f"regime check failed: {e}"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_positions": {}, "history": []}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def _scan_overbought() -> list[dict]:
    """Pairs with RSI > threshold, most overbought first."""
    out = []
    for pair in UNIVERSE:
        try:
            df = data.ohlcv_extended(pair, days_back=60)
            if df.empty or len(df) < 30:
                continue
            rsi = _rsi(df["close"])
            recent_10d_high = float(df["high"].iloc[-10:].max())
            current_price = float(df["close"].iloc[-1])
            out.append({
                "pair": pair,
                "rsi": rsi,
                "price": current_price,
                "recent_10d_high": recent_10d_high,
            })
        except Exception:
            continue
    # Most overbought first (highest RSI)
    out.sort(key=lambda x: -x["rsi"])
    return out


def cycle(mode: str = "paper") -> dict:
    """One cycle: regime check → exits → entries (if regime confirms)."""
    state = load_state()
    open_pos = state.get("open_positions", {})

    # === Regime gate (hard) ===
    is_bear, regime_reason = _check_bear_regime()

    # === Sleeve circuit breaker + Sharpe gate ===
    perp = PerpBroker(mode=mode, sleeve="overbought_fade")
    short_mtm = 0.0
    for pair, info in list(open_pos.items()):
        base = pair.split("/")[0]
        qty = perp._state.positions.get(base, 0.0)  # negative for short
        if abs(qty) < 1e-12:
            continue
        try:
            current = float(perp.ticker(pair).get("last") or 0)
        except Exception:
            continue
        entry = info.get("entry_price", current)
        short_mtm += qty * (current - entry)  # qty is negative; profit when price drops
    baseline = 100_000.0 * BASKET_ALLOCATION_PCT  # $10k baseline
    current_equity = baseline + short_mtm
    sleeve_scale = apply_sleeve_scaling("overbought_fade", current_equity)
    if is_paused("overbought_fade"):
        return {"status": "sleeve_paused"}
    gates = get_all_gates_scale("overbought_fade")
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused", "event": gates["event_name"]}

    actions = []

    # === Exit logic for open shorts ===
    for pair, info in list(open_pos.items()):
        try:
            df = data.ohlcv_extended(pair, days_back=30)
            if df.empty:
                continue
            current = float(df["close"].iloc[-1])
            current_high = float(df["high"].iloc[-1])
            rsi = _rsi(df["close"])
        except Exception:
            continue

        entry = info["entry_price"]
        qty = info["qty"]  # negative for short
        stop = info.get("stop_loss", entry * (1 + STOP_LOSS_BUFFER))
        target_price = entry * (1 - TARGET_PROFIT_PCT)
        opened_at = datetime.fromisoformat(info["opened_at"])
        age_days = (datetime.now(timezone.utc) - opened_at).days

        exit_reason = None
        if rsi < RSI_EXIT_THRESHOLD:
            exit_reason = "rsi_recovered"
        elif current <= target_price:
            exit_reason = "target_hit"
        elif current_high > stop:
            exit_reason = "stop_loss"
        elif age_days >= TIME_CAP_DAYS:
            exit_reason = "time_cap"

        if exit_reason:
            try:
                result = perp.close_position(pair)
                realized = result.get("realized_pnl", 0.0)
                log_trade(NAME, pair, "close_short", abs(qty), current,
                          realized_pnl=realized, note=f"exit:{exit_reason}")
                untag(f"ofade:{pair}")
                actions.append({
                    "action": "exit", "pair": pair, "reason": exit_reason,
                    "entry": entry, "exit": current, "realized_pnl": realized,
                })
                state.setdefault("history", []).append({
                    **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_price": current, "exit_reason": exit_reason,
                    "realized_pnl": realized,
                })
                del open_pos[pair]
            except Exception as e:
                actions.append({"action": "exit_failed", "pair": pair, "error": str(e)})

    # === Entry logic — only if BEAR regime confirmed ===
    candidates = _scan_overbought()
    overbought = [c for c in candidates if c["rsi"] > RSI_OVERBOUGHT_THRESHOLD]
    log_signal(NAME, "_universe", float(len(overbought)),
               regime="bear" if is_bear else "non-bear",
               note=f"n_overbought={len(overbought)}, regime: {regime_reason}")

    entries_blocked_reason = None
    if not is_bear:
        entries_blocked_reason = f"regime_gate: {regime_reason}"
    elif len(overbought) < MIN_CONFIRMING_PAIRS:
        entries_blocked_reason = f"only {len(overbought)} pairs overbought (need {MIN_CONFIRMING_PAIRS})"

    if not entries_blocked_reason:
        top_n = overbought[:TOP_N_OVERBOUGHT]
        # === Inverse-vol sizing (W10) ===
        cash = float(perp._state.cash_quote)
        basket_total = cash * BASKET_ALLOCATION_PCT * effective_scale
        candidate_pairs = [c["pair"] for c in top_n if c["pair"] not in open_pos]
        vol_allocations = vol_weighted_allocation(candidate_pairs, basket_total)
        for c in top_n:
            pair = c["pair"]
            if pair in open_pos:
                continue
            base_notional = vol_allocations.get(pair, 0)
            if base_notional < 100:
                continue
            # W16.A: Liquidation pressure upsizing.
            # When liquidation_pressure reports edge_direction == fade_long,
            # there's a brewing long-cascade — overbought_fade shorts here
            # have asymmetric reward and should upsize 1.5x.
            lp_scale = 1.0
            lp_edge = None
            try:
                from core.liquidation_pressure import liquidation_pressure as _lp
                lp = _lp(pair)
                lp_edge = lp.get("edge_direction", "no_edge")
                if lp_edge == "fade_long":
                    lp_scale = 1.5
            except Exception:
                pass
            notional = base_notional * lp_scale
            try:
                qty = notional / c["price"]
                perp.open_position(pair, "short", notional)
                stop = c["recent_10d_high"] * (1 + STOP_LOSS_BUFFER)
                entry_record = {
                    "entry_price": c["price"],
                    "qty": -qty,  # negative = short
                    "stop_loss": stop,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "entry_rsi": c["rsi"],
                    "notional_in": notional,
                    "liq_pressure_edge": lp_edge,
                    "liq_pressure_scale": lp_scale,
                }
                open_pos[pair] = entry_record
                _liq_tag = f" liq:{lp_edge}({lp_scale:.1f}x)" if lp_scale != 1.0 else ""
                log_trade(NAME, pair, "open_short", qty, c["price"],
                          note=f"entry:rsi{c['rsi']:.0f}{_liq_tag}")
                tag_entry(f"ofade:{pair}", sleeve=NAME, side="short",
                          entry_price=c["price"], qty=qty)
                actions.append({
                    "action": "entry_short", "pair": pair,
                    "rsi": c["rsi"], "price": c["price"], "qty": qty,
                    "stop": stop, "target": c["price"] * (1 - TARGET_PROFIT_PCT),
                    "liq_edge": lp_edge, "liq_scale": lp_scale,
                })
            except Exception as e:
                actions.append({"action": "entry_failed", "pair": pair, "error": str(e)})

    state["open_positions"] = open_pos
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    state["last_regime_reason"] = regime_reason
    state["last_universe_rsi"] = {c["pair"]: round(c["rsi"], 1) for c in candidates}
    save_state(state)

    return {
        "status": "ok",
        "regime_bear": is_bear,
        "regime_reason": regime_reason,
        "n_overbought": len(overbought),
        "n_open_positions": len(open_pos),
        "entries_blocked_reason": entries_blocked_reason,
        "actions": actions,
    }
