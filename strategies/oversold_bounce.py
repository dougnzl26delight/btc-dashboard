"""Oversold mean-reversion bounce strategy — tactical sleeve.

Designed for bear-market relief-bounce capture. When the universe is in
regime-wide oversold (>=3 pairs at RSI < 25), enter equal-weight long
basket on the 5 most oversold names.

Exit any position when:
  - RSI > 50 (recovered from oversold)
  - +20% from entry (target hit)
  - Daily low breaks the 20-day low at entry (stop-loss)
  - 30 days from entry (time cap)

Position size: 15% of bankroll, equal-weight across selected pairs.

Logic background (from 2022 + 2018 bear-market relief analysis):
  - Bear-market reliefs produce 4-5 oversold-bounce setups before true bottom
  - Each bounce typically 20-50% in 1-3 weeks
  - Hit rate ~65-70% when 3+ pairs simultaneously oversold (cross-section confirms)
  - Hit rate ~40% on single-pair oversold (too noisy)

State file: .oversold_bounce_state.json
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
from core.broker import Broker
from core.pnl_attribution import tag_entry, untag
from core.pnl_db import log_trade, log_signal
from core.vol_sizing import vol_weighted_allocation
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_sharpe_scale, get_all_gates_scale

NAME = "oversold_bounce"
STATE_FILE = REPO_ROOT / ".oversold_bounce_state.json"

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "BNB/USDT",
    "DOT/USDT", "ATOM/USDT",
]

# Entry params
RSI_OVERSOLD_THRESHOLD = 25
MIN_CONFIRMING_PAIRS = 3        # cross-sectional confirmation
TOP_N_OVERSOLD = 5              # take 5 most-oversold
BASKET_ALLOCATION_PCT = 0.15    # 15% of bankroll
PER_PAIR_PCT = BASKET_ALLOCATION_PCT / TOP_N_OVERSOLD  # 3% per pair

# Exit params
RSI_EXIT_THRESHOLD = 50
TARGET_PROFIT_PCT = 0.20
TIME_CAP_DAYS = 30
STOP_LOSS_BUFFER = 0.02         # stop 2% below the recent 20-day low


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_positions": {}, "history": []}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def _scan_oversold() -> list[dict]:
    """Return list of pairs with RSI < threshold, most oversold first."""
    out = []
    for pair in UNIVERSE:
        try:
            df = data.ohlcv_extended(pair, days_back=60)
            if df.empty or len(df) < 30:
                continue
            rsi = _rsi(df["close"])
            recent_20d_low = float(df["low"].iloc[-20:].min())
            current_price = float(df["close"].iloc[-1])
            out.append({
                "pair": pair,
                "rsi": rsi,
                "price": current_price,
                "recent_20d_low": recent_20d_low,
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["rsi"])
    return out


def _reconcile_state_with_broker(state: dict, mode: str) -> dict:
    """Drop state positions that the broker can't support.

    Other sleeves (BAH BTC) share the same spot broker. If BAH rebalances
    and sells positions that oversold_bounce thinks it owns, we'd try to
    sell zero balance. Detect divergence and prune stale entries so the
    next entry cycle has a clean slate. Logs the prune to pnl_db for audit.
    """
    if state.get("open_positions") is None:
        return state
    spot = Broker(mode=mode, long_only=True, sleeve="oversold_bounce")
    bal = spot.get_balance()
    pruned = []
    for pair, info in list(state["open_positions"].items()):
        base = pair.split("/")[0]
        broker_qty = float(bal.get(base, 0))
        claimed_qty = float(info.get("qty", 0))
        # Drop if broker has less than 10% of what we claim (stale state)
        if abs(broker_qty) < abs(claimed_qty) * 0.1:
            try:
                log_trade(NAME, pair, "state_reset", claimed_qty, info.get("entry_price", 0),
                          realized_pnl=0.0,
                          note=f"reconcile: broker has {broker_qty:.6f}, state claimed {claimed_qty:.6f}")
            except Exception:
                pass
            try:
                untag(f"oversold:{pair}")
            except Exception:
                pass
            state.setdefault("history", []).append({
                **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                "exit_reason": "reconcile_broker_divergence",
                "broker_qty_at_reset": broker_qty,
            })
            del state["open_positions"][pair]
            pruned.append(pair)
    if pruned:
        try:
            from ops.alerts import alert as _alert
            _alert(
                f"oversold_bounce reconcile: pruned {len(pruned)} stale state entries "
                f"({pruned}). Broker is source of truth.",
                level="warning",
            )
        except Exception:
            pass
    return state


def cycle(mode: str = "paper") -> dict:
    """One cycle: reconcile state, check exits, then look for new entries."""
    state = load_state()
    state = _reconcile_state_with_broker(state, mode)

    # === Sleeve gating ===
    # Approximate current equity = 15% notional + open MTM
    open_pos = state.get("open_positions", {})
    open_mtm = 0.0
    for pair, info in list(open_pos.items()):
        try:
            current = float(data.ohlcv_extended(pair, days_back=2)["close"].iloc[-1])
        except Exception:
            continue
        entry = info.get("entry_price", 0)
        qty = info.get("qty", 0)
        open_mtm += qty * (current - entry)
    baseline = 100_000.0 * BASKET_ALLOCATION_PCT  # $15k
    current_equity = baseline + open_mtm
    sleeve_scale = apply_sleeve_scaling("oversold_bounce", current_equity)
    if is_paused("oversold_bounce"):
        return {"status": "sleeve_paused"}
    # W10+W16.H: composed scale from ALL gates including BTC dominance regime.
    # alt_regime=True → multiply by 0 in BTC_HEGEMONY, 0.5 in BTC_DOMINANT.
    gates = get_all_gates_scale("oversold_bounce", alt_regime=True)
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused", "event": gates["event_name"]}
    if effective_scale <= 0.0:
        return {
            "status": "dominance_paused",
            "regime": gates.get("dominance_regime"),
            "note": "alt-sleeve fully gated by BTC dominance regime",
        }

    spot = Broker(mode=mode, long_only=True, sleeve="oversold_bounce")
    cash = float(spot.get_balance().get("USDT", 0))
    actions = []

    # === Exit logic for open positions ===
    for pair, info in list(open_pos.items()):
        try:
            df = data.ohlcv_extended(pair, days_back=30)
            if df.empty:
                continue
            current = float(df["close"].iloc[-1])
            current_low = float(df["low"].iloc[-1])
            rsi = _rsi(df["close"])
        except Exception:
            continue

        entry = info["entry_price"]
        qty = info["qty"]
        stop = info.get("stop_loss", entry * (1 - STOP_LOSS_BUFFER))
        target = entry * (1 + TARGET_PROFIT_PCT)
        opened_at = datetime.fromisoformat(info["opened_at"])
        age_days = (datetime.now(timezone.utc) - opened_at).days

        exit_reason = None
        if rsi > RSI_EXIT_THRESHOLD:
            exit_reason = "rsi_recovered"
        elif current >= target:
            exit_reason = "target_hit"
        elif current_low < stop:
            exit_reason = "stop_loss"
        elif age_days >= TIME_CAP_DAYS:
            exit_reason = "time_cap"

        if exit_reason:
            notional = qty * current
            realized = (current - entry) * qty
            try:
                spot.place_market_order(pair, "sell", notional)
                log_trade(NAME, pair, "sell", qty, current, realized_pnl=realized,
                          note=f"exit:{exit_reason}")
                untag(f"oversold:{pair}")
                actions.append({
                    "action": "exit", "pair": pair, "reason": exit_reason,
                    "entry": entry, "exit": current, "realized_pnl": realized,
                })
                # Move to history
                state.setdefault("history", []).append({
                    **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_price": current, "exit_reason": exit_reason,
                    "realized_pnl": realized,
                })
                del open_pos[pair]
            except Exception as e:
                actions.append({"action": "exit_failed", "pair": pair, "error": str(e)})

    # === Entry logic — scan universe for oversold pairs ===
    candidates = _scan_oversold()
    oversold = [c for c in candidates if c["rsi"] < RSI_OVERSOLD_THRESHOLD]

    log_signal(NAME, "_universe", float(len(oversold)),
               note=f"n_oversold={len(oversold)}")

    if len(oversold) >= MIN_CONFIRMING_PAIRS:
        # Regime-wide oversold confirmed — enter top N if not already in
        top_n = oversold[:TOP_N_OVERSOLD]
        # === Inverse-vol sizing (W10) ===
        # Distribute basket capital so each position contributes ~equal risk.
        # Lower-vol pairs (BTC) get MORE capital; higher-vol pairs (alts) get LESS.
        basket_total = cash * BASKET_ALLOCATION_PCT * effective_scale
        candidate_pairs = [c["pair"] for c in top_n if c["pair"] not in open_pos]
        vol_allocations = vol_weighted_allocation(candidate_pairs, basket_total)
        for c in top_n:
            pair = c["pair"]
            if pair in open_pos:
                continue
            base_notional = vol_allocations.get(pair, 0)
            if base_notional < 100:  # min trade size
                continue
            # W16.A: Liquidation pressure upsizing.
            # When edge_direction == fade_short, short squeeze is brewing —
            # oversold_bounce longs have asymmetric reward and upsize 1.5x.
            lp_scale = 1.0
            lp_edge = None
            try:
                from core.liquidation_pressure import liquidation_pressure as _lp
                lp = _lp(pair)
                lp_edge = lp.get("edge_direction", "no_edge")
                if lp_edge == "fade_short":
                    lp_scale = 1.5
            except Exception:
                pass
            notional = base_notional * lp_scale
            try:
                qty = notional / c["price"]
                spot.place_market_order(pair, "buy", notional)
                stop = c["recent_20d_low"] * (1 - STOP_LOSS_BUFFER)
                entry_record = {
                    "entry_price": c["price"],
                    "qty": qty,
                    "stop_loss": stop,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "entry_rsi": c["rsi"],
                    "notional_in": notional,
                    "liq_pressure_edge": lp_edge,
                    "liq_pressure_scale": lp_scale,
                }
                open_pos[pair] = entry_record
                _liq_tag = f" liq:{lp_edge}({lp_scale:.1f}x)" if lp_scale != 1.0 else ""
                log_trade(NAME, pair, "buy", qty, c["price"], note=f"entry:rsi{c['rsi']:.0f}{_liq_tag}")
                tag_entry(f"oversold:{pair}", sleeve=NAME, side="long",
                          entry_price=c["price"], qty=qty)
                actions.append({
                    "action": "entry", "pair": pair,
                    "rsi": c["rsi"], "price": c["price"], "qty": qty,
                    "stop": stop, "target": c["price"] * (1 + TARGET_PROFIT_PCT),
                    "liq_edge": lp_edge, "liq_scale": lp_scale,
                })
            except Exception as e:
                actions.append({"action": "entry_failed", "pair": pair, "error": str(e)})

    state["open_positions"] = open_pos
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    state["last_universe_rsi"] = {c["pair"]: round(c["rsi"], 1) for c in candidates}
    save_state(state)

    return {
        "status": "ok",
        "n_oversold": len(oversold),
        "n_open_positions": len(open_pos),
        "actions": actions,
        "regime_armed": len(oversold) >= MIN_CONFIRMING_PAIRS,
    }


def latest_signal(pair: str) -> float:
    """For orchestrator integration: returns +1 if we'd want to be long this pair."""
    state = load_state()
    if pair in state.get("open_positions", {}):
        return 1.0
    candidates = _scan_oversold()
    by_pair = {c["pair"]: c for c in candidates}
    n_oversold = sum(1 for c in candidates if c["rsi"] < RSI_OVERSOLD_THRESHOLD)
    if n_oversold < MIN_CONFIRMING_PAIRS:
        return 0.0
    if pair in by_pair and by_pair[pair]["rsi"] < RSI_OVERSOLD_THRESHOLD:
        return 1.0
    return 0.0
