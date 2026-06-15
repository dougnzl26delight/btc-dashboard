"""Intraday momentum SHORT sleeve — mirror of intraday_momentum.

Active in BEAR regime. Fires shorts on intraday DOWN-moves with volume.

Entry: 4h TSMOM_3bar < -1.5%, volume spike > 1.5x avg, RSI 25-50.
Exit: +3% gain (price dropped), -1.5% stop, 24h cap, trail stop.

Routes to perp account (true shorts, not synthetic).
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
from core.perp_broker import PerpBroker
from core.pnl_attribution import tag_entry, untag
from core.pnl_db import log_trade, log_signal
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_all_gates_scale


NAME = "intraday_momentum_short"
STATE_FILE = REPO_ROOT / ".intraday_momentum_short_state.json"

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
            "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "BNB/USDT", "ATOM/USDT", "DOT/USDT"]

SLEEVE_BASELINE = 10_000.0
PER_TRADE_PCT = 0.03
MAX_CONCURRENT = 4

TSMOM_3BAR_THRESHOLD = -0.015       # -1.5% over last 12h
VOLUME_SPIKE_MULT = 1.5
RSI_MIN = 25
RSI_MAX = 50

PROFIT_TARGET_PCT = 0.03            # +3% gain = price drops 3%
STOP_LOSS_PCT = 0.015
TIME_CAP_HOURS = 24
TRAIL_ATR_MULT = 1.5

# Regime gate: only fire shorts when bear is confirmed
REGIME_BEAR_THRESHOLD_PCT = -0.05   # BTC 14d > -5% bear


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1]) if not (100 - 100 / (1 + rs)).empty else 50.0


def _atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _check_bear_regime() -> tuple[bool, str]:
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=20)
        if df.empty or len(df) < 15:
            return False, "no_data"
        ret_14d = float(df["close"].iloc[-1] / df["close"].iloc[-15] - 1)
        if ret_14d > REGIME_BEAR_THRESHOLD_PCT:
            return False, f"BTC 14d {ret_14d*100:+.1f}% > {REGIME_BEAR_THRESHOLD_PCT*100:.0f}%"
        return True, f"BTC 14d {ret_14d*100:+.1f}% bear confirmed"
    except Exception as e:
        return False, str(e)


def _scan_for_short_momentum():
    cands = []
    for pair in UNIVERSE:
        try:
            df = data.ohlcv_extended(pair, days_back=10)
            if df.empty or len(df) < 25:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            current = float(df["close"].iloc[-1])
            tsmom_3bar = current / float(df["close"].iloc[-4]) - 1
            vol_20avg = float(df["volume"].iloc[-20:].mean())
            vol_now = float(df["volume"].iloc[-1])
            vol_spike = vol_now / vol_20avg if vol_20avg > 0 else 0
            rsi = _rsi(df["close"])
            atr = _atr(df)
            if (tsmom_3bar < TSMOM_3BAR_THRESHOLD
                and vol_spike > VOLUME_SPIKE_MULT
                and RSI_MIN <= rsi <= RSI_MAX):
                cands.append({
                    "pair": pair, "price": current, "tsmom": tsmom_3bar,
                    "vol_spike": vol_spike, "rsi": rsi, "atr": atr,
                    "signal_strength": abs(tsmom_3bar) * vol_spike,
                })
        except Exception:
            continue
    cands.sort(key=lambda x: -x["signal_strength"])
    return cands


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_positions": {}, "history": []}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def cycle(mode="paper"):
    state = load_state()
    open_pos = state.get("open_positions", {})

    is_bear, regime_reason = _check_bear_regime()
    sleeve_scale = apply_sleeve_scaling("intraday_momentum_short", SLEEVE_BASELINE)
    if is_paused("intraday_momentum_short"):
        return {"status": "sleeve_paused"}
    gates = get_all_gates_scale("intraday_momentum_short")
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused"}

    perp = PerpBroker(mode=mode, sleeve="intraday_momentum_short")
    actions = []

    # Exit logic
    for pair, info in list(open_pos.items()):
        try:
            df = data.ohlcv_extended(pair, days_back=5)
            if df.empty:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            current = float(df["close"].iloc[-1])
        except Exception:
            continue

        entry = info["entry_price"]
        qty = info["qty"]  # negative for short
        opened_at = datetime.fromisoformat(info["opened_at"])
        age_h = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600

        # For short: gain when price DROPS
        pnl_pct = (entry - current) / entry  # positive when price down
        exit_reason = None
        if pnl_pct >= PROFIT_TARGET_PCT:
            exit_reason = "target_hit"
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = "stop_loss"
        elif age_h >= TIME_CAP_HOURS:
            exit_reason = "time_cap"
        else:
            trough = info.get("trough_price", entry)
            new_trough = min(trough, current)
            atr = _atr(df)
            trail_level = new_trough + TRAIL_ATR_MULT * atr
            if current > trail_level and pnl_pct > 0:
                exit_reason = "trail_stop"
            info["trough_price"] = new_trough

        if exit_reason:
            try:
                res = perp.close_position(pair)
                realized = res.get("realized_pnl", 0.0)
                log_trade(NAME, pair, "close_short", abs(qty), current,
                          realized_pnl=realized, note=f"exit:{exit_reason}")
                untag(f"intradayshort:{pair}")
                actions.append({"action": "exit", "pair": pair, "reason": exit_reason,
                                "entry": entry, "exit": current, "realized_pnl": realized,
                                "pnl_pct": pnl_pct})
                state.setdefault("history", []).append({
                    **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_price": current, "exit_reason": exit_reason, "realized_pnl": realized,
                })
                del open_pos[pair]
            except Exception as e:
                actions.append({"action": "exit_failed", "pair": pair, "error": str(e)})

    # Entries only if bear regime confirmed
    if not is_bear:
        save_state(state)
        return {"status": "ok", "regime": "non-bear", "regime_reason": regime_reason,
                "n_candidates": 0, "n_open_positions": len(open_pos), "actions": actions}

    candidates = _scan_for_short_momentum()
    log_signal(NAME, "_universe", float(len(candidates)),
               regime="bear", note=f"n_short_candidates={len(candidates)}")

    available = MAX_CONCURRENT - len(open_pos)
    cash = float(perp._state.cash_quote)
    for c in candidates[:available]:
        pair = c["pair"]
        if pair in open_pos:
            continue
        notional = SLEEVE_BASELINE * PER_TRADE_PCT * effective_scale
        if notional < 100:
            continue
        try:
            qty = notional / c["price"]
            perp.open_position(pair, "short", notional)
            entry_record = {
                "entry_price": c["price"], "qty": -qty,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_tsmom": c["tsmom"], "entry_rsi": c["rsi"],
                "trough_price": c["price"],
                "stop_loss": c["price"] * (1 + STOP_LOSS_PCT),
                "target": c["price"] * (1 - PROFIT_TARGET_PCT),
            }
            open_pos[pair] = entry_record
            log_trade(NAME, pair, "open_short", qty, c["price"],
                      note=f"entry:tsmom{c['tsmom']*100:.1f}%_vol{c['vol_spike']:.1f}x")
            tag_entry(f"intradayshort:{pair}", sleeve=NAME, side="short",
                      entry_price=c["price"], qty=qty)
            actions.append({"action": "entry_short", "pair": pair,
                            "price": c["price"], "qty": qty,
                            "tsmom": c["tsmom"], "rsi": c["rsi"]})
        except Exception as e:
            actions.append({"action": "entry_failed", "pair": pair, "error": str(e)})

    state["open_positions"] = open_pos
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return {"status": "ok", "regime": "bear",
            "n_candidates": len(candidates), "n_open_positions": len(open_pos),
            "actions": actions}
