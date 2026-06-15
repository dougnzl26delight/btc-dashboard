"""Intraday momentum sleeve — 15-min cadence on 4h-bar signals.

Designed for ACTIVITY. Fires multiple trades per day on intraday breakouts.

Entry rules per pair:
    - 4h-bar TSMOM_3 (last 3 4h-bars = 12h return) > +1.5%
    - 4h-bar volume in last bar > 1.5x 20-bar average (volume spike)
    - RSI(14) on 4h between 50-75 (not overbought, momentum confirmed)
    - Long only (matches rig long-bias)

Exit rules:
    - +3% target hit
    - -1.5% stop loss
    - 24h time cap
    - Trail stop: 1.5 ATR below highest mid

Per trade: 3% of sleeve capital (~$300 on $10k baseline).
Max 4 concurrent positions.

Filters:
    - Honors event_calendar (no new entries during high-vol windows)
    - Honors all gates (CB, Sharpe, loss-streak, correlation)
    - Real-time monitor watches stops/targets continuously
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
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_all_gates_scale


NAME = "intraday_momentum"
STATE_FILE = REPO_ROOT / ".intraday_momentum_state.json"

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "BNB/USDT",
    "ATOM/USDT", "DOT/USDT",
]

# Sleeve allocation (sub-account funded with $10k baseline)
SLEEVE_BASELINE = 10_000.0
PER_TRADE_PCT = 0.03               # 3% per trade = ~$300
MAX_CONCURRENT = 4

# Entry filters (4h-bar based)
TSMOM_3BAR_THRESHOLD = 0.015        # 1.5% over last 12h
VOLUME_SPIKE_MULT = 1.5             # 1.5x 20-bar volume avg
RSI_MIN = 50
RSI_MAX = 75

# W16.D: Multi-timeframe confluence floor.
# Single-TF entries chase noise. Require ≥0.5 alignment AND net direction up.
# Backtest from multi_timeframe.py:
#   confluence < 0.5 -> filter out (too noisy)
#   confluence 0.5-0.8 -> normal sizing
#   confluence > 0.8 -> can upsize via meta-confidence
#
# W16.D-regime-gate (added 2026-06-01 from _bt_mtf_bull_regime.py finding):
#   Backtests across 3 regimes showed:
#     2021 H1 BULL  (BTC +109%):  MTF lift +0.27%/sig (HELPS)
#     2023 H2 CHOP  (BTC flat):   MTF lift +0.96%/sig (HELPS most)
#     2022 H1 BEAR  (BTC -60%):   MTF lift -1.99%/sig (HURTS badly)
#     2026 H1 (cur) (BTC -4%):    MTF lift -0.79%/sig (HURTS)
#   Discriminator: BTC 30-day return. Below MTF_DOWNTREND_THRESHOLD = clear bear.
#   Skip the MTF filter in clear bear so mean-reversion bounces aren't rejected.
MIN_MTF_CONFLUENCE = 0.5
MTF_DOWNTREND_THRESHOLD = -0.08  # if BTC 30d < -8%, skip MTF (bear regime)

# Exit
PROFIT_TARGET_PCT = 0.03            # +3%
STOP_LOSS_PCT = 0.015               # -1.5%
TIME_CAP_HOURS = 24
TRAIL_ATR_MULT = 1.5


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _4h_bars(pair: str, days_back: int = 30) -> pd.DataFrame:
    """Resample daily bars to 4h via simulated price walk (daily approximation).
    True 4h bars would need `data.ohlcv_extended(pair, timeframe='4h')` — use that if available.
    """
    try:
        # Try fetching 4h directly if data module supports it
        df = data.ohlcv_extended(pair, days_back=days_back)
        if df.empty:
            return pd.DataFrame()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def _mtf_confluence(pair: str) -> dict | None:
    """W16.D: query multi-timeframe confluence for a pair. Returns None on failure.

    Wrapped in try so a single pair's network failure can't halt the scan.
    """
    try:
        from core.multi_timeframe import confluence
        return confluence(pair)
    except Exception:
        return None


def _mtf_should_apply() -> bool:
    """W16.D regime gate: return True if MTF filter should be applied.

    Backtest evidence (_bt_mtf_bull_regime.py 2026-06-01): MTF filter HELPS in
    BULL and CHOP regimes (lift +0.27% to +0.96% per signal) but HURTS in clear
    BEAR (lift -1.99%/sig in 2022 H1, -0.79%/sig in 2026 H1).

    Discriminator: BTC 30-day return. Skip MTF when BTC is in clear downtrend
    so mean-reversion bounces can still fire through base filters.

    Fail OPEN (apply MTF) on data fetch failure — conservative default.
    """
    try:
        from core import data as _data
        df = _data.ohlcv_extended("BTC/USDT", days_back=35)
        if df.empty or len(df) < 31:
            return True  # default to applying
        ret_30d = float(df["close"].iloc[-1] / df["close"].iloc[-31] - 1)
        # If BTC is in clear downtrend, skip MTF
        return ret_30d > MTF_DOWNTREND_THRESHOLD
    except Exception:
        return True  # fail safe — apply MTF


def _scan_for_momentum() -> list[dict]:
    """Find pairs matching entry filters. Returns sorted by signal strength.

    W16.D: candidates must additionally satisfy multi-timeframe confluence
    (1h/4h/1d/1w) ≥ 0.5 with net_direction > 0. Single-TF entries chase noise.
    Regime-gated (2026-06-01): MTF skipped when BTC 30d < -8% (clear bear)
    because backtests showed MTF rejects winning mean-reversion bounces in bear.
    """
    apply_mtf = _mtf_should_apply()
    candidates = []
    rejected_for_mtf = 0
    for pair in UNIVERSE:
        try:
            df = _4h_bars(pair, days_back=10)
            if df.empty or len(df) < 25:
                continue
            current = float(df["close"].iloc[-1])
            tsmom_3bar = current / float(df["close"].iloc[-4]) - 1  # 3 bars back ~ 12h
            vol_20avg = float(df["volume"].iloc[-20:].mean())
            vol_now = float(df["volume"].iloc[-1])
            vol_spike = vol_now / vol_20avg if vol_20avg > 0 else 0
            rsi = _rsi(df["close"])
            atr = _atr(df)

            # Apply base filters
            if not (tsmom_3bar > TSMOM_3BAR_THRESHOLD
                    and vol_spike > VOLUME_SPIKE_MULT
                    and RSI_MIN <= rsi <= RSI_MAX):
                continue

            # W16.D: multi-timeframe confluence filter — regime-gated.
            # Only apply MTF in non-bear regimes (BTC 30d > -8%).
            mtf_score = 0.0
            mtf = None
            if apply_mtf:
                mtf = _mtf_confluence(pair)
                mtf_score = mtf.get("confluence_score", 0.0) if mtf else 0.0
                mtf_dir = mtf.get("net_direction", 0) if mtf else 0
                # If MTF data available and aligned bearish or weak, reject.
                if mtf is not None and (mtf_score < MIN_MTF_CONFLUENCE or mtf_dir <= 0):
                    rejected_for_mtf += 1
                    continue

            candidates.append({
                "pair": pair, "price": current, "tsmom": tsmom_3bar,
                "vol_spike": vol_spike, "rsi": rsi, "atr": atr,
                "mtf_confluence": mtf_score,
                "mtf_verdict": mtf.get("verdict", "no_data") if mtf else "no_data",
                # Composite: tsmom × vol × MTF confluence → strong multi-TF entries rank first
                "signal_strength": tsmom_3bar * vol_spike * max(mtf_score, 0.5),
            })
        except Exception:
            continue
    candidates.sort(key=lambda x: -x["signal_strength"])
    if rejected_for_mtf:
        try:
            from core.pnl_db import log_signal
            log_signal(NAME, "_mtf_filter", float(rejected_for_mtf),
                       note=f"rejected_for_mtf_confluence<{MIN_MTF_CONFLUENCE}")
        except Exception:
            pass
    return candidates


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"open_positions": {}, "history": []}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def cycle(mode: str = "paper") -> dict:
    """One intraday momentum cycle. Runs every 15 min via cron."""
    state = load_state()
    open_pos = state.get("open_positions", {})

    # 2026-06-01 — empirical regime pause (ops.regime_gate).
    # Backtest: intraday_momentum loses money in clear bear (BTC 30d < -8%).
    # Pause new entries prospectively rather than waiting for DD CB.
    # Existing positions still managed (exit logic runs below).
    try:
        from ops.regime_gate import should_pause_sleeve
        pause = should_pause_sleeve("intraday_momentum")
        if pause["should_pause"]:
            # Still run exit logic on open positions; only skip NEW entries
            entries_blocked_reason = pause["reason"]
        else:
            entries_blocked_reason = None
    except Exception:
        entries_blocked_reason = None

    # Sleeve scaling
    sleeve_scale = apply_sleeve_scaling("intraday_momentum", SLEEVE_BASELINE)
    if is_paused("intraday_momentum"):
        return {"status": "sleeve_paused"}
    gates = get_all_gates_scale("intraday_momentum")
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused", "event": gates["event_name"]}

    spot = Broker(mode=mode, long_only=True, sleeve="intraday_momentum")
    cash = float(spot.get_balance().get("USDT", 0))
    actions = []

    # === Exit logic for open positions ===
    for pair, info in list(open_pos.items()):
        try:
            df = _4h_bars(pair, days_back=5)
            if df.empty:
                continue
            current = float(df["close"].iloc[-1])
        except Exception:
            continue

        entry = info["entry_price"]
        qty = info["qty"]
        opened_at = datetime.fromisoformat(info["opened_at"])
        age_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600

        pnl_pct = (current / entry - 1)
        exit_reason = None
        if pnl_pct >= PROFIT_TARGET_PCT:
            exit_reason = "target_hit"
        elif pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = "stop_loss"
        elif age_hours >= TIME_CAP_HOURS:
            exit_reason = "time_cap"
        else:
            # Trail stop check
            peak = info.get("peak_price", entry)
            new_peak = max(peak, current)
            atr = _atr(df)
            trail_level = new_peak - TRAIL_ATR_MULT * atr
            if current < trail_level and pnl_pct > 0:
                exit_reason = "trail_stop"
            info["peak_price"] = new_peak

        if exit_reason:
            notional = qty * current
            realized = (current - entry) * qty
            try:
                spot.place_market_order(pair, "sell", notional)
                log_trade(NAME, pair, "sell", qty, current, realized_pnl=realized,
                          note=f"exit:{exit_reason}")
                untag(f"intraday:{pair}")
                actions.append({
                    "action": "exit", "pair": pair, "reason": exit_reason,
                    "entry": entry, "exit": current, "realized_pnl": realized,
                    "pnl_pct": pnl_pct,
                })
                state.setdefault("history", []).append({
                    **info, "closed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_price": current, "exit_reason": exit_reason,
                    "realized_pnl": realized,
                })
                del open_pos[pair]
            except Exception as e:
                actions.append({"action": "exit_failed", "pair": pair, "error": str(e)})

    # === Entry logic ===
    # Skip entirely if regime-paused (bear). Exit logic above still ran.
    if entries_blocked_reason == "bear_regime_pause":
        state["open_positions"] = open_pos
        state["last_scan"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        return {
            "status": "bear_regime_paused",
            "n_open_positions": len(open_pos),
            "actions": actions,
            "note": "intraday_momentum paused — BTC 30d < -8% (clear bear). "
                    "Backtest shows -2%/sig in 2022 H1 bear. Resume on regime change.",
        }

    candidates = _scan_for_momentum()
    log_signal(NAME, "_universe", float(len(candidates)),
               note=f"n_candidates={len(candidates)}")

    available_slots = MAX_CONCURRENT - len(open_pos)
    for c in candidates[:available_slots]:
        pair = c["pair"]
        if pair in open_pos:
            continue
        notional = SLEEVE_BASELINE * PER_TRADE_PCT * effective_scale
        if notional < 100 or notional > cash:
            continue
        try:
            qty = notional / c["price"]
            spot.place_market_order(pair, "buy", notional)
            entry_record = {
                "entry_price": c["price"],
                "qty": qty,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_tsmom": c["tsmom"],
                "entry_rsi": c["rsi"],
                "entry_vol_spike": c["vol_spike"],
                "entry_mtf_confluence": c.get("mtf_confluence"),
                "entry_mtf_verdict": c.get("mtf_verdict"),
                "peak_price": c["price"],
                "stop_loss": c["price"] * (1 - STOP_LOSS_PCT),
                "target": c["price"] * (1 + PROFIT_TARGET_PCT),
            }
            open_pos[pair] = entry_record
            log_trade(NAME, pair, "buy", qty, c["price"],
                      note=f"entry:tsmom{c['tsmom']*100:.1f}%_vol{c['vol_spike']:.1f}x")
            tag_entry(f"intraday:{pair}", sleeve=NAME, side="long",
                      entry_price=c["price"], qty=qty)
            actions.append({
                "action": "entry", "pair": pair,
                "price": c["price"], "qty": qty,
                "tsmom": c["tsmom"], "rsi": c["rsi"],
                "mtf_confluence": c.get("mtf_confluence"),
                "mtf_verdict": c.get("mtf_verdict"),
            })
        except Exception as e:
            actions.append({"action": "entry_failed", "pair": pair, "error": str(e)})

    state["open_positions"] = open_pos
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return {
        "status": "ok",
        "n_candidates": len(candidates),
        "n_open_positions": len(open_pos),
        "actions": actions,
    }
