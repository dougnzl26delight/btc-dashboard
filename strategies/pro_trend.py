"""Pro trend follower — generalized to any pair, long AND short via perp broker.

Same mechanics as pro_trend_btc but:
  - Pair-parameterized (BTC, ETH, SOL — proven to work in backtest)
  - LONG via spot broker when price > SMA200
  - SHORT via perp broker when price < SMA200 (Donchian LOW breakout)
  - Per-pair state file: .pro_trend_state_{base}.json

Backtest evidence:
  BTC: +24% / Sharpe 0.50 / DD 13% over 2000 days
  ETH: +20% / Sharpe 0.45 / DD 14% over 2000 days (BEAT BAH which was -14%)
  SOL: +107% / Sharpe 0.92 / DD 22% over 2000 days
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.broker import Broker
from core.catalyst_signals import combined_catalyst_multiplier
from core.perp_broker import PerpBroker
from core.pnl_attribution import tag_entry, untag
from core.swing_backtest import compute_atr


REPO_ROOT = Path(__file__).resolve().parent.parent

# Top 5 by per-pair Sharpe over 1500 days (universe_size_test.py 2026-05-10):
# concentrating beats expanding — shrinking from 11 to 5 raised mean
# annualized from +4.35% to +10.62% (per-pair), and from +2.13% to +45.28%
# at the portfolio level once portfolio_risk_cap is on.
PRO_TREND_PAIRS = ["SOL/USDT", "BTC/USDT", "OP/USDT", "AVAX/USDT", "ETH/USDT"]

# Production parameters (calibrated 2026-05-10)
SMA_FILTER = 200
DONCHIAN_WINDOW = 20
ATR_PERIOD = 14
ATR_STOP_MULT = 4.0
RISK_PCT_PER_UNIT = 0.04
PYRAMID_ATR_STEP = 2.0
MAX_PYRAMID_UNITS = 2
# DD kill at 0.35. Backtest (param_sweep.py 2026-05-10) showed 0.30 had
# Sharpe 1.48 on FULL 6.3y history vs 1.40 here, BUT kept the kill at 0.35
# because: (a) on the 3.4y chop-dominated subwindow 0.30 generated 127
# kills vs 38, dropping to +11% ann; (b) real execution suffers from
# kill+reentry far more than backtest assumes. 0.35 is more robust across
# regimes the rig will actually face in 2026-2028.
DRAWDOWN_KILL_PCT = 0.35
ROUND_TRIP_BPS = 30
# Leverage multiplier — longs route to perp at this notional multiplier.
LEVERAGE_MULTIPLIER = 1.5
# Portfolio risk cap — total active risk across pairs. With 5 pairs at 4% each
# uncapped, simultaneous entries push to 20% concurrent risk and the 35% DD
# kill triggers ~114x in 1500 days. Cap at 15% raises Sharpe from 0.26 to 1.01.
PORTFOLIO_RISK_CAP = 0.15
# Catalyst overlay (halving cycle + ETF flows) — DISABLED. Backtest showed it
# costs 24pp of annualized return for zero Sharpe benefit (catalyst_overlay_test.py).
USE_CATALYST_OVERLAY = False

# W16.D: Multi-timeframe confluence requirement.
# pro_trend fires on Donchian breakouts; require higher-timeframe alignment
# to filter false breakouts within larger consolidations. Below 0.5 = skip.
# Regime-gated (2026-06-01 _bt_mtf_bull_regime.py): skip MTF in clear bear
# (BTC 30d < -8%) where backtest showed MTF rejects winning mean-reversion.
MIN_MTF_CONFLUENCE = 0.5
MTF_DOWNTREND_THRESHOLD = -0.08


def _state_file(pair: str) -> Path:
    base = pair.split("/")[0]
    return REPO_ROOT / f".pro_trend_state_{base}.json"


def load_state(pair: str) -> dict:
    f = _state_file(pair)
    if f.exists():
        return json.loads(f.read_text())
    return {
        "side": None,           # "long" / "short" / None
        "units": [],
        "extreme": 0.0,         # high-water for long, low-water for short
        "trail_stop": 0.0,
        "peak_equity": 100_000.0,
    }


def save_state(pair: str, state: dict) -> None:
    _state_file(pair).write_text(json.dumps(state, indent=2, default=str))


def reset_state(pair: str) -> None:
    f = _state_file(pair)
    if f.exists():
        f.unlink()


def count_active_pairs(exclude_pair: str | None = None) -> int:
    """Count pairs with an open pro_trend position by scanning state files.

    Used to apply PORTFOLIO_RISK_CAP at entry: each new entry gets at most
    PORTFOLIO_RISK_CAP / max(1, n_active) of equity at risk. The exclude_pair
    arg is for the entry path — a pair about to enter shouldn't count itself.

    Corrupt or unreadable state files are skipped (treated as "no position")
    so a single bad file doesn't take down the whole cycle.
    """
    n = 0
    for p in PRO_TREND_PAIRS:
        if p == exclude_pair:
            continue
        try:
            st = load_state(p)
        except Exception:
            continue
        if st.get("units"):
            n += 1
    return n


def effective_risk_pct(pair: str, base_risk: float = RISK_PCT_PER_UNIT) -> float:
    """Apply portfolio cap: total active risk across all pairs <= PORTFOLIO_RISK_CAP."""
    n_active = count_active_pairs(exclude_pair=pair) + 1  # this pair will be active
    per_pair_max = PORTFOLIO_RISK_CAP / n_active
    return min(base_risk, per_pair_max)


def cycle(
    pair: str,
    mode: str = "paper",
    enable_shorts: bool = True,
) -> dict:
    """Run one pro-trend cycle for `pair`. Both directions if enable_shorts."""
    df = data.ohlcv_extended(pair, days_back=400)
    if df.empty or len(df) < SMA_FILTER + 10:
        return {"pair": pair, "status": "insufficient_data"}

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(DONCHIAN_WINDOW).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_WINDOW).min().shift(1)
    df["sma_filter"] = df["close"].rolling(SMA_FILTER).mean()
    df["atr"] = compute_atr(df, ATR_PERIOD)
    # 2026-05-11 entry-filter upgrade: TSMOM_30 and MACD_hist confirmation.
    # Indicator factor mining over 4 years (indicator_strategy_test.py) showed
    # v5 = baseline + TSMOM>0 + MACD_hist>0 beats baseline on walk-forward
    # stability (WF std 0.76 vs 1.27, min fold -0.49 vs -1.45).
    df["tsmom30"] = df["close"].pct_change(30)
    _ema_fast = df["close"].ewm(span=12).mean()
    _ema_slow = df["close"].ewm(span=26).mean()
    _macd = _ema_fast - _ema_slow
    _macd_sig = _macd.ewm(span=9).mean()
    df["macd_hist"] = (_macd - _macd_sig) / df["close"]
    df = df.dropna()

    last = df.iloc[-1]
    price = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    atr = float(last["atr"])
    sma = float(last["sma_filter"])
    donchian_high = float(last["donchian_high"])
    donchian_low = float(last["donchian_low"])
    tsmom30 = float(last["tsmom30"])
    macd_hist = float(last["macd_hist"])
    in_bull = price > sma

    state = load_state(pair)
    side = state.get("side")
    units = state.get("units", [])
    extreme = float(state.get("extreme", 0))
    trail_stop = float(state.get("trail_stop", 0))
    peak_equity = float(state.get("peak_equity", 100_000))

    spot = Broker(mode=mode, long_only=False, sleeve="pro_trend")
    base = pair.split("/")[0]

    # Mark-to-market across all units
    if side == "long":
        pos_qty = sum(u["qty"] for u in units)
        unrealized = pos_qty * price - sum(u["qty"] * u["entry_price"] for u in units)
    elif side == "short":
        pos_qty = -sum(u["qty"] for u in units)
        unrealized = pos_qty * price + sum(u["qty"] * u["entry_price"] for u in units)
    else:
        pos_qty = 0
        unrealized = 0

    # Approximate equity from a snapshot of cash + this asset's exposure
    cash = float(spot.get_balance().get("USDT", 0))
    mtm_eq = cash + (pos_qty * price if side == "long" else 0) + (unrealized if side == "short" else 0)
    if mtm_eq > peak_equity:
        peak_equity = mtm_eq
    equity_dd = max(0.0, 1 - mtm_eq / peak_equity) if peak_equity > 0 else 0

    actions: list[dict] = []

    # === DD KILL ===
    if equity_dd > DRAWDOWN_KILL_PCT and units:
        if side == "long":
            for u in units:
                spot.place_market_order(pair, "sell", u["qty"] * price)
        elif side == "short" and enable_shorts:
            perp = PerpBroker(mode=mode, sleeve="pro_trend")
            perp.close_position(pair)
        actions.append({"action": "dd_kill", "side": side, "n_units": len(units)})
        units = []
        side = None
        extreme = trail_stop = 0
        untag(pair)

    # === EXIT ===
    if side == "long" and units:
        if high > extreme:
            extreme = high
            new_trail = high - ATR_STOP_MULT * atr
            if new_trail > trail_stop:
                trail_stop = new_trail
        if low <= trail_stop or price < sma:
            for u in units:
                spot.place_market_order(pair, "sell", u["qty"] * price)
            actions.append({
                "action": "exit_long", "n_units_closed": len(units),
                "exit_price": price,
                "reason": "trail_stop" if low <= trail_stop else "sma_break",
            })
            units = []
            side = None
            extreme = trail_stop = 0
            untag(pair)

    # NB: short EXIT logic always runs (no enable_shorts gate).
    # Existing short positions (e.g. force-entered) must always be managed.
    # Only NEW short ENTRIES are gated by enable_shorts below.
    elif side == "short" and units:
        if low < extreme or extreme == 0:
            extreme = low
            new_trail = low + ATR_STOP_MULT * atr
            if trail_stop == 0 or new_trail < trail_stop:
                trail_stop = new_trail
        if high >= trail_stop or price > sma:
            perp = PerpBroker(mode=mode, sleeve="pro_trend")
            close_result = perp.close_position(pair)
            actions.append({
                "action": "exit_short", "n_units_closed": len(units),
                "exit_price": price,
                "reason": "trail_stop" if high >= trail_stop else "sma_break",
                "realized": close_result.get("realized_pnl", 0),
            })
            units = []
            side = None
            extreme = trail_stop = 0
            untag(pair)

    # === PYRAMID ===
    if units and len(units) < MAX_PYRAMID_UNITS:
        last_unit = units[-1]
        # Pyramid risk uses the same portfolio-cap as entry — pair already
        # counts as active here, so don't exclude itself.
        n_active = count_active_pairs(exclude_pair=None)
        risk_pct = min(RISK_PCT_PER_UNIT, PORTFOLIO_RISK_CAP / max(n_active, 1))
        if side == "long" and high >= last_unit["entry_price"] + PYRAMID_ATR_STEP * last_unit["entry_atr"]:
            stop_dist = ATR_STOP_MULT * atr
            if stop_dist > 0:
                qty = (mtm_eq * risk_pct) / stop_dist
                qty = min(qty, mtm_eq * 0.25 / price)
                spot.place_market_order(pair, "buy", qty * price)
                units.append({"qty": qty, "entry_price": price, "entry_atr": atr})
                actions.append({"action": "pyramid_long", "qty": qty, "entry": price})
        elif side == "short" and enable_shorts and low <= last_unit["entry_price"] - PYRAMID_ATR_STEP * last_unit["entry_atr"]:
            stop_dist = ATR_STOP_MULT * atr
            if stop_dist > 0:
                qty = (mtm_eq * risk_pct) / stop_dist
                qty = min(qty, mtm_eq * 0.25 / price)
                perp = PerpBroker(mode=mode, sleeve="pro_trend")
                perp.open_position(pair, "short", qty * price)
                units.append({"qty": qty, "entry_price": price, "entry_atr": atr})
                actions.append({"action": "pyramid_short", "qty": qty, "entry": price})

    # === ENTRY (portfolio-capped sizing + leverage routing) ===
    # Honor flash-crash lockout: skip new entries when kill switch is active.
    # Existing position management (trail, pyramid, exit) ran above and is
    # NOT blocked — risk-reducing trades still proceed.
    lock_file = REPO_ROOT / ".kill_switch_lock.json"
    kill_locked = False
    if lock_file.exists():
        try:
            import datetime as _dt
            lock_data = json.loads(lock_file.read_text())
            locked_until = _dt.datetime.fromisoformat(lock_data["locked_until"])
            if _dt.datetime.now(_dt.timezone.utc) < locked_until:
                kill_locked = True
                actions.append({"action": "entry_skipped",
                                "reason": "kill_switch_lockout"})
        except Exception:
            pass

    if not units and not kill_locked:
        catalyst = (
            combined_catalyst_multiplier()["combined_mult"]
            if USE_CATALYST_OVERLAY else 1.0
        )
        # Portfolio cap: each pair gets at most RISK_PCT_PER_UNIT, but if other
        # pairs are already active, scale down so total stays under the cap.
        risk_pct = effective_risk_pct(pair)
        size_mult = LEVERAGE_MULTIPLIER * catalyst

        # v5 entry filters (2026-05-11 / 2026-05-28):
        #   LONG : TSMOM_30 > 0 AND MACD_hist > 0  (filter chop / false breakouts)
        #   SHORT: TSMOM_30 < -0.10 AND MACD_hist < 0  (require DEEP bear momentum;
        #          unfiltered shorts fire at capitulation bottoms — cost ~50bp Sharpe)
        long_filter_pass = tsmom30 > 0 and macd_hist > 0
        short_filter_pass = tsmom30 < -0.10 and macd_hist < 0

        # W16.D: Multi-timeframe confluence — require ≥0.5 confluence with
        # direction matching trade side. Filters single-TF Donchian fakeouts.
        # Regime gate: skip MTF entirely in clear bear (BTC 30d < -8%) where
        # backtest showed MTF rejects winning mean-reversion bounces.
        mtf_ok_long = True
        mtf_ok_short = True
        mtf_score = None
        try:
            # Use BTC 30d return to decide whether MTF applies in current regime
            btc_30d_check = data.ohlcv_extended("BTC/USDT", days_back=35)
            apply_mtf = True
            if not btc_30d_check.empty and len(btc_30d_check) >= 31:
                btc_30d_ret = float(btc_30d_check["close"].iloc[-1] /
                                     btc_30d_check["close"].iloc[-31] - 1)
                apply_mtf = btc_30d_ret > MTF_DOWNTREND_THRESHOLD
            if apply_mtf:
                from core.multi_timeframe import confluence
                mtf = confluence(pair)
                mtf_score = mtf.get("confluence_score", 0.0)
                mtf_dir = mtf.get("net_direction", 0)
                mtf_ok_long = (mtf_score >= MIN_MTF_CONFLUENCE and mtf_dir > 0)
                mtf_ok_short = (mtf_score >= MIN_MTF_CONFLUENCE and mtf_dir < 0)
        except Exception:
            # MTF data unavailable — let base filters handle. Fail open.
            pass

        if in_bull and high >= donchian_high and atr > 0 and long_filter_pass and mtf_ok_long:
            stop_dist = ATR_STOP_MULT * atr
            qty = (mtm_eq * risk_pct * size_mult) / stop_dist
            notional_cap_frac = 0.30 * LEVERAGE_MULTIPLIER
            qty = min(qty, mtm_eq * notional_cap_frac / price)
            if LEVERAGE_MULTIPLIER > 1.0:
                perp = PerpBroker(mode=mode, sleeve="pro_trend")
                perp.open_position(pair, "long", qty * price)
            else:
                spot.place_market_order(pair, "buy", qty * price)
            units = [{"qty": qty, "entry_price": price, "entry_atr": atr,
                      "size_mult": size_mult, "risk_pct": risk_pct}]
            side = "long"
            extreme = high
            trail_stop = price - stop_dist
            actions.append({
                "action": "entry_long", "qty": qty, "entry": price,
                "stop": trail_stop, "size_mult": size_mult, "risk_pct": risk_pct,
            })
            tag_entry(pair, sleeve="systematic_pro_trend", side="long",
                      entry_price=price, qty=qty)
        elif not in_bull and enable_shorts and low <= donchian_low and atr > 0 and short_filter_pass and mtf_ok_short:
            stop_dist = ATR_STOP_MULT * atr
            qty = (mtm_eq * risk_pct * size_mult) / stop_dist
            notional_cap_frac = 0.30 * LEVERAGE_MULTIPLIER
            qty = min(qty, mtm_eq * notional_cap_frac / price)
            perp = PerpBroker(mode=mode, sleeve="pro_trend")
            perp.open_position(pair, "short", qty * price)
            units = [{"qty": qty, "entry_price": price, "entry_atr": atr,
                      "size_mult": size_mult, "risk_pct": risk_pct}]
            side = "short"
            extreme = low
            trail_stop = price + stop_dist
            actions.append({
                "action": "entry_short", "qty": qty, "entry": price,
                "stop": trail_stop, "size_mult": size_mult, "risk_pct": risk_pct,
            })
            tag_entry(pair, sleeve="systematic_pro_trend", side="short",
                      entry_price=price, qty=qty)

    save_state(pair, {
        "side": side, "units": units,
        "extreme": extreme, "trail_stop": trail_stop,
        "peak_equity": peak_equity,
    })

    return {
        "pair": pair,
        "status": "ok",
        "price": price, "sma": sma,
        "donchian_high": donchian_high, "donchian_low": donchian_low,
        "atr": atr, "in_bull": in_bull,
        "side": side, "n_units": len(units),
        "trail_stop": trail_stop,
        "actions": actions,
    }


if __name__ == "__main__":
    import json as _j
    for p in PRO_TREND_PAIRS:
        r = cycle(p)
        print(f"{p}: {r['status']} | side={r.get('side')} | "
              f"price={r.get('price')} | sma={r.get('sma')} | "
              f"in_bull={r.get('in_bull')} | actions={len(r.get('actions',[]))}")
