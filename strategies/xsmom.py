"""Cross-sectional momentum (XSMOM) live cycle.

Long top 2 / short bottom 2 by 14-day return rank. Rebalance weekly.
Wider universe than pro_trend so the cross-section is meaningful.

Capital allocation: 30% of paper account ($30k of $100k), pro_trend gets 70%.

Backtest: standalone Sharpe 0.31, correlation to pro_trend +0.17.
70/30 portfolio: Sharpe 1.40 (same as pro_trend solo), MaxDD 30% (vs 40% solo)
— diversification reduces DD by 10pp at no Sharpe cost.

State file: .xsmom_state.json — current target weights per pair + last rebalance.
Tagged in pnl_attribution under sleeve='xsmom'.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.broker import Broker
from core.perp_broker import PerpBroker
from core.pnl_attribution import tag_entry, untag
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_sharpe_scale, get_all_gates_scale


STATE_FILE = REPO_ROOT / ".xsmom_state.json"

# Universe — broader than pro_trend so cross-section is rich
XSMOM_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ATOM/USDT",
]

# Strategy parameters (validated 2026-05-10)
MOMENTUM_WINDOW_DAYS = 14
REBALANCE_FREQ_DAYS = 14   # rebalance every 2 weeks
LONG_N = 2
SHORT_N = 2
RISK_PER_LEG = 0.20         # 20% allocation per leg = 10% per pair x2 longs
STRATEGY_ALLOCATION = 0.30  # 30% of paper account


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "weights": {},        # pair -> target weight fraction
        "last_rebalance": None,
        "peak_equity": 30_000.0,  # 30% of $100k
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def reset_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def _days_since_rebalance(state: dict) -> int:
    last = state.get("last_rebalance")
    if not last:
        return 999
    try:
        last_dt = datetime.fromisoformat(last)
        return (datetime.now(timezone.utc) - last_dt).days
    except Exception:
        return 999


def _compute_target_weights() -> dict:
    """Rank pairs by 14-day return; long top 2, short bottom 2, equal-weight."""
    momentum = {}
    for pair in XSMOM_UNIVERSE:
        try:
            df = data.ohlcv_extended(pair, days_back=MOMENTUM_WINDOW_DAYS + 5)
            if df.empty or len(df) < MOMENTUM_WINDOW_DAYS:
                continue
            ret = df["close"].iloc[-1] / df["close"].iloc[-MOMENTUM_WINDOW_DAYS] - 1
            momentum[pair] = float(ret)
        except Exception:
            continue

    if len(momentum) < (LONG_N + SHORT_N):
        return {}

    ranked = sorted(momentum.items(), key=lambda x: -x[1])
    weights = {}
    for pair, _ in ranked[:LONG_N]:
        weights[pair] = RISK_PER_LEG / LONG_N
    for pair, _ in ranked[-SHORT_N:]:
        weights[pair] = -RISK_PER_LEG / SHORT_N
    return weights


def cycle(mode: str = "paper") -> dict:
    """One XSMOM cycle. Rebalances if it's been REBALANCE_FREQ_DAYS or more."""
    state = load_state()
    days_since = _days_since_rebalance(state)

    # Honor flash-crash kill-switch lockout
    lock_file = REPO_ROOT / ".kill_switch_lock.json"
    if lock_file.exists():
        try:
            lock_data = json.loads(lock_file.read_text())
            until = datetime.fromisoformat(lock_data["locked_until"])
            if datetime.now(timezone.utc) < until:
                return {"status": "locked_out", "lock_reason": lock_data.get("reason")}
        except Exception:
            pass

    if days_since < REBALANCE_FREQ_DAYS and state.get("weights"):
        return {
            "status": "ok",
            "action": "no_rebalance",
            "days_since_rebalance": days_since,
            "current_weights": state.get("weights", {}),
        }

    # Time to rebalance
    new_weights = _compute_target_weights()
    if not new_weights:
        return {"status": "insufficient_data"}

    # 2026-05-28 W9: spot only used for cash balance lookup. XSMOM allocations
    # are stored in its dedicated perp sub-account so spot ref is reserve.
    spot = Broker(mode=mode, long_only=False, sleeve="spot_reserve")
    perp = PerpBroker(mode=mode, sleeve="xsmom")
    cash = float(spot.get_balance().get("USDT", 0))

    # === Sleeve-level drawdown circuit breaker ===
    # Compute current XSMOM equity = perp cash + mark-to-market of XSMOM positions
    perp_cash = float(perp._state.cash_quote)
    xsmom_mtm = 0.0
    for p in XSMOM_UNIVERSE:
        base = p.split("/")[0]
        qty = perp._state.positions.get(base, 0.0)
        if abs(qty) < 1e-12:
            continue
        try:
            px = float(perp.ticker(p).get("last") or 0)
            entry = perp._state.entry_prices.get(base, px)
            xsmom_mtm += qty * (px - entry)
        except Exception:
            pass
    current_equity = max(perp_cash + xsmom_mtm, 1.0)
    sleeve_scale = apply_sleeve_scaling("xsmom", current_equity)
    if is_paused("xsmom"):
        return {
            "status": "sleeve_paused",
            "reason": "drawdown circuit breaker > 20%",
            "note": "Run: python -m ops.sleeve_circuit_breakers reset xsmom",
        }

    # W16.H: full gates pipeline (CB + Sharpe + loss-streak + correlation + event
    # + meta-confidence + BTC dominance). XSMOM is alt-leaning — gate by
    # dominance regime so it cuts size when capital flees to BTC.
    gates = get_all_gates_scale("xsmom", alt_regime=True)
    if gates["event_active"]:
        return {
            "status": "event_window_paused",
            "event": gates["event_name"],
            "note": "rebalance deferred until after high-vol event",
        }
    if gates["effective"] <= 0.0:
        return {
            "status": "dominance_paused",
            "regime": gates.get("dominance_regime"),
            "note": "XSMOM gated by BTC dominance regime (BTC_HEGEMONY zero allocation)",
        }
    sharpe_scale = gates["sharpe_scale"]
    effective_scale = gates["effective"]
    strategy_capital = cash * STRATEGY_ALLOCATION * effective_scale

    actions = []
    old_weights = state.get("weights", {})

    # Close any positions in OLD universe but not in new weights
    for pair, _ in old_weights.items():
        if pair not in new_weights:
            try:
                perp.close_position(pair)
                untag(f"xsmom:{pair}")
                actions.append({"action": "close_old", "pair": pair})
            except Exception as e:
                actions.append({"action": "close_old_failed", "pair": pair, "error": str(e)})

    # Open / adjust new positions
    for pair, target_w in new_weights.items():
        target_notional = strategy_capital * target_w
        if abs(target_notional) < 100:
            continue
        try:
            current_price = float(perp.ticker(pair).get("last") or 0)
            if current_price <= 0:
                continue
            qty = abs(target_notional) / current_price
            side = "long" if target_w > 0 else "short"
            # Close existing if direction differs, then open fresh
            try:
                perp.close_position(pair)
                untag(f"xsmom:{pair}")
            except Exception:
                pass
            perp.open_position(pair, side, abs(target_notional))
            tag_entry(f"xsmom:{pair}", sleeve="xsmom",
                      side=side, entry_price=current_price, qty=qty)
            actions.append({
                "action": "rebalance", "pair": pair, "side": side,
                "weight": target_w, "notional": target_notional,
            })
        except Exception as e:
            actions.append({"action": "rebalance_failed", "pair": pair, "error": str(e)})

    save_state({
        "weights": new_weights,
        "last_rebalance": datetime.now(timezone.utc).isoformat(),
        "peak_equity": state.get("peak_equity", strategy_capital),
    })

    return {
        "status": "ok",
        "action": "rebalanced",
        "n_actions": len(actions),
        "actions": actions,
        "new_weights": new_weights,
    }


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(cycle(), indent=2, default=str))
