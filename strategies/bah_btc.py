"""BAH BTC sleeve — passive BTC buy-and-hold at 20% target allocation.

Rationale (from strategy_bakeoff.py + three_sleeve_test.py 2026-05-11):

The recent 18-month chop regime was hostile to ALL trend strategies. BAH BTC
delivered Sharpe 0.44 with +10.2% return while pro_trend sat in cash because
no pair was above SMA200. Adding a BAH BTC sleeve at 20% allocation:
  - Improves recent 18-month return from +6.4% to +11.3%
  - Slightly improves full 6.3y Sharpe (1.48 -> 1.49)
  - Modestly increases max drawdown (28.5% -> 32.9%)

Mechanism:
  - Target: 20% of total paper capital ($20k of $100k baseline)
  - Buy BTC on first run, top up monthly if drift below target
  - Rebalance: monthly OR if drift exceeds +/-5pp from target

This is INTENTIONALLY passive — no signal, no exit. The whole point is to
capture the directional moves pro_trend's SMA200 filter prevents.

State file: .bah_btc_state.json — tracks current BTC qty + last rebalance.
Tagged in pnl_attribution under sleeve='bah_btc'.

Scheduled as Crypto_bah_btc_monthly (1st of each month).
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
from core.pnl_attribution import tag_entry, untag
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from ops.sharpe_gate import get_sharpe_scale, get_all_gates_scale


STATE_FILE = REPO_ROOT / ".bah_btc_state.json"

# 2026-05-31 W10: DYNAMIC allocation scaled by MVRV cycle position.
# Replaces static 10%. In deep bear: load up. In euphoria: exit.
# Backtest 2018-2022 with dynamic sizing vs static 10%: +15% terminal value
# over 4-year window. The single biggest long-term return lever in the rig.
BASELINE_CAPITAL = 100_000.0
TARGET_NOTIONAL = BASELINE_CAPITAL * 0.10  # default fallback if cycle data unavailable

# Cycle-position -> allocation pct mapping
# (cycle_score_max, allocation_pct)  — first match wins
CYCLE_ALLOCATION_TIERS = [
    (20,  0.20),   # DEEP_BEAR     -> 20% load up
    (40,  0.15),   # EARLY_BULL    -> 15%
    (60,  0.10),   # MID_BULL      -> 10%
    (80,  0.05),   # LATE_BULL     ->  5% de-risk
    (100, 0.00),   # EUPHORIA      ->  0% exit entirely
]


def _dynamic_target_pct() -> tuple[float, str]:
    """Return (allocation_pct, reason) based on current MVRV cycle + F&G sentiment.

    Primary driver = MVRV cycle position (5 tiers, DEEP_BEAR 20% → EUPHORIA 0%).
    Secondary overlay = Fear & Greed (W16.B):
      F&G <= 20 (EXTREME_FEAR) → bump to MAX_TIER allocation regardless of MVRV.
                                  Historical bottoms cluster here.
      F&G >= 80 (EXTREME_GREED) → halve allocation regardless of MVRV.
                                   Historical tops cluster here.
    """
    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        score = cp.get("score")
        if score is None:
            return 0.10, "fallback: no cycle data"
        for max_score, pct in CYCLE_ALLOCATION_TIERS:
            if score <= max_score:
                base_reason = f"cycle_score={score:.0f} ({cp.get('phase', '?')})"
                break
        else:
            return 0.0, "score > 100 (unexpected)"

        # W16.B: F&G sentiment overlay
        try:
            from core.fear_greed import latest as _fg_latest
            fg = _fg_latest()
            fg_val = fg.get("value")
            if fg_val is not None:
                if fg_val <= 20:
                    # Extreme fear → max conviction accumulation
                    bumped = max(pct, 0.20)
                    if bumped != pct:
                        return bumped, f"{base_reason} + F&G {fg_val} EXTREME_FEAR → bumped to {bumped*100:.0f}%"
                elif fg_val >= 80:
                    # Extreme greed → halve allocation
                    halved = pct * 0.5
                    return halved, f"{base_reason} + F&G {fg_val} EXTREME_GREED → halved to {halved*100:.1f}%"
        except Exception:
            pass

        return pct, base_reason
    except Exception as e:
        return 0.10, f"fallback: {e}"


# Backward-compat constant (read at import for tests; runtime uses _dynamic_target_pct)
TARGET_ALLOCATION_PCT = 0.10

# Drift tolerance for rebalance
DRIFT_REBAL_THRESHOLD = 0.05  # +/- 5pp

# Cycle-aware exit overlay (2026-05-28): if BTC drops below 0.7x SMA200
# (deep bear) AND makes a lower-low for 30 days, halve the position even
# inside the monthly rebalance window. This protects BAH from holding into
# a confirmed cycle-bottom drawdown.
CYCLE_EXIT_MAYER_THRESHOLD = 0.7
CYCLE_EXIT_LOWER_LOW_DAYS = 30


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "btc_qty": 0.0,
        "entry_price": 0.0,
        "target_notional": TARGET_NOTIONAL,
        "last_rebalance": None,
        "n_buys": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def cycle(mode: str = "paper") -> dict:
    """One BAH BTC cycle. Buys/tops-up to maintain ~$10k BTC exposure.

    Rebalances monthly or on drift > +/- 5pp.

    Source-of-truth (2026-05-28 refactor): current_qty is READ FROM THE BROKER,
    not from the state file. Previously the state file diverged from broker
    reality when other sleeves (e.g. oversold_bounce) bought/sold the same
    asset. Broker is now authoritative for "how much do I own"; state file is
    only used for metadata (last_rebalance, entry_price).
    """
    state = load_state()
    # Read broker for the actual current BTC qty — broker is the truth.
    # 2026-05-28 W9: uses isolated sub-account .paper_state_bah_btc.json
    # No more cross-sleeve interference.
    spot_broker = Broker(mode=mode, long_only=False, sleeve="bah_btc")
    bal = spot_broker.get_balance()
    broker_qty = float(bal.get("BTC", 0))
    # Treat anything < 1e-6 BTC ($0.07 at $73k) as zero (dust from rounding)
    current_qty = broker_qty if abs(broker_qty) > 1e-6 else 0.0
    # If state diverges from broker, log warning + sync state
    state_qty = float(state.get("btc_qty", 0))
    if abs(current_qty - state_qty) > 0.001:  # > 0.001 BTC = ~$70 divergence
        try:
            from ops.alerts import alert as _alert
            _alert(
                f"bah_btc state divergence: state claimed {state_qty:.6f} BTC, "
                f"broker has {current_qty:.6f}. Syncing to broker truth.",
                level="warning",
            )
        except Exception:
            pass
        # Update state file with broker truth — state files are metadata only
        state["btc_qty"] = current_qty

    # Honor flash-crash lockout
    lock_file = REPO_ROOT / ".kill_switch_lock.json"
    if lock_file.exists():
        try:
            lock_data = json.loads(lock_file.read_text())
            until = datetime.fromisoformat(lock_data["locked_until"])
            if datetime.now(timezone.utc) < until:
                return {"status": "locked_out",
                        "lock_reason": lock_data.get("reason")}
        except Exception:
            pass

    # Get current BTC price
    try:
        ticker = data._EX.fetch_ticker("BTC/USDT")
        current_price = float(ticker.get("last") or ticker.get("close") or 0)
    except Exception as e:
        return {"status": "ticker_failed", "error": str(e)}

    if current_price <= 0:
        return {"status": "no_price"}

    current_value = current_qty * current_price

    # === Sleeve-level drawdown circuit breaker ===
    # Track peak position value, scale target_notional by allowed dd-tier.
    # On first run (current_value == 0), uses target as the starting baseline.
    seed_equity = current_value if current_value > 0 else TARGET_NOTIONAL
    sleeve_scale = apply_sleeve_scaling("bah_btc", seed_equity)
    if is_paused("bah_btc"):
        return {
            "status": "sleeve_paused",
            "reason": "drawdown circuit breaker > 20%",
            "current_qty": current_qty,
            "current_value": current_value,
            "note": "Run: python -m ops.sleeve_circuit_breakers reset bah_btc",
        }

    # Use config as source of truth — state stored the prior target but
    # the operator may have changed TARGET_ALLOCATION_PCT (e.g., reducing
    # BAH from 20% to 10% on 2026-05-28). Always read from config.
    # === Cycle-driven dynamic allocation (W10) ===
    cycle_pct, cycle_reason = _dynamic_target_pct()
    base_target = BASELINE_CAPITAL * cycle_pct
    # === W10 + W16.H unified gate composition ===
    # btc_regime=True → lean in during BTC_HEGEMONY (1.2x), lighten during ALTSEASON (0.8x).
    # BAH BTC is the textbook btc_regime sleeve.
    gates = get_all_gates_scale("bah_btc", btc_regime=True)
    effective_scale = gates["effective"]
    if gates["event_active"]:
        return {"status": "event_window_paused", "event": gates["event_name"]}

    # === Cycle-aware exit overlay (v2 2026-05-28) ===
    # Uses ON-CHAIN MVRV as primary cycle signal, with Mayer-Multiple fallback.
    # Halves position in EUPHORIA (top zone) or DEEP_BEAR (capitulation hold).
    # Per cycle-position calibration:
    #   MVRV > 3.0  : late-bull/euphoria — halve (de-risk near top)
    #   MVRV < 0.8  : capitulation — halve (preserve capital, redeploy at bottom)
    #   else        : full size
    cycle_exit_triggered = False
    cycle_exit_reason = "none"
    try:
        from core.onchain import get_mvrv
        mvrv_data = get_mvrv()
        mvrv_val = mvrv_data.get("mvrv")
        if mvrv_val is not None:
            if mvrv_val > 3.0:
                cycle_exit_triggered = True
                cycle_exit_reason = f"euphoria (MVRV {mvrv_val:.2f} > 3.0)"
            elif mvrv_val < 0.8:
                cycle_exit_triggered = True
                cycle_exit_reason = f"capitulation (MVRV {mvrv_val:.2f} < 0.8)"
    except Exception:
        # Fallback to price-based Mayer
        try:
            df_btc = data.ohlcv_extended("BTC/USDT", days_back=250)
            if not df_btc.empty and len(df_btc) >= 200:
                sma200 = float(df_btc["close"].rolling(200).mean().iloc[-1])
                mayer = current_price / sma200 if sma200 > 0 else 1.0
                low_30d_ago = float(df_btc["low"].iloc[-30:-1].min())
                current_low = float(df_btc["low"].iloc[-1])
                if mayer < CYCLE_EXIT_MAYER_THRESHOLD and current_low < low_30d_ago:
                    cycle_exit_triggered = True
                    cycle_exit_reason = f"mayer fallback ({mayer:.2f})"
        except Exception:
            pass

    cycle_exit_scale = 0.5 if cycle_exit_triggered else 1.0
    target_notional = base_target * effective_scale * cycle_exit_scale
    drift = (current_value - target_notional) / target_notional if target_notional > 0 else 0

    # Determine if rebalance needed
    last_rebal = state.get("last_rebalance")
    needs_rebal = False
    reason = ""
    if current_qty == 0:
        needs_rebal = True
        reason = "initial_buy"
    elif last_rebal:
        try:
            last_dt = datetime.fromisoformat(last_rebal)
            days_since = (datetime.now(timezone.utc) - last_dt).days
            if days_since >= 30:
                needs_rebal = True
                reason = f"monthly_rebal ({days_since}d since last)"
        except Exception:
            pass
    if abs(drift) > DRIFT_REBAL_THRESHOLD:
        needs_rebal = True
        reason = f"drift_rebal (drift {drift:+.1%})"

    if not needs_rebal:
        return {
            "status": "ok", "action": "no_rebalance",
            "current_qty": current_qty,
            "current_value": current_value,
            "drift": drift,
            "last_rebalance": last_rebal,
        }

    # Compute target qty
    target_qty = target_notional / current_price
    qty_delta = target_qty - current_qty

    spot = spot_broker  # reuse the instance from the broker-truth read above
    actions = []

    if qty_delta > 0:
        # Buy more
        buy_notional = qty_delta * current_price
        try:
            spot.place_market_order("BTC/USDT", "buy", buy_notional)
            actions.append({
                "action": "buy", "qty": qty_delta, "price": current_price,
                "notional": buy_notional, "reason": reason,
            })
            # Update attribution tag — track total BAH BTC position
            untag("bah:BTC/USDT")  # remove old tag
            tag_entry("bah:BTC/USDT", sleeve="bah_btc", side="long",
                       entry_price=current_price, qty=target_qty)
        except Exception as e:
            actions.append({"action": "buy_failed", "error": str(e)})
    elif qty_delta < 0:
        # Sell down (rebalance back to target)
        sell_qty = abs(qty_delta)
        sell_notional = sell_qty * current_price
        try:
            spot.place_market_order("BTC/USDT", "sell", sell_notional)
            actions.append({
                "action": "sell", "qty": sell_qty, "price": current_price,
                "notional": sell_notional, "reason": reason,
            })
            untag("bah:BTC/USDT")
            tag_entry("bah:BTC/USDT", sleeve="bah_btc", side="long",
                       entry_price=current_price, qty=target_qty)
        except Exception as e:
            actions.append({"action": "sell_failed", "error": str(e)})

    save_state({
        "btc_qty": target_qty,
        "entry_price": current_price,
        "target_notional": target_notional,
        "last_rebalance": datetime.now(timezone.utc).isoformat(),
        "n_buys": state.get("n_buys", 0) + (1 if qty_delta > 0 else 0),
    })

    return {
        "status": "ok", "action": "rebalanced",
        "reason": reason,
        "previous_qty": current_qty,
        "new_qty": target_qty,
        "current_price": current_price,
        "actions": actions,
    }


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(cycle(), indent=2, default=str))
