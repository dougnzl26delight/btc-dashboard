"""Discretionary force-entry helper.

Bypasses the Donchian-breakout entry rule but uses the strategy's normal
risk sizing + stop logic. After firing, writes the pro_trend state file so
subsequent daily cycles manage the position via the same exit/pyramid rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data
from core.broker import Broker
from core.catalyst_signals import combined_catalyst_multiplier
from core.perp_broker import PerpBroker
from core.pnl_attribution import tag_entry
from core.swing_backtest import compute_atr
from strategies import pro_trend
from ops import alerts


REPO_ROOT = Path(__file__).resolve().parent


def force_entry(pair: str, side: str, mode: str = "paper") -> dict:
    """Open a forced position at current market price using strategy sizing.
    side = 'long' or 'short'.
    """
    df = data.ohlcv_extended(pair, days_back=400).copy()
    df["atr"] = compute_atr(df, pro_trend.ATR_PERIOD)
    df = df.dropna()
    if df.empty:
        return {"error": "no data"}

    last = df.iloc[-1]
    price = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    atr = float(last["atr"])

    spot = Broker(mode=mode, long_only=False)
    perp = PerpBroker(mode=mode)
    cash = float(perp.get_balance().get("USDT", 0))
    mtm_eq = cash

    catalyst = (
        combined_catalyst_multiplier()["combined_mult"]
        if pro_trend.USE_CATALYST_OVERLAY else 1.0
    )
    size_mult = pro_trend.LEVERAGE_MULTIPLIER * catalyst
    stop_dist = pro_trend.ATR_STOP_MULT * atr

    qty = (mtm_eq * pro_trend.RISK_PCT_PER_UNIT * size_mult) / stop_dist
    notional_cap_frac = 0.30 * pro_trend.LEVERAGE_MULTIPLIER
    qty = min(qty, mtm_eq * notional_cap_frac / price)

    if side == "long":
        if pro_trend.LEVERAGE_MULTIPLIER > 1.0:
            perp.open_position(pair, "long", qty * price)
            broker_used = "perp"
        else:
            spot.place_market_order(pair, "buy", qty * price)
            broker_used = "spot"
        trail_stop = price - stop_dist
        extreme = high
    elif side == "short":
        perp.open_position(pair, "short", qty * price)
        broker_used = "perp"
        trail_stop = price + stop_dist
        extreme = low
    else:
        return {"error": f"invalid side {side}"}

    state_file = REPO_ROOT / f".pro_trend_state_{pair.split('/')[0]}.json"
    state_file.write_text(json.dumps({
        "side": side,
        "units": [{
            "qty": qty,
            "entry_price": price,
            "entry_atr": atr,
            "size_mult": size_mult,
        }],
        "extreme": extreme,
        "trail_stop": trail_stop,
        "peak_equity": mtm_eq,
    }, indent=2))

    # Tag as discretionary so live P&L can be split from systematic.
    tag_entry(pair, sleeve="discretionary", side=side,
              entry_price=price, qty=qty)

    info = {
        "pair": pair,
        "side": side,
        "qty": qty,
        "entry_price": price,
        "atr": atr,
        "stop_dist": stop_dist,
        "stop_distance_pct": stop_dist / price,
        "trail_stop": trail_stop,
        "notional_usdt": qty * price,
        "leverage": pro_trend.LEVERAGE_MULTIPLIER,
        "size_mult": size_mult,
        "broker": broker_used,
        "discretionary_override": True,
    }
    alerts.alert(
        f"FORCED {side.upper()} {pair} @ ${price:,.4f}, "
        f"qty {qty:.6f}, stop ${trail_stop:,.4f} ({stop_dist/price:.1%}), "
        f"notional ${qty*price:,.0f} (lev {pro_trend.LEVERAGE_MULTIPLIER}x)",
        level="trade",
    )
    return info


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python force_entry.py <PAIR> <long|short>")
        print("Example: python force_entry.py NEAR/USDT long")
        sys.exit(1)
    pair = sys.argv[1].upper()
    side = sys.argv[2].lower()
    result = force_entry(pair, side)
    print(json.dumps(result, indent=2, default=str))
