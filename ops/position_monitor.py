"""Per-position stop loss / take profit / trailing stop monitor.

Runs before each orchestrator cycle and (optionally) as a separate hourly
scheduled task for intraday protection. Closes positions that hit any
configured exit threshold and alerts on every action.

Defaults are tuned for crypto vol — see the constants below to adjust.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.broker import Broker
from core.data import ticker
from ops.alerts import alert


REPO_ROOT = Path(__file__).resolve().parent.parent
HWM_FILE = REPO_ROOT / ".position_hwm.json"
TRADES_FILE = REPO_ROOT / "paper_trades.csv"
STATE_FILE = REPO_ROOT / ".paper_state.json"

# === Risk thresholds (tuned for crypto; flip to None to disable any) ===
STOP_LOSS_PCT = 0.07        # close at 7% loss vs entry
TAKE_PROFIT_PCT = None      # disabled — let signal flips take profit
TRAILING_STOP_PCT = 0.08    # 8% drawdown from peak (only if in profit > 2%)
TRAIL_MIN_PROFIT = 0.02     # only trail once 2% in the money


def _load_hwm() -> dict:
    if HWM_FILE.exists():
        return json.loads(HWM_FILE.read_text())
    return {}


def _save_hwm(hwm: dict) -> None:
    HWM_FILE.write_text(json.dumps(hwm, indent=2))


def _avg_entry(trades_df: pd.DataFrame, pair: str) -> float | None:
    asset_trades = trades_df[trades_df["pair"].astype(str) == pair]
    if asset_trades.empty:
        return None
    signed = asset_trades.apply(
        lambda r: float(r["qty"]) if r["side"] == "buy" else -float(r["qty"]), axis=1
    )
    if signed.abs().sum() == 0:
        return None
    return float(
        (signed.abs() * asset_trades["price"].astype(float)).sum() / signed.abs().sum()
    )


def _check_position(
    asset: str, qty: float, price: float, avg_entry: float, hwm: dict
) -> str | None:
    """Returns 'stop' / 'tp' / 'trail' or None."""
    if qty == 0 or avg_entry <= 0 or price <= 0:
        return None

    direction = 1 if qty > 0 else -1
    pnl_pct = (price / avg_entry - 1.0) * direction

    if STOP_LOSS_PCT and pnl_pct < -STOP_LOSS_PCT:
        return "stop"
    if TAKE_PROFIT_PCT and pnl_pct > TAKE_PROFIT_PCT:
        return "tp"

    if TRAILING_STOP_PCT:
        prev_peak = hwm.get(asset, avg_entry)
        if direction > 0:
            new_peak = max(prev_peak, price)
            drawdown_from_peak = (new_peak - price) / new_peak if new_peak > 0 else 0
        else:
            new_peak = min(prev_peak, price)
            drawdown_from_peak = (price - new_peak) / new_peak if new_peak > 0 else 0
        hwm[asset] = new_peak
        if drawdown_from_peak > TRAILING_STOP_PCT and pnl_pct > TRAIL_MIN_PROFIT:
            return "trail"

    return None


def run_monitor(mode: str = "paper", long_only: bool = False) -> dict:
    """Check every open position; close those that hit any threshold."""
    if not STATE_FILE.exists():
        return {"checked": 0, "actions": []}

    state = json.loads(STATE_FILE.read_text())
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")

    trades_df = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    hwm = _load_hwm()
    broker = Broker(mode=mode, long_only=long_only)

    actions: list[dict] = []

    for asset, qty in list(positions.items()):
        if abs(qty) < 1e-9:
            continue
        pair = f"{asset}/{quote}"
        try:
            price = float(ticker(pair)["last"])
        except Exception:
            continue
        avg_entry = _avg_entry(trades_df, pair)
        if avg_entry is None:
            continue

        action = _check_position(asset, qty, price, avg_entry, hwm)
        if not action:
            continue

        close_side = "sell" if qty > 0 else "buy"
        close_value = abs(qty) * price
        pnl_pct = (price / avg_entry - 1.0) * (1 if qty > 0 else -1)
        try:
            broker.place_market_order(pair, close_side, close_value)
            level = "critical" if action == "stop" else "trade"
            alert(
                f"{action.upper()} hit: {asset} closed @ ${price:,.4f} "
                f"pnl={pnl_pct:+.2%} (entry ${avg_entry:,.4f})",
                level=level,
            )
            actions.append({"asset": asset, "action": action, "pnl_pct": pnl_pct})
            # Reset HWM after close
            hwm.pop(asset, None)
        except Exception as e:
            alert(f"Failed to close {asset} on {action}: {e}", level="critical")

    _save_hwm(hwm)
    return {"checked": len(positions), "actions": actions}


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(run_monitor(), indent=2, default=str))
