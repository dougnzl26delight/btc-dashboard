"""Daily P&L snapshot — writes per-sleeve equity to .pnl.db.

Run daily as Crypto_daily_pnl_snapshot. Captures end-of-day equity for each
sleeve. Tomorrow's pnl_db.daily_pnl will compute daily return vs today.

Sleeves snapshotted:
  - spot_orchestrator: cash + spot positions MTM
  - perp_orchestrator: perp cash + open MTM
  - bah_btc:           BTC qty * BTC price (the position itself is the sleeve)
  - xsmom:             XSMOM-managed perp positions MTM
  - pro_trend:         sum of per-pair open MTM + remaining peak_equity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import data
from core.pnl_db import snapshot_daily_equity, init_db
from core.perp_broker import PerpBroker

REPO_ROOT = Path(__file__).resolve().parent


def _safe_price(asset: str, suffix: str = "/USDT") -> float:
    try:
        df = data.ohlcv_extended(f"{asset}{suffix}", days_back=2)
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0


def snapshot_spot():
    f = REPO_ROOT / ".paper_state.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    cash = float(state.get("cash_quote", 0))
    pos_mtm = 0.0
    for asset, qty in state.get("positions", {}).items():
        if abs(qty) < 1e-12:
            continue
        px = _safe_price(asset)
        pos_mtm += qty * px
    equity = cash + pos_mtm
    snapshot_daily_equity("spot_orchestrator", equity, cash=cash, open_mtm=pos_mtm)
    return equity


def snapshot_perp():
    pb = PerpBroker(mode="paper")
    cash = float(pb._state.cash_quote)
    open_pnl = 0.0
    for asset, qty in pb._state.positions.items():
        if abs(qty) < 1e-12:
            continue
        px = _safe_price(asset)
        if px <= 0:
            continue
        entry = pb._state.entry_prices.get(asset, px)
        open_pnl += qty * (px - entry)
    equity = cash + open_pnl
    snapshot_daily_equity("perp_orchestrator", equity, cash=cash, open_mtm=open_pnl)
    return equity


def snapshot_bah_btc():
    f = REPO_ROOT / ".bah_btc_state.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    qty = float(state.get("btc_qty", 0))
    px = _safe_price("BTC")
    equity = qty * px
    snapshot_daily_equity("bah_btc", equity, cash=0, open_mtm=equity)
    return equity


def snapshot_xsmom():
    pb = PerpBroker(mode="paper")
    # XSMOM universe (from strategies.xsmom)
    XSMOM_PAIRS = ["BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "DOT", "ATOM"]
    mtm = 0.0
    notional_allocated = 30_000.0  # XSMOM nominal allocation
    for asset in XSMOM_PAIRS:
        qty = pb._state.positions.get(asset, 0.0)
        if abs(qty) < 1e-12:
            continue
        px = _safe_price(asset)
        entry = pb._state.entry_prices.get(asset, px)
        mtm += qty * (px - entry)
    equity = notional_allocated + mtm  # baseline + open PnL
    snapshot_daily_equity("xsmom", equity, cash=notional_allocated, open_mtm=mtm)
    return equity


def snapshot_pro_trend():
    total = 0.0
    n = 0
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        try:
            st = json.loads(f.read_text())
        except Exception:
            continue
        total += float(st.get("peak_equity", 100_000.0))
        n += 1
    if n == 0:
        return None
    avg_equity = total / n
    snapshot_daily_equity("pro_trend", avg_equity, cash=avg_equity, open_mtm=0)
    return avg_equity


def snapshot_oversold_bounce():
    f = REPO_ROOT / ".oversold_bounce_state.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    baseline = 15_000.0
    mtm = 0.0
    for pair, info in state.get("open_positions", {}).items():
        try:
            px = _safe_price(pair.split("/")[0])
        except Exception:
            continue
        mtm += info.get("qty", 0) * (px - info.get("entry_price", px))
    equity = baseline + mtm
    snapshot_daily_equity("oversold_bounce", equity, cash=baseline, open_mtm=mtm)
    return equity


def snapshot_overbought_fade():
    f = REPO_ROOT / ".overbought_fade_state.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    baseline = 10_000.0
    mtm = 0.0
    for pair, info in state.get("open_positions", {}).items():
        try:
            px = _safe_price(pair.split("/")[0])
        except Exception:
            continue
        # qty is negative for short — short profit when price drops
        mtm += info.get("qty", 0) * (px - info.get("entry_price", px))
    equity = baseline + mtm
    snapshot_daily_equity("overbought_fade", equity, cash=baseline, open_mtm=mtm)
    return equity


def snapshot_subaccount(sleeve_name: str, is_perp: bool = False):
    """W13: enumerate per-sleeve sub-account state files (W9 architecture).

    Sums cash + open_mtm and records to daily_equity table.
    """
    prefix = ".paper_perp_state_" if is_perp else ".paper_state_"
    f = REPO_ROOT / f"{prefix}{sleeve_name}.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    cash = float(state.get("cash_quote", 0))
    entries = state.get("entry_prices", {}) if is_perp else {}
    mtm = 0.0
    for asset, qty in state.get("positions", {}).items():
        if abs(qty) < 1e-9:
            continue
        px = _safe_price(asset)
        if px <= 0:
            continue
        if is_perp:
            entry = entries.get(asset, px)
            mtm += qty * (px - entry)
        else:
            # Spot: mtm is just current value
            mtm += qty * px
    equity = cash + mtm
    snapshot_daily_equity(sleeve_name, equity, cash=cash, open_mtm=mtm)
    return equity


def main():
    init_db()
    results = {}

    # W13: enumerate all per-sleeve sub-accounts from W9 architecture.
    # Spot sub-accounts
    spot_sleeves = ["bah_btc", "oversold_bounce", "orchestrator", "spot_reserve",
                    "grid_trader", "intraday_momentum", "pro_trend"]
    for sleeve in spot_sleeves:
        try:
            eq = snapshot_subaccount(sleeve, is_perp=False)
            results[f"spot:{sleeve}"] = eq
            if eq is not None:
                print(f"  spot:{sleeve:<22s}  equity = ${eq:>12,.2f}")
        except Exception as e:
            print(f"  spot:{sleeve:<22s}  ERROR: {e}")

    # Perp sub-accounts
    perp_sleeves = ["xsmom", "pro_trend", "overbought_fade", "basis_arb",
                    "perp_reserve", "intraday_momentum_short"]
    for sleeve in perp_sleeves:
        try:
            eq = snapshot_subaccount(sleeve, is_perp=True)
            results[f"perp:{sleeve}"] = eq
            if eq is not None:
                print(f"  perp:{sleeve:<22s}  equity = ${eq:>12,.2f}")
        except Exception as e:
            print(f"  perp:{sleeve:<22s}  ERROR: {e}")

    # Compute combined totals
    spot_total = sum(v for k, v in results.items() if k.startswith("spot:") and v is not None)
    perp_total = sum(v for k, v in results.items() if k.startswith("perp:") and v is not None)
    combined = spot_total + perp_total
    print()
    print(f"  SPOT TOTAL:           equity = ${spot_total:>12,.2f}")
    print(f"  PERP TOTAL:           equity = ${perp_total:>12,.2f}")
    print(f"  COMBINED:             equity = ${combined:>12,.2f}  (P&L {combined - 200_000:+,.2f})")
    return results


if __name__ == "__main__":
    print("Daily P&L snapshot")
    print("-" * 50)
    main()
