"""Comprehensive rig health check — aggregates across per-sleeve sub-accounts."""
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data

ROOT = Path(__file__).resolve().parent


def read_json(p):
    f = ROOT / p
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def get_price(asset, pair_suffix="/USDT"):
    try:
        df = data.ohlcv_extended(f"{asset}{pair_suffix}", days_back=2)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def fmt_usd(v):
    return f"${v:,.2f}" if v is not None else "-"


def sleeve_spot_equity(sleeve: str) -> dict | None:
    """Read a spot sub-account file and compute equity."""
    f = ROOT / f".paper_state_{sleeve}.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    cash = float(state.get("cash_quote", 0))
    pos_value = 0.0
    positions = []
    for asset, qty in state.get("positions", {}).items():
        if abs(qty) < 1e-12:
            continue
        px = get_price(asset)
        if px is None:
            continue
        value = qty * px
        pos_value += value
        positions.append({"asset": asset, "qty": qty, "px": px, "value": value})
    return {"cash": cash, "positions": positions, "pos_value": pos_value,
            "equity": cash + pos_value}


def sleeve_perp_equity(sleeve: str) -> dict | None:
    """Read a perp sub-account and compute equity (cash + open MTM)."""
    f = ROOT / f".paper_perp_state_{sleeve}.json"
    if not f.exists():
        return None
    state = json.loads(f.read_text())
    cash = float(state.get("cash_quote", 0))
    entries = state.get("entry_prices", {})
    mtm = 0.0
    positions = []
    for asset, qty in state.get("positions", {}).items():
        if abs(qty) < 1e-12:
            continue
        px = get_price(asset)
        if px is None:
            continue
        entry = entries.get(asset, px)
        pos_pnl = qty * (px - entry)
        mtm += pos_pnl
        positions.append({"asset": asset, "qty": qty, "entry": entry, "px": px, "pnl": pos_pnl})
    return {"cash": cash, "positions": positions, "mtm": mtm, "equity": cash + mtm}


def main():
    print("=" * 95)
    print(f"FULL RIG HEALTH CHECK  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 95)
    print()

    spot_sleeves = ["bah_btc", "oversold_bounce", "orchestrator", "spot_reserve", "pro_trend"]
    perp_sleeves = ["xsmom", "pro_trend", "overbought_fade", "basis_arb", "perp_reserve"]

    print("SPOT SUB-ACCOUNTS")
    print("-" * 95)
    print(f"  {'Sleeve':<22s} {'Cash':>12s} {'Position MTM':>14s} {'Equity':>12s} Positions")
    print("  " + "-" * 92)
    spot_total = 0.0
    spot_total_cash = 0.0
    for s in spot_sleeves:
        d = sleeve_spot_equity(s)
        if d is None:
            print(f"  {s:<22s} (no sub-account)")
            continue
        spot_total += d["equity"]
        spot_total_cash += d["cash"]
        pos_str = ", ".join(f"{p['asset']} {p['qty']:.4f}={fmt_usd(p['value'])}"
                            for p in d["positions"]) or "—"
        print(f"  {s:<22s} {fmt_usd(d['cash']):>12s} {fmt_usd(d['pos_value']):>14s} "
              f"{fmt_usd(d['equity']):>12s} {pos_str[:55]}")
    print(f"  {'TOTAL SPOT':<22s} {fmt_usd(spot_total_cash):>12s} "
          f"{fmt_usd(spot_total - spot_total_cash):>14s} {fmt_usd(spot_total):>12s}")
    print()

    print("PERP SUB-ACCOUNTS")
    print("-" * 95)
    print(f"  {'Sleeve':<22s} {'Cash':>12s} {'Open MTM':>14s} {'Equity':>12s} Positions")
    print("  " + "-" * 92)
    perp_total = 0.0
    perp_total_cash = 0.0
    for s in perp_sleeves:
        d = sleeve_perp_equity(s)
        if d is None:
            print(f"  {s:<22s} (no sub-account)")
            continue
        perp_total += d["equity"]
        perp_total_cash += d["cash"]
        pos_str = ", ".join(f"{p['asset']} {p['qty']:+.2f}@{p['px']:.2f} pnl={p['pnl']:+,.0f}"
                            for p in d["positions"]) or "—"
        print(f"  {s:<22s} {fmt_usd(d['cash']):>12s} {fmt_usd(d['mtm']):>14s} "
              f"{fmt_usd(d['equity']):>12s} {pos_str[:55]}")
    print(f"  {'TOTAL PERP':<22s} {fmt_usd(perp_total_cash):>12s} "
          f"{fmt_usd(perp_total - perp_total_cash):>14s} {fmt_usd(perp_total):>12s}")
    print()

    print("=" * 95)
    print("GRAND TOTAL")
    print("=" * 95)
    print(f"  Spot equity:         {fmt_usd(spot_total)}")
    print(f"  Perp equity:         {fmt_usd(perp_total)}")
    grand = spot_total + perp_total
    print(f"  COMBINED:            {fmt_usd(grand)}")
    started = 200_000
    pnl = grand - started
    print(f"  Started:             {fmt_usd(started)}")
    print(f"  Combined P&L:        {fmt_usd(pnl)}  ({pnl/started*100:+.2f}%)")
    print()
    print("Architecture: per-sleeve paper sub-accounts. No cross-sleeve interference.")


if __name__ == "__main__":
    main()
