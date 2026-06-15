"""Check paper account status — equity, positions, P&L vs start."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data


paper = json.loads(Path(__file__).resolve().parent.joinpath(".paper_state.json").read_text())
cash = paper["cash_quote"]

print("=" * 80)
print("PAPER ACCOUNT STATUS")
print("=" * 80)
print(f"Cash:                ${cash:,.2f}")
print()

print("Open positions:")
print(f"  {'Asset':<6s} {'Qty':>14s} {'Price':>12s} {'Value':>14s} {'% of equity':>12s}")
total_pos_value = 0.0
position_lines = []
for asset, qty in paper["positions"].items():
    if qty == 0:
        continue
    try:
        df = data.ohlcv_extended(f"{asset}/USDT", days_back=2)
        if df.empty:
            continue
        px = float(df["close"].iloc[-1])
    except Exception:
        continue
    val = qty * px
    total_pos_value += val
    position_lines.append((asset, qty, px, val))

equity = cash + total_pos_value

for asset, qty, px, val in position_lines:
    pct = val / equity * 100 if equity > 0 else 0
    print(f"  {asset:<6s} {qty:>14.6f} ${px:>11,.2f} ${val:>13,.2f} {pct:>11.1f}%")

print()
print(f"  Total positions:   ${total_pos_value:,.2f}")
print(f"  Cash:              ${cash:,.2f}")
print(f"  TOTAL EQUITY:      ${equity:,.2f}")
print()

start = 100000
pnl = equity - start
pnl_pct = pnl / start * 100
print(f"  Started at:        ${start:,.2f}")
print(f"  P&L:               ${pnl:+,.2f}  ({pnl_pct:+.2f}%)")
print()

# Other sleeve state files
print("=" * 80)
print("OTHER SLEEVE STATE FILES")
print("=" * 80)
root = Path(__file__).resolve().parent
for name in [".bah_btc_state.json", ".paper_perp_state.json", ".basis_positions.json",
             ".xsmom_state.json", ".pro_trend_state_BTC.json",
             ".pro_trend_state_ETH.json", ".pro_trend_state_SOL.json",
             ".pro_trend_state_AVAX.json", ".pro_trend_state_NEAR.json",
             ".pro_trend_state_OP.json"]:
    f = root / name
    if not f.exists():
        continue
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    print(f"\n  {name}:")
    snippet = json.dumps(d, indent=2, default=str)
    if len(snippet) > 400:
        snippet = snippet[:400] + "\n    ..."
    for line in snippet.split("\n"):
        print(f"    {line}")
