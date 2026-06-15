"""Test what the dashboard sees for perp state."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

sleeve_files = sorted(ROOT.glob(".paper_perp_state_*.json"))
sleeve_files = [f for f in sleeve_files if "legacy_backup" not in f.name]
merged = {"cash_quote": 0.0, "quote_currency": "USDT", "positions": {}, "entry_prices": {}}
for sf in sleeve_files:
    d = json.loads(sf.read_text())
    sleeve = sf.stem.replace(".paper_perp_state_", "")
    merged["cash_quote"] += float(d.get("cash_quote", 0))
    for asset, qty in d.get("positions", {}).items():
        if abs(qty) > 1e-9:
            merged["positions"][asset] = merged["positions"].get(asset, 0) + qty
    for asset, price in d.get("entry_prices", {}).items():
        if asset not in merged["entry_prices"]:
            merged["entry_prices"][asset] = price

print("MERGED PERP STATE (what dashboard sees):")
print(f"Cash: ${merged['cash_quote']:,.2f}")
print(f"\nPositions ({len(merged['positions'])}):")
for asset, qty in merged["positions"].items():
    direction = "LONG" if qty > 0 else "SHORT"
    entry = merged["entry_prices"].get(asset, 0)
    print(f"  {asset:<6s}  {direction:<6s}  qty={qty:>12.4f}  entry=${entry:.4f}")
