"""Fix W11 over-allocation: new sub-accounts inflated total beyond $200k.

Original plan: spot $100k + perp $100k = $200k total.
After W11: added grid_trader ($10k spot) + intraday_momentum ($10k spot)
          + intraday_momentum_short ($10k perp) = +$30k phantom.

Fix: deduct from reserves to bring totals back to $200k.
  spot_reserve $45k -> $25k (deduct $20k for grid + intraday)
  perp_reserve $10k -> $0k  (deduct $10k for intraday_short)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Adjust spot_reserve
f = ROOT / ".paper_state_spot_reserve.json"
d = json.loads(f.read_text())
print(f"spot_reserve before: ${d['cash_quote']:,.2f}")
d["cash_quote"] = 25_000.0  # was $45k - $20k = $25k
f.write_text(json.dumps(d, indent=2))
print(f"spot_reserve after:  ${d['cash_quote']:,.2f}")

# Adjust perp_reserve
f = ROOT / ".paper_perp_state_perp_reserve.json"
d = json.loads(f.read_text())
print(f"perp_reserve before: ${d['cash_quote']:,.2f}")
d["cash_quote"] = 0.0  # was $10k - $10k = $0
f.write_text(json.dumps(d, indent=2))
print(f"perp_reserve after:  ${d['cash_quote']:,.2f}")

print()
print("Total spot expected:  $100,000")
print("Total perp expected:  $100,000")
print("Combined expected:    $200,000")
