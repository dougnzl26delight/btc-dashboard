"""Smoke-test sleeve circuit breaker module with realistic current values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, status

print("Seeding with current actual sleeve values...")
# bah_btc: 0.247 BTC × $73,414 = ~$18,166 (peak was $20,000)
print(f"bah_btc @ $18,166:    scale = {apply_sleeve_scaling('bah_btc', 18166):.3f}")
# xsmom: perp cash ~$98k + small open PnL — too high vs $30k notional, so use position-based
print(f"xsmom @ $28,367:      scale = {apply_sleeve_scaling('xsmom', 28367):.3f}")
# pro_trend: idle at baseline
print(f"pro_trend @ $100,000: scale = {apply_sleeve_scaling('pro_trend', 100000):.3f}")
print()
print("Current state:")
for name, s in status().items():
    peak = s.get("peak")
    cur = s.get("current")
    if peak:
        print(f"  {name:<22s}  peak ${peak:>10,.0f}  cur ${cur:>10,.0f}  "
              f"dd {s['drawdown']*100:>4.1f}%  scale {s['scale']:.2f}x")
