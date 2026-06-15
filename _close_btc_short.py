"""One-off: close the discretionary BTC short to clear the way for BAH BTC sleeve."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.perp_broker import PerpBroker
from core.pnl_attribution import untag


REPO = Path(__file__).resolve().parent

# Read current state
print("=== Pre-close state ===")
perp = PerpBroker(mode="paper")
bal_before = perp.get_balance()
print(f"BTC position before: {bal_before.get('positions', {}).get('BTC', 0)}")
print(f"BTC entry price:     {bal_before.get('entry_prices', {}).get('BTC', 0)}")
print(f"Perp cash before:    ${bal_before.get('USDT', 0):,.2f}")
print()

# Close the position
print("Closing BTC short via PerpBroker.close_position...")
result = perp.close_position("BTC/USDT")
print(f"Close result: {result}")
print()

# Untag from attribution
untag_result = untag("BTC/USDT")
print(f"Untagged from pnl_attribution: {untag_result is not None}")
print()

# Clear pro_trend BTC state
btc_state_file = REPO / ".pro_trend_state_BTC.json"
if btc_state_file.exists():
    btc_state_file.write_text(json.dumps({
        "side": None, "units": [], "extreme": 0.0,
        "trail_stop": 0.0, "peak_equity": 100_000.0,
    }, indent=2))
    print(f"Cleared {btc_state_file.name}")
print()

# Verify
print("=== Post-close state ===")
bal_after = perp.get_balance()
print(f"BTC position after:  {bal_after.get('positions', {}).get('BTC', 0)}")
print(f"Perp cash after:     ${bal_after.get('USDT', 0):,.2f}")
print(f"Cash delta:          ${bal_after.get('USDT', 0) - bal_before.get('USDT', 0):+,.2f}")
