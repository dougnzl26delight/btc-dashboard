"""BAH BTC sleeve cycle. Scheduled monthly as Crypto_bah_btc_monthly."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# W15.H: BAH BTC LEGITIMATELY tops up underwater (cycle accumulation thesis).
# Override Livermore averaging-down block for this sleeve only.
os.environ["ALLOW_AVERAGE_DOWN"] = "1"

from ops import alerts, watchdog
from strategies import bah_btc


def main():
    result = bah_btc.cycle()
    print(json.dumps(result, indent=2, default=str))

    if result.get("action") == "rebalanced":
        reason = result.get("reason", "unknown")
        new_qty = result.get("new_qty", 0)
        price = result.get("current_price", 0)
        notional = new_qty * price
        alerts.alert(
            f"BAH_BTC rebalanced ({reason}): {new_qty:.6f} BTC @ ${price:,.2f} "
            f"= ${notional:,.0f}",
            level="trade",
        )
    elif result.get("status") == "locked_out":
        alerts.alert(f"BAH_BTC SKIPPED (lockout): {result.get('lock_reason')}",
                     level="warning")

    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
