"""Oversold bounce sleeve runner — daily.

Scans universe for cross-sectional RSI<25 regime. Entries on confirmation,
exits on RSI>50/target/stop/time-cap.

Scheduled as Crypto_oversold_bounce_daily.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
from strategies import oversold_bounce


def main():
    result = oversold_bounce.cycle()
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "sleeve_paused":
        alerts.alert("oversold_bounce sleeve paused (drawdown CB)", level="warning")
        watchdog.beat()
        return result

    for action in result.get("actions", []):
        kind = action.get("action")
        pair = action.get("pair")
        if kind == "entry":
            alerts.alert(
                f"OVERSOLD_BOUNCE ENTRY {pair} @ ${action['price']:,.4f} "
                f"(RSI {action['rsi']:.0f}), stop ${action['stop']:,.4f}, "
                f"target ${action['target']:,.4f}",
                level="trade",
            )
        elif kind == "exit":
            alerts.alert(
                f"OVERSOLD_BOUNCE EXIT {pair}: reason={action['reason']}, "
                f"realized ${action['realized_pnl']:+,.2f}",
                level="trade",
            )

    if result.get("regime_armed"):
        n_os = result["n_oversold"]
        alerts.alert(
            f"OVERSOLD_BOUNCE regime ARMED: {n_os} pairs at RSI<25 — scanning entries",
            level="info",
        )

    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
