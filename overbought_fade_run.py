"""Overbought-fade sleeve runner. Scheduled as Crypto_overbought_fade_daily."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
from strategies import overbought_fade


def main():
    result = overbought_fade.cycle()
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "sleeve_paused":
        alerts.alert("overbought_fade sleeve paused (drawdown CB)", level="warning")
        watchdog.beat()
        return result

    for action in result.get("actions", []):
        kind = action.get("action")
        pair = action.get("pair")
        if kind == "entry_short":
            alerts.alert(
                f"OVERBOUGHT_FADE SHORT {pair} @ ${action['price']:,.4f} "
                f"(RSI {action['rsi']:.0f}), stop ${action['stop']:,.4f}, "
                f"target ${action['target']:,.4f}",
                level="trade",
            )
        elif kind == "exit":
            alerts.alert(
                f"OVERBOUGHT_FADE EXIT {pair}: reason={action['reason']}, "
                f"realized ${action['realized_pnl']:+,.2f}",
                level="trade",
            )

    if not result.get("regime_bear"):
        print(f"  regime_gate: {result.get('regime_reason')}")
    elif result.get("entries_blocked_reason"):
        print(f"  entries blocked: {result['entries_blocked_reason']}")

    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
