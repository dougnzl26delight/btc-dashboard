"""XSMOM weekly cycle. Scheduled as Crypto_xsmom_weekly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
from strategies import xsmom


def main():
    result = xsmom.cycle()
    print(json.dumps(result, indent=2, default=str))

    if result.get("action") == "rebalanced":
        n = result.get("n_actions", 0)
        weights_str = ", ".join(f"{p}={w:+.2%}"
                                 for p, w in result.get("new_weights", {}).items())
        alerts.alert(
            f"XSMOM rebalanced: {n} actions. Weights: {weights_str}",
            level="trade",
        )
    elif result.get("status") == "locked_out":
        alerts.alert(f"XSMOM SKIPPED: {result.get('lock_reason')}", level="warning")

    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
