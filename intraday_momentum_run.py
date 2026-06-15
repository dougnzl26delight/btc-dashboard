"""Intraday momentum runner — every 15 min."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
from strategies import intraday_momentum


def main():
    result = intraday_momentum.cycle()
    print(json.dumps(result, indent=2, default=str))

    for action in result.get("actions", []):
        kind = action.get("action")
        pair = action.get("pair")
        if kind == "entry":
            mtf_str = ""
            mtf_conf = action.get("mtf_confluence")
            if mtf_conf is not None:
                mtf_str = f" mtf_conf={mtf_conf:.2f}"
            alerts.alert(
                f"INTRADAY ENTRY {pair} @ ${action['price']:,.4f} "
                f"tsmom={action['tsmom']*100:.1f}% rsi={action['rsi']:.0f}{mtf_str}",
                level="trade",
            )
        elif kind == "exit":
            alerts.alert(
                f"INTRADAY EXIT {pair}: {action['reason']} "
                f"pnl={action['pnl_pct']*100:+.2f}% (${action['realized_pnl']:+,.2f})",
                level="trade",
            )

    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
