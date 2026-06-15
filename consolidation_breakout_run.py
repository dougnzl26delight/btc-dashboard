"""Consolidation breakout runner — Livermore's edge.

Scans daily for tight ranges. Fires on breakout. Manages multi-target exits.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops import alerts, watchdog
from strategies import consolidation_breakout


def main():
    result = consolidation_breakout.cycle()
    print(json.dumps(result, indent=2, default=str))
    for action in result.get("actions", []):
        kind = action.get("action")
        pair = action.get("pair")
        if kind == "entry":
            alerts.alert(
                f"CONSOLIDATION BREAKOUT {pair} @ ${action['price']:,.4f} "
                f"(compr {action['compression_score']:.2f}), "
                f"stop {action['stop']:,.4f}, T1 {action['target_1']:,.4f}, T2 {action['target_2']:,.4f}",
                level="trade")
        elif kind == "T1_partial":
            alerts.alert(f"CONSOLIDATION T1 partial-close {pair}: ${action['realized_pnl']:+,.2f}", level="trade")
        elif kind == "exit":
            alerts.alert(f"CONSOLIDATION EXIT {pair}: {action['reason']} ${action['realized_pnl']:+,.2f}", level="trade")
    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
