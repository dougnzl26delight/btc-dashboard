"""Grid trader runner — fires every 5 min for active fills."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops import alerts, watchdog
from strategies import grid_trader


def main():
    result = grid_trader.cycle()
    n_actions = result.get("n_actions", 0)
    if n_actions:
        for a in result.get("actions", []):
            kind = a.get("action")
            pair = a.get("pair")
            if "fill" in kind:
                alerts.alert(f"GRID {kind.upper()} {pair} @ ${a.get('price', 0):,.4f}", level="trade")
            elif kind == "recenter":
                alerts.alert(f"GRID RECENTER {pair} @ ${a.get('center', 0):,.4f} ({a.get('reason')})", level="info")
    print(json.dumps(result, indent=2, default=str))
    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
