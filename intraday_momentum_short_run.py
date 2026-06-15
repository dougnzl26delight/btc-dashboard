"""Intraday momentum SHORT runner — every 15 min."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops import alerts, watchdog
from strategies import intraday_momentum_short


def main():
    result = intraday_momentum_short.cycle()
    print(json.dumps(result, indent=2, default=str))
    for action in result.get("actions", []):
        kind = action.get("action")
        pair = action.get("pair")
        if kind == "entry_short":
            alerts.alert(f"INTRADAY SHORT {pair} @ ${action['price']:,.4f} "
                         f"tsmom={action['tsmom']*100:.1f}% rsi={action['rsi']:.0f}",
                         level="trade")
        elif kind == "exit":
            alerts.alert(f"INTRADAY SHORT EXIT {pair}: {action['reason']} "
                         f"pnl={action['pnl_pct']*100:+.2f}% (${action['realized_pnl']:+,.2f})",
                         level="trade")
    watchdog.beat()
    return result


if __name__ == "__main__":
    main()
