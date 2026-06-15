"""Exit-signal daily cycle.

Runs the BTC/alt exit-signal monitor and pushes state changes + EMA21
trailing-stop alerts through the standard ops.alerts pipeline.

Schedule daily as Crypto_exit_signal_daily (mirrors the bah_btc_run pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
import btc_exit_signal_alert as monitor


def main():
    prior = monitor.load_state()
    current = monitor.compute_status()
    msgs = monitor.detect_alerts(current, prior)

    # Categorize alerts: BROKEN/NEW FIRE -> trade level; NEAR -> warning; WATCH -> info
    for m in msgs:
        if "BROKEN" in m or "NEW SIGNAL FIRE" in m:
            alerts.alert(f"exit_signal: {m}", level="trade")
        elif "NEAR EMA21" in m or "PEAK ZONE" in m or "EXTREME OB" in m:
            alerts.alert(f"exit_signal: {m}", level="warning")
        else:
            alerts.alert(f"exit_signal: {m}", level="info")

    monitor.save_state(current)
    watchdog.beat()

    # Brief stdout summary
    n_broken = sum(1 for s in current.values() if s["stop_alert"] == "BROKEN")
    n_near = sum(1 for s in current.values() if s["stop_alert"] == "NEAR")
    n_watch = sum(1 for s in current.values() if s["stop_alert"] == "WATCH")
    print(f"exit_signal: {len(current)} pairs scanned. "
          f"BROKEN={n_broken} NEAR={n_near} WATCH={n_watch}  "
          f"alerts_fired={len(msgs)}")
    for m in msgs:
        print(f"  - {m}")
    return {
        "n_pairs": len(current),
        "n_broken": n_broken,
        "n_near": n_near,
        "n_watch": n_watch,
        "alerts": msgs,
    }


if __name__ == "__main__":
    main()
