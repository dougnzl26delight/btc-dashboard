"""Pre-warm the dashboard caches so the user never sees a cold refresh.

Run via Windows scheduled task every 3 hours. This calls all the heavy
functions that the dashboard depends on, keeping their disk caches fresh.

Without this: every 4 hours, the first dashboard refresh after cache
expiry takes 60-90 seconds while it pulls 60+ signals from network APIs.

With this: caches are refreshed every 3 hours in the background. The
dashboard always reads a hot cache (<100ms refresh).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print(f"[{datetime.now().isoformat()}] Cache warm-up starting...")
    t0 = time.time()

    # Force fresh pull (bypass disk cache check) so we always get current data
    from core.btc_prediction import state_of_btc, pull_all_signals
    try:
        # force=True bypasses the disk cache TTL check
        pull_all_signals(force=True)
        print(f"  [{time.time()-t0:.1f}s] state pulled + cached")
    except Exception as e:
        print(f"  ERROR pulling state: {e}")

    # Bottom signals (separate cache file)
    try:
        from core.btc_bottom_signals import all_bottom_signals
        t1 = time.time()
        all_bottom_signals(force=True)
        print(f"  [{time.time()-t1:.1f}s] bottom signals cached")
    except Exception as e:
        print(f"  ERROR pulling bottom signals: {e}")

    # OHLCV used by dashboard 90d chart + Olson layer
    try:
        from core import data
        t2 = time.time()
        # Multiple windows used by various dashboard features
        data.ohlcv_extended("BTC/USDT", days_back=90)    # price chart
        data.ohlcv_extended("BTC/USDT", days_back=1825)  # 5y for Olson MACD
        data.ohlcv_extended("BTC/USDT", days_back=1095)  # 3y for Olson HA
        print(f"  [{time.time()-t2:.1f}s] OHLCV warmed (3 windows)")
    except Exception as e:
        print(f"  ERROR warming OHLCV: {e}")

    total = time.time() - t0
    print(f"[{datetime.now().isoformat()}] WARMED in {total:.1f}s")


if __name__ == "__main__":
    main()
