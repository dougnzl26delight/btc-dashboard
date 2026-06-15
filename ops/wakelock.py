"""Windows wakelock — keep machine awake while crypto strategies run.

CRITICAL: scheduled task name MUST be prefixed `Crypto_` to avoid colliding
with the stocks system's wakelock task on the same machine.
"""

from __future__ import annotations

import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def acquire() -> None:
    """Acquire a wakelock for the lifetime of this process."""
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )


def release() -> None:
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    acquire()
    print("Crypto wakelock acquired. Press Ctrl+C to release.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        release()
        print("Released.")
