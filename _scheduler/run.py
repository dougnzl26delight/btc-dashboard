"""Generic silent task runner: pythonw.exe _scheduler/run.py <module> <log_name> [args...]

Replaces .cmd wrappers entirely. Because pythonw.exe has no console window,
no cmd subshell is ever spawned -> ZERO window flash. Stdout/stderr stream
into logs/<log_name>.log exactly like the .cmd versions did.

Usage from Task Scheduler:
  Program/script:  pythonw.exe
  Arguments:       _scheduler/run.py core.btc_module log_name [--quiet] [...]
  Start in:        C:\\Users\\dougn\\Documents\\CryptoTrading
"""
from __future__ import annotations

import contextlib
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Always run from project root + ensure imports work
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Force UTF-8 so emoji/arrows in print() don't crash mid-task
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if len(sys.argv) < 2:
    sys.exit("usage: pythonw run.py <module> [log_name] [args...]")

module = sys.argv[1]
log_name = sys.argv[2] if len(sys.argv) > 2 else module.replace(".", "_")
extra_args = sys.argv[3:]

log_path = ROOT / "logs" / f"{log_name}.log"
log_path.parent.mkdir(exist_ok=True)

# Set up sys.argv as if module were invoked directly
sys.argv = [module.split(".")[-1]] + extra_args

# Stream all output to log file, both stdout and stderr
with open(log_path, "a", encoding="utf-8") as logf:
    logf.write(f"\n=== {datetime.now(timezone.utc).isoformat()} "
                f"run.py {module} {' '.join(extra_args)} ===\n")
    logf.flush()
    with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as e:
            sys.exit(int(e.code) if e.code is not None else 0)
        except BaseException as e:  # noqa: BLE001
            import traceback
            traceback.print_exc(file=logf)
            sys.exit(1)
