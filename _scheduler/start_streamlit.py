"""Silent streamlit launcher — runs via pythonw.exe so no console ever appears.

When called via pythonw.exe (no console subsystem), this script uses
subprocess.Popen with the CREATE_NO_WINDOW flag on the child streamlit
process — guaranteeing no window flash even when streamlit spawns its
own child processes.

Usage:
  pythonw.exe _scheduler/start_streamlit.py [port]

Or via start-dashboard.vbs (double-clickable, also silent).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

port = sys.argv[1] if len(sys.argv) > 1 else "8511"
log_path = ROOT / "logs" / "streamlit_btc.log"
log_path.parent.mkdir(exist_ok=True)

# Windows API constant — prevents the OS from allocating a console window
# for the child process even though the child is python.exe (console app).
# Documented at https://learn.microsoft.com/.../process-creation-flags
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008

# Use python.exe (not pythonw) so streamlit can write progress lines
# during startup. CREATE_NO_WINDOW prevents the visible console anyway.
python_exe = sys.executable.replace("pythonw.exe", "python.exe")

# Force UTF-8 so emoji etc. in dashboard don't crash mid-render.
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

with open(log_path, "a", encoding="utf-8") as logf:
    from datetime import datetime, timezone
    logf.write(f"\n=== {datetime.now(timezone.utc).isoformat()} "
                f"start_streamlit.py launch (port {port}) ===\n")
    logf.flush()
    subprocess.Popen(
        [python_exe, "-m", "streamlit", "run",
         "btc_prediction_dashboard.py",
         "--server.port", port,
         "--browser.serverAddress", "localhost",
         "--server.headless", "true"],
        stdout=logf, stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
        cwd=str(ROOT),
        env=env,
    )

# Warm the server in the background so the FIRST visitor gets the fast (~3s) render,
# not the ~27s cold one (fresh-process imports + first chart build). warm_dashboard.py
# retries the connect while streamlit boots; pythonw = no console window. Best-effort —
# never blocks launch. Covers every start path (restart script, watchdog, boot).
try:
    pyw = python_exe.replace("python.exe", "pythonw.exe")
    subprocess.Popen(
        [pyw, str(ROOT / "warm_dashboard.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS, cwd=str(ROOT),
    )
except Exception:
    pass

# Exit immediately — children run detached
sys.exit(0)
