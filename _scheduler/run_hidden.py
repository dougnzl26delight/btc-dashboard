"""Run a command with NO console window, ever — Task Scheduler launcher.

Launched via pythonw.exe (which itself has no console). It then starts the target
command with the Windows CREATE_NO_WINDOW flag, so even a console app like
powershell.exe gets NO console window.

Why this and not -WindowStyle Hidden / wscript+Run(...,0): PowerShell re-shows its
OWN console during startup, so those approaches still flash a black "System32"
console for a split second every time. CREATE_NO_WINDOW means there is no console
window for PowerShell to show -- the only fully reliable suppression. (Same flag
_scheduler/start_streamlit.py uses to launch Streamlit silently.)

Returns the child's exit code so Task Scheduler still sees success/failure.

Usage (Task Scheduler):
  Program/script:  <...>\\pythonw.exe
  Arguments:       "<...>\\run_hidden.py" powershell -NoProfile -ExecutionPolicy Bypass -File "<...>\\watchdog.ps1"
"""
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000

if len(sys.argv) < 2:
    sys.exit(1)

try:
    proc = subprocess.run(sys.argv[1:], creationflags=CREATE_NO_WINDOW)
    sys.exit(proc.returncode)
except Exception:
    sys.exit(1)
