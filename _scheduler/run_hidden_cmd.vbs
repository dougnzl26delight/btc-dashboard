' run_hidden_cmd.vbs - run an arbitrary command line with NO visible window.
'
' Task Scheduler launches console apps (powershell.exe, cmd.exe) with a window
' that briefly FLASHES even with -WindowStyle Hidden: the console host (conhost)
' appears before the app can hide itself. Routing through wscript.exe (which has
' no console of its own) + WshShell.Run(cmd, 0, True) creates the target process
' with its window hidden from the very start - no flash - while still blocking so
' Task Scheduler reads the real exit code.
'
' Usage from Task Scheduler:
'   Program/script:  wscript.exe
'   Arguments:       "<this>\run_hidden_cmd.vbs" "<full command line to run hidden>"
Option Explicit
Dim sh, rc
If WScript.Arguments.Count < 1 Then WScript.Quit 1
Set sh = CreateObject("WScript.Shell")
rc = sh.Run(WScript.Arguments(0), 0, True)   ' 0 = hidden window, True = wait for exit code
WScript.Quit rc
