' run_hidden.vbs — invisible-window wrapper for Task Scheduler invocations.
'
' Without this, every cron tick opens a visible cmd window because Task
' Scheduler launches cmd.exe with WindowStyle=Normal. Routing through
' wscript.exe + WshShell.Run(..., 0, False) suppresses the window
' entirely. Output still streams to the log file inside run_python.bat
' via its existing `>> "%LOG%" 2>&1` redirect.
'
' Usage from Task Scheduler:
'   Program/script:  wscript.exe
'   Arguments:       "C:\Users\dougn\Documents\Trading\_scheduler\run_hidden.vbs"
'                    "C:\Users\dougn\Documents\Trading\_scheduler\run_python.bat"
'                    <task_name> <python_args...>
'
' The first VBS argument is the path to run_python.bat; the rest get
' forwarded to it verbatim.

Set WshShell = CreateObject("WScript.Shell")

If WScript.Arguments.Count < 1 Then
    WScript.Quit 1
End If

' Force UTF-8 stdout/stderr for ALL python children launched by ANY task.
' Windows console default codepage (cp1252) cannot encode common chars like
' ≥, →, ±, °, ², etc.  Without this, the first such print() raises
' UnicodeEncodeError and aborts the cron mid-run.  Insider_trader.py was
' crashing at step [11/12] for months — discovered 2026-05-15.
'
' Setting via WshShell.Environment("Process") affects only THIS wscript
' instance and its child cmd.exe + child python.exe — never persists to the
' user environment.  Belt-and-suspenders: both vars set so python>=3.7 (PYTHONUTF8)
' and earlier (PYTHONIOENCODING) both pick up the UTF-8 default.
Set procEnv = WshShell.Environment("Process")
procEnv("PYTHONIOENCODING") = "utf-8"
procEnv("PYTHONUTF8") = "1"

' Build the command: first arg is the wrapped script, rest are its args.
cmd = """" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
    cmd = cmd & " """ & WScript.Arguments(i) & """"
Next

' WshShell.Run(command, windowStyle, waitForCompletion)
'   windowStyle = 0     → hidden window (no console flash)
'   waitForCompletion = True   → block so Task Scheduler reads exit code correctly
'
' IMPORTANT: cmd /c has a quote-stripping quirk — if the command line starts
' AND ends with a quote, it strips both. With multiple internal quoted args,
' this corrupts the command. The fix is the doubled-outer-quote pattern:
'   cmd /c ""<full quoted command>""
' cmd strips the outermost pair, leaving exactly what we want.
rc = WshShell.Run("cmd /c """ & cmd & """", 0, True)
WScript.Quit rc
