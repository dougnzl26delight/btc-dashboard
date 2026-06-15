' Silent dashboard launcher — double-click to start the BTC prediction
' dashboard with zero window flash.
'
' Internally calls pythonw.exe + _scheduler/start_streamlit.py, which
' launches streamlit with CREATE_NO_WINDOW so even the streamlit child
' process is invisible.
'
' After launch, open http://localhost:8511 in your browser.

Set sh = CreateObject("WScript.Shell")
ROOT = "C:\Users\dougn\Documents\CryptoTrading"
sh.CurrentDirectory = ROOT

' WshShell.Run(command, windowStyle, waitForCompletion)
'   windowStyle = 0  -> hidden (no window for pythonw)
'   waitForCompletion = False  -> fire-and-forget (don't block on streamlit)
sh.Run "pythonw.exe """ & ROOT & "\_scheduler\start_streamlit.py""", 0, False
