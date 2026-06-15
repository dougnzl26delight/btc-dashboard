@echo off
REM Crypto paper-rig dashboard. Runs on port 8510 to coexist with the
REM stocks dashboard on 8501.
REM Browser: http://localhost:8510
cd /d "%~dp0"
streamlit run dashboard.py --server.port 8510 --browser.serverAddress localhost
