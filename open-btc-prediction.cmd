@echo off
REM BTC Prediction Engine dedicated dashboard. Runs on port 8511 alongside
REM the main rig dashboard on 8510.
REM Browser: http://localhost:8511
cd /d "%~dp0"
streamlit run btc_prediction_dashboard.py --server.port 8511 --browser.serverAddress localhost
