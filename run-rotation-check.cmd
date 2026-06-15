@echo off
REM Daily rotation phase check — emails on phase change.
REM Runs via Windows Task Scheduler.
cd /d "%~dp0"
python -m core.btc_rotation_planner check >> logs\rotation_check.log 2>&1
exit /b %ERRORLEVEL%
