@echo off
REM Dashboard panel precompute — refreshes disk cache every 15 min.
cd /d "%~dp0"
python precompute_dashboard.py --quiet >> logs\precompute.log 2>&1
exit /b %ERRORLEVEL%
