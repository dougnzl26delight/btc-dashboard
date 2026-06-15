@echo off
REM Daily equity-top scorecard check — emails on phase change.
cd /d "%~dp0"
python -m core.btc_top_check >> logs\top_check.log 2>&1
exit /b %ERRORLEVEL%
