@echo off
REM BTC bottom countdown — daily run via Windows Task Scheduler.
REM Sends emails at T-60, T-30, T-10 days from projected bottom.
REM Also fires immediate alert when scorecard reaches 6/8 hard criteria.
cd /d "%~dp0"
python -m core.btc_bottom_countdown >> logs\btc_bottom_countdown.log 2>&1
exit /b %ERRORLEVEL%
