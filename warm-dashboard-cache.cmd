@echo off
REM Pre-warm BTC dashboard caches every 3h via Windows Task Scheduler.
REM Without this, the first dashboard refresh after 4h cache expiry takes
REM 60-90 seconds. With this, caches are always fresh and refreshes are <100ms.
cd /d "%~dp0"
python -m core.warm_dashboard_cache >> logs\warm_dashboard_cache.log 2>&1
exit /b %ERRORLEVEL%
