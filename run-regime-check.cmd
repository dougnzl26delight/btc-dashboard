@echo off
REM Daily regime-shift check — emails on RISK_ON/LATE_CYCLE/BEAR transition.
cd /d "%~dp0"
python -m core.btc_regime_alert >> logs\regime_check.log 2>&1
exit /b %ERRORLEVEL%
