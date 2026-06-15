@echo off
REM One-click launcher for the Crypto Claude Code session.
REM Double-click this file to open a new PowerShell window in the
REM CryptoTrading directory and start Claude Code there.
cd /d "%~dp0"
pwsh -NoExit -Command "cd '%~dp0'; claude"
