# Set up crypto-rig Windows scheduled tasks. Run ONCE as Administrator.
#
# Creates three tasks, all prefixed `Crypto_` to avoid collision with the
# stocks-rig tasks already on this machine.
#
#   Crypto_orchestrator_daily  - runs run.py once a day at 14:00 UTC (~02:00 NZT)
#   Crypto_eval_weekly         - runs strict eval on Sundays at 14:30 UTC
#   Crypto_daily_log           - snapshots equity + tickers daily at 14:05 UTC
#
# To remove later: schtasks /Delete /TN Crypto_orchestrator_daily /F
#                  schtasks /Delete /TN Crypto_eval_weekly /F
#                  schtasks /Delete /TN Crypto_daily_log /F

$ErrorActionPreference = "Continue"  # 2026-05-28: was Stop, but schtasks /Delete on non-existent task spuriously errors
$repo = "C:\Users\dougn\Documents\CryptoTrading"
# Windowless interpreter so scheduled tasks don't flash a black console window
# every time they fire (pythonw.exe == python.exe minus the console). All tasks
# below are background jobs, so none of them want a visible console.
$python = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = "C:\Users\dougn\AppData\Local\Microsoft\WindowsApps\pythonw.exe" }

if (-not (Test-Path $repo)) {
    throw "Repo not found at $repo"
}

function Register-CryptoTask {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Schedule,    # "DAILY" or "WEEKLY"
        [string]$Time,        # HH:mm
        [string]$Days = ""    # "SUN" for weekly
    )
    Write-Host "Registering $Name ..."
    cmd /c "schtasks /Delete /TN `"$Name`" /F >nul 2>&1"

    $cmd = "/Create /TN $Name /SC $Schedule /ST $Time /TR `"$python $repo\$Script`""
    if ($Days) { $cmd += " /D $Days" }
    $cmd += " /RU $env:USERNAME /F"

    Invoke-Expression "schtasks $cmd"
    Write-Host "  Done: $Name @ $Time"
}

Register-CryptoTask -Name "Crypto_orchestrator_daily" `
                    -Script "run.py" `
                    -Schedule "DAILY" `
                    -Time "17:00"

Register-CryptoTask -Name "Crypto_daily_log" `
                    -Script "ops\daily_log.py" `
                    -Schedule "DAILY" `
                    -Time "17:05"

# Exit-signal monitor — daily. Tracks MACD bear cross + EMA21 break across
# BTC + alts, fires alerts when pairs hit NEAR (within 2% of EMA21) or BROKEN.
# Reads/writes btc_exit_signal_state.json; dashboard consumes it.
Register-CryptoTask -Name "Crypto_exit_signal_daily" `
                    -Script "exit_signal_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:10"

# Daily email report — 07:00 UTC = 19:00 NZT (standard time) / 20:00 NZDT.
# Emails a full snapshot: overnight alerts, trailing-stop ranking,
# BTC peak countdown, recommended actions. Archives to daily_reports_crypto/.
Register-CryptoTask -Name "Crypto_daily_report" `
                    -Script "daily_report.py" `
                    -Schedule "DAILY" `
                    -Time "07:00"

# Simpleton Summary daily brief — plain-English "what changed in the last 24h"
# for the public dashboard's Simpleton tab. Fires 06:00 NZ (machine = NZST local).
Register-CryptoTask -Name "Crypto_simpleton_brief_daily" `
                    -Script "simpleton_brief_run.py" `
                    -Schedule "DAILY" `
                    -Time "06:00"

# Weekly red-team — the in-house devil's advocate. Emails the strongest bear
# case against the rotation campaign every Sunday 08:00 NZ so conviction is
# stress-tested, not reinforced. Routes via email=True (BTC-dashboard channel).
Register-CryptoTask -Name "Crypto_red_team_weekly" `
                    -Script "red_team_run.py" `
                    -Schedule "WEEKLY" `
                    -Time "08:00" `
                    -Days "SUN"

# Olson signal scorecard — daily: capture his new directional calls + grade any
# that have matured (30d) by forward return -> hit-rate + payoff(R) + expectancy.
# Builds a real track record over months to judge his paid tier on data.
Register-CryptoTask -Name "Crypto_olson_scorecard_daily" `
                    -Script "olson_scorecard_run.py" `
                    -Schedule "DAILY" `
                    -Time "06:30"

# Generic guru signal scorecard — daily: same auto-grading as Olson, for the
# other monitored analysts (Benjamin Cowen, + any added to guru_scorecard.GURU_CFGS).
Register-CryptoTask -Name "Crypto_guru_scorecard_daily" `
                    -Script "guru_scorecard_run.py" `
                    -Schedule "DAILY" `
                    -Time "06:35"

# Pro_trend sleeve — daily. Manages per-pair trail-stops + entry signals.
Register-CryptoTask -Name "Crypto_pro_trend_daily" `
                    -Script "pro_trend_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:20"

# XSMOM sleeve — daily. Internal logic gates 14-day rebalance.
Register-CryptoTask -Name "Crypto_xsmom_daily" `
                    -Script "xsmom_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:25"

# BAH BTC sleeve — daily check. Strategy itself rebalances monthly internally.
Register-CryptoTask -Name "Crypto_bah_btc_daily" `
                    -Script "bah_btc_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:30"

# Oversold bounce — tactical mean-reversion sleeve.
# Scans for regime-wide RSI<25, enters basket of 5 most-oversold pairs.
Register-CryptoTask -Name "Crypto_oversold_bounce_daily" `
                    -Script "oversold_bounce_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:35"

# Overbought fade — tactical SHORT sleeve. Only fires in BEAR regime.
# Shorts top 3 most-overbought pairs when 3+ pairs RSI>70 simultaneously.
Register-CryptoTask -Name "Crypto_overbought_fade_daily" `
                    -Script "overbought_fade_run.py" `
                    -Schedule "DAILY" `
                    -Time "17:40"

# Daily P&L snapshot to SQLite — feeds Sharpe gates + walk-forward analysis.
# Runs LAST so all sleeve runners have settled.
Register-CryptoTask -Name "Crypto_daily_pnl_snapshot" `
                    -Script "daily_pnl_snapshot.py" `
                    -Schedule "DAILY" `
                    -Time "17:55"

# Heartbeat + Telegram /halt poller — every 5 min.
# Catches dead rigs (no watchdog beat in 6h => SMS alert).
# Polls Telegram for /halt commands from operator's phone.
Write-Host "Registering Crypto_heartbeat_5min ..."
cmd /c "schtasks /Delete /TN `"Crypto_heartbeat_5min`" /F >nul 2>&1"
$cmdhb = "/Create /TN Crypto_heartbeat_5min /SC MINUTE /MO 5 /TR `"$python $repo\ops\heartbeat.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdhb"
Write-Host "  Done: Crypto_heartbeat_5min (every 5 minutes)"

# Dashboard watchdog — every 5 min: if btcdelight (:8511) isn't serving, relaunch
# it (process died / machine rebooted). The public Cloudflare tunnel auto-starts;
# this keeps a live backend behind it so an outage self-heals within ~5 min.
Write-Host "Registering Crypto_dashboard_watchdog ..."
cmd /c "schtasks /Delete /TN `"Crypto_dashboard_watchdog`" /F >nul 2>&1"
$cmdwd = "/Create /TN Crypto_dashboard_watchdog /SC MINUTE /MO 5 /TR `"powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $repo\dashboard_watchdog.ps1`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdwd"
Write-Host "  Done: Crypto_dashboard_watchdog (every 5 minutes)"

Register-CryptoTask -Name "Crypto_eval_weekly" `
                    -Script "strategies\diverse_mom_ethbtc.py" `
                    -Schedule "WEEKLY" `
                    -Time "14:30" `
                    -Days "SUN"

# Intraday safety net — runs every 30 minutes, applies stops/TP/trailing.
# Catches adverse moves between daily orchestrator cycles.
Write-Host "Registering Crypto_position_monitor_30min ..."
cmd /c "schtasks /Delete /TN `"Crypto_position_monitor_30min`" /F >nul 2>&1"
$cmd30 = "/Create /TN Crypto_position_monitor_30min /SC MINUTE /MO 30 /TR `"$python $repo\ops\position_monitor.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmd30"
Write-Host "  Done: Crypto_position_monitor_30min (every 30 minutes)"

# Funding-rate basis arb cycle — runs every 4 hours.
# Funding events fire every 8h, so 4h cadence catches signal flips quickly.
Write-Host "Registering Crypto_basis_arb_4hourly ..."
cmd /c "schtasks /Delete /TN `"Crypto_basis_arb_4hourly`" /F >nul 2>&1"
$cmd4h = "/Create /TN Crypto_basis_arb_4hourly /SC HOURLY /MO 4 /TR `"$python $repo\basis_run.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmd4h"
Write-Host "  Done: Crypto_basis_arb_4hourly (every 4 hours)"

# Kill-criteria daily check — enforces Strategy Charter rules.
# Logs equity, computes rolling Sharpe + DD, alerts on K1/K2/K3/K4 triggers.
Write-Host "Registering Crypto_kill_criteria ..."
cmd /c "schtasks /Delete /TN `"Crypto_kill_criteria`" /F >nul 2>&1"
$cmdkc = "/Create /TN Crypto_kill_criteria /SC DAILY /ST 17:15 /TR `"$python $repo\ops\kill_criteria.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdkc"
Write-Host "  Done: Crypto_kill_criteria (daily at 14:15)"

# === TOP-1% MONITORING TIER (added 2026-05-10) ===

# Intraday trail stop manager — pro_trend positions only.
# Updates trail stops every 15 min; closes if price has breached.
Write-Host "Registering Crypto_pro_trend_intraday_15min ..."
cmd /c "schtasks /Delete /TN `"Crypto_pro_trend_intraday_15min`" /F >nul 2>&1"
$cmdptin = "/Create /TN Crypto_pro_trend_intraday_15min /SC MINUTE /MO 15 /TR `"$python $repo\ops\pro_trend_intraday.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdptin"
Write-Host "  Done: Crypto_pro_trend_intraday_15min (every 15 minutes)"

# Real-time kill switch — flash-crash protection. Polls every 5 min,
# tracks rolling MTM velocity, flattens + 24h lockout on RT1/RT2/RT3.
Write-Host "Registering Crypto_realtime_kill_switch_5min ..."
cmd /c "schtasks /Delete /TN `"Crypto_realtime_kill_switch_5min`" /F >nul 2>&1"
$cmdrtks = "/Create /TN Crypto_realtime_kill_switch_5min /SC MINUTE /MO 5 /TR `"$python $repo\ops\realtime_kill_switch.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdrtks"
Write-Host "  Done: Crypto_realtime_kill_switch_5min (every 5 minutes)"

# Weekly portfolio review — Sunday afternoon, written report.
Write-Host "Registering Crypto_weekly_review ..."
cmd /c "schtasks /Delete /TN `"Crypto_weekly_review`" /F >nul 2>&1"
$cmdwr = "/Create /TN Crypto_weekly_review /SC WEEKLY /D SUN /ST 14:30 /TR `"$python $repo\ops\weekly_review.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdwr"
Write-Host "  Done: Crypto_weekly_review (Sundays at 14:30)"

# Monthly OOS revalidation — first Sunday only, walk-forward + param drift check.
Write-Host "Registering Crypto_monthly_oos ..."
cmd /c "schtasks /Delete /TN `"Crypto_monthly_oos`" /F >nul 2>&1"
$cmdmoos = "/Create /TN Crypto_monthly_oos /SC WEEKLY /D SUN /ST 14:45 /TR `"$python $repo\ops\monthly_oos.py`" /RU $env:USERNAME /F"
Invoke-Expression "schtasks $cmdmoos"
Write-Host "  Done: Crypto_monthly_oos (Sundays at 14:45; first Sunday only)"

Write-Host ""
Write-Host "All SIX Crypto_ tasks registered. View them: schtasks /Query /TN Crypto_orchestrator_daily"
Write-Host "Tasks fire at NZT ~02:00-02:30 (well after US close, before NZ wake)."
Write-Host "Position monitor fires every 30 minutes for intraday stop coverage."
Write-Host "Basis arb fires every 4 hours to catch funding-rate flips."
Write-Host "Kill-criteria fires daily at 14:15, after pro_trend cycle."
