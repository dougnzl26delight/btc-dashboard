# Validate every scheduled task by running its underlying script.
# Captures exit code and first/last lines of stdout for each.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\dougn\Documents\CryptoTrading"
$python = (Get-Command python).Source

$tasks = @(
    @{ Name = "Crypto_orchestrator_daily";         Script = "run.py" }
    @{ Name = "Crypto_daily_log";                  Script = "ops\daily_log.py" }
    @{ Name = "Crypto_exit_signal_daily";          Script = "exit_signal_run.py" }
    @{ Name = "Crypto_daily_report";               Script = "daily_report.py" }
    @{ Name = "Crypto_pro_trend_daily";            Script = "pro_trend_run.py" }
    @{ Name = "Crypto_xsmom_daily";                Script = "xsmom_run.py" }
    @{ Name = "Crypto_bah_btc_daily";              Script = "bah_btc_run.py" }
    @{ Name = "Crypto_oversold_bounce_daily";      Script = "oversold_bounce_run.py" }
    @{ Name = "Crypto_overbought_fade_daily";      Script = "overbought_fade_run.py" }
    @{ Name = "Crypto_consolidation_breakout_daily"; Script = "consolidation_breakout_run.py" }
    @{ Name = "Crypto_loss_lock_check";             Script = "ops\loss_acceptance_lock.py" }
    @{ Name = "Crypto_daily_pnl_snapshot";         Script = "daily_pnl_snapshot.py" }
    @{ Name = "Crypto_basis_arb_4hourly";          Script = "basis_run.py" }
    @{ Name = "Crypto_kill_criteria";              Script = "ops\kill_criteria.py" }
    @{ Name = "Crypto_position_monitor_30min";     Script = "ops\position_monitor.py" }
    @{ Name = "Crypto_pro_trend_intraday_15min";   Script = "ops\pro_trend_intraday.py" }
    @{ Name = "Crypto_realtime_kill_switch_5min";  Script = "ops\realtime_kill_switch.py" }
    @{ Name = "Crypto_eval_weekly";                Script = "strategies\diverse_mom_ethbtc.py" }
    @{ Name = "Crypto_weekly_review";              Script = "ops\weekly_review.py" }
    @{ Name = "Crypto_monthly_oos";                Script = "ops\monthly_oos.py" }
)

$results = @()
$pass = 0; $fail = 0

foreach ($task in $tasks) {
    $name = $task.Name
    $script = $task.Script
    $path = "$repo\$script"

    if (-not (Test-Path $path)) {
        Write-Host "[MISS] $name -> $script (file not found)" -ForegroundColor Yellow
        $results += [pscustomobject]@{ Task = $name; Script = $script; Status = "MISSING"; Exit = "-"; Detail = "file not found" }
        $fail++
        continue
    }

    Write-Host "[RUN ] $name -> $script ..." -NoNewline
    $tmpOut = New-TemporaryFile
    $tmpErr = New-TemporaryFile
    try {
        $proc = Start-Process -FilePath $python -ArgumentList "`"$path`"" -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr -NoNewWindow -PassThru -Wait
        $code = $proc.ExitCode
    } catch {
        $code = -1
    }
    $out = Get-Content $tmpOut -ErrorAction SilentlyContinue
    $err = Get-Content $tmpErr -ErrorAction SilentlyContinue
    Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue

    $firstLine = ($out | Select-Object -First 1) -as [string]
    $lastLine = ($out | Select-Object -Last 1) -as [string]
    $errSummary = if ($err) { ($err | Select-Object -First 2) -join " | " } else { "" }

    if ($code -eq 0) {
        Write-Host " OK" -ForegroundColor Green
        $pass++
        $detail = if ($lastLine) { $lastLine.Substring(0, [Math]::Min($lastLine.Length, 90)) } else { "<no stdout>" }
        $results += [pscustomobject]@{ Task = $name; Script = $script; Status = "OK"; Exit = $code; Detail = $detail }
    } else {
        Write-Host " FAIL (exit $code)" -ForegroundColor Red
        $fail++
        $detail = if ($errSummary) { $errSummary.Substring(0, [Math]::Min($errSummary.Length, 120)) } else { "<no stderr>" }
        $results += [pscustomobject]@{ Task = $name; Script = $script; Status = "FAIL"; Exit = $code; Detail = $detail }
    }
}

Write-Host ""
Write-Host "=== Summary ==="
$results | Format-Table -AutoSize Task, Status, Exit, Detail
Write-Host ""
Write-Host "Passed: $pass / $($tasks.Count)   Failed: $fail" -ForegroundColor $(if ($fail -eq 0) { "Green" } else { "Yellow" })
