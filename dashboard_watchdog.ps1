# BTC dashboard watchdog -- keeps btcdelight.com self-healing.
# Pings the local Streamlit server on :8511. If it is NOT serving (process died,
# machine rebooted, crash) AND no btc-streamlit process is already starting up,
# it relaunches via start-dashboard.vbs. Runs every 5 min (Crypto_dashboard_watchdog).
# ALSO supervises cloudflared (the public tunnel): if that process has died, it
# relaunches it via _scheduler\start_cloudflared.vbs. Process-level only.
#
# KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 reads a BOM-less .ps1 as cp1252,
# so a stray Unicode dash/emoji (e.g. an em-dash) becomes a PARSER ERROR and the whole
# watchdog silently dies (exit 1) -- which is exactly what stops the site self-healing.
$ErrorActionPreference = "SilentlyContinue"
$root = "C:\Users\dougn\Documents\CryptoTrading"
$log = "$root\logs\dashboard_watchdog.log"

function Test-Serving {
    try { return (Invoke-WebRequest -Uri "http://localhost:8511/_stcore/health" -UseBasicParsing -TimeoutSec 8).StatusCode -eq 200 }
    catch { return $false }
}

# Tunnel supervision (process-level only): relaunch cloudflared if it DIED.
# Do NOT restart on a mere public-URL failure -- cloudflared has its own connection
# retry, and restart churn can trip Cloudflare rate-limiting. Just keep it alive.
if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  cloudflared DOWN - relaunching via _scheduler\start_cloudflared.vbs" |
        Out-File -Append -Encoding utf8 $log
    & wscript.exe "$root\_scheduler\start_cloudflared.vbs"
}

if (Test-Serving) { exit 0 }   # backend healthy -- nothing more to do

# Not serving. Is a btc-streamlit process already starting up? (avoid relaunch thrash)
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*streamlit*btc_prediction*" }
if ($proc) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  not serving but a btc-streamlit process exists (starting?) - skip" |
        Out-File -Append -Encoding utf8 $log
    exit 0
}

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  dashboard DOWN - relaunching via start-dashboard.vbs" |
    Out-File -Append -Encoding utf8 $log
& wscript.exe "$root\start-dashboard.vbs"   # wscript = no console window (cscript flashed one)
exit 0
