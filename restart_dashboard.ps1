# Cleanly restart the Streamlit dashboard so code changes always take.
# The launcher uses pythonw -> python child; killing only pythonw leaves the
# real server (python.exe) holding port 8511 with STALE code. This kills the
# actual port-holder, then relaunches.
$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping any process holding port 8511..."
$conn = Get-NetTCPConnection -LocalPort 8511 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conn) { Stop-Process -Id $c.OwningProcess -Force }

# Belt-and-braces: kill any streamlit-hosting python/pythonw
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like "*streamlit*btc_prediction*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Start-Sleep -Seconds 2

Write-Host "Relaunching dashboard (silent)..."
$root = "C:\Users\dougn\Documents\CryptoTrading"
& wscript.exe "$root\start-dashboard.vbs"   # wscript = no console window (cscript flashed one)

Start-Sleep -Seconds 14
$c = Get-NetTCPConnection -LocalPort 8511 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) {
    $p = Get-Process -Id $c.OwningProcess
    Write-Host ("OK - serving PID {0} started {1}" -f $p.Id, $p.StartTime)
} else {
    Write-Host "WARNING - nothing listening on 8511 yet; check logs\streamlit_btc.log"
}
