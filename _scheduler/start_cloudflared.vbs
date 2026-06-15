' Silent launcher for the NAMED Cloudflare tunnel (dave-btc).
' Serves https://btcdelight.com -> http://localhost:8511 with a PERMANENT URL.
' Config + credentials: C:\Users\dougn\.cloudflared\
' A copy lives in the Windows Startup folder so it auto-runs at logon.
Set sh = CreateObject("WScript.Shell")
ROOT = "C:\Users\dougn\Documents\CryptoTrading"
sh.CurrentDirectory = ROOT
' --edge-ip-version 4 forces edge dials onto IPv4. This network has no working
' IPv6 path to Cloudflare (region*.v2.argotunnel.com resolves IPv6-only), so the
' default "auto" mode picked dead IPv6 and every dial timed out. Forcing IPv4
' makes the connectivity prechecks pass and connections register.
sh.Run """" & ROOT & "\bin\cloudflared.exe"" tunnel --edge-ip-version 4 --logfile """ & ROOT & "\logs\cloudflared.log"" run dave-btc", 0, False
