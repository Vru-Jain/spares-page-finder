<#
.SYNOPSIS
    Installs the Spares Page Finder as a Windows service that starts with the PC, and
    optionally publishes it to the internet through Cloudflare Tunnel.

.DESCRIPTION
    Run once from an elevated PowerShell (right-click > Run as Administrator), from the
    tool's folder:   powershell -ExecutionPolicy Bypass -File scripts\install-service.ps1 ...

    What it does:
      - registers the tool as a service with NSSM: starts at boot, restarts after a
        crash, writes rotating logs to data\logs
      - with -PublicHost: lets the tool answer to that address (FINDER_HOSTS)
      - with -TunnelToken: installs Cloudflare Tunnel's connector (cloudflared) as a
        service, so the tool is reachable at -PublicHost from anywhere
      - with -Office: also opens the port to the office network (Private and Domain
        networks only), for colleagues in the office who use http://<this PC>:8600
      - schedules backup.bat every night at 02:00
    Re-running it is safe: it reconfigures what is already there.

.PARAMETER PublicHost
    The address Cloudflare publishes the tool at, e.g. spares.example.com.

.PARAMETER TunnelToken
    The token Cloudflare shows when you create the tunnel (Zero Trust > Networks >
    Tunnels > Create a tunnel > Cloudflared > Windows). Needs cloudflared installed:
    winget install --id Cloudflare.cloudflared

.EXAMPLE
    .\scripts\install-service.ps1 -PublicHost spares.example.com -TunnelToken eyJh...
    .\scripts\install-service.ps1 -Office
#>
[CmdletBinding()]
param(
    [int]$Port = 8600,
    [string]$ServiceName = "SparesFinder",
    [string]$PublicHost = "",
    [string]$TunnelToken = "",
    [switch]$Office,
    [switch]$SkipBackupTask
)

$ErrorActionPreference = "Stop"

# --- preflight -------------------------------------------------------------

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Run this from an elevated PowerShell (Right-click > Run as Administrator)." }

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "data\logs"
if (-not (Test-Path $Python)) { throw "No .venv at $Python. Run install.bat first." }
$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) { throw "nssm not found. Install it with: winget install NSSM.NSSM (then open a new PowerShell)." }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Bound to this PC only, unless the office network should reach it directly too: the
# Cloudflare connector runs on this PC and talks to it locally.
$BindHost = if ($Office) { "0.0.0.0" } else { "127.0.0.1" }

# --- the tool as a service ---------------------------------------------------

if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Service '$ServiceName' exists - stopping and reconfiguring."
    & $nssm stop $ServiceName | Out-Null
    Start-Sleep -Seconds 3
} else {
    & $nssm install $ServiceName $Python | Out-Null
}
& $nssm set $ServiceName Application $Python | Out-Null
& $nssm set $ServiceName AppParameters "-m uvicorn app:app --host $BindHost --port $Port" | Out-Null
& $nssm set $ServiceName AppDirectory $Root | Out-Null
& $nssm set $ServiceName DisplayName "Spares Page Finder" | Out-Null
& $nssm set $ServiceName Description "Finds the spare-parts pages of ship manuals" | Out-Null
& $nssm set $ServiceName Start SERVICE_AUTO_START | Out-Null
# Restart after a crash, but not in a tight loop.
& $nssm set $ServiceName AppExit Default Restart | Out-Null
& $nssm set $ServiceName AppRestartDelay 5000 | Out-Null
& $nssm set $ServiceName AppThrottle 10000 | Out-Null
# Logs rotate at 10 MB so they cannot fill the disk.
& $nssm set $ServiceName AppStdout (Join-Path $LogDir "service-out.log") | Out-Null
& $nssm set $ServiceName AppStderr (Join-Path $LogDir "service-err.log") | Out-Null
& $nssm set $ServiceName AppRotateFiles 1 | Out-Null
& $nssm set $ServiceName AppRotateOnline 1 | Out-Null
& $nssm set $ServiceName AppRotateBytes 10485760 | Out-Null
$environment = @("PYTHONIOENCODING=utf-8")
if ($PublicHost) { $environment += "FINDER_HOSTS=$PublicHost" }
& $nssm set $ServiceName AppEnvironmentExtra @environment | Out-Null
& $nssm start $ServiceName | Out-Null

# --- office network ----------------------------------------------------------

$ruleName = "Spares Page Finder (TCP $Port)"
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Office) {
    # Private and Domain networks only: never reachable from a public network.
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort $Port -Profile Private,Domain | Out-Null
    Write-Host "Firewall: office network may reach port $Port."
}

# --- Cloudflare Tunnel -------------------------------------------------------

if ($TunnelToken) {
    $cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
    if (-not $cloudflared) {
        Write-Warning "cloudflared not found. Install it (winget install --id Cloudflare.cloudflared), open a new PowerShell and run this again."
    } else {
        if (Get-Service -Name cloudflared -ErrorAction SilentlyContinue) {
            & $cloudflared service uninstall | Out-Null
            Start-Sleep -Seconds 2
        }
        & $cloudflared service install $TunnelToken | Out-Null
        Write-Host "Cloudflare Tunnel connector installed as the 'cloudflared' service."
    }
}

# --- nightly backup ----------------------------------------------------------

if (-not $SkipBackupTask) {
    $action = New-ScheduledTaskAction -Execute (Join-Path $Root "backup.bat") -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -Daily -At 2am
    Register-ScheduledTask -TaskName "Spares Page Finder backup" -Action $action -Trigger $trigger `
        -User "SYSTEM" -RunLevel Highest -Force | Out-Null
    Write-Host "Backup scheduled every night at 02:00 (data\backups, the newest 14 kept)."
}

# --- check -------------------------------------------------------------------

Start-Sleep -Seconds 8
Write-Host ""
Write-Host "Service '$ServiceName' is $((Get-Service -Name $ServiceName).Status)."
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 20
    Write-Host "Health check: ok, model trained $($health.'model trained'), keeps manuals $($health.keep_days) days."
} catch {
    Write-Warning "Health check failed: $($_.Exception.Message). See $LogDir\service-err.log"
}
if ($PublicHost) { Write-Host "From anywhere: https://$PublicHost (sign-in by Cloudflare Access)" }
if ($Office) { Write-Host "In the office: http://$($env:COMPUTERNAME):$Port" }
Write-Host "Updates: git pull, then  nssm restart $ServiceName"
