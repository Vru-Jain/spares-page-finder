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
        networks only, and only from -AllowFrom: this PC's own subnet unless told
        otherwise), for colleagues in the office who use http://<this PC>:8600
      - schedules backup.bat every night at 02:00
      - lets only administrators, SYSTEM and the person installing change the tool's
        files: the service and the backup run as SYSTEM, so anyone who could edit them
        could run their own code as SYSTEM
    Re-running it is safe: it reconfigures what is already there.

.PARAMETER AllowFrom
    Who on the office network may reach the tool (with -Office): "LocalSubnet" (the
    default: this PC's own network), or addresses and ranges such as
    "192.168.1.0/24" or "192.168.1.20,192.168.1.21".

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
    [string]$AllowFrom = "LocalSubnet",
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
# Over HTTPS once scripts\make_certs.py has made certs\server.crt (README.md, "HTTPS"): Chrome
# warns about every download from a plain HTTP address.
$Tls = Test-Path (Join-Path $Root "certs\server.crt")
$Scheme = if ($Tls) { "https" } else { "http" }
$TlsArgs = if ($Tls) { " --ssl-keyfile certs\server.key --ssl-certfile certs\server.crt" } else { "" }
& $nssm set $ServiceName AppParameters "-m uvicorn app:app --host $BindHost --port $Port$TlsArgs" | Out-Null
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

# --- who may change the tool's files --------------------------------------------

# The service and the nightly backup run as SYSTEM. Windows gives every signed-in user
# Modify on a folder made at the root of a data drive (D:\...), which would let any of them
# edit app.py, backup.bat or the Python in .venv and have SYSTEM run it. Keep write access
# for administrators, SYSTEM and the person installing (who updates it with git pull);
# everyone else may read and run it.
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls $Root /inheritance:d /Q | Out-Null
& icacls $Root /remove:g "*S-1-5-11" /T /C /Q | Out-Null              # Authenticated Users
& icacls $Root /grant:r "*S-1-5-32-545:(OI)(CI)RX" /Q | Out-Null      # Users: read and run
& icacls $Root /grant "${me}:(OI)(CI)M" /Q | Out-Null
Write-Host "Files: only administrators, SYSTEM and $me may change them."
# The certificates' private keys: nobody else may even read them. The office authority's key
# could otherwise sign a certificate every office PC trusts (for this PC's names only).
$Certs = Join-Path $Root "certs"
if (Test-Path $Certs) {
    & icacls $Certs /inheritance:r /grant:r "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" "${me}:(OI)(CI)F" /Q | Out-Null
    Write-Host "Certificates: only administrators, SYSTEM and $me may read them."
}

& $nssm start $ServiceName | Out-Null

# --- office network ----------------------------------------------------------

$ruleName = "Spares Page Finder (TCP $Port)"
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Office) {
    # Private and Domain networks only, never a public one; and only from -AllowFrom (this
    # PC's own subnet by default), not from every network the office routes to it.
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort $Port -Profile Private,Domain -RemoteAddress ($AllowFrom -split ",") | Out-Null
    Write-Host "Firewall: port $Port open to $AllowFrom on the office network."
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
    # this PC asking itself, which may not trust the office authority yet
    if ($Tls) { [Net.ServicePointManager]::ServerCertificateValidationCallback = { $true } }
    $health = Invoke-RestMethod -Uri "${Scheme}://127.0.0.1:$Port/api/health" -TimeoutSec 20
    Write-Host "Health check: ok, model trained $($health.'model trained'), keeps manuals $($health.keep_days) days."
} catch {
    Write-Warning "Health check failed: $($_.Exception.Message). See $LogDir\service-err.log"
} finally {
    [Net.ServicePointManager]::ServerCertificateValidationCallback = $null
}
if ($PublicHost) { Write-Host "From anywhere: https://$PublicHost (sign-in by Cloudflare Access)" }
if ($Office) { Write-Host "In the office: ${Scheme}://$($env:COMPUTERNAME):$Port" }
if ($Tls) { Write-Host "Each PC trusts the office authority once: ${Scheme}://$($env:COMPUTERNAME):$Port/SparesFinder-CA.crt (README.md, HTTPS)" }
Write-Host "Updates: git pull, then  nssm restart $ServiceName"
