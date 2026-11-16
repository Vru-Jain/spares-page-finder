<#
.SYNOPSIS
    Once on each PC that opens Spares Page Finder: so that it opens over HTTPS without warnings
    and its Excel downloads open at once.

.DESCRIPTION
    1. Trusts the office's certificate authority for the whole PC (Trusted Root Certification
       Authorities of the computer). Antivirus that looks inside HTTPS, such as Sophos, checks
       sites against that list, not the user's own; Chrome reads it too. The certificate is
       fetched from the tool and installed only if its SHA-256 fingerprint is the one given.
    2. Removes Excel start-up add-ins that would hang Excel: any whose file no longer exists,
       and the AllocationTool's Highlight add-in. Excel then shows nothing at start and hung
       with no window, so that no workbook opened (2026-11-16).
    Run it in PowerShell opened as Administrator, signed in as the person who uses the PC (the
    Excel part is theirs); without Administrator it does the Excel part only. Running it again
    is safe. -Check only reports.

    Downloaded from the tool (https://<office PC>:8600/setup-pc.ps1), it knows its address and
    fingerprint; run it with: powershell -ExecutionPolicy Bypass -File .\setup-pc.ps1
#>
[CmdletBinding()]
param(
    [string]$Server = "__SERVER__",
    [string]$Fingerprint = "__FINGERPRINT__",
    [switch]$Check
)
$ErrorActionPreference = "Stop"
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$restart = $false

# --- 1. the office's certificate authority ---------------------------------------------
$trusted = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -eq "CN=Spares Page Finder office CA" }
$sha256 = { param($c) -join ([Security.Cryptography.SHA256]::Create().ComputeHash($c.RawData) | ForEach-Object { $_.ToString("x2") }) }
if ($trusted | Where-Object { (& $sha256 $_) -eq $Fingerprint }) {
    Write-Host "Certificate: the office authority is trusted already."
} elseif ($Check) {
    Write-Host "Certificate: NOT trusted yet for the whole PC."
} elseif (-not $isAdmin) {
    Write-Host "Certificate: NOT installed - open PowerShell as Administrator (right-click > Run as administrator) and run this again."
} else {
    # Fetched before this PC trusts the tool, so the connection is not checked; the
    # certificate is, by its fingerprint. The check is C#: Windows calls it on a thread where
    # a PowerShell script block cannot run.
    Add-Type -TypeDefinition @"
using System.Net.Security;
public static class SparesFinderFetch {
    static bool Any(object s, System.Security.Cryptography.X509Certificates.X509Certificate c,
                    System.Security.Cryptography.X509Certificates.X509Chain ch, SslPolicyErrors e) { return true; }
    public static RemoteCertificateValidationCallback Callback() { return Any; }
}
"@
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    [Net.ServicePointManager]::ServerCertificateValidationCallback = [SparesFinderFetch]::Callback()
    $file = Join-Path $env:TEMP "SparesFinder-CA.crt"
    try { Invoke-WebRequest -UseBasicParsing -Uri "https://${Server}:8600/SparesFinder-CA.crt" -OutFile $file -TimeoutSec 60 }
    finally { [Net.ServicePointManager]::ServerCertificateValidationCallback = $null }
    $ca = New-Object Security.Cryptography.X509Certificates.X509Certificate2 $file
    if ((& $sha256 $ca) -ne $Fingerprint) { throw "The certificate from $Server is not the office's (fingerprint differs): nothing installed." }
    & certutil -addstore -f Root $file | Out-Null
    Remove-Item $file
    Write-Host "Certificate: the office authority is now trusted for the whole PC."
    $restart = $true
}

# --- 2. Excel start-up add-ins ----------------------------------------------------------
$removed = 0
foreach ($options in Get-ChildItem "HKCU:\Software\Microsoft\Office" -ErrorAction SilentlyContinue |
         ForEach-Object { Join-Path $_.PSPath "Excel\Options" } | Where-Object { Test-Path $_ }) {
    $values = (Get-ItemProperty $options).PSObject.Properties | Where-Object { $_.Name -match '^OPEN\d*$' }
    foreach ($v in $values) {
        # "/R "C:\...\x.xlam"" is a file; "/A "Name"" a COM add-in by name, and a bare file name
        # one in Excel's own folders: those are left alone unless they are the Highlight add-in.
        $path = ($v.Value -replace '^\s*(/\w+\s+)*', '').Trim('"', ' ')
        $file = $v.Value -notmatch '^\s*/A\s' -and [IO.Path]::IsPathRooted($path)
        $why = if ($path -match 'HighlightAddIn') { "the Highlight add-in" }
               elseif ($file -and -not (Test-Path -LiteralPath $path)) { "its file is missing" } else { $null }
        if (-not $why) { Write-Host "Excel add-in kept: $path"; continue }
        if ($Check) { Write-Host "Excel add-in to remove ($why): $path"; continue }
        Remove-ItemProperty -Path $options -Name $v.Name
        Write-Host "Excel add-in removed ($why): $path"
        $removed++
    }
}
# Its line in Excel's list of add-ins (File > Options > Add-ins), and a copy in the start-up
# folder, which Excel opens at every start: moved beside that folder, not deleted.
foreach ($list in Get-ChildItem "HKCU:\Software\Microsoft\Office" -ErrorAction SilentlyContinue |
         ForEach-Object { Join-Path $_.PSPath "Excel\Add-in Manager" } | Where-Object { Test-Path $_ }) {
    foreach ($v in (Get-ItemProperty $list).PSObject.Properties | Where-Object { $_.Name -match 'HighlightAddIn' }) {
        if ($Check) { Write-Host "Excel add-in list entry to remove: $($v.Name)"; continue }
        Remove-ItemProperty -Path $list -Name $v.Name
        Write-Host "Excel add-in list entry removed: $($v.Name)"
        $removed++
    }
}
$xlstart = Join-Path $env:APPDATA "Microsoft\Excel\XLSTART"
foreach ($f in Get-ChildItem $xlstart -Filter "*HighlightAddIn*" -ErrorAction SilentlyContinue) {
    if ($Check) { Write-Host "Excel start-up file to move out: $($f.FullName)"; continue }
    Move-Item -LiteralPath $f.FullName -Destination (Join-Path (Split-Path $xlstart) "$($f.Name).removed") -Force
    Write-Host "Excel start-up file moved out of XLSTART: $($f.Name)"
    $removed++
}
if (-not $removed -and -not $Check) { Write-Host "Excel: no add-in to remove." }
if (Get-Process EXCEL -ErrorAction SilentlyContinue) { Write-Host "Excel is open: close it (or end it in Task Manager if it hangs) before opening a sheet again." }

Write-Host ""
if ($restart) {
    Write-Host "Done. Restart the PC once, so the antivirus reads the new certificate,"
    Write-Host "then open https://${Server}:8600 in Chrome."
} elseif (-not $Check) {
    Write-Host "Done. Open https://${Server}:8600 in Chrome."
}
