<#
.SYNOPSIS
  Stop the Heliograph backend and web UI started by start.ps1.

.NOTES
  Only processes recorded in .run\*.pid are stopped, after checking that their
  command line belongs to Heliograph (uvicorn app.main / vite). Child processes
  (training workers) are stopped with them. As a fallback, a process still
  listening on port 8000 or 5173 is stopped only if its command line matches.
#>
[CmdletBinding()]
param()

$Root = $PSScriptRoot
$runDir = Join-Path $Root ".run"

function Get-CommandLine([int]$ProcessId) {
    try { return (Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop).CommandLine } catch { return $null }
}

function Test-Ours([string]$Cmd) {
    if (-not $Cmd) { return $false }
    return ($Cmd -match "uvicorn" -and $Cmd -match "app\.main") -or ($Cmd -match "vite" -and $Cmd -match "5173")
}

function Stop-Tree([int]$ProcessId, [string]$Name) {
    $cmd = Get-CommandLine $ProcessId
    if (-not $cmd) { Write-Host "  $Name (PID $ProcessId) is not running."; return }
    if (-not (Test-Ours $cmd)) { Write-Host "  PID $ProcessId does not look like Heliograph's $Name; leaving it alone." -ForegroundColor Yellow; return }
    # /T stops the whole tree (uvicorn + training worker processes), /F forces it.
    & taskkill.exe /PID $ProcessId /T /F | Out-Null
    Write-Host "  stopped $Name (PID $ProcessId)" -ForegroundColor Green
}

$stoppedAny = $false
foreach ($name in @("backend", "frontend")) {
    $pidFile = Join-Path $runDir "$name.pid"
    if (Test-Path $pidFile) {
        $procId = [int](Get-Content $pidFile | Select-Object -First 1)
        Stop-Tree $procId $name
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        $stoppedAny = $true
    }
}

# Fallback: something still listening on our ports that is ours (e.g. pid file deleted)
foreach ($port in 8000, 5173) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            if ($c.OwningProcess -gt 0) { Stop-Tree ([int]$c.OwningProcess) "process on port $port"; $stoppedAny = $true }
        }
    } catch { }
}

Start-Sleep -Milliseconds 500
$busy = @()
foreach ($port in 8000, 5173) {
    try { if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $busy += $port } } catch { }
}
if ($busy.Count -gt 0) {
    Write-Host "Ports still in use: $($busy -join ', '). Another program may be using them." -ForegroundColor Yellow
} elseif ($stoppedAny) {
    Write-Host "Heliograph stopped." -ForegroundColor Green
} else {
    Write-Host "Heliograph was not running."
}
