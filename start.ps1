<#
.SYNOPSIS
  Start Heliograph: FastAPI backend on http://localhost:8000 and the web UI on
  http://localhost:5173. API documentation: http://localhost:8000/docs

.EXAMPLE
  .\start.ps1             # production build of the UI (vite preview)
  .\start.ps1 -Dev        # UI with hot reload (vite dev server)
  .\start.ps1 -NoBrowser
  .\start.ps1 -Device cpu # force CPU even if CUDA is available

.NOTES
  No training starts automatically. Stop everything with .\stop.ps1
#>
[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$NoBrowser,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$venvPy = Join-Path $Root "backend\.venv\Scripts\python.exe"
$runDir = Join-Path $Root ".run"
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $runDir, $logDir | Out-Null

function Fail($msg) { Write-Host "START FAILED: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $venvPy)) { Fail "backend\.venv not found. Run .\setup.ps1 first." }
if (-not $Dev -and -not (Test-Path (Join-Path $Root "frontend\dist\index.html"))) { Fail "frontend\dist not built. Run .\setup.ps1 first (or use -Dev)." }
$vite = Join-Path $Root "frontend\node_modules\vite\bin\vite.js"
if (-not (Test-Path $vite)) { Fail "Frontend dependencies missing. Run .\setup.ps1 first." }
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) { Fail "Node.js not found on PATH." }

function Test-PortInUse([int]$Port) {
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return [bool]$c
    } catch {
        return [bool](netstat -ano | Select-String -Pattern "[:.]$Port\s+.*LISTENING")
    }
}

foreach ($p in 8000, 5173) {
    if (Test-PortInUse $p) { Fail "Port $p is already in use. Is Heliograph already running? Use .\stop.ps1 (or close the program using port $p)." }
}

$env:HELIOGRAPH_DEVICE = $Device
$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting backend (FastAPI on :8000)..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $venvPy `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory (Join-Path $Root "backend") `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError (Join-Path $logDir "backend.err.log") `
    -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $runDir "backend.pid") -Value $backend.Id

$mode = "preview"
if ($Dev) { $mode = "dev" }
Write-Host "Starting web UI (vite $mode on :5173)..." -ForegroundColor Cyan
$frontend = Start-Process -FilePath $nodeCmd.Source `
    -ArgumentList @("`"$vite`"", $mode, "--port", "5173", "--strictPort", "--host", "127.0.0.1") `
    -WorkingDirectory (Join-Path $Root "frontend") `
    -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $logDir "frontend.err.log") `
    -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $runDir "frontend.pid") -Value $frontend.Id

function Wait-Url([string]$Url, [string]$Name, $Proc, [string]$Log) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Proc.HasExited) {
            Write-Host "$Name exited early. Last log lines:" -ForegroundColor Red
            if (Test-Path $Log) { Get-Content $Log -Tail 25 }
            & (Join-Path $Root "stop.ps1") | Out-Null
            Fail "$Name did not start."
        }
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 700
    }
    & (Join-Path $Root "stop.ps1") | Out-Null
    Fail "$Name did not respond at $Url within $TimeoutSeconds s. See $Log"
}

Wait-Url "http://127.0.0.1:8000/api/health" "Backend" $backend (Join-Path $logDir "backend.err.log")
Write-Host "  backend ready   http://localhost:8000   (API docs: http://localhost:8000/docs)" -ForegroundColor Green
Wait-Url "http://127.0.0.1:5173/" "Web UI" $frontend (Join-Path $logDir "frontend.err.log")
Write-Host "  web UI ready    http://localhost:5173" -ForegroundColor Green

if (-not $NoBrowser) { Start-Process "http://localhost:5173" }
Write-Host ""
Write-Host "Heliograph is running. Logs: logs\   Stop with: .\stop.ps1"
