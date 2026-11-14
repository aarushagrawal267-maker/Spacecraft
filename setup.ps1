<#
.SYNOPSIS
  One-time setup of Heliograph on Windows: Python virtual environment, backend
  dependencies, frontend build, database, and (optionally) the NASA dataset.

.EXAMPLE
  .\setup.ps1 -DownloadData          # typical first run (CPU)
  .\setup.ps1 -DownloadData -Cuda    # with an NVIDIA GPU (installs the CUDA build of PyTorch)
  .\setup.ps1 -Python "C:\Python312\python.exe"

.NOTES
  Requires Python 3.11-3.13 (64-bit) and Node.js 18+. Compatible with Windows
  PowerShell 5.1 and PowerShell 7. If script execution is blocked, run once:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
[CmdletBinding()]
param(
    [switch]$DownloadData,
    [switch]$Cuda,
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu126",
    [string]$Python = "",
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    WARNING: $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host ""; Write-Host "SETUP FAILED: $msg" -ForegroundColor Red; exit 1 }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$What (exit code $LASTEXITCODE)" }
}

# ----------------------------------------------------------------------------- location checks
Step "Checking project location"
if ($Root -match "OneDrive") {
    Warn "The project is inside a OneDrive folder. File syncing can lock the SQLite database and model files."
    Warn "Moving the folder outside OneDrive (e.g. C:\Projects\Spacecraft) is recommended."
} else { Ok $Root }

# ----------------------------------------------------------------------------- Python
Step "Locating Python 3.11 - 3.13 (64-bit)"
$py = $null
$candidates = @()
if ($Python) { $candidates += ,@($Python) }
foreach ($v in @("3.12", "3.11", "3.13")) { $candidates += ,@("py", "-$v") }
$candidates += ,@("python")
foreach ($c in $candidates) {
    try {
        $exe = $c[0]
        $pre = @()
        if ($c.Length -gt 1) { $pre = $c[1..($c.Length - 1)] }
        $out = & $exe @pre -c "import sys, struct; print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $parts = $out.Trim().Split(" ")
            $ver = [version]$parts[0]
            if ($ver -ge [version]"3.11" -and $ver -lt [version]"3.14" -and $parts[1] -eq "64") {
                $py = @{ Exe = $exe; Pre = $pre; Version = $parts[0] }
                break
            }
        }
    } catch { }
}
if (-not $py) { Fail "No suitable Python found. Install 64-bit Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." }
Ok ("Using Python {0} ({1} {2})" -f $py.Version, $py.Exe, ($py.Pre -join " "))

# ----------------------------------------------------------------------------- Node
if (-not $SkipFrontend) {
    Step "Checking Node.js"
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { Fail "Node.js not found. Install the LTS version from https://nodejs.org/" }
    $nodeVer = (& node --version).TrimStart("v")
    if ([version]$nodeVer -lt [version]"18.0.0") { Fail "Node.js $nodeVer is too old; version 18 or newer is required." }
    Ok "Node.js $nodeVer"
}

# ----------------------------------------------------------------------------- virtual environment
Step "Creating the backend virtual environment (backend\.venv)"
$venvPy = Join-Path $Root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Invoke-Checked $py.Exe ($py.Pre + @("-m", "venv", "backend\.venv")) "creating the virtual environment"
}
Ok $venvPy
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip", "--quiet") "upgrading pip"

Step "Installing PyTorch"
if ($Cuda) {
    Write-Host "    Installing the CUDA build from $CudaIndex (large download)..."
    Invoke-Checked $venvPy @("-m", "pip", "install", "torch", "--index-url", $CudaIndex) "installing CUDA PyTorch"
} else {
    # The default PyPI wheel for Windows is the CPU build.
    Invoke-Checked $venvPy @("-m", "pip", "install", "torch") "installing PyTorch"
}

Step "Installing backend dependencies"
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "backend\requirements-dev.txt") "installing backend requirements"
$torchInfo = & $venvPy -c "import torch; print(torch.__version__, 'CUDA available' if torch.cuda.is_available() else 'CPU only')"
Ok "PyTorch $torchInfo"

# ----------------------------------------------------------------------------- frontend
if (-not $SkipFrontend) {
    Step "Installing and building the frontend"
    Push-Location (Join-Path $Root "frontend")
    try {
        if (Test-Path "package-lock.json") { Invoke-Checked "npm" @("ci", "--no-audit", "--no-fund") "npm ci" }
        else { Invoke-Checked "npm" @("install", "--no-audit", "--no-fund") "npm install" }
        Invoke-Checked "npm" @("run", "build") "building the frontend"
    } finally { Pop-Location }
    Ok "frontend\dist built"
}

# ----------------------------------------------------------------------------- data + database
New-Item -ItemType Directory -Force -Path (Join-Path $Root "data\incoming") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

if ($DownloadData) {
    Step "Downloading and verifying the NASA SMAP / MSL benchmark"
    Invoke-Checked $venvPy @("scripts\acquire_data.py") "dataset acquisition"
} else {
    Write-Host ""
    Write-Host "    Dataset not downloaded (run .\setup.ps1 -DownloadData, or use the Dataset Explorer page)." -ForegroundColor Yellow
}

Step "Initialising the database"
Invoke-Checked $venvPy @("scripts\init_db.py") "database initialisation"

# ----------------------------------------------------------------------------- resources
Step "Checking resources"
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $totalGb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    Ok "RAM: $freeGb GB free of $totalGb GB; $env:NUMBER_OF_PROCESSORS logical CPUs"
    if ($freeGb -lt 4) { Warn "Little free memory: use the Quick profile and few channels per experiment." }
} catch { }

Write-Host ""
Write-Host "Setup complete. Start the application with:  .\start.ps1" -ForegroundColor Green
