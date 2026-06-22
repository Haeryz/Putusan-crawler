<#
.SYNOPSIS
    One-shot bootstrap + run for the Sinergi Codex extractors (Windows).

.DESCRIPTION
    Installs only what is missing (Python 3, Node.js + the Codex CLI) using
    winget, checks Codex login and the raw-text inputs, then runs both the TPPO
    and Anak extraction loops via LLM-aggregator/run-all-extractions.ps1.

    Codex login (browser) and the downloads/ inputs (kept out of git) cannot be
    fully automated and are checked with clear guidance.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Target 20
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -StatusOnly
#>
param(
    [ValidateRange(1, 1000)]
    [int]$Target = 1,
    [switch]$StatusOnly
)

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location -LiteralPath $ROOT

function Write-Step($m) { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "[ ok ] $m" -ForegroundColor Green }
function Write-Warn2($m){ Write-Host "[warn] $m" -ForegroundColor Yellow }
function Test-Cmd($n)   { $null -ne (Get-Command $n -ErrorAction SilentlyContinue) }

function Install-WithWinget($id, $human) {
    if (-not (Test-Cmd "winget")) {
        throw "winget is not available. Install '$human' manually, then re-run. (winget ships with App Installer from the Microsoft Store.)"
    }
    Write-Step "Installing $human via winget ($id)..."
    winget install --id $id -e --source winget --accept-source-agreements --accept-package-agreements
    # Refresh PATH for this session so the just-installed tool is found.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# ---- Python 3 -------------------------------------------------------------
if ((Test-Cmd "python") -or (Test-Cmd "python3") -or (Test-Cmd "py")) {
    Write-Ok "Python 3 present"
} else {
    Install-WithWinget "Python.Python.3.12" "Python 3.12"
}

# ---- Node + Codex CLI -----------------------------------------------------
if (Test-Cmd "codex") {
    Write-Ok "Codex CLI present"
} else {
    if (-not (Test-Cmd "npm")) {
        Install-WithWinget "OpenJS.NodeJS.LTS" "Node.js LTS"
    }
    if (-not (Test-Cmd "npm")) { throw "npm still not on PATH; open a new terminal and re-run." }
    Write-Step "Installing the Codex CLI (npm i -g @openai/codex)..."
    npm install -g "@openai/codex"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Test-Cmd "codex")) { Write-Warn2 "Codex installed but not on PATH; open a new terminal if the run fails." }
}

# ---- Codex auth -----------------------------------------------------------
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
if (Test-Path -LiteralPath (Join-Path $codexHome "auth.json")) {
    Write-Ok "Codex is authenticated ($codexHome\auth.json)"
} else {
    Write-Warn2 "Codex is not logged in yet."
    if (-not $StatusOnly) {
        Write-Step "Launching 'codex login' (opens a browser)..."
        codex login
        if ($LASTEXITCODE -ne 0) { throw "Codex login did not complete. Run 'codex login' manually, then re-run." }
    }
}

# ---- input data check -----------------------------------------------------
$tppoIn = "downloads/TPPO/raw-text"
$anakIn = "downloads/kasus anak/raw-text"
$tppoN = @(Get-ChildItem -LiteralPath $tppoIn -Filter *.txt -File -ErrorAction SilentlyContinue).Count
$anakN = @(Get-ChildItem -LiteralPath $anakIn -Filter *.txt -File -ErrorAction SilentlyContinue).Count
if ($tppoN -gt 0 -or $anakN -gt 0) {
    Write-Ok "Inputs: TPPO=$tppoN  Anak=$anakN raw-text file(s)"
} else {
    Write-Warn2 "No raw-text inputs found under downloads/ (these are NOT in git)."
    Write-Warn2 "Sync '$tppoIn' and '$anakIn' from your other device, or generate them via the crawler (README), then re-run."
    if (-not $StatusOnly) { throw "Nothing to extract without inputs." }
}

# ---- run ------------------------------------------------------------------
$runner = Join-Path $ROOT "LLM-aggregator/run-all-extractions.ps1"
$hostExe = (Get-Process -Id $PID -ErrorAction SilentlyContinue).Path
if (-not $hostExe) { $hostExe = "powershell.exe" }
if ($StatusOnly) {
    Write-Step "Status for both corpora:"
    & $hostExe -NoProfile -File $runner -StatusOnly
} else {
    Write-Step "Running both extractors (target $Target source(s) per corpus)..."
    & $hostExe -NoProfile -File $runner -Target $Target
}
exit $LASTEXITCODE
