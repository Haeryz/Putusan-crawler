param(
    [ValidateSet("Run", "Status", "Pause", "Resume", "RetryEmpty")]
    [string]$Action = "Run",
    [ValidateRange(1, 16)]
    [int]$Workers = 8,
    [int]$MaxFiles = 0,
    [int]$TimeoutSeconds = 1200,
    [int]$MaxAttempts = 2,
    [int]$MaxOutputTokens = 32768,
    [int]$NetworkFailureThreshold = 3,
    [int]$NetworkCooldownSeconds = 60,
    [ValidateSet("off", "low", "medium", "high", "xhigh")]
    [string]$ReasoningEffort = "off",
    [string[]]$Source = @(),
    [switch]$NoTui
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location -LiteralPath $RepositoryRoot

$PauseFile = "LLM-aggregator/Anak/Qwen/pause"
$InputDir = "downloads/kasus anak/raw-text"
$OutputDir = "LLM-aggregator/Anak/Qwen/output"
$StateFile = "LLM-aggregator/Anak/Qwen/progress.jsonl"
$EnvFile = "LLM-aggregator/Anak/Deepseek/.env"

if ($Action -eq "Pause") {
    New-Item -ItemType File -Force -Path $PauseFile | Out-Null
    Write-Host "Pause requested. Active API calls will finish; no new calls will start."
    exit 0
}

if ($Action -eq "Resume") {
    Remove-Item -LiteralPath $PauseFile -Force -ErrorAction SilentlyContinue
    $Action = "Run"
}

$PythonArguments = @(
    "-m", "llm_aggregator.anak_qwen",
    "--input-dir", $InputDir,
    "--output-dir", $OutputDir,
    "--state", $StateFile,
    "--env-file", $EnvFile,
    "--pause-file", $PauseFile,
    "--workers", "$Workers",
    "--timeout", "$TimeoutSeconds",
    "--max-attempts", "$MaxAttempts",
    "--max-output-tokens", "$MaxOutputTokens",
    "--network-failure-threshold", "$NetworkFailureThreshold",
    "--network-cooldown", "$NetworkCooldownSeconds"
)

if ($Action -eq "Status") {
    $PythonArguments += "--dry-run"
}

if ($Action -eq "RetryEmpty") {
    $PythonArguments += "--retry-empty-sections"
}

if ($MaxFiles -gt 0) {
    $PythonArguments += @("--max-files", "$MaxFiles")
}

foreach ($SourceFile in $Source) {
    if ($SourceFile) {
        $PythonArguments += @("--source", $SourceFile)
    }
}

if ($NoTui) {
    $PythonArguments += "--no-tui"
}

$PythonArguments += @("--reasoning-effort", $ReasoningEffort)

Write-Host "Action=$Action Workers=$Workers MaxFiles=$MaxFiles ReasoningEffort=$ReasoningEffort"
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython @PythonArguments
} else {
    & uv run python @PythonArguments
}
exit $LASTEXITCODE
