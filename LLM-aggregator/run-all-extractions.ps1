<#
.SYNOPSIS
    Run both the TPPO and Anak Codex span-extraction loops, one after the other.

.DESCRIPTION
    Thin orchestrator over the two per-corpus run-codex-extraction.ps1 launchers.
    Each launcher is started in its OWN PowerShell process so its internal
    `exit` cannot tear this script down before the second corpus runs. Works on
    Windows PowerShell 5.1 and PowerShell 7+ (macOS/Linux).

.EXAMPLE
    pwsh -File LLM-aggregator/run-all-extractions.ps1 -Target 5
.EXAMPLE
    pwsh -File LLM-aggregator/run-all-extractions.ps1 -StatusOnly
#>
param(
    [ValidateRange(1, 1000)]
    [int]$Target = 1,
    [string]$Model = "",
    [ValidateSet("Span", "Legacy")]
    [string]$Mode = "Span",
    [string]$ReasoningEffort = "low",
    [switch]$NoPause,
    # Just print pending/completed counts for both corpora and exit.
    [switch]$StatusOnly
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# Same PowerShell host that is running this script (pwsh on macOS/Linux).
$IsWindowsPlatform = if (Test-Path Variable:IsWindows) { [bool]$IsWindows } else { $true }
$PwshExe = (Get-Process -Id $PID -ErrorAction SilentlyContinue).Path
if (-not $PwshExe) { $PwshExe = if ($IsWindowsPlatform) { "powershell.exe" } else { "pwsh" } }

$corpora = @(
    [pscustomobject]@{ Name = "TPPO"; Script = (Join-Path $here "TPPO/GPT/run-codex-extraction.ps1") }
    [pscustomobject]@{ Name = "Anak"; Script = (Join-Path $here "Anak/GPT/run-codex-extraction.ps1") }
)

$failures = @()
foreach ($corpus in $corpora) {
    if (-not (Test-Path -LiteralPath $corpus.Script)) {
        Write-Error "Launcher not found: $($corpus.Script)"
        $failures += $corpus.Name
        continue
    }

    $childArgs = @("-NoProfile", "-File", $corpus.Script, "-Mode", $Mode)
    if ($StatusOnly) {
        $childArgs += @("-Action", "Status")
    } else {
        $childArgs += @("-Target", $Target)
        if ($Model.Trim().Length -gt 0) { $childArgs += @("-Model", $Model) }
        if ($PSBoundParameters.ContainsKey("ReasoningEffort")) { $childArgs += @("-ReasoningEffort", $ReasoningEffort) }
        if ($NoPause) { $childArgs += "-NoPause" }
    }

    Write-Host ""
    Write-Host "==================================================================="
    Write-Host " $($corpus.Name) extractor"
    Write-Host "==================================================================="
    & $PwshExe @childArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Warning "$($corpus.Name) extractor exited with code $code."
        $failures += $corpus.Name
    }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Error "Completed with failures: $($failures -join ', '). See the per-corpus logs above."
    exit 1
}
Write-Host "Both TPPO and Anak extractors finished."
exit 0
