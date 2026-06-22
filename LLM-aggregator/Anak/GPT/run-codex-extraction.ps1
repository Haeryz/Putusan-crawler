param(
    [ValidateSet("Run", "Status", "Prompt")]
    [string]$Action = "Run",
    [Alias("MaxFiles")]
    [ValidateRange(1, 1000)]
    [int]$Target = 10,
    [string]$Model = "",
    [ValidateSet("Span", "Legacy")]
    [string]$Mode = "Span",
    # Boundary location with explicit line numbers + a deterministic verifier
    # does not need high reasoning. "low" is a safe, cheaper default; pass
    # -ReasoningEffort "" to use the model's own default.
    [string]$ReasoningEffort = "low",
    # MCP servers are not needed for extraction and only add failing-auth noise,
    # startup latency, and tool-schema input tokens. Disabled by default for
    # these runs; pass -DisableMcp:$false to keep them.
    [bool]$DisableMcp = $true,
    [switch]$JsonEvents,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
# Forward slashes work on Windows PowerShell and PowerShell 7+ (macOS/Linux).
# Backslashes would be treated as literal filename characters on Unix.
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location -LiteralPath $RepositoryRoot
. (Join-Path $PSScriptRoot "../../lib/codex-no-mcp.ps1")

# Cross-platform host detection (Windows PowerShell 5.1 has no $IsWindows).
$IsWindowsPlatform = if (Test-Path Variable:IsWindows) { [bool]$IsWindows } else { $true }
$TempDir = [System.IO.Path]::GetTempPath()
# Re-launch sub-sessions with the same PowerShell host (pwsh on macOS/Linux,
# powershell.exe or pwsh on Windows); fall back to a name on PATH.
$PwshExe = (Get-Process -Id $PID -ErrorAction SilentlyContinue).Path
if (-not $PwshExe) { $PwshExe = if ($IsWindowsPlatform) { "powershell.exe" } else { "pwsh" } }

$InputDir = "downloads/kasus anak/raw-text"
$OutputDir = "LLM-aggregator/Anak/GPT/output"
$ReportsDir = "LLM-aggregator/Anak/GPT/reports"
$LogsDir = "LLM-aggregator/Anak/GPT/logs"
$SpansDir = "LLM-aggregator/Anak/GPT/.spans"
$ProgressFile = "LLM-aggregator/Anak/GPT/progress.jsonl"
$SchemaFile = "LLM-aggregator/Anak/GPT/Anak.json"
$InstructionFile = "LLM-aggregator/Anak/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md"
$SpecFile = "LLM-aggregator/Anak/GPT/SPAN_EXTRACTION_SPEC.md"
$LibScript = "LLM-aggregator/Anak/GPT/lib/anak_extract.py"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $SpansDir | Out-Null
if (-not (Test-Path -LiteralPath $ProgressFile)) {
    New-Item -ItemType File -Force -Path $ProgressFile | Out-Null
}

function Resolve-PythonCommand {
    # Comma operator preserves the array so a single-element result is not
    # unwrapped to a scalar string by PowerShell's return semantics.
    foreach ($candidate in @("python", "python3")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $cmd) { return , @([string]$cmd.Source) }
    }
    $py = Get-Command "py" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $py) { return , @([string]$py.Source, "-3") }
    throw "Python 3 was not found on PATH. Install Python 3 to run the span-extraction pipeline."
}

function Invoke-Python {
    # $PythonCommand is @(exe) or @(exe, "-3"); pass any leading args after the exe.
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $pc = @($PythonCommand)
    $exe = $pc[0]
    $prefix = @()
    if ($pc.Count -gt 1) { $prefix = $pc[1..($pc.Count - 1)] }
    return & $exe @prefix @Arguments
}

function Get-CompletedCount {
    if (-not (Test-Path -LiteralPath $ProgressFile)) { return 0 }
    $lines = Get-Content -LiteralPath $ProgressFile -ErrorAction SilentlyContinue
    if ($null -eq $lines) { return 0 }
    return @($lines | Where-Object { $_.Trim().Length -gt 0 }).Count
}

function Get-SourceCount {
    if (-not (Test-Path -LiteralPath $InputDir)) { return 0 }
    return @(
        Get-ChildItem -LiteralPath $InputDir -Filter "*.txt" -File -ErrorAction SilentlyContinue
    ).Count
}

function Get-PendingSources {
    if (-not (Test-Path -LiteralPath $InputDir)) { return @() }
    $completed = @{}
    if (Test-Path -LiteralPath $ProgressFile) {
        Get-Content -LiteralPath $ProgressFile -ErrorAction SilentlyContinue |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object {
                try {
                    $record = $_ | ConvertFrom-Json
                    if ($record.status -eq "completed" -and $record.source_file) {
                        $completed[$record.source_file] = $true
                    }
                } catch {
                    # Ignore malformed progress lines during queue discovery.
                }
            }
    }
    $outputs = @{}
    if (Test-Path -LiteralPath $OutputDir) {
        Get-ChildItem -LiteralPath $OutputDir -Filter "*.json" -File -ErrorAction SilentlyContinue |
            ForEach-Object { $outputs[$_.BaseName] = $true }
    }
    return @(
        Get-ChildItem -LiteralPath $InputDir -Filter "*.txt" -File -ErrorAction SilentlyContinue |
            Sort-Object Name |
            Where-Object {
                -not $completed.ContainsKey($_.Name) -and
                -not $outputs.ContainsKey($_.BaseName)
            }
    )
}

if ($Action -eq "Status") {
    $sourceCount = Get-SourceCount
    $completedCount = Get-CompletedCount
    $outputCount = @(
        Get-ChildItem -LiteralPath $OutputDir -Filter "*.json" -File -ErrorAction SilentlyContinue
    ).Count
    $pendingCount = @(Get-PendingSources).Count
    Write-Host "Sources:   $sourceCount"
    Write-Host "Progress:  $completedCount completed checkpoint record(s)"
    Write-Host "Outputs:   $outputCount JSON file(s)"
    Write-Host "Pending:   $pendingCount source file(s)"
    Write-Host "Mode:      $Mode"
    Write-Host "State:     $ProgressFile"
    Write-Host "Output:    $OutputDir"
    exit 0
}

$pauseInstruction = if ($NoPause) {
    "Do not pause for user confirmation. Make reasonable assumptions and keep the extraction loop moving."
} else {
    "Do not pause for user confirmation unless the next action would be destructive outside the GPT extraction paths."
}

# ---------------------------------------------------------------------------
# Span-extraction prompt (token-optimized default).
#
# The model receives the cleaned, line-numbered source INLINE (no file reads,
# no PDF, no heavy guides) and emits ONLY a small spans JSON pointing into the
# line numbers. A deterministic post-processor slices the exact contiguous
# excerpts and writes the schema-conforming output + checkpoint. The model
# still performs every section/boundary decision; nothing is offloaded.
# ---------------------------------------------------------------------------
function New-SpanPrompt {
    param(
        [Parameter(Mandatory = $true)][string]$SourceName,
        [Parameter(Mandatory = $true)][string]$SpansPath,
        [Parameter(Mandatory = $true)][string]$NumberedSource
    )
    $spec = Get-Content -LiteralPath $SpecFile -Raw
    $spansForward = $SpansPath.Replace("\", "/")
    return @"
You are Codex running the token-optimized Anak span-extraction task in:
$RepositoryRoot

Assigned source: $InputDir/$SourceName
The cleaned, line-numbered source is provided INLINE below. Do NOT open the
source file, the SKKMA PDF, $InstructionFile, or any other guide -- everything
you need is inline. Do not re-read or search files.

YOUR ONLY OUTPUT: write a spans JSON file to exactly this path and nothing else:
  $spansForward
Do NOT write the final output JSON. Do NOT edit $ProgressFile. A deterministic
post-processor expands your spans into the schema-conforming artifact and the
checkpoint. After writing the spans file, stop.

$spec

=== CLEANED LINE-NUMBERED SOURCE (1-based; point your line ranges into these) ===
$NumberedSource
=== END SOURCE ===

Work in a single pass: do not re-read or re-verify files. Write the spans JSON
to $spansForward covering all 31 section keys, then stop. $pauseInstruction
"@
}

# Legacy generative prompt (kept as a fallback via -Mode Legacy).
function New-CodexPrompt {
    param([string]$SourceName = "")
    $sourceInstruction = if ($SourceName.Trim().Length -gt 0) {
        "Process this exact source file in this session: $InputDir/$SourceName. Do not choose or process any other source file."
    } else {
        "Select exactly one pending source file in deterministic filename order."
    }
    return @"
You are Codex running the Anak GPT extraction loop in repository:
$RepositoryRoot

This is not a documentation task. Execute the extraction loop.

Authoritative files:
- Instructions: $InstructionFile
- JSON Schema: $SchemaFile
- Anak section guide: LLM-aggregator/Anak/GPT/Putusan-schema.md
- Raw text input: $InputDir
- Per-source output directory: $OutputDir
- Checkpoint JSONL: $ProgressFile
- Run reports: $ReportsDir

Session assignment:
$sourceInstruction

Loop contract:
1. Read $InstructionFile, LLM-aggregator/Anak/GPT/Putusan-schema.md, and $SchemaFile before extracting.
2. Confirm the assigned source is not already completed in $ProgressFile and does not already have a JSON file in $OutputDir.
3. Process exactly one pending source in this Codex session.
4. For the current source, manually extract all 31 sections as exact contiguous source excerpts.
5. Write exactly one JSON output per source at $OutputDir/<source-stem>.json, conforming to $SchemaFile.
6. Verify the output has all 31 section keys, accurate empty_sections, and non-empty values copied from the source.
7. Append exactly one checkpoint JSONL record to $ProgressFile after a source is verified.
8. Stop this Codex session after exactly one source is completed and verified.
9. $pauseInstruction
"@
}

function Get-NumberedSource {
    param([Parameter(Mandatory = $true)][string]$SourcePath)
    $numbered = Invoke-Python -Arguments @($LibScript, "clean", $SourcePath)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clean source '$SourcePath' (python exit $LASTEXITCODE)."
    }
    return ($numbered -join "`n")
}

function Invoke-SpanExpand {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$SpansPath,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][string]$SourceName
    )
    if (-not (Test-Path -LiteralPath $SpansPath)) {
        throw "Spans file was not produced by Codex: $SpansPath"
    }
    $summary = Invoke-Python -Arguments @(
        $LibScript, "expand",
        "--source", $SourcePath,
        "--spans", $SpansPath,
        "--out", $OutputPath,
        "--source-file", $SourceName,
        "--source-path", "$InputDir/$SourceName"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Span expansion failed for '$SourceName' (python exit $LASTEXITCODE)."
    }
    return ($summary | Select-Object -Last 1)
}

function Invoke-CodexWithPrompt {
    # Run `codex exec` reading the prompt from stdin. Codex writes its banner and
    # progress to stderr; under ErrorActionPreference=Stop PowerShell 5.1 would
    # promote that text to a terminating NativeCommandError, so relax the
    # preference around the native call and rely on the real exit code instead.
    param([Parameter(Mandatory = $true)][string]$PromptPath)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Stream Codex's stdout to the host so it stays visible but does NOT
        # become this function's return value; return only the real exit code.
        Get-Content -LiteralPath $PromptPath -Raw | & $CodexCommand.Source @CodexArguments | Out-Host
        return [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Add-Checkpoint {
    param([Parameter(Mandatory = $true)][string]$SummaryJson)
    # Single-writer append guarded by a short retry to tolerate parallel sessions.
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            Add-Content -LiteralPath $ProgressFile -Value $SummaryJson -Encoding UTF8
            return
        } catch {
            Start-Sleep -Milliseconds 150
        }
    }
    throw "Could not append checkpoint to $ProgressFile after retries."
}

if ($Action -eq "Prompt") {
    $pending = @(Get-PendingSources)
    $sourceName = if ($pending.Count -gt 0) { $pending[0].Name } else { "example.txt" }
    if ($Mode -eq "Legacy") {
        Write-Host (New-CodexPrompt -SourceName $sourceName)
    } else {
        $script:PythonCommand = Resolve-PythonCommand
        $sourcePath = if ($pending.Count -gt 0) { $pending[0].FullName } else { "" }
        $numbered = if ($sourcePath) { Get-NumberedSource -SourcePath $sourcePath } else { "<numbered source inline here>" }
        Write-Host (New-SpanPrompt -SourceName $sourceName -SpansPath "$SpansDir/<stem>.spans.json" -NumberedSource $numbered)
    }
    exit 0
}

$CodexCommand = Get-Command "codex.cmd" -ErrorAction SilentlyContinue
if ($null -eq $CodexCommand) {
    $CodexCommand = Get-Command "codex" -ErrorAction SilentlyContinue
}
if ($null -eq $CodexCommand) {
    throw "Codex CLI was not found on PATH. Install or log in to Codex CLI, then rerun this launcher."
}

$CodexArguments = @("exec", "--cd", $RepositoryRoot, "--sandbox", "workspace-write")
if ($Model.Trim().Length -gt 0) {
    $CodexArguments += @("--model", $Model)
}
if ($ReasoningEffort.Trim().Length -gt 0) {
    # Lower reasoning effort is a legitimate token lever for this bounded,
    # deterministically-verified task. Opt-in so an unknown key never breaks runs.
    $CodexArguments += @("-c", "model_reasoning_effort=`"$ReasoningEffort`"")
}
if ($JsonEvents) {
    $CodexArguments += @("--json")
}
$CodexArguments += @("-")

$script:PythonCommand = Resolve-PythonCommand

if ($DisableMcp) {
    $noMcpHome = Initialize-CodexNoMcpHome
    if ($noMcpHome) {
        $env:CODEX_HOME = $noMcpHome
        Write-Host "MCP:       disabled (CODEX_HOME=$noMcpHome)"
    }
}

Write-Host "Starting Codex Anak extraction loop..."
Write-Host "Mode:      $Mode (token-optimized span extraction)"
Write-Host "Input:     $InputDir"
Write-Host "Output:    $OutputDir"
Write-Host "Progress:  $ProgressFile"
Write-Host "Target:    $Target new Codex session(s), one source per session"

$pendingSources = @(Get-PendingSources | Select-Object -First $Target)
if ($pendingSources.Count -eq 0) {
    Write-Host "No pending sources."
    exit 0
}
if ($pendingSources.Count -lt $Target) {
    Write-Host "Only $($pendingSources.Count) pending source(s) available; reducing target."
}

$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"

if ($Mode -eq "Legacy") {
    # Original generative loop: the model writes the final output + checkpoint.
    for ($sessionIndex = 1; $sessionIndex -le $pendingSources.Count; $sessionIndex++) {
        $source = $pendingSources[$sessionIndex - 1]
        $promptPath = Join-Path $TempDir ("sinergi-codex-anak-legacy-{0}-{1}.md" -f $runStamp, $sessionIndex)
        Set-Content -LiteralPath $promptPath -Value (New-CodexPrompt -SourceName $source.Name) -Encoding UTF8
        Write-Host ""
        Write-Host "Legacy session $sessionIndex of $($pendingSources.Count): $($source.Name)"
        $exitCode = Invoke-CodexWithPrompt -PromptPath $promptPath
        Remove-Item -LiteralPath $promptPath -Force -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) {
            Write-Error "Codex legacy session $sessionIndex failed with exit code $exitCode."
            exit $exitCode
        }
    }
    exit 0
}

# --- Span-extraction loop (default) ---------------------------------------
$parallel = $pendingSources.Count -gt 1 -and $pendingSources.Count -lt 10

if ($parallel) {
    Write-Host "Execution: parallel ($($pendingSources.Count) Codex sessions)"
    $processes = @()
    for ($sessionIndex = 1; $sessionIndex -le $pendingSources.Count; $sessionIndex++) {
        $source = $pendingSources[$sessionIndex - 1]
        $numbered = Get-NumberedSource -SourcePath $source.FullName
        $spansPath = Join-Path $SpansDir ($source.BaseName + ".spans.json")
        Remove-Item -LiteralPath $spansPath -Force -ErrorAction SilentlyContinue
        $promptPath = Join-Path $TempDir ("sinergi-codex-anak-span-{0}-{1}.md" -f $runStamp, $sessionIndex)
        Set-Content -LiteralPath $promptPath -Value (New-SpanPrompt -SourceName $source.Name -SpansPath $spansPath -NumberedSource $numbered) -Encoding UTF8
        $stdoutPath = Join-Path $LogsDir ("codex-span-{0}-{1}.stdout.log" -f $runStamp, $sessionIndex)
        $stderrPath = Join-Path $LogsDir ("codex-span-{0}-{1}.stderr.log" -f $runStamp, $sessionIndex)
        # Generate a small driver script per session instead of a -Command string.
        # A PowerShell array literal preserves argument quoting (e.g. the
        # model_reasoning_effort="low" config value) that a space-joined string
        # would mangle.
        $argsLiteral = ($CodexArguments | ForEach-Object { "'" + ([string]$_).Replace("'", "''") + "'" }) -join ', '
        $driverPath = Join-Path $TempDir ("sinergi-codex-anak-driver-{0}-{1}.ps1" -f $runStamp, $sessionIndex)
        $codexHomeLine = if ($env:CODEX_HOME) { "`$env:CODEX_HOME = '$($env:CODEX_HOME.Replace("'", "''"))'" } else { "" }
        $driverContent = @"
`$ErrorActionPreference = 'Continue'
$codexHomeLine
Get-Content -LiteralPath '$($promptPath.Replace("'", "''"))' -Raw | & '$($CodexCommand.Source.Replace("'", "''"))' @($argsLiteral)
exit `$LASTEXITCODE
"@
        Set-Content -LiteralPath $driverPath -Value $driverContent -Encoding UTF8
        Write-Host "Starting parallel session $sessionIndex/$($pendingSources.Count): $($source.Name)"
        # -ExecutionPolicy/-WindowStyle are Windows-only; pwsh on macOS/Linux
        # rejects -WindowStyle, so add those switches only on Windows.
        $driverArgs = if ($IsWindowsPlatform) {
            @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $driverPath)
        } else {
            @("-NoProfile", "-File", $driverPath)
        }
        $startInfo = @{
            FilePath = $PwshExe
            ArgumentList = $driverArgs
            WorkingDirectory = $RepositoryRoot
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
        }
        if ($IsWindowsPlatform) { $startInfo.WindowStyle = "Hidden" }
        $processes += [pscustomobject]@{
            Index = $sessionIndex
            Source = $source.Name
            SourcePath = $source.FullName
            OutputPath = Join-Path $OutputDir ($source.BaseName + ".json")
            SpansPath = $spansPath
            PromptPath = $promptPath
            DriverPath = $driverPath
            StdoutPath = $stdoutPath
            StderrPath = $stderrPath
            Process = (Start-Process @startInfo -PassThru)
        }
    }
    $failed = @()
    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        Remove-Item -LiteralPath $entry.PromptPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $entry.DriverPath -Force -ErrorAction SilentlyContinue
        try {
            $summary = Invoke-SpanExpand -SourcePath $entry.SourcePath -SpansPath $entry.SpansPath `
                -OutputPath $entry.OutputPath -SourceName $entry.Source
            Add-Checkpoint -SummaryJson $summary
            Write-Host "Completed: $($entry.Source) -> $($entry.OutputPath)"
        } catch {
            Write-Error "Session $($entry.Index) for $($entry.Source) failed: $_  (see $($entry.StderrPath))"
            $failed += $entry
        }
    }
    if ($failed.Count -gt 0) { exit 1 }
} else {
    Write-Host "Execution: sequential ($($pendingSources.Count) Codex session(s))"
    for ($sessionIndex = 1; $sessionIndex -le $pendingSources.Count; $sessionIndex++) {
        $source = $pendingSources[$sessionIndex - 1]
        $numbered = Get-NumberedSource -SourcePath $source.FullName
        $spansPath = Join-Path $SpansDir ($source.BaseName + ".spans.json")
        Remove-Item -LiteralPath $spansPath -Force -ErrorAction SilentlyContinue
        $outputPath = Join-Path $OutputDir ($source.BaseName + ".json")
        $promptPath = Join-Path $TempDir ("sinergi-codex-anak-span-{0}-{1}.md" -f $runStamp, $sessionIndex)
        Set-Content -LiteralPath $promptPath -Value (New-SpanPrompt -SourceName $source.Name -SpansPath $spansPath -NumberedSource $numbered) -Encoding UTF8
        Write-Host ""
        Write-Host "Session $sessionIndex of $($pendingSources.Count): $($source.Name)"
        $exitCode = Invoke-CodexWithPrompt -PromptPath $promptPath
        Remove-Item -LiteralPath $promptPath -Force -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) {
            Write-Error "Codex session $sessionIndex failed with exit code $exitCode."
            exit $exitCode
        }
        $summary = Invoke-SpanExpand -SourcePath $source.FullName -SpansPath $spansPath `
            -OutputPath $outputPath -SourceName $source.Name
        Add-Checkpoint -SummaryJson $summary
        Write-Host "Completed: $($source.Name) -> $outputPath"
    }
}

exit 0
