param(
    [ValidateSet("Run", "Status", "Prompt")]
    [string]$Action = "Run",
    [Alias("MaxFiles")]
    [ValidateRange(1, 1000)]
    [int]$Target = 4,
    [string]$Model = "",
    [switch]$JsonEvents,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location -LiteralPath $RepositoryRoot

$InputDir = "downloads/kasus anak/raw-text"
$OutputDir = "LLM-aggregator/Anak/GPT/output"
$ReportsDir = "LLM-aggregator/Anak/GPT/reports"
$LogsDir = "LLM-aggregator/Anak/GPT/logs"
$ProgressFile = "LLM-aggregator/Anak/GPT/progress.jsonl"
$SchemaFile = "LLM-aggregator/Anak/GPT/Anak.json"
$InstructionFile = "LLM-aggregator/Anak/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
if (-not (Test-Path -LiteralPath $ProgressFile)) {
    New-Item -ItemType File -Force -Path $ProgressFile | Out-Null
}

function Get-CompletedCount {
    if (-not (Test-Path -LiteralPath $ProgressFile)) {
        return 0
    }
    $lines = Get-Content -LiteralPath $ProgressFile -ErrorAction SilentlyContinue
    if ($null -eq $lines) {
        return 0
    }
    return @($lines | Where-Object { $_.Trim().Length -gt 0 }).Count
}

function Get-SourceCount {
    if (-not (Test-Path -LiteralPath $InputDir)) {
        return 0
    }
    return @(
        Get-ChildItem -LiteralPath $InputDir -Filter "*.txt" -File -ErrorAction SilentlyContinue
    ).Count
}

function Get-PendingSources {
    if (-not (Test-Path -LiteralPath $InputDir)) {
        return @()
    }
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
    Write-Host "State:     $ProgressFile"
    Write-Host "Output:    $OutputDir"
    exit 0
}

$pauseInstruction = if ($NoPause) {
    "Do not pause for user confirmation. Make reasonable assumptions and keep the extraction loop moving."
} else {
    "Do not pause for user confirmation unless the next action would be destructive outside the GPT extraction paths."
}

function New-CodexPrompt {
    param(
        [string]$SourceName = ""
    )

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
- Anak template PDF: LLM-aggregator/Anak/SKKMA Pidsus Anak-1.pdf
- Raw text input: $InputDir
- Per-source output directory: $OutputDir
- Checkpoint JSONL: $ProgressFile
- Run reports: $ReportsDir

Session assignment:
$sourceInstruction

Loop contract:
1. Read $InstructionFile, LLM-aggregator/Anak/GPT/Putusan-schema.md, and $SchemaFile before extracting.
2. Confirm the assigned source is not already completed in $ProgressFile and does not already have a JSON file in $OutputDir. If it is already complete, write a report and stop without processing another file.
3. Process exactly one pending source in this Codex session.
4. For the current source, manually extract all 31 sections as exact contiguous source excerpts. Do not summarize, translate, normalize, or add reasoning inside section values.
5. Write exactly one JSON output per source at $OutputDir/<source-stem>.json, conforming to $SchemaFile.
6. Verify the output has all 31 section keys, accurate empty_sections, and non-empty values copied from the source.
7. Append exactly one checkpoint JSONL record to $ProgressFile after a source is verified. If the progress file is temporarily locked by another parallel session, retry the append after a short delay instead of abandoning the completed output.
8. Stop this Codex session after exactly one source is completed and verified. Do not continue to a second source inside this same Codex session.
9. The report must include stop reason, usage remaining, reset timing if visible, processed count in this run, completed output paths, last source handled, pending count, failed/skipped sources, and next resume action.
10. Check current Codex/session usage before doing any expensive work. If remaining usage is below 10% of the active five-hour reset window, stop before starting the source and create a Markdown report in $ReportsDir.
11. $pauseInstruction

Do not create one large aggregate JSON. Resume safely from existing outputs and progress records.
"@
}

if ($Action -eq "Prompt") {
    $pending = @(Get-PendingSources)
    $sourceName = if ($pending.Count -gt 0) { $pending[0].Name } else { "" }
    Write-Host (New-CodexPrompt -SourceName $sourceName)
    exit 0
}

$CodexCommand = Get-Command "codex.cmd" -ErrorAction SilentlyContinue
if ($null -eq $CodexCommand) {
    $CodexCommand = Get-Command "codex" -ErrorAction SilentlyContinue
}
if ($null -eq $CodexCommand) {
    throw "Codex CLI was not found on PATH. Install or log in to Codex CLI, then rerun this launcher."
}

$CodexArguments = @(
    "exec",
    "--cd", $RepositoryRoot,
    "--sandbox", "workspace-write",
    "-"
)

if ($Model.Trim().Length -gt 0) {
    $CodexArguments = @("exec", "--cd", $RepositoryRoot, "--sandbox", "workspace-write", "--model", $Model, "-")
}

if ($JsonEvents) {
    $insertIndex = $CodexArguments.Count - 1
    $CodexArguments = $CodexArguments[0..($insertIndex - 1)] + @("--json") + $CodexArguments[$insertIndex..($CodexArguments.Count - 1)]
}

Write-Host "Starting Codex Anak extraction loop..."
Write-Host "Input:     $InputDir"
Write-Host "Output:    $OutputDir"
Write-Host "Progress:  $ProgressFile"
Write-Host "Reports:   $ReportsDir"
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
$parallel = $pendingSources.Count -gt 1 -and $pendingSources.Count -lt 10

if ($parallel) {
    Write-Host "Mode:      parallel ($($pendingSources.Count) Codex sessions)"
    $processes = @()
    for ($sessionIndex = 1; $sessionIndex -le $pendingSources.Count; $sessionIndex++) {
        $source = $pendingSources[$sessionIndex - 1]
        $promptPath = Join-Path $env:TEMP ("sinergi-codex-anak-loop-{0}-{1}.md" -f $runStamp, $sessionIndex)
        Set-Content -LiteralPath $promptPath -Value (New-CodexPrompt -SourceName $source.Name) -Encoding UTF8
        $stdoutPath = Join-Path $LogsDir ("codex-target-{0}-{1}.stdout.log" -f $runStamp, $sessionIndex)
        $stderrPath = Join-Path $LogsDir ("codex-target-{0}-{1}.stderr.log" -f $runStamp, $sessionIndex)
        $command = "Get-Content -LiteralPath '$($promptPath.Replace("'", "''"))' -Raw | & '$($CodexCommand.Source.Replace("'", "''"))' $($CodexArguments -join ' ')"
        Write-Host "Starting parallel target $sessionIndex/$($pendingSources.Count): $($source.Name)"
        $processes += [pscustomobject]@{
            Index = $sessionIndex
            Source = $source.Name
            OutputPath = Join-Path $OutputDir ($source.BaseName + ".json")
            PromptPath = $promptPath
            StdoutPath = $stdoutPath
            StderrPath = $stderrPath
            Process = Start-Process -FilePath "powershell.exe" `
                -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
                -WorkingDirectory $RepositoryRoot `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath `
                -WindowStyle Hidden `
                -PassThru
        }
    }
    $failed = @()
    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        $entry.Process.Refresh()
        Remove-Item -LiteralPath $entry.PromptPath -Force -ErrorAction SilentlyContinue
        $exitCode = $entry.Process.ExitCode
        if ($null -eq $exitCode) {
            $stderrText = if (Test-Path -LiteralPath $entry.StderrPath) {
                Get-Content -Raw -LiteralPath $entry.StderrPath
            } else {
                ""
            }
            $stdoutText = if (Test-Path -LiteralPath $entry.StdoutPath) {
                Get-Content -Raw -LiteralPath $entry.StdoutPath
            } else {
                ""
            }
            $exitCode = if (
                $stdoutText -match "Completed exactly one assigned source" -and
                $stderrText -notmatch "(?m)^(ERROR|error:|Exception|Traceback)"
            ) {
                0
            } else {
                1
            }
        }
        if ($exitCode -ne 0) {
            $checkpointExists = $false
            if (Test-Path -LiteralPath $ProgressFile) {
                $escapedSource = [regex]::Escape($entry.Source)
                $checkpointExists = [bool](
                    Select-String -LiteralPath $ProgressFile -Pattern $escapedSource -Quiet
                )
            }
            if ((Test-Path -LiteralPath $entry.OutputPath) -and $checkpointExists) {
                $exitCode = 0
            }
        }
        $entry | Add-Member -NotePropertyName ExitCode -NotePropertyValue $exitCode
        if ($exitCode -ne 0) {
            $failed += $entry
        }
    }
    if ($failed.Count -gt 0) {
        foreach ($entry in $failed) {
            Write-Error "Parallel target $($entry.Index) for $($entry.Source) failed with exit code $($entry.ExitCode). See $($entry.StdoutPath) and $($entry.StderrPath)."
        }
        exit 1
    }
} else {
    Write-Host "Mode:      sequential ($($pendingSources.Count) Codex session(s))"
    for ($sessionIndex = 1; $sessionIndex -le $pendingSources.Count; $sessionIndex++) {
        $source = $pendingSources[$sessionIndex - 1]
        $promptPath = Join-Path $env:TEMP ("sinergi-codex-anak-loop-{0}-{1}.md" -f $runStamp, $sessionIndex)
        Set-Content -LiteralPath $promptPath -Value (New-CodexPrompt -SourceName $source.Name) -Encoding UTF8
        Write-Host ""
        Write-Host "Starting target session $sessionIndex of $($pendingSources.Count): $($source.Name)"
        Get-Content -LiteralPath $promptPath -Raw | & $CodexCommand.Source @CodexArguments
        $exitCode = $LASTEXITCODE
        Remove-Item -LiteralPath $promptPath -Force -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) {
            Write-Error "Codex target session $sessionIndex failed with exit code $exitCode."
            exit $exitCode
        }
    }
}

exit 0
