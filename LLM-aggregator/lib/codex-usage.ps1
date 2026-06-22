# Shared Codex usage guard for extraction launchers.

function ConvertFrom-CodexUsageText {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Text)

    if (-not $Text) { return $null }
    $clean = $Text -replace "`e\[[0-9;?]*[A-Za-z]", ""
    $line = ($clean -split "`r?`n" | Where-Object { $_ -match '5h\s+limit' } | Select-Object -First 1)
    if (-not $line) { return $null }

    $percent = $null
    if ($line -match '(\d+)\s*%\s*left') {
        $percent = [int]$Matches[1]
    } elseif ($line -match '(\d+)\s*%') {
        $percent = [int]$Matches[1]
    } else {
        return $null
    }

    $reset = ""
    if ($line -match 'resets\s+([^)]+)') {
        $reset = $Matches[1].Trim()
    }

    return [pscustomobject]@{
        PercentLeft = $percent
        Reset = $reset
        Source = "status-text"
        RawLine = $line.Trim()
    }
}

function Get-CodexUsageStatus {
    [CmdletBinding()]
    param([string]$StatusFile = "")

    $fromLogs = Get-CodexUsageFromLogs
    if ($fromLogs) { return $fromLogs }

    if ($env:CODEX_5H_LIMIT_PERCENT_LEFT -match '^\d+$') {
        return [pscustomobject]@{
            PercentLeft = [int]$env:CODEX_5H_LIMIT_PERCENT_LEFT
            Reset = $env:CODEX_5H_LIMIT_RESET
            Source = "CODEX_5H_LIMIT_PERCENT_LEFT"
            RawLine = "5h limit: $($env:CODEX_5H_LIMIT_PERCENT_LEFT)% left"
        }
    }

    if ($env:CODEX_STATUS_TEXT) {
        $parsed = ConvertFrom-CodexUsageText -Text $env:CODEX_STATUS_TEXT
        if ($parsed) {
            $parsed.Source = "CODEX_STATUS_TEXT"
            return $parsed
        }
    }

    if ($StatusFile.Trim().Length -gt 0 -and (Test-Path -LiteralPath $StatusFile)) {
        $parsed = ConvertFrom-CodexUsageText -Text (Get-Content -LiteralPath $StatusFile -Raw)
        if ($parsed) {
            $parsed.Source = $StatusFile
            return $parsed
        }
    }

    $codex = Get-Command "codex.cmd" -ErrorAction SilentlyContinue
    if ($null -eq $codex) { $codex = Get-Command "codex" -ErrorAction SilentlyContinue }
    if ($null -ne $codex) {
        foreach ($candidateArgs in @(@("status"), @("debug", "usage"))) {
            try {
                $output = & $codex.Source @candidateArgs 2>&1 | Out-String
                if ($LASTEXITCODE -eq 0 -and $output) {
                    $parsed = ConvertFrom-CodexUsageText -Text $output
                    if ($parsed) {
                        $parsed.Source = "codex $($candidateArgs -join ' ')"
                        return $parsed
                    }
                }
            } catch {
                # Current Codex versions do not expose a non-interactive status
                # command. Ignore failures and let the caller decide policy.
            }
        }
    }

    return $null
}

function Get-CodexUsageFromLogs {
    [CmdletBinding()]
    param()

    $sqlite = Get-Command "sqlite3" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $sqlite) { return $null }

    $logsDb = Join-Path $HOME ".codex/logs_2.sqlite"
    if (-not (Test-Path -LiteralPath $logsDb)) { return $null }

    $sql = @"
select ts || char(9) || feedback_log_body
from logs
where target = 'codex_api::endpoint::responses_websocket'
  and feedback_log_body like '%websocket event:%'
  and feedback_log_body like '%codex.rate_limits%'
order by ts desc, ts_nanos desc, id desc
limit 50;
"@
    try {
        $rows = @(& $sqlite.Source $logsDb $sql 2>$null)
    } catch {
        return $null
    }
    if ($rows.Count -eq 0) { return $null }

    $event = $null
    $ts = 0
    foreach ($row in $rows) {
        $tab = $row.IndexOf("`t")
        if ($tab -lt 1) { continue }
        $candidateTs = 0
        if (-not [int64]::TryParse($row.Substring(0, $tab), [ref]$candidateTs)) { continue }
        $body = $row.Substring($tab + 1)
        $marker = "websocket event: "
        $idx = $body.IndexOf($marker)
        if ($idx -lt 0) { continue }
        $jsonText = $body.Substring($idx + $marker.Length).Trim()
        try {
            $candidate = $jsonText | ConvertFrom-Json
        } catch {
            continue
        }
        if ($candidate.type -eq "codex.rate_limits" -and $candidate.rate_limits -and $candidate.rate_limits.primary) {
            $event = $candidate
            $ts = $candidateTs
            break
        }
    }
    if ($null -eq $event) { return $null }

    $used = [int]$event.rate_limits.primary.used_percent
    $left = [math]::Max(0, 100 - $used)
    $reset = ""
    if ($event.rate_limits.primary.reset_at) {
        try {
            $resetAt = [DateTimeOffset]::FromUnixTimeSeconds([int64]$event.rate_limits.primary.reset_at).LocalDateTime
            $reset = $resetAt.ToString("HH:mm on dd MMM")
        } catch {
            $reset = ""
        }
    } elseif ($event.rate_limits.primary.reset_after_seconds) {
        $reset = "in $([int][math]::Ceiling([double]$event.rate_limits.primary.reset_after_seconds / 60)) min"
    }

    return [pscustomobject]@{
        PercentLeft = [int]$left
        Reset = $reset
        Source = $logsDb
        RawLine = "5h limit: $left% left ($used% used)"
        EventAgeSeconds = [int]([DateTimeOffset]::Now.ToUnixTimeSeconds() - $ts)
    }
}

function Test-CodexUsageAllowsStart {
    [CmdletBinding()]
    param(
        [int]$StopPercent = 10,
        [string]$StatusFile = ""
    )

    $status = Get-CodexUsageStatus -StatusFile $StatusFile
    if ($null -eq $status) {
        return [pscustomobject]@{
            Allowed = $true
            Status = $null
            Reason = "usage status unavailable"
        }
    }

    $allowed = $status.PercentLeft -ge $StopPercent
    $reason = if ($allowed) {
        "5h usage $($status.PercentLeft)% left"
    } else {
        "5h usage $($status.PercentLeft)% left, below stop threshold $StopPercent%"
    }

    return [pscustomobject]@{
        Allowed = $allowed
        Status = $status
        Reason = $reason
    }
}
