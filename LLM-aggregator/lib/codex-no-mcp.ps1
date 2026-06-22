# Shared helper: run Codex with MCP servers disabled for extraction sessions.
#
# Why: the extraction loop needs no MCP servers, but Codex loads every server
# from ~/.codex/config.toml on each `codex exec`. The remote ones (figma,
# github) fail auth and spam fatal transport errors, and all of them inject
# tool schemas into the model context (wasted input tokens) and slow startup.
#
# How: neither `-c mcp_servers={}` (it merges, not replaces) nor
# `--ignore-user-config` (it also drops the Windows sandbox feature flags, which
# turns workspace-write read-only) works. Instead we build a dedicated
# CODEX_HOME containing a copy of the user config with only the [mcp_servers.*]
# tables stripped (model, reasoning, sandbox features preserved) plus a copy of
# auth.json, and point CODEX_HOME at it for the Codex child only. This disables
# MCP ONLY for these runs and never mutates the user's real config.

function Initialize-CodexNoMcpHome {
    [CmdletBinding()]
    param()

    # $HOME is defined by both Windows PowerShell 5.1 and PowerShell 7+ (macOS/
    # Linux), so it resolves the user's Codex home on every platform.
    $realHome = if ($env:CODEX_HOME -and (Test-Path -LiteralPath $env:CODEX_HOME)) {
        $env:CODEX_HOME
    } else {
        Join-Path $HOME ".codex"
    }
    $configSrc = Join-Path $realHome "config.toml"
    if (-not (Test-Path -LiteralPath $configSrc)) {
        Write-Warning "Codex config.toml not found at $configSrc; running with MCP enabled."
        return $null
    }

    # Per-user data dir for the MCP-stripped Codex home. LOCALAPPDATA on Windows;
    # XDG_CACHE_HOME or ~/.cache on macOS/Linux.
    $dataRoot = if ($env:LOCALAPPDATA) {
        $env:LOCALAPPDATA
    } elseif ($env:XDG_CACHE_HOME) {
        $env:XDG_CACHE_HOME
    } else {
        Join-Path $HOME ".cache"
    }
    $altHome = Join-Path $dataRoot "sinergi-codex-nomcp"
    New-Item -ItemType Directory -Force -Path $altHome | Out-Null

    # Drop every MCP-providing table block. MCP servers come from three places:
    #   [mcp_servers.*]   - directly configured servers
    #   [plugins.*]       - plugin-provided servers (e.g. figma@..., the remote
    #                       URL servers that fail auth and spam the logs)
    #   [marketplaces.*]  - plugin sources that codex syncs plugins from
    # A TOML table runs until the next top-level [header], so skip from a
    # matching header until the next header that is not one of these.
    $dropPattern = '^\s*\[(mcp_servers|plugins|marketplaces)'
    $lines = Get-Content -LiteralPath $configSrc
    $kept = New-Object System.Collections.Generic.List[string]
    $skip = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*\[') { $skip = ($line -match $dropPattern) }
        if (-not $skip) { $kept.Add($line) }
    }
    Set-Content -LiteralPath (Join-Path $altHome "config.toml") -Value $kept -Encoding UTF8

    # Codex may have synced a plugins/ directory into the alt home on earlier
    # runs; remove it so nothing reloads a plugin MCP independently of config.
    $altPlugins = Join-Path $altHome "plugins"
    if (Test-Path -LiteralPath $altPlugins) {
        Remove-Item -LiteralPath $altPlugins -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Refresh auth + identity files each run so the alt home never goes stale.
    foreach ($name in @("auth.json", "installation_id", "version.json")) {
        $src = Join-Path $realHome $name
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $altHome $name) -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not (Test-Path -LiteralPath (Join-Path $altHome "auth.json"))) {
        Write-Warning "auth.json not found in $realHome; MCP-disabled home may not be authenticated."
    }
    return $altHome
}
