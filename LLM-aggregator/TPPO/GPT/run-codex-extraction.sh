#!/usr/bin/env bash
# macOS/Linux launcher — mirrors run-codex-extraction.cmd.
# Requires PowerShell 7+ (pwsh), Python 3, and the Codex CLI on PATH.
#   pwsh:   brew install --cask powershell
# Usage: ./run-codex-extraction.sh [-Action Run|Status|Prompt] [-Target N] ...
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec pwsh -NoProfile -File "$DIR/run-codex-extraction.ps1" "$@"
