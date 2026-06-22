#!/usr/bin/env bash
#
# One-shot bootstrap + run for the Sinergi Codex extractors (macOS / Linux).
#
#   ./setup.sh            # install prerequisites, then run 1 source per corpus
#   ./setup.sh 20         # ...run 20 sources per corpus
#   ./setup.sh --status   # install nothing extra, just show pending counts
#
# Installs (only what is missing): Python 3, PowerShell 7 (pwsh), Node + the
# Codex CLI. Then runs the TPPO and Anak extraction loops via
# LLM-aggregator/run-all-extractions.ps1.
#
# Two things cannot be fully automated and are checked with clear guidance:
#   - Codex login (opens a browser; run once and it is cached in ~/.codex)
#   - the raw-text inputs under downloads/ (kept out of git; sync them first)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

c_info='\033[1;36m'; c_ok='\033[1;32m'; c_warn='\033[1;33m'; c_err='\033[1;31m'; c_off='\033[0m'
log()  { printf "${c_info}[setup]${c_off} %s\n" "$*"; }
ok()   { printf "${c_ok}[ ok ]${c_off} %s\n" "$*"; }
warn() { printf "${c_warn}[warn]${c_off} %s\n" "$*"; }
die()  { printf "${c_err}[fail]${c_off} %s\n" "$*" >&2; exit 1; }

# ---- args -----------------------------------------------------------------
TARGET=1
STATUS_ONLY=0
for a in "$@"; do
  case "$a" in
    --status) STATUS_ONLY=1 ;;
    ''|*[!0-9]*) die "Unknown argument: '$a' (use a number for target, or --status)" ;;
    *) TARGET="$a" ;;
  esac
done

OS="$(uname -s)"
have() { command -v "$1" >/dev/null 2>&1; }

# ---- package-manager abstraction -----------------------------------------
PKG=""
if [ "$OS" = "Darwin" ]; then
  if ! have brew; then
    die "Homebrew is required on macOS. Install it from https://brew.sh then re-run."
  fi
  PKG="brew"
elif have apt-get; then
  PKG="apt"
elif have dnf; then
  PKG="dnf"
else
  warn "No supported package manager found (brew/apt/dnf). Prerequisites must be installed manually."
fi

pkg_install() {
  # pkg_install <human-name> <brew-formula> <apt-pkg> <dnf-pkg>
  local name="$1" brewf="$2" aptf="$3" dnff="$4"
  case "$PKG" in
    brew) brew install $brewf ;;
    apt)  sudo apt-get update -y && sudo apt-get install -y $aptf ;;
    dnf)  sudo dnf install -y $dnff ;;
    *)    die "Cannot auto-install $name; install it manually and re-run." ;;
  esac
}

# ---- Python 3 -------------------------------------------------------------
if have python3 || have python; then
  ok "Python 3 present ($(python3 --version 2>/dev/null || python --version))"
else
  log "Installing Python 3..."
  pkg_install "Python 3" "python" "python3" "python3"
fi

# ---- PowerShell 7 (pwsh) --------------------------------------------------
if have pwsh; then
  ok "PowerShell present ($(pwsh -NoProfile -Command '$PSVersionTable.PSVersion.ToString()' 2>/dev/null))"
else
  log "Installing PowerShell 7 (pwsh)..."
  if [ "$PKG" = "brew" ]; then
    brew install --cask powershell
  elif [ "$PKG" = "apt" ]; then
    # Microsoft package feed (Ubuntu/Debian).
    sudo apt-get update -y
    sudo apt-get install -y wget apt-transport-https software-properties-common
    source /etc/os-release 2>/dev/null || true
    wget -q "https://packages.microsoft.com/config/${ID:-ubuntu}/${VERSION_ID:-22.04}/packages-microsoft-prod.deb" -O /tmp/pmc.deb
    sudo dpkg -i /tmp/pmc.deb && sudo apt-get update -y && sudo apt-get install -y powershell
  elif [ "$PKG" = "dnf" ]; then
    sudo dnf install -y powershell || die "Install PowerShell manually: https://learn.microsoft.com/powershell/scripting/install/install-rhel"
  else
    die "Install PowerShell 7 manually: https://learn.microsoft.com/powershell/scripting/install/installing-powershell"
  fi
  have pwsh || die "pwsh still not on PATH after install."
fi

# ---- Node + Codex CLI -----------------------------------------------------
if have codex; then
  ok "Codex CLI present ($(codex --version 2>/dev/null | head -1))"
else
  if ! have npm; then
    log "Installing Node.js (for the Codex CLI)..."
    pkg_install "Node.js" "node" "nodejs npm" "nodejs npm"
  fi
  log "Installing the Codex CLI (npm i -g @openai/codex)..."
  npm install -g @openai/codex || die "Failed to install @openai/codex. Install it manually, then re-run."
  have codex || warn "Codex installed but not on PATH; open a new shell or add npm global bin to PATH."
fi

# ---- Codex auth -----------------------------------------------------------
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
if [ -f "$CODEX_HOME_DIR/auth.json" ]; then
  ok "Codex is authenticated ($CODEX_HOME_DIR/auth.json)"
else
  warn "Codex is not logged in yet. Launching 'codex login' (opens a browser)..."
  if [ "$STATUS_ONLY" -eq 0 ]; then
    codex login || die "Codex login did not complete. Run 'codex login' manually, then re-run."
  fi
fi

# ---- input data check -----------------------------------------------------
tppo_in="downloads/TPPO/raw-text"
anak_in="downloads/kasus anak/raw-text"
count_txt() { ls "$1"/*.txt >/dev/null 2>&1 && ls "$1"/*.txt 2>/dev/null | wc -l | tr -d ' ' || echo 0; }
tppo_n="$(count_txt "$tppo_in")"
anak_n="$(count_txt "$anak_in")"
if [ "$tppo_n" -gt 0 ] || [ "$anak_n" -gt 0 ]; then
  ok "Inputs: TPPO=$tppo_n  Anak=$anak_n raw-text file(s)"
else
  warn "No raw-text inputs found under downloads/ (these are NOT in git)."
  warn "Sync the folders '$tppo_in' and '$anak_in' from your other device,"
  warn "or generate them with the crawler+extractor (see README), then re-run."
  [ "$STATUS_ONLY" -eq 0 ] && die "Nothing to extract without inputs."
fi

# ---- run ------------------------------------------------------------------
RUNNER="LLM-aggregator/run-all-extractions.ps1"
if [ "$STATUS_ONLY" -eq 1 ]; then
  log "Status for both corpora:"
  exec pwsh -NoProfile -File "$RUNNER" -StatusOnly
else
  log "Running both extractors (target $TARGET source(s) per corpus)..."
  exec pwsh -NoProfile -File "$RUNNER" -Target "$TARGET"
fi
