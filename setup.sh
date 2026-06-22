#!/usr/bin/env bash
#
# Native one-shot bootstrap + run for the Sinergi Codex extractors (macOS/Linux).
#
#   ./setup.sh            # install prerequisites, then run 1 source per corpus
#   ./setup.sh 20         # ...run 20 sources per corpus
#   ./setup.sh --status   # just show pending/completed counts
#
# Native: no PowerShell. The orchestrator is plain Python (run_extractions.py),
# so the only things this installs (if missing) are Python 3 and the Codex CLI
# (via Node). The raw-text inputs, progress, and outputs are committed, so a
# fresh clone runs without any data sync. The one step that can't be automated
# is the interactive Codex login (opens a browser; cached in ~/.codex after).
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
PYTHON=""
if have python3; then PYTHON="python3"
elif have python; then PYTHON="python"
else
  log "Installing Python 3..."
  pkg_install "Python 3" "python" "python3" "python3"
  have python3 && PYTHON="python3" || PYTHON="python"
fi
ok "Python: $($PYTHON --version 2>&1)"

# ---- Codex CLI (via Node) -------------------------------------------------
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
  warn "No raw-text inputs found under '$tppo_in' or '$anak_in'."
  warn "These are committed to the repo, so your clone may be incomplete -- try"
  warn "'git pull', or regenerate them with the crawler+extractor (see README)."
  [ "$STATUS_ONLY" -eq 0 ] && die "Nothing to extract without inputs."
fi

# ---- run ------------------------------------------------------------------
if [ "$STATUS_ONLY" -eq 1 ]; then
  log "Status for both corpora:"
  exec "$PYTHON" run_extractions.py --status
else
  log "Running both extractors (target $TARGET source(s) per corpus)..."
  exec "$PYTHON" run_extractions.py --target "$TARGET"
fi
