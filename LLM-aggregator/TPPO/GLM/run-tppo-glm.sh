#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ACTION="Run"
WORKERS=8
MAX_FILES=0
TIMEOUT_SECONDS=1200
MAX_ATTEMPTS=2
MAX_OUTPUT_TOKENS=32768
NETWORK_FAILURE_THRESHOLD=3
NETWORK_COOLDOWN_SECONDS=60
REASONING_EFFORT="off"
NO_TUI=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --action) ACTION="$2"; shift 2 ;;
    --status) ACTION="Status"; shift ;;
    --pause) ACTION="Pause"; shift ;;
    --resume) ACTION="Resume"; shift ;;
    --retry-empty) ACTION="RetryEmpty"; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
    --network-failure-threshold) NETWORK_FAILURE_THRESHOLD="$2"; shift 2 ;;
    --network-cooldown) NETWORK_COOLDOWN_SECONDS="$2"; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
    --no-tui) NO_TUI=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PAUSE_FILE="LLM-aggregator/TPPO/GLM/pause"
INPUT_DIR="downloads/TPPO/raw-text"
OUTPUT_DIR="LLM-aggregator/TPPO/GLM/output"
STATE_FILE="LLM-aggregator/TPPO/GLM/progress.jsonl"
ENV_FILE="LLM-aggregator/TPPO/Deepseek/.env"

case "$ACTION" in
  Pause)
    mkdir -p "$(dirname "$PAUSE_FILE")"
    : > "$PAUSE_FILE"
    echo "Pause requested. Active API calls will finish; no new calls will start."
    exit 0
    ;;
  Resume)
    rm -f "$PAUSE_FILE"
    ACTION="Run"
    ;;
  Run|Status|RetryEmpty) ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac

ARGS=(
  -m llm_aggregator.tppo_glm
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --state "$STATE_FILE"
  --env-file "$ENV_FILE"
  --pause-file "$PAUSE_FILE"
  --workers "$WORKERS"
  --timeout "$TIMEOUT_SECONDS"
  --max-attempts "$MAX_ATTEMPTS"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --network-failure-threshold "$NETWORK_FAILURE_THRESHOLD"
  --network-cooldown "$NETWORK_COOLDOWN_SECONDS"
  --reasoning-effort "$REASONING_EFFORT"
)

[ "$ACTION" = "Status" ] && ARGS+=(--dry-run)
[ "$ACTION" = "RetryEmpty" ] && ARGS+=(--retry-empty-sections)
[ "$MAX_FILES" -gt 0 ] && ARGS+=(--max-files "$MAX_FILES")
[ "$NO_TUI" -eq 1 ] && ARGS+=(--no-tui)

echo "Action=$ACTION Workers=$WORKERS MaxFiles=$MAX_FILES ReasoningEffort=$REASONING_EFFORT Model=zai-org/GLM-5.2"
if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python "${ARGS[@]}"
elif [ -x ".venv/Scripts/python.exe" ]; then
  exec .venv/Scripts/python.exe "${ARGS[@]}"
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 -c "import llm_aggregator.tppo_glm" >/dev/null 2>&1; then
    exec python3 "${ARGS[@]}"
  fi
fi
if command -v python >/dev/null 2>&1; then
  if python -c "import llm_aggregator.tppo_glm" >/dev/null 2>&1; then
    exec python "${ARGS[@]}"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python "${ARGS[@]}"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "${ARGS[@]}"
elif command -v python >/dev/null 2>&1; then
  exec python "${ARGS[@]}"
else
  echo "Python 3 was not found on PATH." >&2
  exit 127
fi
