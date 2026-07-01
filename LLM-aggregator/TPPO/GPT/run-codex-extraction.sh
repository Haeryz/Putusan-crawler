#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CORPUS="TPPO"
INPUT_DIR="downloads/TPPO/raw-text"
OUTPUT_DIR="LLM-aggregator/TPPO/GPT/output"
REPORTS_DIR="LLM-aggregator/TPPO/GPT/reports"
SPANS_DIR="LLM-aggregator/TPPO/GPT/.spans"
PROGRESS_FILE="LLM-aggregator/TPPO/GPT/progress.jsonl"
SCHEMA_FILE="LLM-aggregator/TPPO/GPT/TPPO.json"
INSTRUCTION_FILE="LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md"
SPEC_FILE="LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md"
LIB_SCRIPT="LLM-aggregator/TPPO/GPT/lib/court_extract.py"
FORMAT_GUIDE="TPPO Format PDF"

ACTION="Run"
TARGET=0
MODEL="gpt-5.4-mini"
MODE="Span"
REASONING_EFFORT="low"
DISABLE_MCP=1
USAGE_STOP_PERCENT=10
USAGE_STATUS_FILE=""
MAX_RUN_MINUTES=270
DISABLE_USAGE_GUARD=0
DISABLE_WALL_CLOCK_GUARD=0
JSON_EVENTS=0
NO_PAUSE=0
JOBS=1
NO_TUI=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --action|-Action) ACTION="$2"; shift 2 ;;
    --status|-Status) ACTION="Status"; shift ;;
    --prompt|-Prompt) ACTION="Prompt"; shift ;;
    --target|-Target|--max-files|-MaxFiles) TARGET="$2"; shift 2 ;;
    --model|-Model) MODEL="$2"; shift 2 ;;
    --mode|-Mode) MODE="$2"; shift 2 ;;
    --reasoning-effort|-ReasoningEffort) REASONING_EFFORT="$2"; shift 2 ;;
    --disable-mcp|-DisableMcp) DISABLE_MCP=1; shift ;;
    --keep-mcp) DISABLE_MCP=0; shift ;;
    --usage-stop-percent|-UsageStopPercent) USAGE_STOP_PERCENT="$2"; shift 2 ;;
    --usage-status-file|-UsageStatusFile) USAGE_STATUS_FILE="$2"; shift 2 ;;
    --max-run-minutes|-MaxRunMinutes) MAX_RUN_MINUTES="$2"; shift 2 ;;
    --disable-usage-guard|-DisableUsageGuard) DISABLE_USAGE_GUARD=1; shift ;;
    --disable-wall-clock-guard|-DisableWallClockGuard) DISABLE_WALL_CLOCK_GUARD=1; shift ;;
    --json-events|-JsonEvents) JSON_EVENTS=1; shift ;;
    --no-pause|-NoPause) NO_PAUSE=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-tui|-NoTui) NO_TUI=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${ACTION,,}" in run) ACTION="Run" ;; status) ACTION="Status" ;; prompt) ACTION="Prompt" ;; *) echo "Unknown action: $ACTION" >&2; exit 2 ;; esac
case "${MODE,,}" in span) MODE="Span" ;; legacy) MODE="Legacy" ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac

resolve_python() {
  if [ -x ".venv/bin/python" ]; then PYTHON_CMD=(".venv/bin/python"); return; fi
  if [ -x ".venv/Scripts/python.exe" ]; then PYTHON_CMD=(".venv/Scripts/python.exe"); return; fi
  if command -v python3 >/dev/null 2>&1 && python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
    PYTHON_CMD=(python3); return
  fi
  if command -v python >/dev/null 2>&1 && python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
    PYTHON_CMD=(python); return
  fi
  if command -v py >/dev/null 2>&1; then PYTHON_CMD=(py -3); return; fi
  if command -v uv >/dev/null 2>&1; then PYTHON_CMD=(uv run python); return; fi
  echo "Python 3 was not found on PATH." >&2
  exit 127
}

first_pending_source() {
  "${PYTHON_CMD[@]}" - "$INPUT_DIR" "$PROGRESS_FILE" "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
progress = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
done = set()
if progress.exists():
    for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "completed" and record.get("source_file"):
            done.add(record["source_file"])
outputs = {p.stem for p in output_dir.glob("*.json")} if output_dir.exists() else set()
for path in sorted(input_dir.glob("*.txt"), key=lambda item: item.name):
    if path.name not in done and path.stem not in outputs:
        print(path.name)
        raise SystemExit(0)
PY
}

pending_sources() {
  "${PYTHON_CMD[@]}" - "$INPUT_DIR" "$PROGRESS_FILE" "$OUTPUT_DIR" "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
progress = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
target = int(sys.argv[4])
done = set()
if progress.exists():
    for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "completed" and record.get("source_file"):
            done.add(record["source_file"])
outputs = {p.stem for p in output_dir.glob("*.json")} if output_dir.exists() else set()
pending = [
    path.name
    for path in sorted(input_dir.glob("*.txt"), key=lambda item: item.name)
    if path.name not in done and path.stem not in outputs
]
if target > 0:
    pending = pending[:target]
print("\n".join(pending))
PY
}

resolve_codex() {
  if command -v codex >/dev/null 2>&1; then
    CODEX_CMD=(codex)
    return
  fi
  echo "Codex CLI was not found on PATH. Install it and run 'codex login', then retry." >&2
  exit 127
}

run_legacy() {
  resolve_codex
  local codex_args=(exec --cd "$ROOT" --sandbox workspace-write)
  [ -n "$MODEL" ] && codex_args+=(--model "$MODEL")
  [ -n "$REASONING_EFFORT" ] && codex_args+=(-c "model_reasoning_effort=\"$REASONING_EFFORT\"")
  [ "$JSON_EVENTS" -eq 1 ] && codex_args+=(--json)
  codex_args+=(-)

  mapfile -t sources < <(pending_sources)
  if [ "${#sources[@]}" -eq 0 ]; then
    echo "No pending sources."
    return 0
  fi

  local index=1
  for source_name in "${sources[@]}"; do
    echo
    echo "Legacy session $index of ${#sources[@]}: $source_name"
    legacy_prompt "$source_name" | "${CODEX_CMD[@]}" "${codex_args[@]}"
    index=$((index + 1))
  done
}

numbered_source_for() {
  "${PYTHON_CMD[@]}" "$LIB_SCRIPT" clean "$INPUT_DIR/$1"
}

pause_instruction() {
  if [ "$NO_PAUSE" -eq 1 ]; then
    printf '%s' "Do not pause for user confirmation. Make reasonable assumptions and keep the extraction loop moving."
  else
    printf '%s' "Do not pause for user confirmation unless the next action would be destructive outside the GPT extraction paths."
  fi
}

span_prompt() {
  local source_name="$1"
  local numbered="$2"
  local spans_path="$3"
  local spec
  spec="$(<"$SPEC_FILE")"
  cat <<EOF
You are Codex running the token-optimized TPPO span-extraction task in:
$ROOT

Assigned source: $INPUT_DIR/$source_name
The cleaned, line-numbered source is provided INLINE below. Do NOT open the
source file, the $FORMAT_GUIDE, $INSTRUCTION_FILE, or any other guide --
everything you need is inline. Do not re-read or search files.

YOUR ONLY OUTPUT: write a spans JSON file to exactly this path and nothing else:
  $spans_path
Do NOT write the final output JSON. Do NOT edit $PROGRESS_FILE. A deterministic
post-processor expands your spans into the schema-conforming artifact and the
checkpoint. After writing the spans file, stop.

$spec

=== CLEANED LINE-NUMBERED SOURCE (1-based; point your line ranges into these) ===
$numbered
=== END SOURCE ===

Work in a single pass: do not re-read or re-verify files. Write the spans JSON
to $spans_path covering all 31 section keys, then stop. $(pause_instruction)
EOF
}

legacy_prompt() {
  local source_name="$1"
  cat <<EOF
You are Codex running the TPPO GPT extraction loop in repository:
$ROOT

This is not a documentation task. Execute the extraction loop.

Authoritative files:
- Instructions: $INSTRUCTION_FILE
- JSON Schema: $SCHEMA_FILE
- Raw text input: $INPUT_DIR
- Per-source output directory: $OUTPUT_DIR
- Checkpoint JSONL: $PROGRESS_FILE
- Run reports: $REPORTS_DIR

Session assignment:
Process this exact source file in this session: $INPUT_DIR/$source_name. Do not choose or process any other source file.

Loop contract:
1. Read $INSTRUCTION_FILE and $SCHEMA_FILE before extracting.
2. Confirm the assigned source is not already completed in $PROGRESS_FILE and does not already have a JSON file in $OUTPUT_DIR.
3. Process exactly one pending source in this Codex session.
4. For the current source, manually extract all 31 sections as exact contiguous source excerpts.
5. Write exactly one JSON output per source at $OUTPUT_DIR/<source-stem>.json, conforming to $SCHEMA_FILE.
6. Verify the output has all 31 section keys, accurate empty_sections, and non-empty values copied from the source.
7. Append exactly one checkpoint JSONL record to $PROGRESS_FILE after a source is verified.
8. Stop this Codex session after exactly one source is completed and verified.
9. $(pause_instruction)
EOF
}

resolve_python
mkdir -p "$OUTPUT_DIR" "$REPORTS_DIR" "$SPANS_DIR"
touch "$PROGRESS_FILE"

if [ "$ACTION" = "Status" ]; then
  exec "${PYTHON_CMD[@]}" run_extractions.py --corpus "$CORPUS" --status
fi

if [ "$ACTION" = "Prompt" ]; then
  source_name="$(first_pending_source || true)"
  source_name="${source_name:-example.txt}"
  if [ "$MODE" = "Legacy" ]; then
    legacy_prompt "$source_name"
  elif [ -f "$INPUT_DIR/$source_name" ]; then
    numbered="$(numbered_source_for "$source_name")"
    span_prompt "$source_name" "$numbered" "$SPANS_DIR/<stem>.spans.json"
  else
    span_prompt "$source_name" "<numbered source inline here>" "$SPANS_DIR/<stem>.spans.json"
  fi
  exit 0
fi

if [ "$MODE" = "Legacy" ]; then
  run_legacy
  exit 0
fi

ARGS=(run_extractions.py --corpus "$CORPUS" --target "$TARGET" --reasoning-effort "$REASONING_EFFORT" --usage-stop-percent "$USAGE_STOP_PERCENT" --max-run-minutes "$MAX_RUN_MINUTES" --jobs "$JOBS")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
[ -n "$USAGE_STATUS_FILE" ] && ARGS+=(--usage-status-file "$USAGE_STATUS_FILE")
[ "$DISABLE_MCP" -eq 0 ] && ARGS+=(--keep-mcp)
[ "$DISABLE_USAGE_GUARD" -eq 1 ] && ARGS+=(--disable-usage-guard)
[ "$DISABLE_WALL_CLOCK_GUARD" -eq 1 ] && ARGS+=(--disable-wall-clock-guard)
[ "$JSON_EVENTS" -eq 1 ] && ARGS+=(--json-events)
[ "$NO_PAUSE" -eq 1 ] && ARGS+=(--no-pause)
[ "$NO_TUI" -eq 1 ] && ARGS+=(--no-tui)

exec "${PYTHON_CMD[@]}" "${ARGS[@]}"
