# Putusan TPPO DeepSeek Aggregator

This pipeline reads every `.txt` decision in
`downloads/TPPO/raw-text`, sends a cleaned line-numbered source to
`deepseek-ai/DeepSeek-V4-Pro` through W&B Inference, and asks the model to
return only small line spans using the same contract as the Codex extractor:
`{"sections": {"field": {"lines": [[start, end]]}}}` / `text` / `empty`.
Python then expands those spans into exact source excerpts, validates them, and
writes one JSON file per decision.

The model is not allowed to summarize or generate cell content. A response is
accepted only when all 31 section keys are present, every line range is valid,
and every expanded or short literal value is an exact contiguous source
excerpt. Invalid JSON, malformed spans, empty HTTP 200 responses, missing
choices, rate limits, server errors, and non-verbatim output are retried with
exponential backoff. An HTTP 200 response with all sections empty, or with an
obvious labeled identity field omitted, is also rejected and retried with
corrective feedback.

Equivalent-but-malformed line-span forms the model commonly emits are coerced
rather than rejected, so a single shape slip no longer discards an otherwise
correct 31-section answer: `{"lines": []}`/`{"text": []}` become empty, a flat
`{"lines": [9, 10]}` and a string `{"lines": ["9-10"]}` both become the range
`[[9, 10]]`, and a single-line `{"lines": [[9]]}` becomes `[[9, 9]]`. Every
coerced range is still bounds-checked and expanded to a verbatim source
excerpt.

The user prompt embeds `LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md`,
sanitized field guidance from
`LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md`,
`LLM-aggregator/Putusan-schema.md`, and the cleaned line-numbered source.
Launcher, checkpoint, usage-guard, and Codex agent-loop directions are removed
before the prompt is sent. DeepSeek only locates line ranges and short
literals; deterministic Python code writes the final extraction JSON.

## Configuration

Put the W&B key in this directory's `.env`:

```dotenv
api_key=YOUR_WANDB_API_KEY
```

Alternatively set `WANDB_API_KEY` or `OPENAI_API_KEY`. To associate usage with
a W&B project, set `WANDB_PROJECT=entity/project` or pass `--project`.

## Commands

### One-click launchers

The Windows and Unix launchers contain all normal configuration in one place:

```powershell
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1
```

```bash
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh
```

Its defaults run eight parallel workers with reasoning off, a 32,768-token
output budget per request, and the Rich live dashboard. Common controls:

```powershell
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Action Status
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Action Pause
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Action Resume
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Action RetryEmpty -MaxFiles 20
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Workers 8 -MaxFiles 100
```

Unix equivalents:

```bash
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh --status
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh --pause
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh --resume
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh --retry-empty --max-files 20
./LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.sh --workers 8 --max-files 100
```

Edit the parameter defaults at the top of
`LLM-aggregator/TPPO/Deepseek/run-tppo-deepseek.ps1` for literal single-click
use.

Inspect the queue without calling the API:

```powershell
uv run tppo-deepseek-aggregate --dry-run
```

Run or resume all pending files:

```powershell
uv run tppo-deepseek-aggregate
```

The default is eight concurrent API requests. Adjust bounded concurrency with
`--workers` (1 to 16):

```powershell
uv run tppo-deepseek-aggregate --workers 8
```

Each worker owns its HTTP session and output file. The main thread serializes
JSONL checkpoint writes, so parallel requests cannot interleave state lines.
The default Rich dashboard shows corpus and batch progress, active worker
files, failed requests, API token usage, and recent events. Each active worker
also reports its current observable lifecycle stage:

- preparing the strict extractive request;
- waiting for W&B, including attempt number and timeout;
- parsing the HTTP response;
- validating JSON properties and exact source spans;
- retry backoff with the rejection/error reason and delay;
- comparing a `RetryEmpty` result; and
- atomically saving the accepted JSON.

Elapsed time and time spent in the current stage update continuously, including
while W&B is still generating a response.

Normal runs stream `choices[0].delta.content`, so the worker table shows JSON
characters arriving while the extraction is being generated. The launcher and
the direct Python CLI both default to `off` (no thinking). Available levels are
`off`, `low`, `medium`, `high`, and `xhigh`:

```powershell
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -ReasoningEffort off
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -ReasoningEffort low
.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -ReasoningEffort medium
```

The default is `off` because the model only emits small line-span JSON: live
probes across small and very large (200 KB) decisions found that `medium`/`high`
thinking gave **no recall gain** over `off` while costing 5-13x the latency and
reserving most of the output budget for reasoning tokens. That budget pressure
is the only path by which W&B's `max_tokens` could truncate a response, so `off`
also removes the truncation risk. If a streamed response ever finishes with
`finish_reason=length`, the worker automatically disables thinking and retries
so the full budget is reserved for the span JSON. When reasoning is not `off`,
live `choices[0].delta.reasoning` chunks are shown as a bounded preview in the
active-worker table. Reasoning is never written into the extraction JSON; only
the separately streamed content is schema-checked and source-validated.

With reasoning `off`, small documents use a smaller dynamic output budget to
avoid unnecessary token reservation. Eight workers can improve corpus
throughput, but it also means up to eight expensive requests may run
simultaneously. Reduce `-Workers` if W&B starts returning rate limits or
timeouts.

Never-attempted decisions are processed from smallest to largest so a very
large or repeatedly failing case cannot block early progress. Previously
failed cases remain pending at the end of the queue and are retried after fresh
files.

Process a bounded batch:

```powershell
uv run tppo-deepseek-aggregate --max-files 10
```

Process or retry one exact decision:

```powershell
uv run tppo-deepseek-aggregate --source 10_Pid.Sus_2025_PN_End.txt
```

Create `LLM-aggregator/TPPO/Deepseek/pause` to stop before the next request.
Delete that file and run the same command to resume. `Ctrl+C` also stops
cleanly after preserving completed requests.

Progress is append-only JSONL at
`LLM-aggregator/TPPO/Deepseek/progress.jsonl`. Each accepted response is saved
atomically as `LLM-aggregator/TPPO/Deepseek/output/<source-stem>.json`.
Every file contains all 31 section arrays plus an explicit `empty_sections`
list, making missing content visible without opening an aggregate CSV. Failed
files remain pending and are retried on the next run. If a source file changes,
its stored SHA-256 no longer matches and it is processed again.

### Empty section policy

- A response with all 31 sections empty is considered broken and is
  automatically retried during the same request.
- A partially empty result is accepted because many decisions genuinely omit
  sections such as arrest, experts, or documentary evidence.
- Run
  `.\LLM-aggregator\TPPO\Deepseek\run-tppo-deepseek.ps1 -Action RetryEmpty`
  to retry completed files
  with partial empty sections later. This action does not process unfinished
  files. The existing JSON is replaced only when the new extraction has fewer
  empty sections, so a retry cannot make the file less complete.
- `status: "no_text"` files are not retried by `RetryEmpty`; their raw text
  contained only site boilerplate.

When a raw `.txt` contains only Mahkamah Agung headers and disclaimer text, the
pipeline does not spend an API call. It writes a normal per-source JSON with
`status: "no_text"`, all sections empty, `model: null`, and
`request_attempts: 0`.

Use `--help` for timeout, retry, path, and project options.
