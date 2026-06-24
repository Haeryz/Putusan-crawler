# Putusan TPPO GLM Aggregator

This is the GLM replication of the TPPO DeepSeek aggregator. It uses the same
prompt construction, span validation, retry behavior, checkpointing, output
format, and launch controls as `LLM-aggregator/TPPO/Deepseek`, but sends API
requests with:

```json
{"model": "zai-org/GLM-5.2"}
```

Outputs and state are kept separate under `LLM-aggregator/TPPO/GLM`. The API
key is intentionally shared from `LLM-aggregator/TPPO/Deepseek/.env`.

## Concurrency limit (why GLM needs `-Workers 1`)

GLM-5.2 and DeepSeek run through the identical code, endpoint, and key, but W&B
enforces a far lower project-level concurrency cap on `zai-org/GLM-5.2`
(effectively ~1 in-flight request). With the DeepSeek default of 8 parallel
workers, every GLM request comes back as:

```
HTTP 429 rate_limit_exceeded:
"concurrency limit reached for requests: zai-org/GLM-5.2-project limit reached"
```

The workers requeue and cool down repeatedly, so the run livelocks and makes no
progress — this is why GLM "didn't work" while DeepSeek did. The GLM runners
therefore default to `-Workers 1` (DeepSeek stays at 8). At 1 worker the
occasional transient 429 is absorbed by the existing requeue/cooldown path and
each file completes normally. Raise `-Workers` only if W&B grants GLM more
concurrency.

## Commands

```powershell
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1 -Action Status
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1 -Action Pause
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1 -Action Resume
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1 -Action RetryEmpty -MaxFiles 20
```

```bash
./LLM-aggregator/TPPO/GLM/run-tppo-glm.sh --status
./LLM-aggregator/TPPO/GLM/run-tppo-glm.sh --retry-empty --max-files 20
```

Before a non-dry-run extraction starts, the runner sends a tiny preflight
request to `zai-org/GLM-5.2`. If W&B returns a model/project concurrency error,
the command exits before starting corpus workers or writing failed extraction
state. To bypass that check deliberately:

```powershell
.\LLM-aggregator\TPPO\GLM\run-tppo-glm.ps1 -SkipPreflight
```

```bash
./LLM-aggregator/TPPO/GLM/run-tppo-glm.sh --skip-preflight
```

Direct CLI:

```powershell
uv run tppo-glm-aggregate --dry-run
uv run tppo-glm-aggregate --max-files 10
```

The default input is `downloads/TPPO/raw-text`, output is
`LLM-aggregator/TPPO/GLM/output`, state is
`LLM-aggregator/TPPO/GLM/progress.jsonl`, and pause file is
`LLM-aggregator/TPPO/GLM/pause`.
