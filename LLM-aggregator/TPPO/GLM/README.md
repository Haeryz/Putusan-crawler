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

Direct CLI:

```powershell
uv run tppo-glm-aggregate --dry-run
uv run tppo-glm-aggregate --max-files 10
```

The default input is `downloads/TPPO/raw-text`, output is
`LLM-aggregator/TPPO/GLM/output`, state is
`LLM-aggregator/TPPO/GLM/progress.jsonl`, and pause file is
`LLM-aggregator/TPPO/GLM/pause`.
