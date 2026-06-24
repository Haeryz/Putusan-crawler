# Putusan Anak GLM Aggregator

This is the GLM replication of the Anak DeepSeek aggregator. It uses the same
prompt construction, span validation, retry behavior, checkpointing, output
format, and launch controls as `LLM-aggregator/Anak/Deepseek`, but sends API
requests with:

```json
{"model": "zai-org/GLM-5.2"}
```

Outputs and state are kept separate under `LLM-aggregator/Anak/GLM`. The API
key is intentionally shared from `LLM-aggregator/Anak/Deepseek/.env`.

## Commands

```powershell
.\LLM-aggregator\Anak\GLM\run-anak-glm.ps1
.\LLM-aggregator\Anak\GLM\run-anak-glm.ps1 -Action Status
.\LLM-aggregator\Anak\GLM\run-anak-glm.ps1 -Action Pause
.\LLM-aggregator\Anak\GLM\run-anak-glm.ps1 -Action Resume
.\LLM-aggregator\Anak\GLM\run-anak-glm.ps1 -Action RetryEmpty -MaxFiles 20
```

```bash
./LLM-aggregator/Anak/GLM/run-anak-glm.sh --status
./LLM-aggregator/Anak/GLM/run-anak-glm.sh --retry-empty --max-files 20
```

Direct CLI:

```powershell
uv run anak-glm-aggregate --dry-run
uv run anak-glm-aggregate --max-files 10
```

The default input is `downloads/kasus anak/raw-text`, output is
`LLM-aggregator/Anak/GLM/output`, state is
`LLM-aggregator/Anak/GLM/progress.jsonl`, and pause file is
`LLM-aggregator/Anak/GLM/pause`.
