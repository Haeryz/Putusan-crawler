# GLM 5.2 W&B Inference API

The GLM aggregator uses the same W&B OpenAI-compatible chat-completions
endpoint and API key file as the Anak DeepSeek aggregator.

```python
client.chat.completions.create(
    model="zai-org/GLM-5.2",
    messages=[...],
)
```

Local scripts read `api_key=...` from `LLM-aggregator/Anak/Deepseek/.env` unless
`WANDB_API_KEY` or `OPENAI_API_KEY` is already set.
