# Putusan TPPO Gemini Template

This directory is only an instruction template for a future Gemini extractor.
It intentionally contains no API caller, launcher, state file writer, retry
logic, or output implementation yet.

Use the same extraction contract as the DeepSeek span pipeline:

- input corpus: `downloads/TPPO/raw-text`
- span spec: `LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md`
- extraction reference: sanitized field guidance from
  `LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md`
- schema guide: `LLM-aggregator/Putusan-schema.md`
- intended output directory, when implemented later:
  `LLM-aggregator/TPPO/Gemini/output`

Gemini should return only the compact span JSON:

```json
{"sections": {
  "judul": {"text": ["P U T U S A N"]},
  "nomor_putusan": {"text": ["Nomor 1008/Pid.Sus/2025/PN Mdn"]},
  "tuntutan": {"lines": [[45, 60]]},
  "penangkapan": {"empty": true}
}}
```

The runner that gets implemented later should expand line spans into exact
source excerpts and validate them the same way as the current DeepSeek code.
Do not ask Gemini to write final per-file JSON, progress records, reports, or
filesystem changes.

Prompt template: `PROMPT_TEMPLATE.md`.
