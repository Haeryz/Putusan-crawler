---
language:
- id
license: cc-by-4.0
task_categories:
- text-generation
tags:
- legal
- indonesian
- court-decisions
- structured-extraction
- long-documents
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: sft/train.parquet
  - split: validation
    path: sft/val.parquet
  - split: test
    path: sft/test.parquet
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train.parquet
  - split: validation
    path: grpo/val.parquet
  - split: test
    path: grpo/test.parquet
- config_name: rag
  data_files:
  - split: train
    path: rag/train.parquet
  - split: validation
    path: rag/val.parquet
  - split: test
    path: rag/test.parquet
---

# Putusan windowed line-anchored extraction dataset (Plan B)

Built 2026-07-09T13:01:27+00:00 by `notebooks/build_windowed_dataset.py` from the
legacy `Haeryz/putusan-structured-extraction` dataset (same documents, same
leakage-safe purpose/split assignment, seed 3407).

Each legacy document row (~34K tokens median — longer than a 32K context)
is re-expressed as overlapping line-numbered windows of <= 6400
content tokens (measured with `Qwen/Qwen3.5-9B`; fits a
max_seq_length of 8192 with margin for other tokenizers). The supervision
target per window is a compact JSON of
GLOBAL inclusive line ranges per canonical section:

```json
{"sections": {"dakwaan": [[120, 187]]}, "sections_absent": ["ahli", "..."]}
```

The model copies line numbers it can see (never counts, never quotes
document text); a deterministic assembler slices verbatim text from the
ranges and merges overlapping ranges across windows, reproducing the legacy
31-section `target_json` format exactly (asserted at build time by a gold
round-trip for every document). Hallucination-free by construction.

## Schema (windowed rows)

| column | type | description |
|---|---|---|
| id | str | `legacy_id#wNNN` |
| doc_id | str | legacy row id (`corpus/annotator_model/sha16`) |
| corpus / annotator_model / source_file / source_sha256 | str | provenance |
| purpose / split / split_seed | str/int | identical to legacy dataset |
| window_index / n_windows | int | window position within document |
| line_start / line_end / n_doc_lines | int | global 1-based inclusive line span of this window |
| input_text | str | line-numbered window text (`NNNNN|content`) |
| target_json | str | window target: sections -> [[start, end], ...] + sections_absent |
| messages | list | system/user/assistant conversation (SFT) |
| prompt | list | system+user (GRPO rollout) |
| answer | str | = target_json (GRPO reference) |
| n_sections_present / n_input_chars / n_target_chars | int | statistics |

## Row counts (windows; documents in parentheses)

| purpose | train | val | test |
|---|---|---|---|
| sft | 13657 (2468) | 1606 (311) | 1661 (296) |
| grpo | 3061 (534) | 376 (68) | 334 (65) |
| rag | 2848 (529) | 319 (68) | 340 (68) |

Total: 24202 windows from 4407 document rows.

Reassembly reference implementation: `assemble_document` in
`notebooks/build_windowed_dataset.py`.
