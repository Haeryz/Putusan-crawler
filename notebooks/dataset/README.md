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

# Putusan structured-extraction dataset

Built 2026-07-08T23:02:28+00:00 by `notebooks/build_dataset.py` (seed 3407).

Indonesian court-decision (putusan) extractive-structuring dataset over three
corpora (Anak, Asusila, TPPO). Each row is one model extraction of one source
document into 31 canonical sections of verbatim spans. Empty sections were
completed from sibling model extractions of the same document where available
(`cross_model_fill_json` records per-section donor provenance).

## Files

`{sft,grpo,rag}/{train,val,test}.parquet` — purposes are document-disjoint
(SFT 70% / GRPO 15% / RAG 15% of unique documents per corpus), then each
purpose is split train 80% / val 10% / test 10%. All rows of a document
(one per annotator model) share the same purpose+split, so there is zero
document leakage across any pair of files.

## Schema (uniform across all files)

| column | type | description |
|---|---|---|
| id | str | `corpus/annotator_model/sha16` |
| corpus | str | Anak, Asusila, or TPPO |
| annotator_model | str | model that produced the extraction (GPT/Deepseek/Qwen) |
| source_file / source_sha256 | str | source decision text identity |
| extraction_method | str | extraction protocol tag |
| purpose / split | str | sft-grpo-rag / train-val-test |
| split_seed | int | RNG seed used for the deterministic split |
| input_text | str | decision body reconstructed from section spans |
| target_json | str | full 31-key supervision target (JSON) |
| sections_json | str | the 31 sections alone (JSON) |
| messages | list<struct{role,content}> | system/user/assistant conversation (SFT) |
| prompt | list<struct{role,content}> | system+user only (GRPO rollout prompt) |
| answer | str | reference answer for reward computation (GRPO) |
| n_nonempty_sections / n_empty_sections / empty_sections | int/list | section coverage |
| cross_model_fill_json | str | {section: donor_model} completion provenance |
| n_sections_filled_cross_model | int | sections completed from a sibling model |
| models_covering_doc | list<str> | all models that extracted this document |
| n_input_chars / n_input_words / n_target_chars | int | size statistics |

## Row counts

| purpose | train | val | test |
|---|---|---|---|
| sft | 2468 | 311 | 296 |
| grpo | 534 | 68 | 65 |
| rag | 529 | 68 | 68 |

Total rows: 4407 (expected 4500; shortfall is missing/failed
extractions — see `dataset_info.json` `per_dir_file_counts`).
