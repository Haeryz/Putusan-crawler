# IMPLEMENTATION DIRECTIVE — READ THIS FIRST (Final, 2026-07-11)

This section is the **single authoritative contract** for implementing Plan D.
Everything below it — the original Plan D, Revision 1 (flaw audit), and
Revision 2 (fixes) — is the historical record and design rationale. An
implementing agent follows **only this directive**; where any text below
disagrees with it, this directive wins. All citations referenced here were
venue-verified in Revisions 1 and 2 (NeurIPS 2023/2024, ICLR 2024, ICML 2025,
ACL 2024/2025/2026, EMNLP 2024/2025); do not re-litigate them.

## D0. Absolute prohibitions

These are non-negotiable. Any statement elsewhere in this file that appears
to permit one of these is void.

1. **NO smoke tests. NO dry runs. NO rehearsal runs. NO ten-step trial runs.
   NO tiny training subsets. NO smoke/debug flags. NO blocking GPU asserts.**
   There is exactly one GPU run: the real training run. Data-builder logic is
   validated **locally on CPU** before the GPU session; correctness is
   enforced by the build gates in D4, which run as part of the real pipeline.
   Training throughput (tokens per second) is read from the **first logged
   steps of the real run itself** — those steps count toward training and are
   never a separate rehearsal.
2. No truncation reliance anywhere: every built example must fit the ceiling
   before training starts, or the build fails.
3. No gold-label-constructed inputs in the main method: `sections_json` is
   used for **targets only**, never to select or assemble context text.
4. No line numbers, character offsets, anchors, or span IDs in model output.
5. No plaintext credentials in the notebook. `WANDB_API_KEY` comes from Colab
   Secrets or an environment variable; the previously leaked key is treated
   as compromised and must be rotated before the run.
6. `notebooks/dataset/windowed_dataset`, GRPO, RAG serving, and Plan A/B/C
   mechanisms are out of scope.

## D1. Fixed configuration

- Model: `Qwen/Qwen3.5-9B` (a 9.65B multimodal `qwen3_5` checkpoint; text
  tokenizer is accessed through the processor). Loading: 4-bit QLoRA,
  bfloat16 compute, `max_seq_length` 32768, Unsloth gradient checkpointing,
  no full finetuning.
- LoRA: r 32, alpha 32, dropout 0, bias none, seed 3407; language, attention,
  and MLP layers on; vision layers off; **no** explicit q/k/v/o target-module
  list (the hybrid architecture has differently named linear-attention
  projections). The adapter audit prints resolved parent module names and
  asserts coverage of full attention, linear attention, and MLP.
- Trainer: micro-batch 1, gradient accumulation 16, one epoch (budgeted per
  D5), learning rate 2e-4, warmup ratio 0.05, adamw_8bit, weight decay 0.001,
  linear schedule, max grad norm 1.0, `max_length` 32768, packing off,
  response-only loss with the ChatML user/assistant markers, logging every 10
  steps, eval and save every 100 steps, save limit 2, eval micro-batch 1,
  load best model at end.
- Token ceilings: build ceiling 32,256; trainer limit 32,768; single unified
  safety margin 512. The 256-token allowance in original Section 6 is void.
- Splits: transform train, validation, **and test** partitions identically.
  Training attaches a two-way evaluation dictionary (validation and test);
  model selection reads validation only; the transformed test split drives
  post-training inference. No new random splits; zero cross-split leakage.

## D2. Conversation contract

- Single-turn conversations. System prompt: the Indonesian extractor prompt
  from original Section 5.1, amended to state that the provided context may
  be a bounded excerpt of a longer judgment and that requested sections must
  be extracted **as they appear in this excerpt**, verbatim, with absent
  sections going to an empty list plus `empty_sections`.
- User turn ordering: **judgment context first, requested-fields line last**,
  so the long document prefix is identical across all 13 group requests for
  one document and is reusable by RadixAttention-style prefix caching at
  inference (Zheng et al., NeurIPS 2024). Training and serving templates are
  identical.
- Assistant target: one JSON object with `sections` (exactly the requested
  keys, canonical order, values are lists of verbatim strings) and
  `empty_sections`. No provenance fields, no markdown, no explanations.
- The 13 semantic field groups are exactly the table in original Section 5.2;
  together they cover all 31 canonical fields exactly once.

## D3. Data build pipeline (in this order)

1. **Verbatim audit and target relocation.** For every non-empty
   `sections_json` value: if it is an exact substring of `input_text`, keep
   it; otherwise locate it via symmetric whitespace-run canonicalization and
   **re-extract the target from `input_text` at the located span** (targets
   are document-verbatim by construction); if it still cannot be located,
   exclude it and count it. Report the audit and exclusion rates per corpus
   and per field. Also report section coverage (fraction of `input_text`
   characters inside annotated blocks) as an analysis statistic.
2. **Tier A — full input.** For each row × each of the 13 groups, format the
   full conversation with the complete `input_text` and the partial target,
   tokenize with the actual chat template, and accept if at most 32,256
   tokens. `context_mode` full.
3. **Tier W — raw contiguous windows** (the only overflow mechanism). Slice
   the raw, unmodified `input_text` into contiguous windows sized to the
   available context budget with roughly one-quarter overlap between
   neighbours. One conversation per window per group: targets are the
   requested values lying fully inside the window; a value that fits in no
   single window contributes its maximal document-verbatim portion per
   window; requested fields absent from a window go to `[]` and
   `empty_sections`. `context_mode` window, with `window_index` and
   `window_count` as metadata only. There is no Tier B gold aggregation and
   no Tier D packet splitting in the main build (gold aggregation exists only
   as the optional D2 ablation, not part of the first implementation).
4. **Schema.** Original Section 8 columns, replacing the continuation columns
   with `window_index`/`window_count` and allowing `context_mode` values
   `full` and `window`.

## D4. Build gates (fail, never warn — these run inside the real pipeline)

- The 13 groups cover all 31 keys exactly once; every assistant `sections`
  object equals `requested_sections` in canonical order; `empty_sections` is
  consistent; no provenance keys in targets.
- Every non-empty target string is an exact substring of its `context_text`.
- Every row's full templated sequence is at most 32,256 tokens and ends with
  the complete assistant answer and end marker.
- Response-only masking leaves at least one supervised token in every row;
  the trainer's response-masking step must drop **zero** rows.
- Deterministic rebuild: same inputs and seed 3407 give byte-identical rows,
  ordering, and token counts.
- Provenance (`source_sha256`, corpus, annotator) preserved; no row from
  outside the intended partition.

## D5. Compute budget and run protocol

1. After the build, sum `n_total_tokens` over the training set and record it.
2. Fix the epoch token budget in advance from the available session
   wall-clock (anchor from Revision 1: roughly 590–710M tokens for the
   exhaustive set is 33–130+ hours on one A100-40GB and is **not** feasible;
   a budget near 150M tokens is roughly 14–28 hours at 1,500–3,000 tokens
   per second).
3. Meet the budget by stratified per-epoch sampling: keep **all** Tier W
   window examples; cap full-input conversations at G groups per document per
   epoch, with deterministic seed-3407 rotation so all 13 groups are covered
   corpus-wide (LIMA, NeurIPS 2023: curated coverage beats exhaustive
   duplication). Choose G so the summed tokens fit the budget.
4. Run protocol, one real run: verify the GPU and usable memory; load the
   model and adapters; run the adapter-coverage audit; load the built data;
   assert the D4 gates passed; start training. The first logged steps supply
   the measured tokens-per-second for the wall-clock projection; if the
   projection exceeds the session, continue to the checkpoint boundary and
   resume in a later session from the last checkpoint with sampler state
   restored deterministically. If the longest row OOMs, lower the build
   ceiling in 1,024-token decrements, rebuild affected rows, and restart —
   never truncate.

## D6. Evaluation (after training, same session or a resumed one)

- Headline metrics are computed on the **merged raw-window condition**:
  overflow documents are evaluated by presenting their raw windows,
  collecting per-window outputs, and merging per field by document-order
  concatenation with overlap deduplication (a deterministic string
  operation). Metrics: the original Section 12.1 list, with
  "reconstructed-field F1" replaced by **merged-field F1**.
- Breakdowns: corpus (Anak/Asusila/TPPO), group, field, context mode, token
  bucket, requested-field count, empty versus non-empty, annotator model;
  source-level aggregation weights each `source_sha256` equally.
- Baselines when resources allow, in priority order: B1 (partial targets,
  full-input only, overflow discarded) versus D1 (main method); D2 (gold-
  aggregated contexts) is an oracle upper bound only; D3 (windows only) is
  optional. B0 is documented by the already-saved failed run and need not be
  re-run.
- Acceptance thresholds from original Section 13 stand as **targets with a
  decision rule**: report per-field, ship fields that pass, and iterate
  window size/overlap for fields that miss. They are not promises.

## D7. Deliverable

A training notebook (or notebook-generating source script, following the
repo's existing build convention) that executes D3 → D4 → D5 → D6 top to
bottom as one real run, plus the built dataset artifact with its audit
report. Commit to master and push when done. Nothing in the deliverable may
implement anything listed in D0.

---
---

# Plan D — 32K Partial-Field Conversational SFT by Semantic Data Aggregation

*(Historical record below: original plan, then Revision 1 — flaw audit, then
Revision 2 — citation-backed fixes. Superseded where the directive above
differs.)*

## Executive Summary

Plan D keeps `Qwen/Qwen3.5-9B` as a normal generative language model and keeps
`max_seq_length = 32768`. It does not extend the model context, replace the LM
head, introduce a CRF, or make the model predict line numbers or character
offsets. The model is fine-tuned to generate the actual verbatim legal text.

The only training source is:

```text
notebooks/dataset/sft/train.parquet
```

`notebooks/dataset/windowed_dataset` is explicitly out of scope. The GRPO and
RAG partitions are not mixed into SFT.

The current training unit is the problem: one row contains the complete
`input_text` followed by an assistant answer containing all 31 sections. Because
the answer repeats most of the legal text, the complete causal-LM sequence is
too long. Plan D changes the training unit to an independent conversation that
requests only one semantic group of approximately two or three fields. The
assistant returns partial JSON containing only those requested fields.

For each document/field-group pair, Plan D first tries to retain the complete
`input_text`. If the fully formatted conversation does not fit, it constructs a
field-conditioned semantic aggregate containing every positive span plus
same-document distractors. Only the residual oversized fields are expressed as
semantic continuation examples. No example is allowed to rely on tokenizer or
trainer truncation.

The resulting objective is:

```text
requested 1–3 fields + judgment context
    -> partial JSON containing verbatim text for those fields
```

This is real generative text-extraction fine-tuning. The model emits the text,
not references to the text.

---

## 1. Scope and Non-Goals

### In scope

- Reshape `notebooks/dataset/sft/train.parquet` for 32K SFT.
- Keep the existing Qwen3.5-9B language-model architecture.
- Preserve conversational `system` / `user` / `assistant` training.
- Ask for approximately two or three canonical fields per conversation.
- Generate the corresponding legal text verbatim in partial JSON.
- Retain the complete document input whenever the complete conversation fits.
- Use semantic, field-conditioned aggregation only for over-budget examples.
- Split irreducibly long field values into semantic continuation examples.
- Use response-only causal language-model loss.
- Fit training on one A100-SXM4-40GB.

### Out of scope

- `notebooks/dataset/windowed_dataset`.
- Plan B line-number targets and fixed line windows.
- Plan A model-generated character offsets or anchors.
- Plan C custom encoders, BiGRUs, boundary heads, semi-CRFs, Viterbi decoding,
  representation-gradient caching, or removal of the generative LM head.
- Changing Qwen positional encoding, attention implementation, or underlying
  context architecture.
- RAG implementation, PDF ingestion, serving, and inference orchestration.
- GRPO.
- Generating all 31 fields in one assistant response.

---

## 2. Correct Diagnosis of the Current Training Failure

### 2.1 The 32K limit covers the complete training sequence

For causal-LM SFT, `max_length = 32768` does not mean 32,768 source tokens plus
an unlimited answer. It covers everything:

```text
system prompt
+ user request
+ input context
+ assistant target
+ chat-template and special tokens
<= 32,768 tokens
```

The Plan D builder therefore uses a stricter build-time ceiling:

```text
BUILD_TOKEN_LIMIT = 32,256
TRAINER_MAX_LENGTH = 32,768
SAFETY_MARGIN = 512
```

The safety margin covers tokenizer/template boundary effects and prevents a
dependency upgrade from silently pushing a borderline example over the trainer
limit.

### 2.2 Existing fully formatted sequence lengths

The saved notebook audit measured the complete current conversation—system,
whole input, and full 31-field answer—with Qwen's inner text tokenizer:

| Percentile | Complete sequence tokens |
|---|---:|
| p50 | 34,499 |
| p90 | 74,328 |
| p95 | 90,106 |
| maximum | 448,111 |

The median already exceeds 32K. Raising a percentile heuristic cannot fix this
while `max_seq_length` remains 32K.

### 2.3 The exact corruption mode

The previous statement that every over-32K example becomes entirely `-100` is
too broad. The saved run shows two failure modes:

1. **Response marker lost completely.** Unsloth removed 342 of 2,468 training
   rows and 26 of 311 validation rows because truncation removed the assistant
   marker and left no supervised response tokens.
2. **Response retained only partially.** Other long rows keep the beginning of
   the assistant response but lose its tail. They train the model on incomplete,
   frequently invalid JSON and systematically remove later canonical fields.

Both are unacceptable. Plan D requires every formatted training example and
its complete assistant answer to fit before training begins.

### 2.4 The saved run also failed on memory

The notebook was written as if an A100 80GB were available, but the saved run
used:

```text
NVIDIA A100-SXM4-40GB
39.494 GB total
```

It loaded 16-bit LoRA, used a micro-batch of two, reserved 28.227 GB before
training, and failed on the first backward pass while trying to allocate another
1.46 GB. Plan D therefore uses 4-bit QLoRA and a micro-batch of one. This is a
precision/configuration change, not a model-architecture change. QLoRA is the
established memory-efficient fine-tuning basis for this choice
([Dettmers et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)).

---

## 3. Corpus Token Audit

The following values were measured directly from the 2,468 rows in
`notebooks/dataset/sft/train.parquet`, using the cached Qwen3.5 inner tokenizer
and tokenizing only `input_text`—before adding the system prompt or answer.

| Corpus | Rows | Mean | Median | p90 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Anak | 835 | 17,329 | 15,702 | 28,493 | 34,594 | 68,557 |
| Asusila | 815 | 15,029 | 13,112 | 23,338 | 30,111 | 99,587 |
| **TPPO** | **818** | **30,282** | **24,513** | **51,008** | **71,993** | **222,778** |

TPPO is highest by total, mean, median, p90, p95, p99, and maximum. Its largest
training input is:

```text
555_Pid.Sus_2023_PN_Stb.txt — 222,778 input tokens
```

Consequences:

- Partial answers alone are sufficient for many Anak and Asusila rows.
- TPPO requires the semantic aggregation fallback much more often.
- Even a tiny header target cannot make a 222K input fit; input-side selection
  is unavoidable for the TPPO tail.
- A single global character or word cutoff would be unreliable. Every decision
  must be based on the actual Qwen chat-template token count.

---

## 4. Research Basis

Plan D combines recent findings without copying any paper's architecture.

### 4.1 Legal knowledge-aware selective compression

**Less is More: Knowledge-Aware Compression for Long Legal Judgment Prediction**
selects sentence-level semantic units using legal relevance and an adaptive
threshold rather than retaining a fixed fraction of every legal document. It
reports improved accuracy and computational efficiency across legal datasets and
jurisdictions. Plan D adopts the data-level principle: preserve task-relevant
legal units and remove redundant material only when the full sequence cannot
fit. It does not adopt the paper's prediction task or model components.

- Lou et al., Findings of ACL 2026:
  [paper](https://aclanthology.org/2026.findings-acl.1450/)

### 4.2 Arbitrary-length evidence copying

**Unstructured Evidence Attribution for Long Context Query Focused
Summarization** shows that language models struggle to copy arbitrary-length
evidence spans without adaptation, and that explicit fine-tuning improves exact
copy behavior and evidence quality. It also uses modular document sections and
position variation to reduce lost-in-the-middle behavior. Plan D applies this to
legal extraction: the assistant is explicitly trained to copy variable-length
verbatim spans from supplied context.

- Wright et al., EMNLP 2025:
  [paper](https://aclanthology.org/2025.emnlp-main.95/)

### 4.3 Pinpoint retrieval in long legal judgments

**LexCLiPR** studies paragraph-level retrieval from multilingual legal judgments
instead of retrieving complete cases. It finds that targeted post-training is
important, especially for low-resource languages and unseen legal concepts.
Plan D is training-only and does not implement retrieval, but its semantic
evidence packages use the same pinpoint-information principle.

- Upadhya and T.y.s.s, ACL 2025:
  [paper](https://aclanthology.org/2025.acl-long.683/)

### 4.4 Realistic context and diverse distractors

**Understanding Synthetic Context Extension via Retrieval Heads** finds that
training on real data is stronger than purely synthetic long-context data and
that downstream performance correlates with learned retrieval-head recall.
Plan D therefore uses real spans and real same-document distractors from the
Parquet row rather than invented filler or unrelated text.

- Zhao, Yin, and Durrett, ICML 2025:
  [paper](https://proceedings.mlr.press/v267/zhao25ad.html)

### 4.5 Position-agnostic decomposition

**Never Lost in the Middle** improves long-context QA by decomposing tasks and
varying the location of relevant information. Plan D decomposes one 31-field
task into semantic partial-field requests and includes validation by evidence
position.

- He et al., ACL 2024:
  [paper](https://aclanthology.org/2024.acl-long.736/)

### 4.6 Long-tail sequence handling

**ChunkFlow** observes that long-context datasets contain many short sequences
and a small long tail, and improves efficiency by consolidating short examples
and splitting long examples. Plan D uses the data-design implication—treat the
tail separately instead of configuring every sample for the longest row—but
does not implement ChunkFlow's distributed scheduling system.

- Yuan et al., ICML 2025:
  [paper](https://proceedings.mlr.press/v267/yuan25m.html)

**LongAlign** additionally supports length-aware batching and careful handling
of varied-length instruction data. Plan D keeps one conversation per sequence
for its first implementation because response-only masking must remain simple
and auditable; packing is reserved for a later ablation.

- Bai et al., Findings of EMNLP 2024:
  [paper](https://aclanthology.org/2024.findings-emnlp.74/)

### 4.7 Legal evaluation must be capability-specific

LegalBench and LexEval demonstrate that legal-model performance varies by task
and capability. Plan D therefore reports each canonical field separately rather
than relying only on one aggregate score.

- Guha et al., NeurIPS 2023 Datasets and Benchmarks:
  [LegalBench](https://proceedings.neurips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)
- Li et al., NeurIPS 2024 Datasets and Benchmarks:
  [LexEval](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2cb40fc022ca7bdc1a9a78b793661284-Abstract-Datasets_and_Benchmarks_Track.html)

---

## 5. Plan D Training Task

### 5.1 Conversation contract

Each derived row is an independent single-turn conversation. It is not a flat
instruction triple and does not ask for all 31 fields.

Recommended system prompt:

```text
Anda adalah pengekstrak teks putusan pengadilan Indonesia. Berdasarkan hanya
konteks putusan yang diberikan, ekstrak hanya bagian yang diminta. Setiap nilai
harus berupa daftar kutipan verbatim yang disalin persis dari konteks; jangan
memparafrasekan, meringkas, menambah, atau mengarang. Jika bagian yang diminta
tidak terdapat dalam konteks, gunakan daftar kosong dan cantumkan namanya pada
empty_sections. Keluarkan hanya satu objek JSON tanpa markdown atau penjelasan.
```

Recommended user template:

```text
Bagian yang diminta: ["tuntutan", "dakwaan"]

Konteks putusan:
<context_text>
```

Recommended assistant target:

```json
{
  "sections": {
    "tuntutan": ["kutipan verbatim dari konteks"],
    "dakwaan": ["kutipan verbatim dari konteks"]
  },
  "empty_sections": []
}
```

Rules:

- `sections` contains exactly the requested keys, in their canonical order.
- No unrequested key appears.
- A requested but absent key maps to `[]` and is listed in `empty_sections`.
- `status`, `source_file`, and `source_sha256` stay in dataset metadata; the
  model does not waste tokens regenerating them.
- Every non-empty target string must be an exact substring of `context_text`.
- The assistant emits the legal text itself. It never emits line numbers,
  character offsets, page indexes, anchors, or span IDs.

### 5.2 The 13 semantic field groups

| Group | Requested fields |
|---|---|
| `header` | `judul`, `nomor_putusan`, `irah_irah` |
| `court_case` | `nama_pengadilan_negeri`, `keterangan_perkara` |
| `identity_a` | `nama_lengkap`, `tempat_lahir`, `umur_tanggal_lahir` |
| `identity_b` | `jenis_kelamin`, `kebangsaan`, `tempat_tinggal` |
| `identity_c` | `agama`, `pekerjaan` |
| `custody` | `penangkapan`, `penahanan` |
| `prosecution` | `tuntutan`, `dakwaan` |
| `testimony` | `saksi`, `ahli` |
| `evidence` | `terdakwa`, `surat`, `petunjuk_barang_bukti` |
| `analysis` | `fakta_hukum`, `pertimbangan_hukum` |
| `ruling` | `amar_putusan` |
| `date` | `hari`, `tanggal`, `tahun` |
| `bench` | `siapa_yang_memutus`, `panitera_pengganti`, `tanda_tangan_majelis` |

Long semantic sections are intentionally not combined into one giant narrative
request. A group may be split into individual fields for a specific document if
the grouped positives and grouped answer do not fit.

---

## 6. Measured Feasibility of Partial Targets

Every original training row was paired with all 13 groups, yielding 32,084
candidate conversations. The assistant target was reduced to the requested
group, while `input_text` remained complete. Token estimates included a
conservative 256-token prompt/template allowance.

| Group | Complete input fits | Overflow pairs | Full-sequence p50 | p90 | p95 |
|---|---:|---:|---:|---:|---:|
| header | 85.45% | 359 | 17,232 | 37,067 | 45,002 |
| court_case | 85.33% | 362 | 17,239 | 37,139 | 45,001 |
| identity_a | 85.37% | 361 | 17,222 | 37,062 | 45,001 |
| identity_b | 85.41% | 360 | 17,222 | 37,073 | 45,004 |
| identity_c | 85.49% | 358 | 17,191 | 37,028 | 44,956 |
| custody | 84.89% | 373 | 17,525 | 37,673 | 45,992 |
| prosecution | 77.59% | 553 | 20,759 | 45,247 | 55,553 |
| testimony | 74.68% | 625 | 21,218 | 48,378 | 61,090 |
| evidence | 81.16% | 465 | 19,273 | 41,399 | 50,558 |
| analysis | 72.93% | 668 | 23,128 | 47,050 | 57,083 |
| ruling | 84.81% | 375 | 17,582 | 37,859 | 46,202 |
| date | 85.53% | 357 | 17,192 | 37,024 | 44,966 |
| bench | 85.17% | 366 | 17,370 | 37,263 | 45,166 |

Overall:

```text
26,502 / 32,084 document/group pairs fit with complete input_text: 82.60%
 5,582 / 32,084 require semantic input aggregation:              17.40%
```

By corpus:

| Corpus | Complete-input pair fit rate |
|---|---:|
| Anak | 90.23% |
| Asusila | 94.70% |
| TPPO | 62.76% |

These measurements justify a tiered policy. Partial output is sufficient for
most candidates, so input compression should not be applied indiscriminately.
The fallback is concentrated in TPPO and in `analysis`, `testimony`, and
`prosecution`.

---

## 7. Deterministic Data-Reshaping Algorithm

### 7.1 Inputs

For each row of `notebooks/dataset/sft/train.parquet`, use:

- `id`, `corpus`, `annotator_model`, `source_file`, `source_sha256`;
- `input_text` as the authoritative source context;
- `sections_json` as the 31-field supervision source;
- the existing purpose and split metadata.

Do not use the row's current full `messages`, `target_json`, or `answer` as the
new target because they contain all 31 fields.

### 7.2 Build all base candidates

For every source row and each of the 13 groups:

1. Select only that group's values from `sections_json`.
2. Construct the partial assistant object.
3. Construct the system and user turns.
4. Initially use the complete `input_text` as `context_text`.
5. Apply Qwen's actual inner tokenizer chat template with
   `add_generation_prompt = false`.
6. Record the exact complete sequence length.

This produces 32,084 base candidates before rare continuation expansion. No
canonical field is discarded.

### 7.3 Tier A — complete input

If the fully formatted sequence is at most 32,256 tokens:

- retain the complete `input_text` without alteration;
- set `context_mode = "full"`;
- set `continuation_index = 0` and `continuation_count = 1`.

This is the default and applies to 82.60% of measured candidates.

### 7.4 Tier B — field-conditioned semantic aggregation

If the complete input overflows:

1. Reconstruct the 31 canonical source blocks from `sections_json`, using the
   same newline and blank-line separators as `input_text`.
2. Mark every requested non-empty block as mandatory positive evidence.
3. Compute the available context budget from the already-tokenized system,
   request, assistant target, template, and 512-token safety margin.
4. Add all mandatory positive blocks.
5. Rank non-requested blocks as distractors:
   - immediately preceding and following canonical sections first;
   - other sections in the same coarse legal family second;
   - remaining non-empty sections in deterministic seeded order third.
6. Greedily add complete distractor blocks while the exact formatted sequence
   remains within 32,256 tokens.
7. Sort selected blocks back into their original canonical/source order before
   rendering the context.
8. Do not print section names around the blocks; labels in the input would leak
   the answer.
9. Set `context_mode = "semantic_aggregate"`.

The coarse families used only for distractor ranking are:

| Family | Sections |
|---|---|
| header | `judul` through `keterangan_perkara` |
| identity | `nama_lengkap` through `pekerjaan` |
| procedure | `penangkapan`, `penahanan`, `tuntutan`, `dakwaan` |
| evidence | `saksi` through `petunjuk_barang_bukti` |
| decision | `fakta_hukum`, `pertimbangan_hukum`, `amar_putusan` |
| closing | `hari` through `tanda_tangan_majelis` |

Using real same-document distractors ensures the model learns selection and
boundaries rather than simply copying the entire supplied context.

### 7.5 Tier C — split the requested group

If all requested positive blocks plus the grouped target still exceed 32,256:

- split the two- or three-field group into one field per conversation;
- rerun Tier A and Tier B for each field;
- retain the original group name in metadata and add `subgroup_key`.

This handles combinations such as `saksi + ahli` where either field may be long
even though the grouping is semantically correct for most documents.

### 7.6 Tier D — semantic continuation for one oversized field

If one field's positive text plus its response still exceeds the build limit:

1. Pack the field's existing list items greedily without modifying their text.
2. If one existing list item is individually too long, split it at blank-line
   paragraph boundaries.
3. If a paragraph is individually too long, split at preserved newline
   boundaries.
4. Only if a single line remains too long, split at Indonesian sentence
   boundaries while preserving the exact substrings and delimiters needed for
   reconstruction.
5. Never split by a predicted line number and never expose offsets to the model.
6. Create one conversation per semantic packet.
7. Store `continuation_id`, `continuation_index`, `continuation_count`, original
   item index, and source ordering only as dataset metadata.
8. Assert that concatenating packets with their recorded original separators
   reconstructs the original annotated value exactly.

The assistant target still contains legal text:

```json
{
  "sections": {
    "saksi": ["verbatim continuation packet"]
  },
  "empty_sections": []
}
```

### 7.7 Empty-field supervision

When a requested field is empty:

- keep the requested key with `[]`;
- include it in `empty_sections`;
- supply full context when it fits;
- otherwise supply same-document distractors selected by the Tier B policy;
- never insert invented positive text.

This is particularly important for `ahli` and `penangkapan`, which have genuine
empty cases, and teaches abstention within the normal generative objective.

---

## 8. Derived Dataset Schema

The transformed SFT rows retain original provenance and add the following
training-specific columns:

| Column | Type | Meaning |
|---|---|---|
| `parent_id` | string | original Parquet row ID |
| `request_group` | string | one of the 13 semantic groups |
| `requested_sections` | list[string] | exact assistant key set |
| `context_mode` | string | `full` or `semantic_aggregate` |
| `context_text` | string | complete or aggregated source context |
| `target_json` | string | partial requested-field answer |
| `messages` | chat list | system, user, assistant turns |
| `continuation_id` | nullable string | groups packets from one oversized value |
| `continuation_index/count` | integer | deterministic packet position |
| `n_context_tokens` | integer | context token count |
| `n_response_tokens` | integer | assistant target token count |
| `n_total_tokens` | integer | complete templated sequence count |
| `context_sections` | list[string] | internal audit of selected source blocks |
| `distractor_sections` | list[string] | internal audit of selected negatives |

`context_sections` and `distractor_sections` are metadata only and must never be
rendered into the user context.

The transformed artifact should preserve the original train split rather than
creating a new random split. Validation uses the same transformation on
`sft/val.parquet`; validation rows are never selected as training examples.

---

## 9. Build-Time Correctness Gates

The data build must fail—not warn—if any invariant is violated.

### Schema gates

- The 13 base groups cover all 31 canonical keys exactly once.
- Every assistant `sections` object equals `requested_sections` exactly.
- Keys remain in canonical order.
- `empty_sections` equals the requested keys whose lists are empty.
- No provenance field is generated by the assistant.

### Grounding gates

- Every non-empty assistant string is an exact substring of `context_text`.
- No target is paraphrased, normalized, or synthesized.
- Every semantic continuation packet is an exact substring of its original item.
- Continuation round-trip reconstruction passes exactly for every split item.

### Token gates

- Tokenization uses `text_tokenizer = getattr(processor, "tokenizer", processor)`.
- `n_total_tokens <= 32256` for every row.
- The complete assistant target and final `<|im_end|>` marker are present.
- The collator produces at least one non-`-100` label for every row.
- Trainer truncation is disabled; no example depends on truncation behavior.

### Split and determinism gates

- No row outside the original SFT train partition enters training.
- `source_sha256`, corpus, and annotator provenance are preserved.
- All seeded distractor ordering is reproducible with seed 3407.
- Rebuilding with the same inputs produces identical row IDs, messages, token
  counts, and Parquet content ordering.

---

## 10. Fine-Tuning Configuration

### 10.1 Model loading

```text
model: Qwen/Qwen3.5-9B
max_seq_length: 32768
load_in_4bit: true
compute dtype: bfloat16
full_finetuning: false
gradient checkpointing: unsloth
```

The notebook filename still contains `(4B)`, but the actual and intended model
is Qwen3.5-9B.

### 10.2 LoRA

```text
r: 32
alpha: 32
dropout: 0
bias: none
random_state: 3407
```

Keep:

```text
finetune_vision_layers = false
finetune_language_layers = true
finetune_attention_modules = true
finetune_mlp_modules = true
```

Do not restore an explicit `q_proj/k_proj/v_proj/o_proj` target list. Qwen3.5 is
a hybrid model whose linear-attention layers use different projection names.
The adapter audit must print resolved parent module names and assert coverage of
full attention, linear attention, and MLP modules. The current
`Counter({'lora_A': 128})` output does not verify that coverage.

### 10.3 Trainer defaults

```text
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
effective example batch: 16
num_train_epochs: 1 for the first full run
learning_rate: 2e-4
warmup_ratio: 0.05
optimizer: adamw_8bit
weight_decay: 0.001
lr_scheduler_type: linear
max_grad_norm: 1.0
seed: 3407
max_length: 32768
packing: false
response-only loss: true
```

One epoch is the initial default because the source dataset expands from 2,468
rows to 32,084 base group conversations plus continuations. A second epoch is
permitted only after validation shows continuing improvement without a rise in
verbatim-error or extra-key rate.

`packing = false` is deliberate for the first implementation:

- it keeps each conversation and its response mask independently auditable;
- it prevents one packed conversation from attending to unrelated previous
  examples;
- it makes the no-truncation invariant simple to verify.

Length-aware ordering may group similar-length rows for throughput, but it must
not change row weights or mix labels.

### 10.4 Evaluation and saving cadence

The current `eval_steps = 1` is unnecessarily expensive. Recommended defaults:

```text
logging_steps: 10
eval_steps: 100
save_steps: 100
save_total_limit: 2
per_device_eval_batch_size: 1
load_best_model_at_end: true
```

Run a 10-step smoke test first, followed by the one-epoch full run.

### 10.5 Response-only masking

Continue using ChatML markers:

```text
instruction_part = "<|im_start|>user\n"
response_part = "<|im_start|>assistant\n"
```

After masking, explicitly compute supervised-token counts. The run must abort if
any row has zero supervised labels. Unsloth must remove zero rows.

---

## 11. Training Preflight

Before the full run:

1. Verify the GPU is `NVIDIA A100-SXM4-40GB` and report usable memory.
2. Load 4-bit Qwen3.5-9B and attach LoRA.
3. Verify adapter coverage by actual module names.
4. Load the transformed train and validation data.
5. Confirm maximum `n_total_tokens <= 32256`.
6. Re-tokenize a random sample and the longest example inside the notebook to
   detect tokenizer-version drift.
7. Apply response-only masking and confirm minimum supervised-token count > 0.
8. Run forward and backward on the longest example with micro-batch one.
9. Require peak reserved GPU memory below 39.494 GB.
10. Run 10 optimizer steps and require finite loss and gradients.
11. Only then begin the full epoch.

If the longest valid row still OOMs under 4-bit QLoRA and micro-batch one, the
response is not to truncate it. Lower the build-time total ceiling in increments
of 1,024 tokens, rebuild only affected aggregated examples, and repeat the
preflight while leaving `max_seq_length = 32768` unchanged.

---

## 12. Evaluation

Validation must use independently transformed `sft/val.parquet` rows. Model
selection never reads the test split.

### 12.1 Core metrics

- JSON validity rate.
- Exact requested-key-set accuracy.
- Extra-key rate.
- Missing-requested-key rate.
- Exact-substring/verbatim rate for generated strings.
- Character-level precision, recall, and F1 per canonical field.
- Exact field match rate.
- Empty-field precision, recall, and F1.
- Continuation packet character F1 and reconstructed-field F1.
- Macro average across all 31 fields.
- Micro average across all emitted characters.

### 12.2 Required breakdowns

Report metrics by:

- corpus: Anak, Asusila, TPPO;
- semantic group;
- individual field;
- `context_mode`: full versus semantic aggregate;
- total-token bucket: `<4K`, `4–8K`, `8–16K`, `16–24K`, `24–32K`;
- requested field count: one, two, or three;
- empty versus non-empty target;
- continuation versus ordinary example;
- annotator model.

Each `source_sha256` receives equal weight in source-level aggregate reporting so
multiple annotator rows do not silently dominate the headline metric.

### 12.3 Training-only baselines

| ID | Training format | Purpose |
|---|---|---|
| B0 | current whole input + full 31-field target | quantify truncation/OOM failure |
| B1 | partial-field target + complete input only; discard overflow | isolate benefit of target decomposition |
| D1 | Plan D full-input tier + semantic aggregation | main method |
| D2 | D1 without same-document distractors | measure whether negatives teach selection |
| D3 | D1 without full-input examples | measure value of retaining whole context where possible |

Plan B and Plan C are not baselines because they change the output mechanism or
model architecture rather than only the SFT data organization.

---

## 13. Acceptance Criteria

### Dataset acceptance

- 100% of derived rows satisfy `n_total_tokens <= 32256`.
- 100% preserve the complete assistant response.
- 100% have at least one supervised response token.
- 100% of non-empty targets are verbatim substrings of their contexts.
- 100% of continuation examples pass exact round-trip reconstruction.
- All 31 fields are represented in training and validation.
- Zero training examples are removed by `train_on_responses_only`.
- Zero document leakage is introduced.

### Runtime acceptance

- Longest-row forward/backward succeeds on the 40GB A100.
- Ten-step smoke training completes without OOM, NaN, or infinite loss.
- Peak reserved memory remains below the physical limit.
- The full run completes without trainer truncation warnings.

### Model acceptance

- JSON validity >= 99% on validation.
- Requested-key-set accuracy >= 99.5%.
- Extra-key rate <= 0.5%.
- Generated-string verbatim validity >= 99% before any external validation.
- Empty-field F1 and per-field character F1 are reported, not hidden by a
  single aggregate.
- Plan D beats B1 on source-weighted macro character F1, showing that semantic
  aggregation recovers overflow data rather than merely fitting shorter rows.
- TPPO results are reported separately and must not regress behind Anak/Asusila
  solely because more TPPO rows require aggregation.

---

## 14. Security and Reproducibility

The current notebook contains a plaintext W&B API key in a markdown cell. That
credential must be considered compromised and rotated before another run.

Requirements:

- Remove saved credentials and interactive login transcript from the notebook.
- Read `WANDB_API_KEY` from Colab Secrets or an environment variable.
- Clear all saved notebook outputs before committing.
- Pin model, tokenizer, Transformers, TRL, Unsloth, and dataset revisions.
- Log the Plan D configuration, canonical groups, build token limit, tokenizer
  revision, dataset fingerprint, and seed.
- Save the LoRA adapter and tokenizer, not credentials or cached datasets.

---

## 15. Risks and Mitigations

### Target text remains long for narrative fields

`saksi`, `dakwaan`, `ahli`, and `pertimbangan_hukum` can be too long even when
requested alone.

**Mitigation:** semantic continuation packets with exact reconstruction, never
target truncation.

### Semantic aggregation could make the task too easy

If context contains only the requested text, the model may learn to copy all
input rather than classify legal sections.

**Mitigation:** include real adjacent and same-family distractors and evaluate
the no-distractor ablation.

### Full-input examples may dominate compute

Many short-output conversations retain a long full document.

**Mitigation:** use one initial epoch, length-aware ordering, and report tokens
processed per group. If compute is excessive, select a balanced full-context
subset as a documented ablation rather than silently truncating rows.

### TPPO dominates the long tail

Only 62.76% of TPPO group candidates fit with full input, versus more than 90%
for Anak and Asusila.

**Mitigation:** stratify all validation metrics by corpus and ensure semantic
aggregation does not reduce TPPO recall.

### Generative copying can still make mistakes

Plan D intentionally trains an actual generative extractor, so verbatim output
is learned rather than structurally guaranteed by an offset assembler.

**Mitigation:** exact-substring supervision, response-only loss, explicit
verbatim metrics, and strict validation of generated strings during evaluation.

### Dataset annotation variants

Different annotator-model rows for the same `source_sha256` may contain different
yet self-consistent text selections.

**Mitigation:** preserve annotator provenance, keep source-level split safety,
and weight headline evaluation by unique source rather than raw row count.

---

## 16. Final Recommendation

Plan D should replace whole-document/31-field SFT with partial-field
conversational SFT while leaving Qwen's architecture untouched.

The decisive workflow is:

```text
train.parquet row
  -> choose one of 13 semantic field groups
  -> build partial verbatim JSON target
  -> try complete input_text
  -> if over 32,256: aggregate mandatory positives + real distractors
  -> if still over: split fields
  -> if still over: semantic continuation packets
  -> assert complete chat <= 32,256
  -> 4-bit QLoRA response-only SFT at max_seq_length 32768
```

This directly addresses the actual failure:

- the model no longer generates all 31 sections at once;
- all supervision survives tokenization;
- most examples retain the complete input;
- TPPO overflow rows remain usable through semantic aggregation;
- the long narrative tail remains usable through continuations;
- the trained system remains a normal chat-capable, generative Qwen LLM that
  outputs the actual legal text.

---
---

# Revision 1 — Independent Verification and Flaw Audit (2026-07-11)

Everything above this line is the original Plan D, preserved unchanged. This
revision records an independent verification pass: every citation was checked
against its publisher page (ACL Anthology, NeurIPS proceedings, PMLR), the
model repository was checked on the Hugging Face Hub, and the plan's own
numbers were re-derived. Verdict: **the citations are real, but the plan is
too good to be true in four specific ways.** The training-data reshaping is
sound; the claims about what the trained model can do, and how long training
takes, are not.

## R1. Citation verification — all pass

Each link was fetched and the title, authors, and venue were compared against
the publisher's page on 2026-07-11. No citation is fabricated.

| Plan D citation | Publisher page says | Verdict |
|---|---|---|
| Dettmers et al., QLoRA, NeurIPS 2023 | "QLoRA: Efficient Finetuning of Quantized LLMs", Dettmers, Pagnoni, Holtzman, Zettlemoyer, NeurIPS 2023 main track | Correct |
| Lou et al., Findings of ACL 2026 | "Less is More: Knowledge-Aware Compression for Long Legal Judgment Prediction", Fanghao Lou et al., Findings of ACL 2026 (San Diego, July 2026) | Correct |
| Wright et al., EMNLP 2025 | "Unstructured Evidence Attribution for Long Context Query Focused Summarization", Wright, Mujahid, Wang, Augenstein, Jurgens, EMNLP 2025 main (Suzhou, Nov 2025) | Correct |
| Upadhya and T.y.s.s, ACL 2025 | "LexCLiPR: Cross-Lingual Paragraph Retrieval from Legal Judgments", ACL 2025 Long Papers (Vienna) | Correct |
| He et al., ACL 2024 | "Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training", ACL 2024 Long Papers, pp. 13628–13642 | Correct |
| Zhao, Yin, Durrett, ICML 2025 | "Understanding Synthetic Context Extension via Retrieval Heads", PMLR v267, pp. 77885–77910 | Correct |
| Yuan et al., ICML 2025 | "Efficient Long Context Fine-tuning with Chunk Flow", Xiulong Yuan et al., PMLR v267, pp. 73732–73742 | Correct |
| Bai et al., Findings of EMNLP 2024 | "LongAlign: A Recipe for Long Context Alignment of Large Language Models", Findings of EMNLP 2024 (Miami) | Correct |
| Guha et al., NeurIPS 2023 D&B | "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models", Datasets and Benchmarks Track | Correct |
| Li et al., NeurIPS 2024 D&B | "LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models", Datasets and Benchmarks Track | Correct |

`Qwen/Qwen3.5-9B` also verifies on the Hugging Face Hub: 9,653M parameters,
`qwen3_5` architecture, Apache-2.0. One correction to the plan's framing: the
Hub lists it as an **image-text-to-text multimodal model** loaded through the
multimodal auto-class, which is why `finetune_vision_layers = false` and the
"tokenizer inside the processor" access pattern matter. Section 9's token-gate
wording already anticipates this; the model-loading section should state it
explicitly rather than describing the model as a plain text LLM.

## R2. Flaw 1 — Tier B builds inputs from the answer key, which does not exist at inference

This is the central too-good-to-be-true problem. Tier B (Section 7.4)
constructs the training context by reconstructing canonical blocks **from
`sections_json`** — the gold annotations. At inference time there is no
`sections_json`; there is only a raw court document. Three consequences:

1. **The deployment gap is unsolved.** A 222,778-token TPPO judgment still
   cannot be given to the model at inference. Section 16's claim that "TPPO
   overflow rows remain usable" is true only for training. Some out-of-scope
   component (chunking, retrieval, sliding windows) must select which 32K of
   the document the model sees, and Plan D trains the model on a context
   distribution that component will never produce: gold spans plus curated
   same-document distractors, with the connective boilerplate between sections
   stripped out. Real windows contain that boilerplate, partial sections, and
   text belonging to no canonical section at all.
2. **Validation is graded on the same privileged inputs.** Section 12
   transforms `val.parquet` with the same gold-derived aggregation, so the
   reported metrics for aggregated rows measure "extraction from a
   label-constructed context", not "extraction from a real long document".
   The headline numbers will overstate deployed performance by construction.
3. **Coverage is assumed, not shown.** Tier B step 1 assumes the 31 blocks
   concatenated with recovered separators approximate `input_text`. The plan
   never reports what fraction of `input_text` characters fall inside some
   annotated section. If coverage is low, aggregated contexts are far shorter
   and cleaner than anything seen in deployment.

**Required revision.** Keep Tier B, but state honestly that it is a
training-data recovery device, not a long-document solution. Add to Section 12
a mandatory evaluation condition in which overflow validation documents are
presented as raw contiguous windows (no gold-derived selection) and metrics
are reported per window position. Add to Section 3 a measured
section-coverage statistic (fraction of `input_text` inside annotated blocks)
before Tier B is trusted. The Zhao/Yin/Durrett citation (Section 4.4) actually
argues this direction: their finding is that models trained on data that
diverges from realistic long-context structure underperform on real
long-context tasks — the paper supports raw-window evaluation, not
gold-aggregated evaluation.

## R3. Flaw 2 — Tier D continuation packets are ill-posed as a generation task

Section 7.6 stores `continuation_index` and `continuation_count` **only as
dataset metadata** and never exposes them to the model. If two packets from
the same oversized field share the same system prompt and the same user
request, the model is trained to produce different answers to inputs that
differ only in context composition — and at inference nothing tells the model
(or the caller) which packet is being requested or how many exist. The
round-trip reconstruction gate proves the dataset is self-consistent; it does
not make the inference task well-defined. Section 12.1's "reconstructed-field
F1" quietly presumes an orchestration layer that re-slices documents into the
same packets, which is out of scope and unspecified.

**Required revision.** Either (a) make the packet identity visible in the user
turn in natural language — the request names the section and states that the
context is a bounded excerpt, and the instruction is to extract the requested
section's text *as it appears in this excerpt* — which turns every
continuation example into an honest, self-contained task; or (b) drop
reconstructed-field F1 from Section 12 and report continuation packets only as
standalone examples. Option (a) is consistent with the Wright et al. (EMNLP
2025) framing the plan already cites: evidence extraction is always relative
to the supplied context.

## R4. Flaw 3 — the plan never sums its own token table; one epoch does not fit a Colab session

Section 6 reports per-group full-sequence medians of roughly 17K–23K tokens
across 32,084 candidate conversations. Multiplying the plan's own numbers:
32,084 conversations at a ~18.5K-token median is roughly **590–710 million
tokens per epoch** (using the median as a lower bound; the length distribution
is right-skewed, so the true mean is higher). On one A100-40GB running 4-bit
QLoRA on a 9B model at 32K context, plausible training throughput is on the
order of 1,500–5,000 tokens per second, giving **roughly 33–130 hours for the
single epoch Section 10.3 calls the default**. The original document expands
by ~13× because the same full `input_text` is re-encoded once per field group,
and Section 15 waves at this ("full-input examples may dominate compute")
without ever computing it. A plan whose acceptance criteria include "the full
run completes" must show this arithmetic and a session strategy; as written,
the run cannot finish inside any single Colab session.

**Required revision.** Add a compute budget subsection to Section 10 that (a)
sums measured `n_total_tokens` over the built dataset before training, (b)
converts it to wall-clock using a measured tokens-per-second figure from the
preflight's ten optimizer steps, and (c) chooses one documented reduction if
the result exceeds the available session budget — the honest options are
sampling a fixed number of groups per document per epoch (with all groups
covered across the corpus), or capping full-input duplicates per document and
letting Tier B contexts (which are shorter) carry the rest. Checkpoint-resume
across sessions must be specified, not assumed. Silent truncation remains
forbidden.

## R5. Flaw 4 — the verbatim-substring gate is asserted, never measured

Sections 9 and 13 require 100% of non-empty targets to be exact substrings of
`context_text`, and Tier B step 1 requires reconstructing blocks "using the
same newline and blank-line separators as `input_text`". But the gold
`sections_json` values were produced by an annotator LLM, and LLM annotators
routinely normalize whitespace, dashes, OCR artifacts, and line breaks. The
plan presents the 100% gate as if it will simply pass. If a material fraction
of annotations fail exact-substring matching, the build halts — or, worse, a
"repair" step silently edits targets and the verbatim guarantee becomes
circular. Nowhere does the plan report the actual current exact-substring rate
of `sections_json` against `input_text`.

**Required revision.** Add to Section 3 a measured audit: for every row and
every non-empty section value, report the exact-substring pass rate per corpus
and per field, before any other build work. Define the policy for failures in
advance: values that match after a defined, lossless canonicalization
(whitespace-run collapse used symmetrically on both sides for matching only,
with the original document text — not the annotation — always used as the
training target) are retained; values that still fail are excluded and
counted, and the exclusion rate is reported alongside every headline metric.
Excluding failures silently would inflate the verbatim scores the plan brags
about in Section 13.

## R6. Secondary corrections

- **Section 10.4 and Section 11 prescribe smoke tests.** This project has an
  explicit standing rule: no smoke-test scaffolding in training notebooks;
  logic is validated locally on CPU and the real run is the only GPU run. The
  ten-step smoke run and "10-step smoke test first" language must be removed;
  the preflight's build-time gates (Section 9) and the local CPU validation of
  the data builder replace them. The single legitimate GPU pre-step is the
  longest-example forward/backward memory check, folded into the start of the
  real run rather than run as a separate rehearsal.
- **Section 12 ignores the test split.** Project convention is that both
  validation and test splits are attached during training as a two-way
  evaluation dictionary and the test split drives post-training inference.
  Plan D transforms only `val.parquet`. The same transformation must be
  applied to the test partition, with the same no-leakage guarantee.
- **Inference cost is multiplied thirteenfold and never stated.** Extracting
  all 31 fields from one document now requires 13 generation calls, most of
  which re-encode the same full document. This is an acceptable trade, but it
  belongs in Section 15 as a stated cost, with prefix caching noted as the
  standard mitigation, rather than being left for the reader to discover.
- **Internal inconsistency:** Section 6 uses a 256-token prompt/template
  allowance for feasibility estimates while Sections 2.1 and 7.4 use a
  512-token safety margin. The feasibility percentages (82.60% fit) are
  therefore slightly optimistic relative to the actual build rule. Re-state
  Section 6 as an estimate and let the build's exact-tokenization numbers be
  the reported figures.
- **Acceptance thresholds in Section 13 are targets, not predictions.** JSON
  validity ≥ 99% and verbatim validity ≥ 99% are plausible for short fields
  but optimistic for multi-thousand-token narrative copies such as
  `pertimbangan_hukum`, where greedy decoding of very long verbatim spans
  drifts. Keep the thresholds, but add the missing decision rule: what happens
  when a long-field group misses the bar (report per-field, ship the fields
  that pass, and iterate on the long-field tiers) — otherwise the criteria
  read as guaranteed outcomes, which is exactly the too-good-to-be-true tone
  this revision exists to remove.

## R7. Revised verdict

Plan D's diagnosis (Section 2) is correct and its data-reshaping direction is
the right one: partial-field targets, no truncation reliance, exact token
accounting, and response-only loss are all sound and all citation-supported.
What must change before implementation: raw-window evaluation and a coverage
audit for Tier B (R2), a well-posed continuation contract (R3), an explicit
compute budget with session strategy (R4), a measured verbatim-substring audit
with a declared failure policy (R5), and the removal of smoke-test scaffolding
plus inclusion of the test split (R6). With those amendments Plan D is
credible; without them, its acceptance criteria promise results its own
evaluation design cannot honestly certify.

---
---

# Revision 2 — Citation-Backed Fixes (2026-07-11)

This revision resolves the four flaws identified in Revision 1. Every new
citation below was verified on its publisher page (NeurIPS proceedings,
ICLR virtual site, ACL Anthology) on 2026-07-11, the same standard applied in
Revision 1. Where this revision contradicts the original Sections 7, 10, 12,
or 13, this revision governs; the original text is retained above for the
record.

## New verified references

| Reference | Verified venue | Used for |
|---|---|---|
| Zhou et al., "LIMA: Less Is More for Alignment" — [NeurIPS 2023 main track](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) | NeurIPS 2023 | Fix 3: per-epoch sampling instead of exhaustive 13× duplication |
| Chang, Lo, Goyal, Iyyer, "BooookScore: A systematic exploration of book-length summarization in the era of LLMs" — [ICLR 2024 oral](https://iclr.cc/virtual/2024/oral/19789) | ICLR 2024 (oral) | Fixes 1–2: chunk-then-merge workflow for documents longer than the context window |
| Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" — [NeurIPS 2024 main track](https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | NeurIPS 2024 | Fix 5: RadixAttention prefix caching to absorb the 13-request inference cost |

A sliding-window document-parsing paper (Kumaravel et al., "Slide, Constrain,
Parse, Repeat") was considered but found to be arXiv-only with no verified
conference record, so it is deliberately **not** cited. All other support
comes from references already verified in Revision 1: He et al. (ACL 2024),
Zhao, Yin, Durrett (ICML 2025), Wright et al. (EMNLP 2025), Yuan et al.
(ICML 2025), Bai et al. (Findings of EMNLP 2024).

## Fix 1 — Replace gold-built contexts with raw contiguous windows (resolves R2)

Tier B as originally specified is demoted from the primary overflow mechanism
to an ablation arm. The primary overflow mechanism becomes **Tier W — raw
window examples**, which never uses gold labels to construct the input:

1. For every document/group pair whose fully formatted conversation exceeds
   the build ceiling, slice the **raw, unmodified `input_text`** into
   contiguous windows sized to the available context budget, with a fixed
   overlap of roughly one quarter of the window between neighbours so that no
   section boundary is only ever seen at a window edge.
2. Each window becomes one conversation with the same group request. The
   target contains exactly those requested field values that are exact
   substrings lying fully inside the window; requested fields with no value
   inside the window go to `[]` and `empty_sections`.
3. A field value that fits in no single window contributes, per window, the
   maximal document-verbatim portion present in that window, under the
   excerpt-relative contract of Fix 2. This subsumes and replaces Tier D:
   window boundaries, not ad-hoc packet splitting, define continuations.
4. Gold annotations are used only as **targets**, exactly as in ordinary SFT
   — never to select, reorder, or filter the input text.

Rationale from the verified literature: Zhao, Yin, and Durrett (ICML 2025)
show that fine-tuning data whose contexts diverge from realistic long-context
structure yields weaker retrieval behaviour on real tasks — raw windows are
the realistic distribution, gold aggregates are not. He et al. (ACL 2024)
show that varying the position of the relevant evidence across training
examples is itself beneficial, which overlapping windows provide for free.
BooookScore (ICLR 2024) establishes chunk-level processing followed by
merging as the standard workflow for documents that exceed the context
window; Tier W trains the model for precisely the chunk-level step of that
workflow.

**Inference contract (now in scope, one paragraph, no orchestration system).**
Deployment presents the same raw windows with the same group requests and
merges per-field outputs across windows by document-order concatenation with
overlap deduplication (a string operation, not a model call). This closes the
gap identified in R2: the model is trained on exactly the input distribution
it will see, including the 222K-token TPPO tail.

**Evaluation.** The headline metrics in Section 12 are computed on the
**merged raw-window condition** for overflow documents. The gold-aggregated
condition may still be reported, but only labelled as an oracle upper bound.
A section-coverage audit (fraction of `input_text` characters inside
annotated blocks) is added to Section 3 as Revision 1 required; it now
informs analysis rather than gating the method, since Tier W no longer
depends on coverage.

## Fix 2 — Excerpt-relative extraction contract (resolves R3)

The user turn for every Tier W conversation states, in Indonesian, that the
context is a bounded excerpt of a longer judgment and asks for the requested
sections **as they appear in this excerpt**. With that single wording change,
every conversation is a self-contained, well-posed task: identical requests
never map to different answers, because the context itself differs, and the
model needs no hidden knowledge of packet indexes. `continuation_id`,
`continuation_index`, and `continuation_count` are deleted from the schema in
Section 8 and replaced by `window_index` and `window_count`, still
metadata-only. Wright et al. (EMNLP 2025) support this framing directly:
evidence extraction is defined relative to the supplied context, and explicit
fine-tuning on that task is what improves verbatim copying. "Reconstructed-
field F1" in Section 12.1 is renamed **merged-field F1** and is computed
after the deterministic merge described in Fix 1, making it a deployment-
faithful metric rather than one that presumed an unspecified orchestrator.

## Fix 3 — Explicit compute budget with per-epoch group sampling (resolves R4)

The exhaustive 13-groups-per-document design implied roughly 590–710 million
tokens per epoch (Revision 1, R4), i.e. tens to over a hundred hours on the
single A100-40GB. The fix has three parts:

1. **Budget first.** After the build, sum `n_total_tokens` over the training
   set. The epoch token budget is fixed in advance from the available session
   wall-clock and the tokens-per-second figure read from the first logged
   steps of the real training run (no separate rehearsal run of any kind).
   As an anchor: a 150-million-token epoch is roughly 14–28 hours at
   1,500–3,000 tokens per second — feasible in one to two sessions.
2. **Stratified per-epoch sampling.** Each document contributes all of its
   Tier W window examples (they carry the hard long-tail signal and are
   individually short) plus at most G full-input group conversations per
   epoch, where G is set so the budget holds. Group selection per document
   rotates deterministically with seed 3407 across epochs, so all 13 groups
   are covered corpus-wide every epoch and per-document over successive
   epochs. LIMA (NeurIPS 2023) is the venue-verified basis for preferring a
   curated, diverse subset over exhaustive duplication: alignment quality is
   driven by coverage and quality, not raw example count, and every sampled
   conversation here remains full-quality supervision. ChunkFlow (ICML 2025)
   already cited in Section 4.6 supports handling the length tail separately
   rather than letting it dictate the whole run's shape.
3. **Session strategy.** Checkpoint-resume across sessions is mandatory:
   resume from the last saved checkpoint with the sampler's epoch state
   restored deterministically. The acceptance criterion "the full run
   completes" in Section 13 is amended to "the full budgeted epoch completes
   across one or more resumed sessions with no truncation and no sampler
   drift". Silent truncation remains forbidden.

## Fix 4 — Measured verbatim audit with a lossless repair policy (resolves R5)

Before any conversation is built, run a corpus audit reporting, per corpus
and per field, the fraction of non-empty `sections_json` values that are
exact substrings of their `input_text`. The build then applies one fixed
policy:

- **Exact match:** use the annotation as-is.
- **Match after symmetric whitespace-run canonicalization:** the
  canonicalization is used **only to locate** the span; the training target
  is then re-extracted from the original `input_text` at the located
  position. Targets are therefore document-verbatim by construction, which is
  strictly stronger than trusting the annotator string.
- **Still no match:** the value is excluded, counted, and the exclusion rate
  is reported next to every headline metric in Section 12. No silent
  exclusion, no annotation-side editing.

This keeps the 100% verbatim gate of Section 9 honest: it now holds because
targets are re-extracted from the document, and the residual failure rate is
a published number instead of an unexamined assumption. Wright et al.
(EMNLP 2025), already verified, is the evidential basis for treating
verbatim drift as an expected failure mode to be measured rather than assumed
away.

## Fix 5 — Secondary corrections applied (resolves R6)

- **Smoke tests removed.** The ten-step smoke run in Sections 10.4 and 11 is
  deleted. Builder logic is validated locally on CPU; the only GPU pre-steps
  are the longest-example forward/backward memory check and the
  tokens-per-second measurement, both folded into the start of the single
  real run, which then continues uninterrupted.
- **Test split included.** The Plan D transformation is applied identically
  to the test partition; training attaches a two-way evaluation dictionary
  (validation and test), and the transformed test split drives post-training
  inference, per project convention. Model selection still reads only
  validation.
- **Inference cost stated and mitigated.** Extracting all 31 fields requires
  13 requests per document (more for windowed documents). The user template
  in Section 5.1 is reordered so the judgment context precedes the
  requested-fields line; the long document prefix is then identical across
  all 13 requests and is served from KV cache by RadixAttention-style prefix
  reuse (Zheng et al., NeurIPS 2024), reducing the marginal cost of each
  additional group request to roughly the cost of its short suffix and
  generation. This reordering is applied in training too, so the training and
  serving templates stay identical.
- **Margin unified.** Section 6's 256-token allowance is superseded: all
  feasibility numbers are re-derived during the build with the exact
  tokenizer and the single 512-token margin. Section 6's table is retained
  above as an estimate only.
- **Acceptance thresholds get a decision rule.** The Section 13 model
  thresholds stand as targets, evaluated on the merged raw-window condition.
  If a long-narrative field misses its bar, the shipped extractor covers the
  passing fields and the failing field's window size and overlap are the
  first iteration knobs — the criteria are commitments about process, not
  promises about outcomes.

## Revised baselines (supersedes Section 12.3)

| ID | Training format | Purpose |
|---|---|---|
| B0 | current whole input + full 31-field target | quantify truncation/OOM failure |
| B1 | partial-field target + complete input only; overflow discarded | isolate benefit of target decomposition |
| D1 | full-input tier + Tier W raw windows (main method) | deployment-faithful main result |
| D2 | D1 with gold-aggregated contexts replacing Tier W | oracle upper bound; measures the R2 gap directly |
| D3 | D1 without full-input examples (windows only) | value of retaining whole context where possible |

The D1-versus-D2 comparison turns Revision 1's central criticism into a
measured quantity: the difference between the two is exactly the optimism
that gold-built contexts would have injected into the headline numbers.

## Closing statement

With Fixes 1–5, every mechanism in Plan D is trained on inputs that exist at
inference time, every conversation is a well-posed task, the compute budget
is derived from measured throughput before the run starts, the verbatim
guarantee is enforced by construction and audited by publication, and the
evaluation certifies deployment behaviour rather than an oracle setting. All
supporting citations are venue-verified: NeurIPS 2023/2024, ICLR 2024,
ICML 2025, ACL 2024/2025/2026, and EMNLP 2024/2025.
