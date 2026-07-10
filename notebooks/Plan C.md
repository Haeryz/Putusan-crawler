# Plan C — Schema-Constrained Hierarchical Span SFT

**Task:** recover the 31 canonical Putusan MA sections as verbatim source spans.

**Training environment:** Google Colab, one NVIDIA A100-SXM4-40GB. The local
repository is used to author the notebook only. All durable run state is stored as
Weights & Biases Artifacts; Colab's `/content` filesystem is treated as disposable.

**Primary model:** `Qwen/Qwen3.5-9B`.

**Dataset:** `Haeryz/putusan-structured-extraction`, config `sft`. Plan C does not
use `notebooks/dataset/windowed_dataset`.

**Status:** implementation specification for `Qwen3_5_(4B).ipynb`.

---

## 1. Problem Definition

The current notebook trains a causal LM on:

```text
system prompt + complete input_text + complete 31-section JSON
```

The audit in `datalog.md` found fully templated lengths of p50 34,499, p90
74,328, p95 90,106, and max 448,111 tokens. The notebook caps training at 32,768
tokens while masking loss to the assistant response. For most long rows the JSON
response is truncated, leaving no supervised labels. This is not corrected by a
different percentile or batch size.

The target is also almost a duplicate of the input: across the local SFT files,
the median `n_target_chars / n_input_chars` ratio is about 1.03. Autoregressively
reproducing the court text consumes memory and generation time without adding
information, while creating a hallucination opportunity at every output token.

Plan C changes the learning problem from text regeneration to structured span
prediction:

```text
input_text
  -> Qwen semantic unit representations
  -> whole-document contextualization
  -> joint span-boundary and section-label prediction
  -> verbatim JSON serialization
```

The model chooses the boundaries and labels. Deterministic code only validates
the selected character ranges and copies their source substrings into the public
JSON schema.

### Dataset limitation

`build_dataset.py` constructs `input_text` by concatenating the annotated section
spans in canonical section order. It does not contain the original unfiltered
judgment body. The checkout also does not contain the corresponding raw `.txt`
corpus. Therefore:

- Plan C evaluates **reconstructed-document segmentation**, not production raw-
  judgment extraction.
- Canonical-order transition constraints are valid for this dataset but must not
  be presented as a universal property of Indonesian judgments.
- Position-only and whitespace-only baselines are mandatory. If they match the
  learned model, the experiment demonstrates a dataset artifact rather than legal
  semantic learning.

RAG, GRPO, raw-PDF processing, and downstream legal search are out of scope.

---

## 2. Research Basis and Novelty

### Direct technical basis

1. **Semi-Markov Conditional Random Fields for Information Extraction** — NIPS
   2004. Semi-CRFs jointly choose variable-length spans and their labels under a
   globally normalized structured objective.
   https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html
2. **Filtered Semi-Markov CRF** — Findings of EMNLP 2023. A learned boundary/span
   filter prunes the quadratic candidate space while retaining global structured
   decoding.
   https://aclanthology.org/2023.findings-emnlp.17/
3. **Gradient Cache** — RepL4NLP 2021. Representation gradients can be cached and
   replayed through encoder micro-batches, making a loss that depends on a large
   logical batch trainable with near-constant encoder activation memory.
   https://aclanthology.org/2021.repl4nlp-1.31/

### Direct legal-NLP basis

1. **Joint Span Segmentation and Rhetorical Role Labeling with Data Augmentation
   for Legal Documents** — ECIR 2023. This is the closest prior task: a
   hierarchical encoder and semi-CRF jointly segment long judgments and assign
   rhetorical labels.
   https://doi.org/10.1007/978-3-031-28238-6_54
2. **Structural Text Segmentation of Legal Documents** — ICAIL 2021. Transformer
   boundary classifiers outperform non-contextual segmentation baselines on long
   legal documents.
   https://doi.org/10.1145/3462757.3466085
3. **Segmenting U.S. Court Decisions into Functional and Issue Specific Parts** —
   JURIX 2018. Sequence-aware CRF classification is effective for functional court-
   decision structure.
   https://doi.org/10.3233/978-1-61499-935-5-111
4. **LegalSeg** — Findings of NAACL 2025. Hierarchical and sequence-aware models
   outperform isolated sentence classification for judgment segmentation.
   https://aclanthology.org/2025.findings-naacl.63/
5. **HiCuLR** — Findings of EMNLP 2024. Coarse-to-fine role and document curricula
   improve legal rhetorical-role learning.
   https://aclanthology.org/2024.findings-emnlp.433/

### Why context-extension methods are not the main solution

LongLoRA (ICLR 2024), PoSE (ICLR 2024), and Activation Beacon (ICLR 2025) are
important long-context methods, but none changes the redundant target objective.
Even if they made every input trainable, the model would still be optimized to
regenerate nearly the whole judgment. Integrating a new context-compression
architecture into Qwen3.5's hybrid Gated DeltaNet stack would also dominate the
engineering risk without directly supervising legal boundaries.

### Plan C novelty hypothesis

Plan C combines components not evaluated together in the cited work:

- a 9B Qwen3.5 QLoRA semantic encoder rather than a frozen BERT sentence encoder;
- full-document end-to-end gradients on a single 40GB GPU through representation-
  gradient caching;
- a learned boundary filter with exact semi-Markov normalization and Viterbi
  decoding over an ordered 31-section legal schema;
- same-label transitions for multiple verbatim list entries and skipped labels for
  empty sections;
- coarse-to-fine Indonesian legal section training;
- explicit artifact baselines that test whether the reconstructed formatting, not
  legal semantics, explains performance.

This is a research hypothesis, not a novelty claim established by implementation
alone. It must be supported by the ablations in Section 7.

---

## 3. Supervision Construction

### Exact alignment

For each original SFT row:

1. Parse `sections_json` in canonical section order.
2. Sequentially exact-match each non-empty span in `input_text`, starting after the
   previously matched span.
3. Store `(section_key, item_index, start_char, end_char)`.
4. Reject the row on any missing, overlapping, or out-of-order match.
5. Assert that slicing each range exactly reproduces its annotated string.

The local audit covered all 3,075 SFT rows and found zero sequential matching
failures. All 87,889 inter-section transitions had an inserted blank-line
separator. These values must be recomputed and logged by the Colab notebook rather
than trusted as constants.

### Semantic units and local chunks

- Start from non-empty source lines, retaining their exact character positions and
  the intervening newline text.
- Tokenize with the Qwen fast text tokenizer and offset mappings.
- Split a line only when it exceeds 512 tokens. A forced split is an encoding
  boundary, not automatically a predicted legal boundary.
- Assign every unit the unique gold section and list-item identity covering it.
- Pack consecutive units, plus their original intervening whitespace, into local
  chunks of at most 4,096 tokens.
- Do not print or add line numbers. Do not retrieve, discard, or reorder units.

The global model always receives the ordered representation of every unit in the
document. Chunking only bounds Qwen activation memory.

### Source-balanced sampling

Several source documents have more than one annotator-model row. Give each row
weight `1 / rows_for_source_sha256` so repeated annotations do not give a document
extra optimization or evaluation weight. Do not expose `annotator_model` to the
model; use it only for stratified diagnostics.

---

## 4. Model

### Local Qwen encoder

- Load `Qwen/Qwen3.5-9B` with 4-bit NF4 weights and BF16 computation.
- Freeze the vision tower and causal LM output head.
- Use the text backbone's last hidden states; never compute vocabulary logits.
- Apply QLoRA with `r=32`, `alpha=32`, dropout `0`, and bias `none`.
- Discover and verify adapters on the language backbone's full-attention
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`), linear-attention (`in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`), and MLP (`gate_proj`,
  `up_proj`, `down_proj`) projections.
- Attention-pool each unit's token states and project 4,096 dimensions to 512.

### Whole-document context

Pass the complete ordered unit sequence through a two-layer bidirectional GRU with
256 hidden units per direction and dropout 0.1. This layer is linear in the number
of units and provides left and right document context missing from independent
Qwen chunks.

### Learned boundary filter

For each gap between adjacent units, score a possible span boundary from the left
and right contextual vectors, their elementwise difference, and normalized document
position. Train with focal binary cross entropy (`gamma=2`).

During training, always include every gold boundary. During inference:

1. retain start and end;
2. retain every gap above the validation-calibrated threshold;
3. add the highest-scoring remaining gaps up to a cap of 256;
4. if validation gold-boundary recall is below 99.5%, raise the cap to 512 and then
   1,024.

### Schema-constrained filtered semi-CRF

For every retained pair of boundaries, build a span representation from:

- first contextual unit;
- last contextual unit;
- mean contextual vector computed by prefix sums;
- log-bucketed span duration;
- learned candidate-label embedding.

The semi-CRF supports:

```text
same label  -> another list item in that section
later label -> next non-empty canonical section
skipped key -> empty section
backward label transition -> forbidden
```

Training minimizes exact negative log-likelihood over every valid labeled
segmentation in the filtered graph. Inference uses exact Viterbi decoding with
backpointers.

---

## 5. Training Curriculum

Use these six coarse classes:

| coarse class | canonical sections |
|---|---|
| header | `judul` through `keterangan_perkara` |
| identity | `nama_lengkap` through `pekerjaan` |
| procedure | `penangkapan`, `penahanan`, `tuntutan`, `dakwaan` |
| evidence | `saksi` through `petunjuk_barang_bukti` |
| decision | `fakta_hukum`, `pertimbangan_hukum`, `amar_putusan` |
| closing | `hari` through `tanda_tangan_majelis` |

### Stage 1 — local QLoRA curriculum

1. One epoch of coarse unit classification plus boundary loss.
2. Up to two epochs of 31-way unit classification plus boundary loss.
3. Validate after each epoch and retain the best checkpoint.

### Stage 2 — global structured head

1. Freeze Qwen and the local unit projector.
2. Cache projected train/validation unit embeddings under `/content`.
3. Train the bidirectional GRU, boundary filter, presence head, and filtered semi-
   CRF for at most 30 epochs with patience 5.
4. Log the completed embedding cache as its own versioned W&B Artifact so a new
   Colab runtime can resume without recomputing it.

### Stage 3 — joint full-document tuning

Use representation-gradient caching:

1. Run all document chunks without an autograd graph and collect projected unit
   embeddings.
2. Treat those embeddings as leaf tensors and backpropagate the full structured
   loss to obtain `dL/dEmbedding` for every unit.
3. Re-run one Qwen chunk at a time with autograd and backpropagate the dot product
   between the recomputed embeddings and their cached gradients.
4. Update QLoRA, the projector, and the structured head only after the complete
   logical document has been processed.

With zero Qwen/LoRA dropout, the recomputation is deterministic. A tiny-model test
must show that cached and ordinary backpropagation agree within BF16 tolerance.

Train for at most five joint epochs with patience 1. Use an effective document
batch of four through gradient accumulation.

### Optimization defaults

| parameter | value |
|---|---:|
| QLoRA learning rate | `1e-4` |
| structured-head learning rate | `5e-4` |
| optimizer | AdamW 8-bit |
| weight decay | `0.01` |
| warmup | 5% |
| schedule | cosine |
| gradient norm | `1.0` |
| local chunk | 4,096 tokens |
| maximum unit | 512 tokens |
| primary seed | 3407 |
| reporting seeds | 3407, 3408, 3409 |

Final joint objective:

```text
L = 1.0 * filtered_semi_crf_nll
  + 0.5 * boundary_focal_loss
  + 0.2 * fine_unit_cross_entropy
  + 0.1 * coarse_unit_cross_entropy
  + 0.2 * section_presence_bce
```

---

## 6. Colab and W&B Artifact Contract

The notebook must be runnable from a fresh Colab runtime. It must:

1. install pinned dependencies;
2. verify CUDA and report the actual GPU name and usable memory;
3. refuse a full run on a non-A100 runtime unless `SMOKE_TEST=True`;
4. read `WANDB_API_KEY` from Colab Secrets;
5. download the model and dataset into ephemeral Colab caches;
6. store all temporary files under `/content`;
7. use W&B as the sole durable checkpoint and final-artifact backend;
8. never mount Google Drive or push a model to Hugging Face Hub.

`report_to=wandb` logs Trainer metrics but does not, by itself, persist this
custom model's complete state. Plan C therefore logs explicit W&B Artifacts.

Each checkpoint bundle contains:

- QLoRA adapter and tokenizer;
- unit projector, global encoder, boundary filter, semi-CRF, and auxiliary heads;
- optimizer and scheduler states;
- stage, epoch, document cursor, accumulation cursor, and global step;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- configuration, canonical schema, coarse mapping, filter threshold/cap;
- dataset fingerprint and exact model/tokenizer revisions;
- W&B run ID and current best validation metrics.

Artifact policy:

- log `latest` every 100 optimizer steps and at every stage boundary;
- log `best` whenever validation span macro-F1 improves;
- block on `artifact.wait()` before removing or replacing local checkpoint files;
- resume with the same W&B run ID, `resume=must`, and the `latest` Artifact;
- restore the data cursor and every RNG/optimizer state before the next update;
- call `wandb.finish()` after final evaluation.

The committed notebook must have no saved outputs or credentials. Authentication
must use Colab Secrets. Any credential previously stored in notebook state must be
rotated.

Official integration reference:
https://docs.wandb.ai/models/integrations/huggingface_transformers

---

## 7. Evaluation and Ablations

Use validation only for thresholds, early stopping, ablations, and model selection.
Load and evaluate the test split once after freezing the selected configuration.

### Required systems

| ID | system |
|---|---|
| A0 | position-only canonical segmentation |
| A1 | whitespace/newline-only boundary model |
| A2 | frozen Qwen plus independent unit classifier |
| A3 | local QLoRA without global context |
| A4 | QLoRA plus bidirectional context and linear-chain CRF |
| A5 | unfiltered semi-CRF |
| C | full Plan C |

Run component ablations for the global encoder, boundary filter, semi-CRF,
coarse curriculum, section-presence loss, monotonic mask, and Stage 3 joint QLoRA.

### Metrics

- boundary precision, recall, and F1, separating section and same-section item
  boundaries;
- unit macro/micro F1;
- character-level macro/micro F1 across the 31 keys;
- exact labeled-span precision, recall, and F1;
- per-section exact match and character F1;
- empty-section precision, recall, and F1;
- canonical document JSON exact match;
- candidate-filter gold recall and candidate count;
- peak GPU memory, tokens/sec, documents/hour, and inference time;
- fraction of emitted characters not present at the claimed source offsets;
- W&B checkpoint upload and fresh-runtime resume success.

Report metrics by corpus, annotator model, input-length bucket, and section.
Aggregate source-level results with each `source_sha256` receiving equal weight.

### Promotion gates

Plan C is successful only if:

- every training row aligns and contributes supervised loss;
- no label is lost to truncation;
- gold spans round-trip to canonical JSON for 100% of rows;
- validation boundary-candidate recall is at least 99.5%;
- serialized JSON validity is 100%;
- emitted legal text always equals its claimed source slice;
- peak memory stays below the Colab A100 limit;
- a fresh Colab runtime resumes from W&B and reproduces the next update within
  numerical tolerance;
- full Plan C beats the strongest non-generative baseline in source-level macro
  span F1 with a paired-bootstrap 95% confidence interval excluding zero.

If the position-only or whitespace-only baseline matches Plan C, the result must be
reported as a failed semantic-learning experiment.

---

## 8. Public Output

The learned decoder returns internal records:

```json
{
  section_key: pertimbangan_hukum,
  start_char: 42117,
  end_char: 58732,
  confidence: 0.97
}
```

The model does not generate or count these integers. They are tokenizer/source
indices attached to the boundaries chosen by the structured decoder.

The serializer validates the path, copies exact source substrings, preserves
same-label list items, fills skipped keys with `[]`, computes `empty_sections`, and
emits the existing contract:

```text
status
source_file
source_sha256
sections (31 canonical keys)
empty_sections
```

No downstream consumer receives a schema change.

---

## 9. What the Trained Model Is

The trained system is **not a generative LLM**. Qwen3.5-9B is used as a frozen
4-bit reading encoder; its vocabulary head is never called, and the notebook never
invokes `model.generate()`. There is no prompt, no chat template, no sampling, and
no decoding temperature anywhere in training or inference.

### Components of the built model

| component | role | trained? | approximate size |
|---|---|---|---:|
| Qwen3.5-9B backbone (4-bit NF4) | reads text, produces hidden states | frozen | 9B parameters |
| QLoRA adapter (`r=32`) | adapts the reader to legal text | yes | ~100–200 MB |
| unit pooler (attention pool + 4096→512 projection) | one vector per source line | yes | small |
| two-layer bidirectional GRU (256/direction) | whole-document context | yes | ~5M parameters |
| boundary filter MLP | scores each gap between lines | yes | small |
| filtered semi-CRF (span scorer, label embedding, transitions) | picks the global best segmentation | yes | small |
| auxiliary heads (fine/coarse unit classifiers, presence head) | training signal only | yes | small |

### Forward pass

```text
document text
  -> Qwen encoder (per 4,096-token chunk): one 512-dim vector per source line
  -> bidirectional GRU over all line vectors of the document
  -> boundary filter: which gaps between lines may be section boundaries
  -> semi-CRF + Viterbi: best (span, label) segmentation under the canonical order
  -> deterministic serializer: slice the source text at the chosen offsets
```

The neural network's output is numbers — boundary scores, span-label potentials,
and segment posteriors — never words. The only text in the final JSON is verbatim
substrings copied from the input by ordinary Python slicing.

### Two output layers

1. **Internal records** (Section 8): `(section_key, start_char, end_char,
   confidence)` per predicted span. The confidence is the exact marginal
   probability of that segment under the globally normalized semi-CRF.
2. **Public JSON**: the existing 31-key contract. All 31 keys are always present;
   skipped sections are `[]` and listed in `empty_sections`; repeated labels become
   list items (for example, multiple `saksi` entries).

### Deployment implications

- The model answers exactly one question — "where are the 31 canonical sections in
  this document" — and cannot chat, summarize, or answer free-form queries.
- The durable artifact (W&B checkpoint) is the QLoRA adapter plus the structured
  head weights. It is not a standalone Hugging Face generation model and only runs
  inside this notebook's stack: base Qwen + adapter + heads + Viterbi + serializer.
- The output is guaranteed valid JSON with guaranteed verbatim legal text: the
  model makes ~31 structured decisions per document instead of being correct at
  every one of tens of thousands of generated tokens, so inference takes seconds
  per judgment rather than minutes, and hallucinated text is structurally
  impossible.
- This is the same architectural relationship BERT-based extractors have to BERT —
  an encoder with task heads — with a 9B QLoRA-adapted Qwen as the encoder.
