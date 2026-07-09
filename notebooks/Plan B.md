# Plan B — Line-Anchored Windowed Extraction with Deterministic Assembly

**Task:** structured extraction of 31 canonical sections (verbatim spans) from Indonesian court
decisions (putusan), dataset `Haeryz/putusan-structured-extraction`.
**Constraint set:** 1× A100-SXM4-40GB (39.5 GB usable), models < 10B, three models trained in
order: `Qwen/Qwen3.5-9B` → `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` → `google/gemma-4-E4B-it`.
**Status:** proposal. Companion to `Plan A.md` (char-offset variant); supersedes the
whole-document SFT currently in `Qwen3_5_(4B).ipynb` / `qwen_GRPO.ipynb`.

---

## 1. Problem statement

The token-length audit in `datalog.md` (inner text tokenizer, fully-templated
system + putusan + assistant JSON):

| percentile | tokens  |
| ---------- | ------- |
| p50        | 34,499  |
| p90        | 74,328  |
| p95        | 90,106  |
| max        | 448,111 |

With `max_seq_length = 32768`, **the median example already exceeds the context window**.
Truncation keeps the front of the sequence; the assistant JSON target sits at the end;
`train_on_responses_only` masks loss to that trailing JSON. Consequence: for the majority of
examples the entire target is truncated away, all labels become `-100`, and those steps
contribute zero (or NaN) learning signal. This is silent data corruption, not a hyperparameter
problem.

A second structural pathology: because the task is *extractive* — `build_dataset.py` asserts
every gold span is a verbatim substring of `input_text` — the target JSON re-embeds nearly the
whole document (`n_target_chars ≈ n_input_chars`, p50 ≈ 62k chars). The model is asked to
*copy out ~60k characters verbatim*, which (a) doubles sequence length for no informational
gain, (b) makes every generated token a hallucination opportunity, and (c) makes inference
cost scale with document length twice.

## 2. Why the obvious fixes fail (options considered)

### Option 0 — raise `max_seq_length`
Not attainable. Published Unsloth measurements: a 9B model with QLoRA + offloaded gradient
checkpointing reaches ~11K context on 24 GB; a 27B reaches ~9.7K on 40 GB (vs ~3K for
HF + FlashAttention-2). Extrapolating, a 9B on 40 GB trains comfortably at 8–16K, perhaps
~30K with 4-bit and aggressive offload — still under half of p90 (74K) and 1/13 of max.
Activation memory scales linearly with sequence length even with FlashAttention
(LoRA-FA, arXiv:2308.03303; Unsloth long-context blog). **Rejected: physically infeasible.**

### Option 1 — long-context adaptation (RoPE scaling / YaRN / offload systems)
Systems work (e.g. "Extending Language Model Context Up to 3 Million Tokens on a Single GPU",
arXiv:2502.08910) targets *inference*, not gradient-based training; million-token *training*
systems assume multi-node clusters. Same memory wall as Option 0. **Rejected.**

### Option 2 — RAG: retrieve-then-extract per section
Standard in legal NLP (LegalBench-RAG, arXiv:2408.10343; AQgR for Indian case law,
arXiv:2508.04710). Works well for short identity fields (`nomor_putusan`, `agama`, dates) but
the long narrative sections (`dakwaan`, `saksi`, `pertimbangan_hukum`, `fakta_hukum`) span
dozens of pages of *contiguous* text; any retrieval miss silently truncates recall, and the
"Long Context vs. RAG" evaluation (arXiv:2501.01880) finds chunk-based retrieval lags whenever
long-distance dependencies matter. **Kept as a baseline on the `rag` split, not the main
method.**

### Option 3 — Plan A: section-level extraction with character offsets
Plan A has the right *shape* (compact span references + deterministic assembler) but the wrong
*anchor*: it asks the model to emit character offsets `{"start": 52012, "end": 54490}`.
Transformers cannot count characters reliably — the tokenizer hides character positions, and
position arithmetic over tens of thousands of characters is exactly the kind of symbolic
counting sub-10B models fail at. The model would have to *compute* something it never sees.
**Refined into Option 4.**

### Option 4 — RECOMMENDED: line-anchored windowed extraction + deterministic assembly
Print line numbers *in the input*, and ask the model to **copy** the numbers it can literally
see — never to count. This is:

1. **The pattern this repo already validated.** The raw annotation stage
   (`LLM-aggregator/*/*/SPAN_EXTRACTION_SPEC.md`) gave the extractor line-numbered source and
   received `{"lines": [[start, end]]}` / `{"text": [...]}` / `{"empty": true}` per section,
   then a deterministic post-processor sliced the verbatim text. All 4,420 usable outputs in
   `LLM-aggregator/*/*/output/` were produced this way. Plan B makes that exact I/O contract
   the *training objective*.
2. **The pattern the literature converged on for beyond-context documents.**
   LLM×MapReduce (ACL 2025, aclanthology 2025.acl-long.1341) formalizes chunk → map →
   collapse → reduce, naming the two failure modes we must engineer around: *inter-chunk
   dependency* and *inter-chunk conflict*. Our reduce step is deterministic (merge line
   ranges), which is strictly easier than their generative reduce.
3. **Verifiable end-to-end**, which is precisely what the GRPO stage needs: every reward
   (range validity, overlap-F1 vs gold, schema adherence) is computable in pure Python, in
   line with "Think Inside the JSON" (arXiv:2502.14905) and ThinkJSON-style multi-reward GRPO
   for schema adherence on ≤8B models.
4. **Hallucination-free by construction.** The model never generates document text; the
   assembler copies substrings from the source. Post-assembly hallucination rate is 0 by
   definition — a headline property for a legal application.

---

## 3. Recommended pipeline

### 3.1 Data reshaping (`build_dataset_planb.py`, or a `planb` config in `build_dataset.py`)

Input: the existing parquet datasets (or re-run from `LLM-aggregator/*/*/output/*.json`).
Keep the existing document-disjoint purpose/split assignment **unchanged** (same seeds) so
results remain comparable with the whole-doc baseline and no leakage is introduced.

1. **Line-number the document.** Split `input_text` on `\n`; render each line as
   `NNNNN|content` (5-digit, zero-padded — a fixed-width prefix tokenizes consistently).
   Measured overhead is ~4 tokens/line; putusan lines average ~10–15 tokens, so expect
   ~25–35% token overhead — budget for it (hence 6K-token content windows inside an 8K cap,
   see below).
2. **Window.** Greedily pack whole lines into windows of ≤ ~5,500 tokens of numbered content,
   with ~15% overlap (trailing lines of window *k* reappear at the head of window *k+1*).
   Never split a line. p50 documents (~34.5K tokens raw) → ~7–8 windows; p95 (~90K) → ~20;
   the 448K outlier → ~100 windows (fine — each is an independent training row).
3. **Gold targets per window.** Every gold span is a verbatim substring of `input_text`
   (asserted at build time), so its global line range is recoverable by exact string matching.
   For each window, intersect all gold ranges with the window's line interval. Target JSON,
   mirroring SPAN_EXTRACTION_SPEC:

   ```json
   {
     "sections": {
       "dakwaan":      {"lines": [[120, 187]]},
       "nomor_putusan": {"text": ["Nomor 12/Pid.Sus-Anak/2023/PN Unh"]},
       "saksi":        {"lines": [[201, 240], [244, 251]]}
     },
     "sections_absent": ["ahli", "penangkapan", "..."]
   }
   ```

   - `lines` uses **global** line numbers exactly as printed in the window (copy, don't
     transform — no window-local re-indexing for the model to get wrong).
   - `text` form only for the 18 short identity/header fields (per the SPEC's split between
     line-form and text-form sections); everything long is `lines`.
   - Sections with no gold content in this window go to `sections_absent` (teaches abstention;
     most windows contain only 3–8 of the 31 sections).
4. **Row schema.** Keep the 25-column shape where possible; `messages` becomes
   system + windowed-user + window-target; add `doc_id`, `window_index`, `n_windows`,
   `line_start`, `line_end` so the assembler and eval can regroup windows into documents.
   Emit the same three configs (`sft`, `grpo`, `rag`) from the same document partitions.

Resulting SFT training rows: ~2,468 docs × ~8 windows ≈ **~20K rows of ≤8K tokens each —
100% usable, zero truncation**, and targets are a few hundred tokens instead of ~35K.

### 3.2 SFT stage (per model; order Qwen → DeepSeek → Gemma)

Common: `max_seq_length = 8192`, LoRA r=32, α=32, dropout 0, seed 3407,
`train_on_responses_only`, loss only on the (short) response → the truncation pathology is
gone by construction. At 8K on 40 GB there is headroom: 16-bit LoRA fits with
`per_device_train_batch_size = 2–4` (vs 1 today), gradient accumulation to effective batch
16. lr 2e-4, 2 epochs, adamw_8bit, linear schedule — unchanged from the current notebook.

Model-specific (all three quirks already documented in `datalog.md`):

| model | notes |
|---|---|
| `Qwen/Qwen3.5-9B` | Hybrid attention (3/4 linear-attn layers with `in_proj_qkvz`/`in_proj_ba`/`out_proj`): do **not** pass explicit `target_modules`; use Unsloth filters (`finetune_vision_layers=False`, language/attention/MLP=True). Ships `Qwen3VLProcessor` → use inner `text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)` everywhere text-only. |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Standard Qwen3 dense arch: explicit `target_modules=[q,k,v,o,gate,up,down]_proj` as in `qwen_GRPO.ipynb`. R1-style `<think>` tags optional at SFT (targets are copy-tasks; keep reasoning off for SFT, on for GRPO). |
| `google/gemma-4-E4B-it` | ~8B raw / ~4.5B effective (MatFormer + per-layer embeddings) — the cheapest of the three to train; multimodal like Qwen3.5 → freeze vision/audio towers, language layers only; watch for the same processor-vs-tokenizer gotcha. |

SFT success gates (window level, validation split): >98% valid JSON, >99% section keys drawn
from the canonical 31, >97% of emitted `lines` ranges inside the window's printed range,
line-overlap F1 as the tracked metric.

### 3.3 GRPO stage (`grpo` split; reuse `qwen_GRPO.ipynb` scaffolding)

Same windowing. Because completions are now hundreds of tokens (not 8K+), `num_generations`
can go from 4 to 8 within the same memory, which materially improves GRPO's group baseline.
All rewards verifiable in pure Python:

1. **JSON validity** (+/−, as today).
2. **Schema adherence** — only canonical keys, correct `{lines|text|empty}` form per section
   type (ThinkJSON-style multi-reward).
3. **Range validity** — `line_start ≤ start ≤ end ≤ line_end` of the window; malformed −.
4. **Line-overlap F1 vs gold** per section, mean over sections present in the union — the
   dense task reward (replaces today's substring-based `check_sections`).
5. **Abstention correctness** — F1 over `sections_absent` vs gold absent set.
6. **No-duplicate / minimal-cover penalty** — overlapping or redundant ranges penalized.
7. **Drop `format_and_language_reward_func`** (langid): output is JSON line references, not
   Indonesian prose; the reward is meaningless in this objective and only adds noise.

Keep `<think>` format rewards for DeepSeek-R1 (its native mode); for Qwen/Gemma run without
reasoning tags.

### 3.4 Deterministic assembler (pure Python, no model)

Per document: parse each window's JSON → validate → map ranges to global lines (they already
are global) → slice verbatim text from the stored `input_text` → **reduce**: union ranges per
section across windows, merging overlapping/adjacent ranges; where overlapping windows
disagree, majority vote, tie → keep (recall-biased; precision is recovered by eval, and legal
review prefers over- to under-extraction) → dedupe `text`-form values case/whitespace-exactly
→ emit the original `target_json` schema (`status, source_file, source_sha256, sections,
empty_sections`) with sections in canonical order. Downstream consumers see **no format
change** versus today.

### 3.5 Baselines (for the comparison table / paper)

| # | system | purpose |
|---|---|---|
| B0 | current whole-doc truncated SFT (Qwen3.5-9B) | the broken baseline; quantifies the damage |
| B1 | zero-shot per model, windowed prompt, no fine-tune | how much SFT buys |
| B2 | `gemma-4-E4B-it` zero/few-shot **whole-document at 128K context, inference-only** (covers ≈ p97 of docs) | the "just use long context" alternative — inference is feasible on 40 GB even though training is not |
| B3 | retrieve-then-extract on the `rag` split (hybrid BM25+dense, per-section queries) | the RAG alternative (LegalBench-RAG-style) |
| M1–M3 | windowed SFT per model | main results |
| M1'–M3' | windowed SFT + GRPO per model | does verifiable-reward RL help |

The `rag` split remains training-untouched, reserved for B3 and retrieval-index material.

### 3.6 Evaluation protocol

Document level, **post-assembly**, on the document-disjoint test splits as built:
- Per-section char-level span precision / recall / F1 (gold spans vs assembled spans), macro-F1
  over the 31 sections, micro-F1 overall.
- Exact-section match rate; `empty_sections` accuracy.
- Hallucination rate = fraction of emitted characters not present verbatim in source — must be
  **0.000** for M-systems (report it; it will not be 0 for B0/B2).
- Window level (diagnostic): JSON validity, range validity, line-overlap F1.
- Efficiency: tokens generated per document (expect ~50–100× reduction vs whole-doc JSON),
  wall-clock per document.

### 3.7 Compute budget (A100-40GB, Colab)

| stage | model | ctx | precision | est. VRAM | est. wall-clock |
|---|---|---|---|---|---|
| SFT | Qwen3.5-9B | 8K | bf16 LoRA | ~30–35 GB (bs 2) | ~20K rows × 2 ep ≈ 12–18 h |
| SFT | DS-R1-Qwen3-8B | 8K | bf16 LoRA | ~28–32 GB | similar |
| SFT | gemma-4-E4B | 8K | bf16 LoRA | ~18–24 GB (bs 4) | ~6–10 h |
| GRPO | each | 8K prompt + short completions | 4-bit + vLLM | ~35 GB | 100–300 steps, few h |
| B2 inference | gemma-4-E4B | ≤128K | 4-bit | fits (KV-cache dominated) | eval-only |

(Estimates from Unsloth's published scaling: 9B ≈ 11K ctx @ 24 GB with QLoRA; 8K bf16 LoRA on
40 GB leaves batch headroom. Verify with a 50-step smoke run before committing to full runs.)

### 3.8 Risks & mitigations

- **Sections straddling window boundaries** → 15% overlap + range-union in the assembler;
  audit boundary recall specifically in eval.
- **Line-number overhead** (~25–35% tokens) → accounted in the 5.5K-content / 8K-total budget;
  if too costly, drop to 4-digit numbering or number every line but strip leading whitespace.
- **Model emits window-relative or fabricated line numbers** → range-validity reward (GRPO)
  and hard validation in the assembler (invalid ranges dropped, counted as misses, surfaced as
  a metric).
- **Duplicate/conflicting spans from overlapping windows** → deterministic merge rules
  (§3.4), unit-tested on gold: assembling *gold* window targets must reproduce `target_json`
  exactly — this round-trip test is the correctness gate for the whole data pipeline.
- **Gemma/Qwen multimodal processors** → inner-tokenizer pattern from `datalog.md` applied
  from day one.
- **Distribution shift vs raw `.txt`** — `input_text` is span-reconstructed, not raw OCR. Same
  caveat as today (unchanged by Plan B); flag for future work: re-run windowing on true raw
  text using the offsets in `LLM-aggregator` outputs.

### 3.9 Execution order

1. `build_dataset_planb.py` + round-trip assembler test (gold windows → exact `target_json`).
2. Push `planb` configs to the HF repo alongside existing ones.
3. Adapt SFT notebook for Qwen3.5-9B → smoke run → full run → window+doc eval.
4. Repeat SFT for DeepSeek-R1-0528-Qwen3-8B, then gemma-4-E4B-it.
5. GRPO on top of each SFT checkpoint.
6. Baselines B0–B3; assemble the comparison table.

---

## 4. References

**Long-document processing / memory**
- LLM×MapReduce: Simplified Long-Sequence Processing (ACL 2025) — https://aclanthology.org/2025.acl-long.1341.pdf
- Long Context vs. RAG for LLMs: An Evaluation and Revisits — https://arxiv.org/html/2501.01880v1
- LoRA-FA: Memory-Efficient Low-Rank Fine-tuning — https://arxiv.org/pdf/2308.03303
- Unsloth offloaded gradient checkpointing (context-length measurements) — https://unsloth.ai/blog/long-context ; https://unsloth.ai/blog/gemma2
- Extending LM Context to 3M Tokens on a Single GPU (inference) — https://arxiv.org/html/2502.08910v1
- ChuLo: Chunk-Level Key Information Representation for Long Documents — https://arxiv.org/pdf/2410.11119

**Structured output / RL**
- Think Inside the JSON: RL for Strict Schema Adherence — https://arxiv.org/abs/2502.14905
- ThinkJSON: Multi-Reward GRPO for JSON Schema Adherence in SLMs — https://openreview.net/forum?id=J0j01U1D2c
- RL-Struct: Lightweight RL for Reliable Structured Output — https://arxiv.org/pdf/2512.00319

**Legal NLP**
- LegalBench-RAG: RAG Benchmark for the Legal Domain — https://arxiv.org/pdf/2408.10343
- AQgR: Question-guided Retrieval of Indian Case Law (RAG + structured summaries) — https://arxiv.org/pdf/2508.04710
- LLMs for Judicial Entity Extraction — https://arxiv.org/pdf/2407.05786
- COLIEE 2025 (incl. Rationale Extraction pilot; NOWJ system paper) — https://arxiv.org/pdf/2509.08025
- IndoLER: Indonesian Legal Entity Recognition dataset (IJECE) — https://ijece.iaescore.com/index.php/IJECE/article/view/35275
- Extracting legal entities from Indonesian judicial decisions (MethodsX) — https://www.sciencedirect.com/science/article/pii/S2215016125004546

**Models**
- google/gemma-4-E4B-it (128K ctx, ~4.5B effective / ~8B raw, MatFormer) — https://huggingface.co/google/gemma-4-E4B-it ; https://ai.google.dev/gemma/docs/core
