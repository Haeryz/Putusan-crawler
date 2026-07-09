# Plan A: Span-Indexed Section Extraction for Putusan MA

## Executive Summary

The current whole-document SFT setup is not viable on the available hardware. The audit in
`notebooks/datalog.md` shows the fully templated SFT sequence lengths are:

| percentile | tokens |
|---|---:|
| p50 | 34,499 |
| p90 | 74,328 |
| p95 | 90,106 |
| max | 448,111 |

The notebook caps training at `max_seq_length = 32768`, while the assistant JSON target sits at
the end of the sequence. Since more than half of examples exceed the cap, the model often sees the
front of the prompt but loses the supervised JSON target. With response-only loss, those rows
become zero-signal or corrupt-signal training examples.

The best solution is not to raise context length. The best solution is to stop asking the model to
regenerate the entire verbatim legal document as JSON. Instead, train the model to identify section
spans with compact references, then use a deterministic assembler to copy the exact source text
into the final 31-section JSON.

This keeps the output schema the same, removes the truncation failure, fits A100 40GB, and makes
hallucinated legal text impossible after assembly because final spans are copied from the source.

## Research Basis

### AI / LLM Technical Basis

1. **Retrieval-Augmented Generation, NeurIPS 2020**
   - Lewis et al. show that external retrieval improves factuality and provenance for
     knowledge-intensive tasks.
   - Relevance here: legal extraction should not depend on the model memorizing or regenerating
     long source text. The model should operate over retrieved/bounded evidence windows.

2. **QLoRA, NeurIPS 2023**
   - Dettmers et al. show that 4-bit quantized LoRA can fine-tune large models with much lower
     memory while preserving strong performance.
   - Relevance here: A100 40GB is enough for <10B QLoRA, but not for naive long-context 9B
     whole-document SFT at 74k-90k token lengths.

3. **LongLoRA / efficient long-context fine-tuning**
   - Long-context fine-tuning can extend context, but it still has high memory and optimization
     cost.
   - Relevance here: even if 64k or 100k context were possible, the target JSON can be hundreds of
     thousands of characters. Context extension does not solve the “regenerate all verbatim spans”
     objective.

4. **Lost in the Middle, TACL 2023**
   - Liu et al. show that long-context models often underuse information placed in the middle of
     long inputs.
   - Relevance here: Indonesian court decisions contain relevant fields spread across long bodies.
     A bounded section/window approach is more reliable than forcing one huge prompt.

5. **GRPO / DeepSeekMath**
   - GRPO works well where rewards are objectively verifiable and no learned critic is needed.
   - Relevance here: extraction rewards can be computed exactly: valid JSON, valid offsets,
     copied spans, section match, empty-section correctness, and no out-of-source text.

6. **Instruction Tuning With Loss Over Instructions, NeurIPS 2024**
   - The paper warns that response-only loss can underperform in some long-input settings.
   - Relevance here: response-only loss remains the right default after conversion because the
     target becomes compact. A prompt-loss ablation can be tested later, but it is not the main
     fix.

### Legal / Law-AI Basis

1. **LegalBench, NeurIPS 2023 Datasets & Benchmarks**
   - LegalBench emphasizes task-specific legal evaluation and per-task capability reporting.
   - Relevance here: the benchmark should report per-section extraction quality, not only one
     aggregate score.

2. **COLIEE legal retrieval and entailment tasks**
   - COLIEE-style legal AI tasks emphasize retrieval, evidence selection, and legal text
     grounding.
   - Relevance here: the model should be evaluated on whether extracted spans are grounded in
     source evidence.

3. **ICAIL / JURIX-style legal information extraction work**
   - The legal AI literature generally treats legal extraction as a high-precision, evidence-bound
     task rather than open-ended generation.
   - Relevance here: deterministic verification and evidence citations are more important than
     fluent natural-language generation.

## Recommended Architecture

### Replace Whole-Document JSON Generation

Current objective:

```text
full putusan body -> full 31-section JSON containing all verbatim spans
```

Recommended objective:

```text
section key + bounded source window(s) -> compact span references
deterministic assembler -> final 31-section JSON with copied verbatim spans
```

The final external output remains:

```text
{
  "status": "completed",
  "source_file": "...",
  "source_sha256": "...",
  "sections": {
    "judul": [...],
    ...
    "tanda_tangan_majelis": [...]
  },
  "empty_sections": [...]
}
```

The model no longer emits the long legal text directly. It emits compact references to text already
present in `input_text`.

### Span Reference Format

Use a compact intermediate format:

```text
{
  "section_key": "dakwaan",
  "empty": false,
  "spans": [
    {
      "start": 12345,
      "end": 18320
    }
  ]
}
```

Offsets are preferred over generated text because:

- They are short.
- They are easy to validate.
- They prevent paraphrase.
- They make hallucination impossible after deterministic copying.
- They can be scored with exact overlap against gold offsets.

If offsets are difficult for a model, use a hybrid:

```text
{
  "section_key": "dakwaan",
  "empty": false,
  "spans": [
    {
      "start_anchor": "Menimbang, bahwa berdasarkan surat dakwaan...",
      "end_anchor": "...sebagaimana diatur dan diancam pidana"
    }
  ]
}
```

Default recommendation: start with offsets, keep anchors as fallback.

## Dataset Reshaping

The current dataset already has the right ingredients:

- `input_text`: reconstructed source body.
- `sections_json`: 31 canonical sections.
- `target_json`: full output object.
- `prompt` / `messages`: current chat format.
- `source_sha256`: document identity.
- document-disjoint splits across `sft`, `grpo`, and `rag`.

Convert each row into section-level tasks:

```text
one document row -> 31 section tasks
```

For each canonical section:

1. Read gold span list from `sections_json`.
2. Exact-match each gold span inside `input_text`.
3. Store `start` and `end` offsets.
4. If a gold section is empty, store `empty = true` and `spans = []`.
5. If a span appears more than once, choose the occurrence that preserves canonical document order.
6. If exact matching fails, mark the example for review instead of silently training on it.

The training target becomes short even when the gold legal span is very long.

### Windowing

Use bounded windows instead of full documents.

Default window policy:

- Window size: 4k-8k tokens.
- Overlap: 10-20%.
- Include document metadata in every task: corpus, source file, section key, and optional court/case number.
- For each section, select candidate windows using lexical cues and section priors.
- If a section spans multiple windows, train multiple span-reference outputs and merge them later.

Long sections that require special handling:

- `saksi`
- `dakwaan`
- `pertimbangan_hukum`
- `fakta_hukum`
- `terdakwa`
- `ahli`
- `surat`

Short identity sections can often be extracted from the first/header window:

- `judul`
- `nomor_putusan`
- `irah_irah`
- `nama_pengadilan_negeri`
- `nama_lengkap`
- `tempat_lahir`
- `umur_tanggal_lahir`
- `jenis_kelamin`
- `kebangsaan`
- `tempat_tinggal`
- `agama`
- `pekerjaan`

Closing sections can usually be extracted from the final window:

- `hari`
- `tanggal`
- `tahun`
- `siapa_yang_memutus`
- `panitera_pengganti`
- `tanda_tangan_majelis`

## Model Training Plan

The constraint is:

- GPU: A100 40GB.
- Model size: below 10B.
- Train three models.
- Training order: Qwen first, then DeepSeek, then Gemma.

### Model 1: Qwen

Primary model:

```text
Qwen/Qwen3.5-9B
```

Role:

- Main extractor candidate.
- Highest-capacity model under the <10B constraint.
- Best first model because the current notebooks already target Qwen and have tokenizer/processor notes.

Training default:

- QLoRA or memory-safe LoRA.
- Sequence length: 8k by default.
- Raise to 16k only after a token audit proves it materially reduces dropped windows.
- Use the inner text tokenizer because Qwen ships a multimodal processor.
- Freeze vision / non-text modules.
- Let Unsloth attach adapters broadly rather than using only `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### Model 2: DeepSeek

Primary model:

```text
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

Role:

- Reasoning-model comparison.
- Useful for difficult boundary decisions and section classification.

Risk:

- It may produce verbose reasoning or extra text.
- Legal extraction requires strict compact JSON, so reasoning text should not be part of the final output.

Training default:

- QLoRA.
- Same reshaped dataset as Qwen.
- Same metrics.
- Penalize any prose outside the compact JSON.

### Model 3: Gemma

Primary model:

```text
google/gemma-4-E4B-it
```

Possible alternative:

```text
google/gemma-4-E4B-it-assistant
```

Role:

- Efficient smaller-model comparison.
- Candidate for lower-cost serving if quality is close to Qwen.

Training default:

- Start with QLoRA.
- Consider 16-bit LoRA only if memory audit shows strong headroom.
- Use the same section-level span-reference task for fair comparison.

Preflight requirement:

- Verify exact tokenizer, chat template, and processor behavior before training.
- Confirm whether the base `it` model or assistant variant is more suitable for strict JSON generation.

## SFT Stage

SFT goal:

```text
Given a section key and bounded source context, emit valid compact span-reference JSON.
```

Do not train on full 31-section JSON generation.

SFT targets should be short:

```text
{
  "section_key": "amar_putusan",
  "empty": false,
  "spans": [
    {"start": 52012, "end": 54490}
  ]
}
```

SFT success criteria:

- JSON validity above 95% on validation.
- Correct `section_key` above 99%.
- Offset validity above 95%.
- Empty-section classification is stable.
- No output prose outside JSON.

## GRPO Stage

GRPO should start only after SFT produces stable compact JSON.

Use verifiable rewards only. No LLM judge.

Reward components:

| Reward | Purpose |
|---|---|
| JSON parse reward | Enforce machine-readable output |
| Exact `section_key` reward | Prevent section drift |
| Offset validity reward | Ensure `0 <= start < end <= len(input_text)` |
| Span overlap reward | Match gold offsets |
| Empty-section reward | Correctly emit `empty = true` when needed |
| No-extra-text reward | Prevent reasoning/prose leakage |
| No-duplicate-span reward | Prevent repeated extraction |
| Length sanity reward | Penalize selecting the whole document |

Core GRPO principle:

```text
Sample multiple outputs for the same section task, score each with deterministic rewards,
and update toward the outputs that are better than their siblings.
```

The reward must strongly punish:

- Invalid JSON.
- Offsets outside the source.
- `start >= end`.
- Extra prose.
- Selecting entire windows as a lazy answer.
- Empty sections predicted as non-empty.
- Non-empty sections predicted as empty.

## Deterministic Assembly

After model inference:

1. Parse compact JSON.
2. Validate offsets or anchors.
3. Copy exact substrings from `input_text`.
4. Merge adjacent or overlapping spans.
5. Deduplicate repeated spans.
6. Preserve canonical 31-section order.
7. Fill missing sections with `[]`.
8. Compute `empty_sections`.
9. Emit the original full `target_json` shape.

This assembler is responsible for legal-text faithfulness. The model is responsible only for
selecting where the evidence is.

## RAG / Retrieval Role

The `rag` split should be used for retrieval and downstream evaluation, not for training leakage.

Recommended retrieval layer:

- Chunk `input_text` by section/window.
- Store metadata: `source_sha256`, `source_file`, `corpus`, `section_key`, `annotator_model`.
- Use hybrid retrieval:
  - exact lexical cues for legal headings and section markers,
  - dense semantic retrieval for noisy or non-standard wording,
  - graph/entity retrieval later for judge, court, article, defendant, and sentence relations.

The serving path should be:

```text
incoming putusan
-> chunk / retrieve section windows
-> model emits compact span references
-> deterministic assembler copies spans
-> final 31-section JSON
-> optional RAG/graph ingestion for legal search and QA
```

Do not use RAG to hide bad extraction. Extraction quality must pass the frozen benchmark before
serving.

## Benchmark Plan

Evaluate on held-out document-disjoint splits only.

### Extraction Metrics

Report:

- JSON validity rate.
- Offset validity rate.
- Exact span match.
- Character-level span F1.
- Per-section precision, recall, and F1.
- Macro average over all 31 sections.
- Empty-section precision, recall, and F1.
- Hallucinated text rate after assembly.

Hallucinated text rate after assembly must be:

```text
0%
```

because the assembler copies only from source text.

### Per-Section Reporting

Report separate results for:

- identity/header sections,
- narrative/evidence sections,
- verdict/closing sections,
- long sections,
- empty-prone sections.

Do not rely only on one aggregate metric. LegalBench-style reporting requires task-level and
capability-level breakdowns.

### Model Comparison

Compare:

1. Current whole-document SFT baseline.
2. Qwen section-level SFT.
3. Qwen section-level SFT + GRPO.
4. DeepSeek section-level SFT.
5. DeepSeek section-level SFT + GRPO.
6. Gemma section-level SFT.
7. Gemma section-level SFT + GRPO.

Promotion criterion:

- Prefer the model with the best long-section F1 and lowest invalid-output rate.
- If Qwen and Gemma are close, prefer Gemma for serving efficiency.
- If DeepSeek produces extra reasoning/prose, do not serve it even if some F1 metrics are strong.

## Acceptance Criteria

The solution is acceptable only if:

- The training target is no longer truncated away.
- Median and p90 training tasks fit comfortably within the chosen sequence length.
- The final public output schema remains unchanged.
- Model-generated legal text is not trusted directly.
- Final legal spans are copied from `input_text`.
- Held-out evaluation reports per-section metrics.
- The frozen evaluation split is not used in SFT or GRPO.
- Qwen, DeepSeek, and Gemma are trained/evaluated under the same reshaped task.

## Why This Is the Best Option

Raising context length tries to make the broken setup bigger. It does not address the fact that the
model is being trained to reproduce enormous verbatim legal spans, which is inefficient,
memory-heavy, and unsafe.

Span-indexed extraction changes the problem into the form the hardware and legal task actually
support:

- The model performs judgment: which source span belongs to which legal section.
- The system performs copying: exact legal text is copied deterministically.
- Evaluation is verifiable.
- Hallucination is structurally prevented.
- A100 40GB becomes enough for all three <10B models.

This should be Plan A.
