# Orchestration — Putusan Extractor: SFT → GRPO → RAG-Anything Relation Serving → Benchmark → Feedback Flywheel

End-to-end plan for turning the repo's extracted court-decision JSON into a fine-tuned
**structured extractor** (`Qwen/Qwen3.5-9B`), served behind a **RAG-Anything / LightRAG**
relation-retrieval layer, gated by a **pre-serve benchmark**, and continuously improved by a
**post-RL production feedback flywheel**.

> Companion design doc: [`README.md`](./README.md) (LightRAG ingestion recipe §4 + query modes §5).
> Section schema: [`../LLM-aggregator/Putusan-schema.md`](../LLM-aggregator/Putusan-schema.md).

---

## 0. Ground truth: what the data is

Each decision is extracted into one JSON file:

```
LLM-aggregator/<CORPUS>/<MODEL>/output/<stem>.json
```

Fields: `status`, `source_file`, `source_path`, `source_sha256`, `method`/`model`,
`empty_sections`, and `sections` — an object of **exactly 31 keys**, each a **list of verbatim
extractive spans** copied from the source (never paraphrased). Canonical section order (used
throughout for input reconstruction):

```
judul, nomor_putusan, irah_irah, nama_pengadilan_negeri, keterangan_perkara,
nama_lengkap, tempat_lahir, umur_tanggal_lahir, jenis_kelamin, kebangsaan,
tempat_tinggal, agama, pekerjaan, penangkapan, penahanan, tuntutan, dakwaan,
saksi, ahli, terdakwa, surat, petunjuk_barang_bukti, fakta_hukum,
pertimbangan_hukum, amar_putusan, hari, tanggal, tahun, siapa_yang_memutus,
panitera_pengganti, tanda_tangan_majelis
```

**Corpora / model inventory (Gemini dropped — no usable output):**

| Corpus | GPT | Deepseek | Qwen | Gemini |
|--------|-----|----------|------|--------|
| TPPO   | 329 | 1        | 0    | — (dropped) |
| Anak   | 500 | 44       | 35   | — (dropped) |

Raw sources (500 each): `downloads/TPPO/raw-text/`, `downloads/kasus anak/raw-text/`.

### Hard constraint

**Training reads only the extracted `/output/*.json` — never the raw `.txt`.**
Resolution: the model *input* is **reconstructed from the JSON's own section spans** (Stage 0),
so no `.txt` file is ever opened during training. The JSON is simultaneously the supervision
target and the source of the input text.

### Locked decisions

- **Model role:** structured section extractor (putusan body → 31-section JSON).
- **Data split:** dedup by `source_sha256`, then **70 / 15 / 15** → SFT / GRPO / **frozen benchmark**.
- **Base model:** `unsloth/Qwen3.5-9B` for **both** SFT and GRPO.

---

## Pipeline at a glance

```
 output/*.json ──(Stage 0: reconstruct input from spans, dedup, split)──► data/train/{sft,grpo,benchmark}.jsonl
                                                                                 │
                          ┌──────────────────────────────────────────────────────┤
                          ▼ (Stage 1 SFT)                                          ▼ (Stage 2 GRPO, verifiable rewards)
                 qwen_extractor_sft_lora ───────────────────────────────► qwen_extractor_grpo_lora ──► merge 16-bit
                                                                                 │
                                    (Stage 4: benchmark on FROZEN split — gate)  │
                                                                                 ▼
   incoming putusan ──► extractor (merged model) ──► 31-section JSON ──► LightRAG dual KG (Stage 3)
                                                                                 │
                          user query ──► relation/graph retrieval (hybrid|global|local|naive) ──► grounded, cited answer
                                                                                 │
                                                 👍/👎/comment/correction ◄───────┘
                                                                                 ▼
                              (Stage 5) triage → DPO (thumbs) + SFT/GRPO refresh (corrections) → canary → promote
```

---

## Stage 0 — Data foundation

**Script:** `scripts/build_dataset.py` (new). Reads only `output/*.json`, skips Gemini and any
file with `status != "completed"`.

1. **Dedup by `source_sha256`.** The same decision extracted by multiple models is **one logical
   document**; `model`/`corpus` are kept as provenance metadata. When >1 model extracted a doc,
   prefer the most complete extraction (fewest `empty_sections`); record cross-model agreement as
   an optional confidence tag.
2. **Reconstruct the training INPUT from JSON (no raw txt).** Join all non-empty section spans in
   canonical order into one text blob — a `.txt`-free reconstruction of the decision body. This
   is the model input.
3. **Target = the `sections` JSON** (all 31 keys) + `empty_sections`. `source_file` /
   `source_sha256` are the record id.
4. **Split by `source_sha256`, per corpus, 70/15/15** → `sft` / `grpo` / `benchmark`. The
   benchmark split is **frozen** and never used for SFT or GRPO. Emit
   `data/train/{sft,grpo,benchmark}.jsonl`.
5. **Row formats:**
   - SFT: chat messages — `system` = Indonesian extraction instruction, `user` = reconstructed
     input, `assistant` = pretty-printed JSON.
   - GRPO: `prompt` + gold `answer` (the JSON) for verifiable reward scoring.

**Invariants (asserted at build time):** zero `source_sha256` overlap across splits; every span
in every target is a substring of its reconstructed input.

---

## Stage 1 — SFT (language-only extractor)

**Notebook:** `notebooks/Qwen3_5_(4B)_Vision.ipynb`, converted from a vision notebook to
language-only SFT.

- Replace `FastVisionModel` → `FastLanguageModel`; remove `UnslothVisionDataCollator`, image
  columns, and `finetune_vision_layers`. Use the standard text `SFTTrainer` path.
- `from_pretrained("unsloth/Qwen3.5-9B", ...)` (was `Qwen3.5-4B`); 16-bit LoRA (or 4-bit QLoRA
  if VRAM-bound), `r = 32`, `lora_alpha = 32`.
- `train_dataset = data/train/sft.jsonl`, formatted via `apply_chat_template`. Drop the LaTeX-OCR
  dataset entirely.
- `num_train_epochs = 1–3`; set `max_length` to the 90th-percentile reconstructed-input length
  (putusan bodies are long — measure; expect ~4k–8k tokens).
- Save `qwen_extractor_sft_lora` for Stage 2 to continue from.

**Goal:** reliably emit valid 31-key JSON with verbatim spans given a putusan body.

---

## Stage 2 — GRPO RL (verifiable extractor rewards)

**Notebook:** `notebooks/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb`, adapted to the same base
`unsloth/Qwen3.5-9B`, loading the Stage-1 SFT LoRA, training on `data/train/grpo.jsonl`.

All rewards are **verifiable** — we hold both the gold JSON and the input text, so no LLM judge
is needed. Each sampled completion is scored by the **sum** of the reward functions below
(mirroring the notebook's `match_format_exactly` / `match_format_approximately` / `check_answer`).

| # | Reward | Signal |
|---|--------|--------|
| 1 | Parseable JSON | `+3.0` clean single-object parse; `+1.0` near-JSON (repairable trailing comma/brace); `-4.0` unparseable or prose leaked outside the object |
| 2 | Schema (31 keys) | `+2.0` exact key set (no missing, no extra); `-0.5` per missing key and `-0.5` per extra key (capped); `+1.0` if top-level shape (`status`, `source_file`, `source_sha256`, `sections`, `empty_sections`) is correct |
| 3 | **Extractive faithfulness (anti-hallucination)** | `+4.0 ×` fraction of emitted spans that are **verbatim substrings of the input**; `-3.0 ×` fraction of spans **not** found in the source |
| 4 | Span accuracy vs gold | `+5.0 ×` mean per-section F1 under LegalBench normalization (lowercase, collapse whitespace, strip punctuation); `+3.5` whitespace-only near-miss; `+1.5–2.0` boundary-overlap partial credit scaled by overlap ratio |
| 5 | `empty_sections` correctness | `+2.0` if reported set exactly equals the truly-empty key set; `-1.0` per inconsistency |
| 6 | No padding | `-2.0 ×` excess-length ratio vs gold (stops "grab the whole document" cheating) |

**Optional language reward:** keep the notebook's `langid` Indonesian check **on the `<think>`
reasoning block only**, never on the JSON payload.

**Config:** unchanged in spirit — `GRPOConfig`, `num_generations = 4`, vLLM sampling
(`min_p = 0.1`). Reward **#3 is the core RL signal**: extractive spans must literally appear in
the input, a cheap per-span boolean that directly optimizes "no fabricated legal text" and is
resistant to gaming.

**Optional strict mode:** any hallucinated span (a span not in the source) caps the whole
completion's reward at `0` — zero-tolerance variant for high-stakes legal use.

Save `qwen_extractor_grpo_lora`, then merge to 16-bit for serving.

---

## Stage 3 — RAG-Anything relation serving

RAG-Anything / LightRAG is **not yet installed** — it must be added (`raganything` / LightRAG
core; the multimodal parser is skipped since our text is already clean). Implements
[`README.md`](./README.md) §4–§5.

**Scripts (new):** `RAG/ingest.py`, `RAG/serve.py`.

1. **Ingest** each `output/*.json` section into the LightRAG **dual knowledge graph**:
   - **Entities:** terdakwa (defendant), hakim/majelis (judges), penuntut umum (prosecutor),
     saksi, ahli, pengadilan negeri (court), cited law articles, monetary amounts.
   - **Relations:** court → decided → case, judge → sentenced → defendant, decision → orders →
     restitution, charge → based-on → article.
   - Enter at the **KG-insertion stage** (bypass the multimodal parser); feed each section's
     joined text tagged with `section_key` + metadata (`corpus`, `model`, `source_sha256`,
     `nomor_putusan`, `nama_pengadilan`). Persist to gitignored `RAG/store/`; key by
     `source_sha256` for idempotent re-ingest.
2. **Serve loop:** incoming putusan → **fine-tuned Qwen3.5-9B extractor** produces the 31-section
   JSON → ingest into the graph → answer user queries via LightRAG **relation/graph modes**
   (`QueryParam(mode=...)`):

   | Mode | Use |
   |------|-----|
   | `hybrid` | **Default** — most legal questions spanning one case + context |
   | `global` | Corpus-wide aggregation ("trends across TPPO convictions") |
   | `local`  | Single-case / single-entity lookups |
   | `naive`  | Vector-only fallback / verbatim passage quoting |

   The graph relation traversal (`local`/`global`/`hybrid`) **is the "relation method."**
3. **Answer synthesis LLM:** a current Claude model (per README) or the fine-tuned model itself;
   every answer cites the decision + `section_key` it came from.

---

## Stage 4 — Benchmark BEFORE serving (the gate)

**Script:** `RAG/benchmark.py` (new). Grounded in top-venue benchmarks/metrics from the last
three years. Two layers.

### A. Extraction quality — fine-tuned model on the **frozen 15% benchmark split** (gold JSON)

- **Per-section span F1 / exact-match** with LegalBench normalization (lowercase, collapse
  whitespace, strip punctuation) — *LegalBench, NeurIPS 2023 Datasets & Benchmarks*.
- **JSON validity rate**, **hallucination rate** (spans not verbatim in source),
  **`empty_sections` accuracy**. Report macro over the 31 sections + per-section breakdown.

### B. End-to-end RAG quality — QA set over frozen docs (incl. cross-doc relation queries)

- **ARES** (context relevance / answer faithfulness / answer relevance, with PPI confidence
  intervals) — *arXiv 2311.09476*.
- **RAGAS** (reference-free faithfulness, answer relevancy, context precision, context recall).
- **CRAG** (accuracy + explicit **hallucination scoring** that penalizes wrong answers; question
  types: simple, conditional, comparative, aggregation, multi-hop, set, post-processing, false
  premise) — *NeurIPS 2024 Datasets & Benchmarks*.
- **MultiHop-RAG**-style multi-hop set for relation-traversal queries (judge / court / article
  joins across decisions).

**Gate:** publish the results table in this document; **promote to serving only if extraction F1
and answer faithfulness clear a preset threshold.**

---

## Stage 5 — Post-RL production feedback flywheel

The served UI collects 👍/👎, comments, and user corrections/edits per answer (as Claude/GPT do).
Feedback is converted to training data **on a schedule — never live**.

**Location (new):** `RAG/feedback/`.

1. **Log** each interaction as a JSONL event: `interaction_id`, `timestamp`, `query`,
   `retrieval_mode`, retrieved unit ids (`source_sha256` + `section_key`), model output, `rating`
   (👍/👎), `comment`, user `correction`, `model_version` / `adapter_hash`. Preserve the
   minor-redaction (`xxxx`) — this is legal PII.
2. **Triage** every 👎 into one of three bug classes, each with a different fix:
   - **Retrieval error** (wrong/missing sections) → fix the LightRAG graph (re-ingest, fix
     entity/relation extraction, adjust mode). **No weight update.**
   - **Extraction error** (JSON fields wrong) → the user-corrected JSON becomes **new verifiable
     gold**; feeds both a periodic SFT refresh and the GRPO verifiable set. Highest leverage.
   - **Generation error** (grounded but poor answer) → preference signal.
3. **Two scheduled training paths:**
   - 👍/👎 answer pairs → **DPO** (`DPOTrainer`): binary thumbs form chosen/rejected preference
     pairs — the standard production-feedback path; cheaper and more stable than re-running GRPO.
   - Corrections → **gold expansion → periodic SFT refresh + verifiable GRPO** (reuses Stage 1–2
     unchanged).
4. **Guardrails (mandatory):**
   - Never train on raw thumbs directly — aggregate, dedup, spam/abuse-filter, and human-review a
     sample before anything enters training.
   - **Batch + canary/A-B:** accumulate feedback, retrain on a cadence (e.g. every N corrections),
     A/B the new adapter vs the current one on the frozen benchmark before promotion.
   - Build a rolling **"hard-cases" eval** from 👎 interactions to catch regressions — kept
     **strictly separate** from the frozen Stage-4 benchmark so feedback never leaks into headline
     metrics.

---

## Files to create / modify

| Path | Action | Purpose |
|------|--------|---------|
| `RAG/ORCHESTRATION.md` | this file | full pipeline doc |
| `scripts/build_dataset.py` | create | Stage 0 — JSON-only reconstruction, dedup, 70/15/15 split |
| `notebooks/Qwen3_5_(4B)_Vision.ipynb` | modify | Vision → language-only SFT, Qwen3.5-9B, JSON data |
| `notebooks/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb` | modify | Qwen3.5-9B, verifiable extractor rewards |
| `RAG/ingest.py`, `RAG/serve.py` | create | LightRAG dual-KG ingest + relation-mode serving |
| `RAG/benchmark.py` | create | extraction + RAG evaluation on the frozen split |
| `RAG/feedback/` | create | feedback capture, triage, and curation |
| `pyproject.toml` | update | add `raganything`/LightRAG + eval deps (ragas/ares) as an extra |

## Verification

- **Stage 0:** `data/train/{sft,grpo,benchmark}.jsonl` produced; assert zero `source_sha256`
  overlap across splits; assert every target span is a substring of its reconstructed input.
- **Stage 1/2:** notebooks run end-to-end on a small subset; SFT emits valid 31-key JSON; GRPO
  reward curve trends up; hallucination-rate reward > 0.
- **Stage 4:** `RAG/benchmark.py` prints the extraction + RAG metric tables on the frozen split.
- **Stage 5:** a simulated 👎 + correction event flows through triage into a DPO/SFT-ready record
  without touching the frozen benchmark.

## References (top venues, last 3 years)

- **RAG-Anything: All-in-One RAG Framework** — [arXiv 2510.12323](https://arxiv.org/abs/2510.12323) — chosen serving framework (LightRAG graph retrieval).
- **LegalBench** — [NeurIPS 2023 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/89e44582fd28ddfea1ea4dcb0ebbf4b0-Paper-Datasets_and_Benchmarks.pdf) — extraction F1 + normalization protocol.
- **CRAG: Comprehensive RAG Benchmark** — [NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1435d2d0fca85a84d83ddcb754f58c29-Abstract-Datasets_and_Benchmarks_Track.html) — RAG accuracy + hallucination scoring.
- **ARES: Automated RAG Evaluation** — [arXiv 2311.09476](https://arxiv.org/abs/2311.09476) — context relevance / answer faithfulness / answer relevance + PPI.
- **RAGAS** — reference-free faithfulness / answer relevancy / context precision-recall.
- **MultiHop-RAG** — multi-hop retrieval evaluation for relation-traversal queries.
- **RAG Foundry** — [arXiv 2408.02545](https://arxiv.org/abs/2408.02545) — retained as an evaluation harness reference only.
