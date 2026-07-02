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

   > See [Stage 1 → **SFT data format**](#sft-data-format-conversation-style-single-turn-prompt-masked)
   > for the exact row schema, a worked example, the rendered chat template, and the
   > instruction-vs-conversation rationale.

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

### SFT data format: conversation-style, single-turn, prompt-masked

**Verdict (from the actual data, not assumed):** the SFT rows are **conversation / chat
format** — the OpenAI-`messages` / ShareGPT style — **not** the Alpaca / Self-Instruct
*instruction-triple* style (`{"instruction","input","output"}`). Each row is a single
`messages` array of role-tagged turns: `system` (the fixed Indonesian extraction
instruction), `user` (the reconstructed putusan body), `assistant` (the gold
pretty-printed 31-key JSON). It is **single-turn** (one user turn, one assistant turn)
and trained **prompt-masked** — loss is computed **only on the assistant JSON**. Source
of truth: `notebooks/build_dataset.py` → `make_sft_row()` (emits the `messages` + `meta`
row) and `notebooks/Qwen3_5_(4B).ipynb`, which renders each row with
`tokenizer.apply_chat_template(...)` and trains with `SFTTrainer` +
`train_on_responses_only`.

This mirrors the format that top-venue instruction-tuning work converged on. LIMA
demonstrates that a small, high-quality set of **role-delimited chat exchanges** (with an
explicit end-of-turn token) is enough to teach a pretrained model to follow a target
response format — *LIMA, NeurIPS 2023*. The Tülu study ("How Far Can Camels Go?")
standardizes open instruction data into a **chat/`messages` schema** and shows the format
and template materially affect downstream quality — *NeurIPS 2023 Datasets & Benchmarks*.
The older **instruction-triple** lineage (Self-Instruct, ACL 2023; the Flan Collection,
ICML 2023) is the alternative we deliberately do **not** use.

| Format | Row shape | Lineage | Used here? |
|--------|-----------|---------|-----------|
| Instruction-triple | flat `instruction` / `input` / `output` fields | Self-Instruct (ACL 2023), Alpaca, Flan Collection (ICML 2023) | **No** |
| **Conversation / chat** | `messages: [{role, content}, …]` + a chat template | LIMA (NeurIPS 2023), Tülu / "How Far Can Camels Go?" (NeurIPS 2023 D&B), Zephyr | **Yes** |

**Why conversation, not instruction-triple, for this project:**

- **Role turns are reused verbatim by GRPO.** Stage 2's `make_grpo_row()` takes the same
  `system` + `user` turns as its `prompt` and scores completions against the gold JSON.
  One chat schema serves both the SFT target and the GRPO verifiable prompt — a flat
  instruction-triple would have to be re-templated for RL.
- **A persistent system instruction** carries the "emit exactly 31 verbatim-span keys,
  never paraphrase" contract as a first-class `system` turn (`SYSTEM_PROMPT` in
  `build_dataset.py`), which is where extraction/anti-hallucination constraints belong.
- **Native template alignment.** `unsloth/Qwen3.5-9B` ships a ChatML template; feeding
  `messages` through `apply_chat_template` matches the base model's pretraining/finetune
  format exactly and lets Unsloth's `train_on_responses_only` locate the assistant span.

**Worked example — one real (abbreviated) `sft.jsonl` row.** Built from
`LLM-aggregator/TPPO/GPT/output/50_Pid.Sus_2024_PN_Psr.json` (long spans elided as `…`;
this is exactly the object `make_sft_row()` writes — a `messages` array plus provenance
`meta`):

```jsonl
{"messages":[{"role":"system","content":"Anda adalah pengekstrak terstruktur putusan pengadilan Indonesia. Diberikan badan teks putusan, keluarkan SATU objek JSON dengan tepat 31 kunci bagian (dalam urutan kanonik). Setiap nilai adalah daftar kutipan verbatim (extractive) … Kunci bagian, dalam urutan: judul, nomor_putusan, irah_irah, …, tanda_tangan_majelis."},{"role":"user","content":"P U T U S A N\n\nNomor 50/Pid.sus/2024/PN Psr\n\nDEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA\n\nPengadilan Negeri Pasuruan\n\n… Terdakwa NOVAL ANDRIAN Bin SUDIRMAN SITUMORANG … [dakwaan …] … [pertimbangan_hukum …] … [amar_putusan …]"},{"role":"assistant","content":"{\n  \"status\": \"completed\",\n  \"source_file\": \"50_Pid.Sus_2024_PN_Psr.txt\",\n  \"source_sha256\": \"20763799…d7f8e\",\n  \"sections\": {\n    \"judul\": [\"P U T U S A N\"],\n    \"nomor_putusan\": [\"Nomor 50/Pid.sus/2024/PN Psr\"],\n    \"irah_irah\": [\"DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA\"],\n    \"nama_pengadilan_negeri\": [\"Pengadilan Negeri Pasuruan\"],\n    …,\n    \"dakwaan\": [\"…\"],\n    \"saksi\": [\"…\"],\n    \"ahli\": [],\n    …,\n    \"amar_putusan\": [\"…\"],\n    …,\n    \"tanda_tangan_majelis\": [\"…\"]\n  },\n  \"empty_sections\": [\"ahli\"]\n}"}],"meta":{"corpus":"TPPO","model":"GPT","source_sha256":"20763799…d7f8e","source_file":"50_Pid.Sus_2024_PN_Psr.txt","also_extracted_by":[]}}
```

Notes: every `sections` value is a **list of verbatim spans** (many are multi-thousand-token
— e.g. `saksi` ≈ 36k chars, `dakwaan` ≈ 16k in this file); truly-empty keys carry `[]` and
are also listed in `empty_sections` (here `["ahli"]`); `meta` is provenance only and is
**not** fed to the model.

**Rendered chat template (what the model actually sees).** `apply_chat_template` flattens
the `messages` into one Qwen ChatML string; `train_on_responses_only` then masks the loss
so only the assistant JSON contributes gradient:

```text
<|im_start|>system
Anda adalah pengekstrak terstruktur putusan … tanda_tangan_majelis.<|im_end|>   ┐
<|im_start|>user                                                                │ loss MASKED
P U T U S A N … Nomor 50/Pid.sus/2024/PN Psr … [amar_putusan …]<|im_end|>        │ (prompt tokens)
<|im_start|>assistant                                                           ┘
{                                                                               ┐
  "status": "completed", … "empty_sections": ["ahli"]                           │ loss SUPERVISED
}<|im_end|>                                                                      ┘ (assistant JSON only)
```

**Loss masking + caveat.** We train on responses only (mask the long putusan prompt).
*Instruction Tuning With Loss Over Instructions* (NeurIPS 2024) reports that prompt-masking
can **underperform** when inputs are long and outputs are short — but here the assistant
JSON target is itself long (all 31 sections of verbatim spans), so completion-only loss is
the correct default. Treat "extend the loss over the input" as a tuning knob to revisit
only if extraction quality plateaus.

**Goal:** reliably emit valid 31-key JSON with verbatim spans given a putusan body.

---

## Stage 2 — GRPO RL (verifiable extractor rewards)

**Notebook:** `notebooks/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb`, adapted to the same base
`unsloth/Qwen3.5-9B`, loading the Stage-1 SFT LoRA, training on `data/train/grpo.jsonl`.

All rewards are **verifiable** — we hold both the gold JSON and the input text, so no LLM judge
is needed. Each sampled completion is scored by the **sum** of the reward functions below
(mirroring the notebook's `match_format_exactly` / `match_format_approximately` / `check_answer`).

### What GRPO is and how it learns

**Technical version.** GRPO (Group Relative Policy Optimization; Shao et al., 2024,
DeepSeekMath, arXiv 2402.03300) is a **critic-free** policy-gradient RL algorithm. For each
prompt `q`, sample a **group** of `G` completions `{o_1 … o_G}` from the current policy and
score each with the verifiable reward `r_i` (here, the sum of the six functions below).
Unlike PPO, GRPO trains **no value/critic network**; it uses the group itself as the
baseline: the advantage is `A_i = (r_i − mean(r)) / std(r)` — each completion is judged
**relative to its siblings on the same prompt**. The policy is nudged (clipped PPO-style) to
raise the probability of above-average completions and lower below-average ones, with a **KL
penalty to the frozen SFT reference** to prevent drift. No critic → less memory, simpler, and
a natural fit for verifiable-reward tasks. Here `G = num_generations = 4`.

**Analogy version (non-technical).** Imagine coaching a law clerk to extract case data. For
each case you ask for **4 draft extractions** (the group). A fixed **rubric** — not a human
judge, just a checklist: valid JSON? all 31 fields? every quote actually in the source? —
scores the 4 drafts. You never need an expert to say "8/10 is good" in the abstract; you just
**compare the 4 drafts to each other**: those above the group average get "do more of this,"
those below get "do less of this." Repeat over many cases and the clerk drifts toward the
habits that score well. A **KL leash** keeps them from forgetting everything they learned in
"training school" (SFT) while chasing rubric points.

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

### 3.0 How RAG-Anything works (per the paper)

**Technical version.** RAG-Anything (Guo et al., 2025, arXiv 2510.12323) is a three-stage
pipeline: **universal indexing → cross-modal hybrid retrieval → knowledge-enhanced synthesis.**

1. **Universal indexing.**
   - *Multimodal Knowledge Unification:* each source `k_i` is decomposed into atomic content
     units `c_j = (t_j, x_j)` — a modality type `t_j ∈ {text, image, table, equation}` plus
     raw content `x_j` — while keeping figures grounded to captions, equations to definitions,
     tables to their narrative.
   - *Dual-graph construction:*
     - **Cross-modal KG** — for each non-text unit an MLLM derives two texts: a *detailed
       description* `d_j` (for retrieval) and an *entity summary* `e_j` (for the graph),
       generated context-aware over a neighborhood window `δ`. The unit becomes an anchor
       node `v_mm^j`; its intra-chunk entities attach via `belongs_to` edges.
     - **Text KG** — standard LightRAG/GraphRAG NER + relation extraction over text chunks.
   - *Graph fusion + index:* the two graphs are merged by **entity-name alignment** into a
     unified `G = (V, E)`; all entities, relations, and chunks are embedded into table `T`;
     the retrieval index is `I = (G, T)`.
2. **Cross-modal hybrid retrieval** over `I = (G, T)`:
   - Modality-aware query encoding: detect lexical modality cues + compute query embedding `e_q`.
   - Two complementary pathways: **Structural Knowledge Navigation** (exact entity match on
     `G` then neighborhood/hop expansion → `C_stru(q)`, good for multi-hop relations) and
     **Semantic Similarity Matching** (dense top-k over `T` → `C_seman(q)`).
   - **Candidate pool unification** `C = C_stru ∪ C_seman`, then **multi-signal fusion scoring**
     (graph structural importance + embedding similarity + modality preference) → ranked `C*(q)`.
3. **Synthesis:** build textual context `P(q)` (concatenate entity summaries + relation
   descriptions + chunk contents with modality delimiters), recover visual content `V*(q)`,
   then `Response = VLM(q, P(q), V*(q))`.

The four LightRAG **dual-level** retrieval modes are the knobs on step 2: *low-level keys*
(concrete entities) → `local`; *high-level keys* (abstract themes) → `global`; both →
`hybrid`; plain vector search with no graph → `naive`.

**How we specialize it (text-only).** Our extractor already outputs clean 31-section JSON —
no images, tables, or equations — so the cross-modal KG, MLLM unit descriptions, and VLM
visual dereferencing are **no-ops**. We run the **text-KG branch only (= LightRAG)**.
Retrieval still uses **both** pathways — structural navigation over the legal entity/relation
graph *and* semantic vector matching, fused — and synthesis uses a **text LLM (Claude)**
instead of a VLM. So we inherit RAG-Anything's graph+vector hybrid retrieval, not its
multimodal front-end.

**Analogy version (non-technical).** Picture a **courthouse records room**:

- *Indexing* = a clerk reads every decision and builds **two** finding aids: (1) a wall of
  index cards cross-linked "Judge X sentenced Defendant Y", "this case cites Article Z" — a
  web of who-relates-to-whom (**the knowledge graph**); and (2) a Google-style "find similar
  wording" search (**the vector index**). RAG-Anything builds both.
- For documents with charts/photos/tables, the clerk also *describes each picture in words*
  and pins that description beside the paragraph it came from, so images become searchable.
  Our legal texts have no pictures, so this step is simply skipped.
- *Answering a question* = **two librarians work at once**: one **follows the cross-links**
  ("start at this judge → their cases → the articles cited" — best for multi-step
  relationship questions), the other does a **fuzzy meaning search** ("find passages that
  sound like the question"). Their finds are pooled, de-duplicated, ranked, and handed to a
  **writer (the LLM)** who composes a cited answer.
- The four **modes** = how wide the librarian looks: `local` = one case / one person;
  `global` = trends across the whole archive; `hybrid` = both (default); `naive` = skip the
  card wall, just fuzzy-search.

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
- **GRPO / DeepSeekMath** — [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization: critic-free, group-relative-advantage RL used in Stage 2.
- **LegalBench** — [NeurIPS 2023 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/89e44582fd28ddfea1ea4dcb0ebbf4b0-Paper-Datasets_and_Benchmarks.pdf) — extraction F1 + normalization protocol.
- **CRAG: Comprehensive RAG Benchmark** — [NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1435d2d0fca85a84d83ddcb754f58c29-Abstract-Datasets_and_Benchmarks_Track.html) — RAG accuracy + hallucination scoring.
- **ARES: Automated RAG Evaluation** — [arXiv 2311.09476](https://arxiv.org/abs/2311.09476) — context relevance / answer faithfulness / answer relevance + PPI.
- **RAGAS** — reference-free faithfulness / answer relevancy / context precision-recall.
- **MultiHop-RAG** — multi-hop retrieval evaluation for relation-traversal queries.
- **RAG Foundry** — [arXiv 2408.02545](https://arxiv.org/abs/2408.02545) — retained as an evaluation harness reference only.

### SFT data-format references (Stage 1)

- **LIMA: Less Is More for Alignment** — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) — small, high-quality **conversation-format** SFT with explicit end-of-turn tokens; basis for the single-turn chat schema.
- **How Far Can Camels Go? (Tülu)** — [NeurIPS 2023 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ec6413875e4ab08d7bc4d8e225263398-Abstract-Datasets_and_Benchmarks.html) — standardizes open instruction data into a chat/`messages` schema; format/template effects.
- **The Flan Collection** — [ICML 2023](https://proceedings.mlr.press/v202/longpre23a.html) — the instruction-format lineage we contrast against.
- **Self-Instruct** — [ACL 2023 / arXiv 2212.10560](https://arxiv.org/abs/2212.10560) — origin of the Alpaca-style instruction-triple format (contrast, not used).
- **Instruction Tuning With Loss Over Instructions** — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ffb43adf37b3eeaba559098bc084cc6-Abstract-Conference.html) — prompt-masking vs loss-over-instruction tradeoff; justifies our `train_on_responses_only` default and its caveat.
