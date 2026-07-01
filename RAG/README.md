# RAG over Putusan Extractions

Design notes for a Retrieval-Augmented Generation layer on top of the structured
Indonesian court-decision (`putusan`) extractions produced in this repo.

> **This is a design document only.** No code lives here yet. It records the
> architecture decision, the paper it is based on, and a concrete ingestion recipe so
> that whoever implements the RAG layer starts from a settled plan.

---

## 1. Purpose & scope

The extractor turns each court decision into one JSON file split into the **31 named
sections** defined in [`../LLM-aggregator/Putusan-schema.md`](../LLM-aggregator/Putusan-schema.md)
(`judul`, `dakwaan`, `saksi`, `amar_putusan`, …). Files live one-per-decision under:

```
LLM-aggregator/<CORPUS>/<MODEL>/output/<source-stem>.json
```

This RAG layer is designed to cover **both corpora and all four models**:

| Corpus | Description                    | Models                          |
|--------|--------------------------------|---------------------------------|
| TPPO   | Human-trafficking decisions    | GPT, Deepseek, Gemini, Qwen     |
| Anak   | Juvenile / child-case decisions| GPT, Deepseek, Gemini, Qwen     |

**Current population (snapshot, not a constraint):** GPT/TPPO ≈ 256 files; Deepseek/TPPO 1;
Gemini and Qwen pending; Anak corpus mirrors the layout. There are also ~503 raw-text
sources under `downloads/TPPO/raw-text/`. The design below is **population-agnostic** — it
discovers whatever `output/` directories exist and ingests them; new model outputs need no
design change.

---

## 2. Paper decision

The choice was between two frameworks:

- **2408.02545 — RAG Foundry**
  ([arXiv](https://arxiv.org/abs/2408.02545)) — a framework that unifies *data creation,
  model training, inference, and evaluation* for RAG. Its center of gravity is
  **fine-tuning** LLMs (Llama-3, Phi-3) for RAG and **benchmarking** configurations.
- **2510.12323 — RAG-Anything**
  ([arXiv](https://arxiv.org/abs/2510.12323)) — an all-in-one, deployable RAG framework
  (LightRAG lineage) built on **dual knowledge-graph construction + cross-modal hybrid
  retrieval** (structural graph navigation combined with semantic vector matching).

### Recommendation: **RAG-Anything (2510.12323).**

| Dimension                     | RAG Foundry (2408.02545)            | RAG-Anything (2510.12323)                 |
|-------------------------------|-------------------------------------|-------------------------------------------|
| Primary purpose               | Train & evaluate RAG models         | **Build & serve a RAG system**            |
| What you get out of the box   | Training/eval workflow, dataset gen | **Query-time retrieval + generation**     |
| Retrieval capability          | Depends on config you assemble      | **Entity/relationship graph + vectors**   |
| Fit to *pre-structured* text  | Low — its data-creation step assumes raw sources you still need to shape | **High — insert clean sections directly** |
| Compute profile               | GPU fine-tuning expected            | Inference-only; no training required      |
| Multimodal need here          | N/A                                 | Present but **not needed** (text-only)    |

**Why not RAG Foundry:** our data is already parsed and semantically sectioned, so its
data-creation and fine-tuning value largely does not apply, and it is a research/training
harness rather than a query-serving system. Adopting it would mean building the actual
retrieval service ourselves anyway.

**Why RAG-Anything:** it is a deployable retriever whose knowledge-graph retrieval maps
naturally onto legal entities and relationships (defendants, judges, charges, sentences,
restitution) that recur across decisions. We simply **skip its multimodal document
parser** — the content is already clean text — and enter at the knowledge-graph insertion
stage, feeding the 31 sections as structured knowledge units.

---

## 3. Why graph retrieval fits legal data

Each putusan already contains entities and relationships implicit in its sections:

- **Entities**: terdakwa (defendant), hakim / majelis (judges), penuntut umum
  (prosecutor), saksi (witnesses), ahli (experts), pengadilan negeri (court), articles of
  law cited, monetary amounts (fines, restitution).
- **Relationships**: *court → decided → case*, *judge → sentenced → defendant*,
  *decision → orders → restitution*, *charge → based-on → article*.

Plain single-vector RAG answers "find the passage most similar to my query," which is
weak for questions that require **connecting facts across many decisions**. A knowledge
graph handles those directly. Examples of questions that motivate the graph:

- "Which decisions ordered **restitution above IDR X**?"
- "List decisions **decided by judge Y**."
- "How many **TPPO convictions cited article Z**, and what sentences resulted?"
- "Which courts appear most often, and what is the sentence distribution per court?"

These are aggregation/traversal questions over entities — the graph's strength — with the
per-section text still available for grounded, quoted answers.

---

## 4. Ingestion recipe (schema-driven)

Conceptual pipeline — **no code**, meant to guide implementation.

**Discover & load.** Walk every `LLM-aggregator/<CORPUS>/<MODEL>/output/*.json`. Each file
is one extraction object with `status`, `source_file`, `source_path`, `source_sha256`,
`method`, `sections`, and `empty_sections`. Skip files whose `status` is not `completed`.

**Document identity & dedup.** Key each decision by `source_sha256` (fall back to
`source_file`). The *same* decision may be extracted by multiple models — treat those as
one logical document and keep `model` + `corpus` as **provenance metadata**, not as
separate documents. (See §8 for using cross-model agreement as a confidence signal.)

**Map the 31 sections → knowledge units.** Read the section keys *from the file* rather
than hard-coding them, so the Anak corpus's schema variant works without a code change.
Two roles:

- **Identity / header sections → entity attributes.** `judul`, `nomor_putusan`,
  `irah_irah`, `nama_pengadilan_negeri`, `keterangan_perkara`, and defendant-identity
  fields (`nama_lengkap`, `tempat_lahir`, `umur_tanggal_lahir`, `jenis_kelamin`,
  `kebangsaan`, `tempat_tinggal`, `agama`, `pekerjaan`), plus the closing block (`hari`,
  `tanggal`, `tahun`, `siapa_yang_memutus`, `panitera_pengganti`, `tanda_tangan_majelis`).
  These populate the case/defendant/judge/court **entities**.
- **Narrative sections → retrievable chunks.** `penangkapan`, `penahanan`, `tuntutan`,
  `dakwaan`, `saksi`, `ahli`, `terdakwa`, `surat`, `petunjuk_barang_bukti`, `fakta_hukum`,
  `pertimbangan_hukum`, `amar_putusan`. These are inserted as text for graph extraction
  and semantic retrieval, each tagged with its `section_key`.

**Bypass the multimodal parser.** RAG-Anything normally parses raw PDFs/images. Here the
content is already clean, section-labeled text, so enter at the **knowledge-graph
insertion stage** and hand each section's text (its array joined) straight in.

**Metadata on every unit.** Carry at minimum:

```
corpus            e.g. "TPPO" | "Anak"
model             e.g. "GPT" | "Deepseek" | "Gemini" | "Qwen"
source_file       e.g. "10_Pid.Sus_2025_PN_End.txt"
source_sha256     content hash (dedup key)
nomor_putusan     e.g. "Nomor 10/Pid.Sus/2025/PN End"
nama_pengadilan   e.g. "Pengadilan Negeri Ende"
section_key       e.g. "amar_putusan"
```

This lets retrieval filter by corpus/model/court and lets every generated answer cite the
exact decision and section it came from.

---

## 5. Retrieval modes

RAG-Anything / LightRAG expose several query modes. Map putusan question types onto them:

| Mode      | What it does                                   | Best for                                                        |
|-----------|------------------------------------------------|-----------------------------------------------------------------|
| `local`   | Entity-centric neighborhood retrieval          | "Tell me about defendant/decision X" — facts about one entity   |
| `global`  | Theme/community-level graph reasoning          | "Trends across TPPO convictions" — aggregation across the corpus|
| `hybrid`  | Combines local + global                        | Default for most legal questions spanning one case and context  |
| `naive`   | Plain vector similarity (no graph)             | Fallback / baseline; quoting a passage by wording               |

Default to **`hybrid`**; use `global` for corpus-wide aggregation and `local` for
single-case lookups.

---

## 6. Query examples

| Question                                                              | Mode     |
|-----------------------------------------------------------------------|----------|
| "Summarize the charges and verdict in decision 10/Pid.Sus/2025/PN End."| `local`  |
| "Which decisions ordered restitution, and how much?"                  | `global` |
| "What sentences did Pengadilan Negeri Ende hand down in TPPO cases?"   | `hybrid` |
| "Find decisions where a defense witness (a de charge) testified."     | `hybrid` |
| "Quote the exact `amar putusan` wording about the 14-day restitution deadline." | `naive` |
| "Compare sentencing between TPPO and Anak corpora."                   | `global` |

---

## 7. Setup outline

High-level only — no code, no config files committed here yet.

1. **Environment.** Create an isolated env; install RAG-Anything (LightRAG core). Multimodal
   parser dependencies are **not** required for this text-only ingestion.
2. **Backends.**
   - *Generation LLM:* default to a current Claude model (e.g. `claude-opus-4-8` for
     answer synthesis, a smaller Claude for cheap graph-extraction steps), consistent with
     the rest of the repo.
   - *Embeddings:* any supported text-embedding model; pick one and keep it fixed so the
     vector index stays consistent.
3. **Build the graph.** Run the §4 ingestion over the `output/` directories to populate the
   knowledge graph + vector store. Persist the store outside `RAG/` (e.g. a gitignored
   `RAG/store/`) so it can be rebuilt.
4. **Query.** Expose a small query entry point using the §5 modes.
5. **Rebuild trigger.** Re-run ingestion when new model outputs land; keying by
   `source_sha256` keeps re-ingestion idempotent.

---

## 8. Open items / future work

- **Populate the other models.** Deepseek/Gemini/Qwen TPPO and all Anak outputs are largely
  empty; the graph grows automatically as they fill in.
- **Cross-model agreement as confidence.** When ≥2 models extract the same section
  identically for one `source_sha256`, treat that as higher-confidence; divergence flags a
  section worth review. Could weight retrieval or surface as metadata.
- **Confirm the Anak schema.** Verify whether the child-case corpus uses the same 31 keys
  or a variant before ingesting; the schema-driven loader tolerates differences but the
  entity/narrative split in §4 may need per-corpus tuning.
- **Raw-text fallback.** Decide whether to also index the ~503 raw-text sources under
  `downloads/TPPO/raw-text/` as fallback context for sections the extractor left empty
  (`empty_sections`).
- **Evaluation.** If retrieval quality needs formal measurement later, RAG Foundry
  (2408.02545) remains useful *as an evaluation harness only* — separate from this serving
  layer.

---

## References

- RAG-Anything: All-in-One RAG Framework — https://arxiv.org/abs/2510.12323 *(chosen)*
- RAG Foundry: A Framework for Enhancing LLMs for RAG — https://arxiv.org/abs/2408.02545
- Section schema: [`../LLM-aggregator/Putusan-schema.md`](../LLM-aggregator/Putusan-schema.md)

