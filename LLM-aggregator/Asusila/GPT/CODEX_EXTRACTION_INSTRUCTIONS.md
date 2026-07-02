# Codex Asusila (Pidana Biasa) Extraction Instructions

Use this file as the full task instruction.

## Token-Optimized Span Pipeline (default since 2026-06-21)

The launcher now defaults to `-Mode Span`, a token-optimized pipeline that cuts
Codex usage by roughly half with no loss of fidelity (measured ~57% fewer
tokens/doc vs the legacy generative loop: ~44.7k vs ~104.2k). The technique is
span/offset extraction (extractive-QA / Pointer Networks): the model emits
*pointers* into the source instead of regenerating the text, and a deterministic
post-processor slices the exact contiguous excerpts. The model still performs
every section/boundary decision; nothing is offloaded to another model.

Per source, the launcher:

1. Runs `lib/asusila_extract.py clean` to strip the repeated Mahkamah Agung
   boilerplate (disclaimer, page headers/footers, form-feeds — ~25-30% of every
   file) and number the remaining lines.
2. Sends the cleaned, line-numbered source INLINE in the prompt with the compact
   `SPAN_EXTRACTION_SPEC.md` (no file reads, no PDF, no large guides, single
   pass, low reasoning effort).
3. Codex writes ONLY a small spans JSON (`.spans/<stem>.spans.json`) of
   per-section line ranges / short literals — a few hundred output tokens
   instead of re-emitting tens of thousands.
4. Runs `lib/asusila_extract.py expand` to slice the exact excerpts, build the
   `Asusila.json`-conforming output, validate it structurally, and append the
   checkpoint. Short literals are snapped back to the exact source substring, so
   excerpts are 100% verbatim.

This also removes the boilerplate that the legacy loop used to copy into section
values, so quality improves. Measure usage anytime with
`python lib/measure_tokens.py`. Use `-Mode Legacy` to fall back to the original
generative loop (the rest of this document).

---

## Legacy generative loop (reference)

## Objective

Manually extract Indonesian ordinary-criminal (Pidana Biasa) court decision sections into individual
JSON artifacts. Each processed raw-text source must produce exactly one JSON
file under:

`LLM-aggregator/Asusila/GPT/output/<source-stem>.json`

Each output file must conform to:

`LLM-aggregator/Asusila/GPT/Asusila.json`

Follow the Asusila section schema and boundary guide in:

`LLM-aggregator/Asusila/GPT/Putusan-schema.md`

Use the Pidana Biasa official format context from:

`LLM-aggregator/Asusila/Pidana Biasa Format KKMA.pdf`

Source documents are already extracted as raw text in:

`downloads/Asusila/raw-text`

Do not waste time extracting from PDFs unless the raw text is missing or
unreadable.

## One-Click Launcher

On macOS/Linux, start or resume the automated Codex extraction loop with the
native Python launcher:

```bash
python3 run_extractions.py --corpus Asusila
```

For single-click/shell-wrapper usage on macOS/Linux, run:

```bash
./LLM-aggregator/Asusila/GPT/run-codex-extraction.sh
```

On Windows PowerShell, use:

```powershell
.\LLM-aggregator\Asusila\GPT\run-codex-extraction.ps1
```

For Windows Explorer/double-click usage, run:

```cmd
LLM-aggregator\Asusila\GPT\run-codex-extraction.cmd
```

The launcher calls `codex exec` non-interactively, gives Codex this extraction
loop, and writes outputs/checkpoints in the Asusila GPT directory. It resumes from
existing `progress.jsonl` records and `GPT/output/*.json` files, so rerunning
the launcher continues pending files.

Useful controls:

```bash
python3 run_extractions.py --corpus Asusila --status
./LLM-aggregator/Asusila/GPT/run-codex-extraction.sh --status
.\LLM-aggregator\Asusila\GPT\run-codex-extraction.ps1 -Action Prompt
python3 run_extractions.py --corpus Asusila --target 1
python3 run_extractions.py --corpus Asusila --target 10
python3 run_extractions.py --corpus Asusila --model gpt-5-codex
```

With no target, the launcher processes all pending sources one at a time until
the 5h usage guard stops it, a failure occurs, or the corpus is complete.
`-Target X` / `--target X` launches up to X new Codex sessions. Each Codex
session processes exactly one pending source file, writes/checkpoints it, then
exits. `-MaxFiles` is accepted as a backward-compatible PowerShell alias for
`-Target`; it does not mean multiple files inside one Codex session.

Guarded AFK runs are sequential so usage can be checked before every next
source. Parallel sessions are available only when the usage guard is disabled.

## Agent Loop

This workflow follows the Codex agent loop pattern: select, extract, verify,
checkpoint, and stop safely without requiring a new prompt for each document.

Process exactly one raw-text source per Codex session. Do not process multiple
source files inside a single Codex session.

Loop steps:

1. Discover pending files by comparing `downloads/Asusila/raw-text/*.txt`
   against completed records in `LLM-aggregator/Asusila/GPT/progress.jsonl` and
   existing JSON files in `LLM-aggregator/Asusila/GPT/output/`.
2. Preassign one distinct source file to each target Codex session in
   deterministic filename order.
3. Each Codex session reads only its assigned source file.
4. Extract all 31 fields into one JSON object using exact contiguous source
   excerpts.
5. Save the result as `LLM-aggregator/Asusila/GPT/output/<source-stem>.json`.
6. Verify the output against `LLM-aggregator/Asusila/GPT/Asusila.json`, including all
   31 section keys and accurate `empty_sections`.
7. Append one completed JSONL checkpoint record to
   `LLM-aggregator/Asusila/GPT/progress.jsonl`.
8. Check current Codex/session usage before starting the next file.
9. Continue with the next pending file only if the usage guard has not
   triggered.

Usage guard:

- If remaining usage is below 10% of the active five-hour reset window, stop
  before starting another source.
- If `/status` text is unavailable to the non-interactive launcher, the
  270-minute wall-clock fallback stops before starting another source.
- Do not begin a large source when usage is already under the 10% threshold.
- Create a Markdown run report under `LLM-aggregator/Asusila/GPT/reports/` before
  the final response.
- The report must include stop reason, usage remaining, reset timing if
  visible, processed count in this run, completed output paths, last source
  handled, pending count, failed or skipped sources, and recommended resume
  command or next action.

Append one JSONL checkpoint record per completed document with:

- `source_file`
- `source_path`
- `source_sha256`
- `output`
- `status`
- `method`
- `empty_sections`

## Manual Extraction Rule

This is Codex doing the extraction, not a generated extractor.

Allowed:

- Read/search raw text files to locate relevant passages.
- Use tools only to inspect text, create/edit JSON output artifacts, and update
  the checkpoint.
- Extract the text exactly as it appears in the source.
- Store long copied excerpts directly in the JSON artifact instead of trying to
  display them in chat.

Not allowed:

- Do not write a program that decides/extracts the legal sections
  automatically.
- Do not hardcode a general extractor.
- Do not use another LLM or external service to do the extraction.
- Do not summarize the text.
- Do not write reasoning inside section values.
- Do not write anything beside actual extraction into section values.

## Pidana Biasa Format Context

The Pidana Biasa PDF format file is not a source decision to extract into the
JSON output. It is a guide for how ordinary-criminal (`Pid.B`) decisions are
usually ordered and where section boundaries normally fall.

Common Pidana Biasa decision variants in the format:

- `Pid.I.A.1.1 Biasa-Vrijspraak`: acquittal.
- `Pid.I.A.1.2 (Format Biasa Lepas)`: release from all legal charges.
- `Pid.I.A.1.3 Format Biasa - Terbukti`: conviction.

Use this expected order to resolve ambiguous boundaries:

1. `P U T U S A N` / `PUTUSAN`, case number, irah-irah, court name, and case
   description (`Pengadilan Negeri … yang mengadili perkara pidana dengan acara
   pemeriksaan biasa … dalam perkara Terdakwa/Para Terdakwa`).
2. Numbered defendant identity fields: `1. Nama lengkap` through `8. Pekerjaan`.
3. Optional arrest sentence and detention history (Rumah Tahanan Negara), up to
   12 detention stages.
4. Counsel status: Penasihat Hukum appointed, no counsel, or refusal of counsel.
5. Procedural review: `Setelah membaca` (Penetapan Penunjukan Majelis Hakim,
   Penetapan Hari Sidang, case file), then the witnesses/experts/defendant/
   evidence hearing sentence.
6. Prosecution demand (`Setelah mendengar pembacaan tuntutan pidana`).
7. Defense (pembelaan), leniency request, prosecution reply, and defendant reply
   if present.
8. Dakwaan.
9. Evidence sequence: prosecution witnesses, experts, documentary/electronic
   evidence, defendant statement, defense witnesses/experts/documents,
   verbalisan witness, and prosecution goods/evidence list.
10. Facts (`diperoleh fakta hukum sebagai berikut`).
11. Legal consideration and element analysis under dakwaan forms (`DAKWAAN
   TUNGGAL/ALTERNATIF/SUBSIDAIRITAS/KUMULATIF/GABUNGAN`), including `Ad.1/Ad.2…`.
12. Evidence-disposition reasoning, aggravating/mitigating factors, costs, any
   restitution/compensation consideration where applicable, and `Mengingat...`.
13. `MENGADILI` operative orders.
14. `Demikianlah diputuskan...` closing paragraph and signature block.

## Field-Level Extraction Context

- Fields 6-13, defendant identity: extract only the value for that identity
  field. For multiple defendants, put all values in the same section array in
  document order, preserving labels such as `Terdakwa I` when present.
- Field 10, Kebangsaan: treat `Kewarganegaraan` as the same field. Stop before
  an optional `Pendidikan` line if present.
- Field 14, Penangkapan: include only arrest wording and dates. Do not include
  detention stages here.
- Field 15, Penahanan: include every detention stage and extension (Penyidik,
  Perpanjangan Penuntut Umum, Ketua Pengadilan Negeri, Hakim/Majelis Hakim,
  Ketua Pengadilan Tinggi — up to 12 stages), plus penangguhan, pembantaran,
  pengalihan penahanan, or detention in another case if present.
- Field 16, Tuntutan: extract the copied prosecution demand after `Setelah
  mendengar pembacaan tuntutan pidana... pada pokoknya sebagai berikut`.
- Field 17, Dakwaan: extract the complete charge text after `didakwa
  berdasarkan surat dakwaan Penuntut Umum... sebagai berikut`, including all
  forms (tunggal, alternatif, subsidairitas, kumulatif, gabungan).
- Field 18, Saksi: include prosecution witnesses, victims, defense witnesses
  (`saksi yang meringankan` / `a de charge`), and verbalisan witnesses when
  present, with oath status and the defendant's response to each.
- Field 19, Ahli: include prosecution and defense experts, including expert
  statements read into court.
- Field 20, Terdakwa: extract the defendant's own courtroom statement beginning
  around `Menimbang bahwa Terdakwa/Para Terdakwa* di persidangan telah
  memberikan keterangan...`.
- Field 21, Surat: include documentary and electronic evidence under `Surat
  (termasuk alat bukti elektronik)` or equivalent.
- Field 22, Petunjuk/Barang Bukti: include the submitted goods/evidence
  inventory, usually after `Penuntut Umum mengajukan barang bukti sebagai
  berikut`.
- Field 23, Fakta Hukum: starts at `diperoleh fakta hukum sebagai berikut` and
  ends before the court starts legal/element analysis.
- Field 24, Pertimbangan Hukum: includes dakwaan element analysis, conclusions
  on whether unsur are fulfilled, any alasan pembenar/pemaaf or restitution/
  compensation reasoning, detention reasoning, evidence disposition reasoning,
  aggravating/mitigating factors, costs, and `Mengingat...` up to but not
  including `MENGADILI`.
- Field 25, Amar Putusan: starts at `MENGADILI` and includes every numbered
  operative order.
- Fields 26-31, closing fields: parse from `Demikianlah diputuskan...` and the
  signature block. `Hari`, `Tanggal`, and `Tahun` refer to the deliberation
  decision date unless the task explicitly asks for pronouncement date.

## JSON Output Rules

Use this output shape:

```json
{
  "status": "completed",
  "source_file": "example.txt",
  "source_path": "downloads/Asusila/raw-text/example.txt",
  "source_sha256": "<64 hex chars>",
  "sections": {
    "judul": [],
    "nomor_putusan": [],
    "irah_irah": [],
    "nama_pengadilan_negeri": [],
    "keterangan_perkara": [],
    "nama_lengkap": [],
    "tempat_lahir": [],
    "umur_tanggal_lahir": [],
    "jenis_kelamin": [],
    "kebangsaan": [],
    "tempat_tinggal": [],
    "agama": [],
    "pekerjaan": [],
    "penangkapan": [],
    "penahanan": [],
    "tuntutan": [],
    "dakwaan": [],
    "saksi": [],
    "ahli": [],
    "terdakwa": [],
    "surat": [],
    "petunjuk_barang_bukti": [],
    "fakta_hukum": [],
    "pertimbangan_hukum": [],
    "amar_putusan": [],
    "hari": [],
    "tanggal": [],
    "tahun": [],
    "siapa_yang_memutus": [],
    "panitera_pengganti": [],
    "tanda_tangan_majelis": []
  },
  "empty_sections": [],
  "method": "codex_manual_extractive"
}
```

Every section value is an array. Put exact copied source excerpts in the array.
Use multiple array items only for multiple Terdakwa or genuinely separate
occurrences. Use `[]` only when no exact source excerpt exists after checking
the field label, schema boundaries, aliases, and OCR variants.

Do not overwrite an existing completed output unless the checkpoint is being
intentionally reset or the user asks to redo that source.

## Required Section Keys

1. Judul
2. Nomor Putusan
3. Irah-irah
4. Nama Pengadilan Negeri
5. Keterangan Perkara
6. Nama Lengkap
7. Tempat Lahir
8. Umur/Tanggal Lahir
9. Jenis Kelamin
10. Kebangsaan
11. Tempat Tinggal
12. Agama
13. Pekerjaan
14. Penangkapan
15. Penahanan
16. Tuntutan
17. Dakwaan
18. Saksi
19. Ahli
20. Terdakwa
21. Surat
22. Petunjuk/Barang Bukti
23. Fakta Hukum
24. Pertimbangan Hukum
25. Amar Putusan
26. Hari
27. Tanggal
28. Tahun
29. Siapa yang Memutus
30. Panitera Pengganti
31. Tanda Tangan Majelis

## Verification Before Finishing

Before final response:

- Confirm the intended number of new documents were processed.
- Confirm each new JSON output has all 31 section keys.
- Confirm each non-empty section value is copied from the source text as a
  contiguous excerpt.
- Confirm `empty_sections` exactly matches the section keys whose arrays are
  empty.
- Confirm `progress.jsonl` has one completed checkpoint entry per processed
  document.
- Report the processed filenames and output paths.
