# Codex TPPO Extraction Instructions

Use this file as the full task instruction.

## Objective

Manually extract Indonesian court decision sections into individual JSON artifacts. Each processed raw-text source must produce exactly one JSON file under:

`LLM-aggregator/TPPO/GPT/output/<source-stem>.json`

Each output file must conform to:

`LLM-aggregator/TPPO/GPT/TPPO.json`

`TPPO.json` is the JSON Schema only. Do not append all 500 extractions into `TPPO.json`, and do not create one single big JSON file as the primary extraction output.

Follow the section schema in:

`LLM-aggregator/Putusan-schema.md`

Use the TPPO official format context from:

`LLM-aggregator/TPPO Format.pdf`

Source documents are already extracted as raw text in:

`downloads/TPPO/raw-text`

Do not waste time extracting from PDFs unless the raw text is missing or unreadable.

## One-Click Launcher

On macOS/Linux, start or resume the automated Codex extraction loop with the
native Python launcher:

```bash
python3 run_extractions.py --corpus TPPO
```

For single-click/shell-wrapper usage on macOS/Linux, run:

```bash
./LLM-aggregator/TPPO/GPT/run-codex-extraction.sh
```

On Windows PowerShell, use:

```powershell
.\LLM-aggregator\TPPO\GPT\run-codex-extraction.ps1
```

For Windows Explorer/double-click usage, run:

```cmd
LLM-aggregator\TPPO\GPT\run-codex-extraction.cmd
```

The launcher calls `codex exec` non-interactively, gives Codex this extraction loop, and writes outputs/checkpoints in the GPT directory. It resumes from existing `progress.jsonl` records and `GPT/output/*.json` files, so rerunning the launcher should continue pending files rather than starting over.

Useful controls:

```bash
python3 run_extractions.py --corpus TPPO --status
./LLM-aggregator/TPPO/GPT/run-codex-extraction.sh --status
python3 run_extractions.py --corpus TPPO --target 1
python3 run_extractions.py --corpus TPPO --target 10
python3 run_extractions.py --corpus TPPO --model gpt-5-codex
```

With no target, the launcher processes all pending sources one at a time until the 5h usage guard stops it, a failure occurs, or the corpus is complete. `-Target X` / `--target X` means launch up to X new Codex sessions. Each Codex session processes exactly one pending source file, writes/checkpoints it, then exits. `-MaxFiles` is accepted as a backward-compatible PowerShell alias for `-Target`; it does not mean multiple files inside one Codex session.

Guarded AFK runs are sequential so usage can be checked before every next source. Parallel sessions are available only when the usage guard is disabled.

## Agent Loop

This workflow follows the current "agent loop" pattern: use a repeatable control loop that keeps selecting, extracting, verifying, checkpointing, and checking stop conditions without requiring a new prompt for each document.

Process exactly one raw-text source per Codex session. Do not process multiple source files inside a single Codex session.

Loop steps:

1. Discover pending files by comparing `downloads/TPPO/raw-text/*.txt` against completed records in `LLM-aggregator/TPPO/GPT/progress.jsonl` and existing JSON files in `LLM-aggregator/TPPO/GPT/output/`.
2. Preassign one distinct source file to each target Codex session in deterministic filename order.
3. Each Codex session reads only its assigned source file.
4. Extract all 31 fields into one JSON object using exact contiguous source excerpts.
5. Save the result as `LLM-aggregator/TPPO/GPT/output/<source-stem>.json`.
6. Verify the output against `LLM-aggregator/TPPO/GPT/TPPO.json`, including all 31 section keys and accurate `empty_sections`.
7. Append one completed JSONL checkpoint record to `LLM-aggregator/TPPO/GPT/progress.jsonl`.
8. Check current Codex/session usage before starting the next file.
9. Continue with the next pending file only if the usage guard has not triggered.

Usage guard:

- If remaining usage is below 10% of the active five-hour reset window, stop before starting another source.
- If `/status` text is unavailable to the non-interactive launcher, the 270-minute wall-clock fallback stops before starting another source.
- Do not begin a large source when usage is already under the 10% threshold.
- Create a Markdown run report under `LLM-aggregator/TPPO/GPT/reports/` before the final response.
- Name the report with a timestamp, for example `20260620-181500-usage-stop.md`.
- The report must include: stop reason, usage remaining, reset timing if visible, processed count in this run, completed output paths, last source handled, pending count, failed or skipped sources, and recommended resume command or next action.

Use this checkpoint to know what has already been completed:

`LLM-aggregator/TPPO/GPT/progress.jsonl`

Append one JSONL record per completed document with:

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
- Use tools only to inspect text, create/edit JSON output artifacts, and update the checkpoint.
- EXTRACT THE TEXT AS IT IS IN THE TEXT U CANNOT AND NOT ALLOWED TO DO ANY SUMMARIZATION
- Since u r limited by output tokens, store long copied excerpts directly in the JSON artifact instead of trying to display them in chat.

Not allowed:

- Do not write a program that decides/extracts the legal sections automatically.
- Do not hardcode a general extractor.
- Do not use another LLM or external service to do the extraction.
- DO NOT SUMMARIZE THE TEXT
- DO NOT WRITE UR REASONING INSIDE SECTION VALUES
- DO NOT WRITE ANYTHING ELSE BESIDE ACTUAL EXTRACTION INTO SECTION VALUES

## TPPO Format Context

The PDF format file is not a source decision to extract into the JSON output. It is a guide for how TPPO decisions are usually ordered and where section boundaries normally fall.

Common TPPO decision variants in the format:

- `Pid.I.A.7.1 Khusus - Vrijspraak`: acquittal. Amar commonly says the defendant is not legally and convincingly proven guilty, releases the defendant from the indictment, restores rights, and charges costs to the state.
- `Pid.I.A.7.2 (Format Khusus Lepas)`: release from all legal charges. Amar commonly says the proven conduct is not a criminal offense and releases the defendant from all legal charges.
- `Pid.I.A.7.3 Format Khusus - Terbukti`: conviction. Amar commonly includes prison/fine, possible TPPO restitution, detention credit/status, evidence disposition, and case costs.

Use this expected order to resolve ambiguous boundaries:

1. `P U T U S A N` / `PUTUSAN`, case number, irah-irah, court name, and case description.
2. Numbered defendant identity fields: `1. Nama lengkap` through `8. Pekerjaan`.
3. Optional arrest sentence and detention history.
4. Counsel status.
5. `Setelah membaca...` procedural materials.
6. `Setelah mendengar pembacaan tuntutan pidana...` prosecution demand.
7. Defense/plea/reply material, if placed before dakwaan.
8. `Menimbang bahwa ... didakwa berdasarkan surat dakwaan...` dakwaan.
9. Evidence sequence: witnesses, experts, documentary/electronic evidence, defendant statement, defense evidence, verbalisan witness, goods/evidence list.
10. `diperoleh fakta hukum sebagai berikut` facts.
11. Legal consideration and element analysis under dakwaan forms.
12. TPPO-specific restitution consideration, detention reasoning, evidence-disposition reasoning, aggravating/mitigating factors, costs, and `Mengingat...`.
13. `MENGADILI` operative orders.
14. `Demikianlah diputuskan...` closing paragraph and signature block.

## Field-Level Extraction Context

- Fields 6-13, defendant identity: extract only the value for that identity field. For multiple defendants, put all defendants' values in the same section array in document order, preserving labels such as `Terdakwa I` when present.
- Field 10, Kebangsaan: treat `Kewarganegaraan` as the same field if the decision uses that wording.
- Field 14, Penangkapan: include only the arrest wording (`ditangkap sejak tanggal... sampai dengan...`, `ditangkap pada...`, or equivalent). Do not include detention stages here.
- Field 15, Penahanan: include every detention stage and extension. Search for the exact TPPO marker `Khusus Penahanan Tindak Pidana TPPO`; TPPO templates may list up to 12 stages, including first/second extensions by Ketua Pengadilan Negeri and Ketua Pengadilan Tinggi. Also include suspension, hospitalization/interruption, transfer, or `ditahan dalam perkara lain` if present.
- Field 16, Tuntutan: extract the copied prosecution demand after `Setelah mendengar pembacaan tuntutan pidana... pada pokoknya sebagai berikut`. Include requested imprisonment, fine, restitution, evidence disposition, and costs if they appear in the demand.
- Field 17, Dakwaan: extract the complete charge text after `didakwa berdasarkan surat dakwaan Penuntut Umum Nomor... tanggal... sebagai berikut`. Keep all forms: tunggal, alternatif, subsidairitas, kumulatif, or gabungan.
- Field 18, Saksi: include prosecution witnesses, defense witnesses (`saksi yang meringankan` / `a de charge`), and verbalisan witnesses when present. Preserve witness names, oath status, testimony bullets, and defendant responses.
- Field 19, Ahli: include prosecution and defense experts, including expert statements read into court.
- Field 20, Terdakwa: extract the defendant's own courtroom statement beginning around `Terdakwa/Para Terdakwa di persidangan telah memberikan keterangan...`.
- Field 21, Surat: include documentary and electronic evidence under `Surat (termasuk alat bukti elektronik)` or equivalent.
- Field 22, Petunjuk/Barang Bukti: include the submitted goods/evidence inventory, usually after `Penuntut Umum mengajukan barang bukti sebagai berikut`. Do not move later evidence-disposition reasoning into this field.
- Field 23, Fakta Hukum: starts at `berdasarkan keterangan saksi-saksi... diperoleh fakta hukum sebagai berikut` and ends before `Majelis Hakim akan mempertimbangkan...`.
- Field 24, Pertimbangan Hukum: includes dakwaan element analysis, conclusions about fulfilled/unfulfilled elements, TPPO restitution consideration, detention reasoning, evidence-disposition reasoning, aggravating/mitigating factors, costs, and `Mengingat...` up to but not including `MENGADILI`.
- Field 25, Amar Putusan: starts at `MENGADILI` and includes every numbered operative order. For TPPO conviction, include restitution orders such as payment to victim/heirs, 14-day warning period, seizure/auction by prosecutor, and substitute imprisonment if present.
- Fields 26-31, closing fields: parse from `Demikianlah diputuskan...` and the signature block. `Hari`, `Tanggal`, and `Tahun` refer to the deliberation decision date unless the task explicitly asks for pronouncement date. `Siapa yang Memutus` is the deciding judges named after `oleh ... sebagai Hakim Ketua ... masing-masing sebagai Hakim Anggota`. `Panitera Pengganti` is the clerk after `dibantu oleh`. `Tanda Tangan Majelis` is the signature block beginning around `Hakim-hakim Anggota` / `Hakim Ketua` through `Panitera Pengganti`.

## Legacy Workbook Note

Legacy workbook extraction is deprecated for the GPT/Codex path. Do not target the deleted workbook unless the user explicitly restores it and asks for spreadsheet output.

## JSON Output Rules

Create one JSON object per processed raw-text file and save it as one separate file in:

`LLM-aggregator/TPPO/GPT/output/`

Use the raw-text source stem as the output filename. For example:

`downloads/TPPO/raw-text/10_Pid.Sus_2025_PN_End.txt`

must be saved as:

`LLM-aggregator/TPPO/GPT/output/10_Pid.Sus_2025_PN_End.json`

The object must validate against:

`LLM-aggregator/TPPO/GPT/TPPO.json`

Use this output shape:

```json
{
  "status": "completed",
  "source_file": "example.txt",
  "source_path": "downloads/TPPO/raw-text/example.txt",
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

Every section value is an array. Put exact copied source excerpts in the array. Use multiple array items only for multiple defendants or genuinely separate occurrences. Use `[]` only when no exact source excerpt exists after checking the field label, schema boundaries, aliases, and OCR variants.

Do not write extraction-status prose, reasoning, or summaries inside `sections`. If a section is absent, leave that section as `[]` and include its key in `empty_sections`.

Do not write multiple source documents into the same JSON file. For the 500-file TPPO corpus, expect up to 500 individual JSON output files in `LLM-aggregator/TPPO/GPT/output/`.

A combined corpus JSON may be generated later as a derived export, but it is not the working extraction format and must not replace the per-source files.

Do not overwrite an existing completed output unless the checkpoint is being intentionally reset or the user asks to redo that source.

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
- Confirm each non-empty section value is copied from the source text as a contiguous excerpt.
- Confirm `empty_sections` exactly matches the section keys whose arrays are empty.
- Confirm `progress.jsonl` has one completed checkpoint entry per processed document.
- Report the processed filenames and output paths.
