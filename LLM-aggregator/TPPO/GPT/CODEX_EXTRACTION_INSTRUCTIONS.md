# Codex TPPO Extraction Instructions

Use this file as the full task instruction.

## Objective

Manually extract Indonesian court decision sections into:

`LLM-aggregator/TPPO/GPT/TPPO_31_sections_500_rows.xlsx`

Follow the section schema in:

`LLM-aggregator/Putusan-schema.md`

Use the TPPO official format context from:

`LLM-aggregator/TPPO Format.pdf`

Source documents are already extracted as raw text in:

`downloads/TPPO/raw-text`

Do not waste time extracting from PDFs unless the raw text is missing or unreadable.

## Scope Per Run

Process the next unprocessed raw-text files one by one. If the user does not specify a count, process 5 documents.

Use this checkpoint to know what has already been completed:

`LLM-aggregator/TPPO/GPT/progress.jsonl`

Append one JSONL record per completed document with:

- `source_file`
- `source_path`
- `workbook`
- `sheet`
- `row`
- `status`
- `method`
- `sections_filled`

## Manual Extraction Rule

This is Codex doing the extraction, not a generated extractor.

Allowed:

- Read/search raw text files to locate relevant passages.
- Use tools only to inspect text, edit the `.xlsx`, and update the checkpoint.
- EXTRACT THE TEXT AS IT IS IN THE TEXT U CANNOT AND NOT ALLOWED TO DO ANY SUMMARIZATION
- Since u r limited by output tokens, therefore u have to store the text temporarely then insert it into the spreadsheet.

Not allowed:

- Do not write a program that decides/extracts the legal sections automatically.
- Do not hardcode a general extractor.
- Do not use another LLM or external service to do the extraction.
- DO NOT SUMMARIZE THE TEXT
- DO NOT WRITE UR REASONING INSIDE COLUMN
- DO NOT WRITE ANYTHING ELSE BESIDE ACTUAL EXTRACTION INTO COLUMNS

## TPPO Format Context

The PDF format file is not a source decision to extract into the workbook. It is a guide for how TPPO decisions are usually ordered and where section boundaries normally fall.

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

- Columns 6-13, defendant identity: extract only the value for that identity field. For multiple defendants, put all defendants' values in the same cell in document order, preserving labels such as `Terdakwa I` when present.
- Column 10, Kebangsaan: treat `Kewarganegaraan` as the same field if the decision uses that wording.
- Column 14, Penangkapan: include only the arrest wording (`ditangkap sejak tanggal... sampai dengan...`, `ditangkap pada...`, or equivalent). Do not include detention stages here.
- Column 15, Penahanan: include every detention stage and extension. Search for the exact TPPO marker `Khusus Penahanan Tindak Pidana TPPO`; TPPO templates may list up to 12 stages, including first/second extensions by Ketua Pengadilan Negeri and Ketua Pengadilan Tinggi. Also include suspension, hospitalization/interruption, transfer, or `ditahan dalam perkara lain` if present.
- Column 16, Tuntutan: extract the copied prosecution demand after `Setelah mendengar pembacaan tuntutan pidana... pada pokoknya sebagai berikut`. Include requested imprisonment, fine, restitution, evidence disposition, and costs if they appear in the demand.
- Column 17, Dakwaan: extract the complete charge text after `didakwa berdasarkan surat dakwaan Penuntut Umum Nomor... tanggal... sebagai berikut`. Keep all forms: tunggal, alternatif, subsidairitas, kumulatif, or gabungan.
- Column 18, Saksi: include prosecution witnesses, defense witnesses (`saksi yang meringankan` / `a de charge`), and verbalisan witnesses when present. Preserve witness names, oath status, testimony bullets, and defendant responses.
- Column 19, Ahli: include prosecution and defense experts, including expert statements read into court.
- Column 20, Terdakwa: extract the defendant's own courtroom statement beginning around `Terdakwa/Para Terdakwa di persidangan telah memberikan keterangan...`.
- Column 21, Surat: include documentary and electronic evidence under `Surat (termasuk alat bukti elektronik)` or equivalent.
- Column 22, Petunjuk/Barang Bukti: include the submitted goods/evidence inventory, usually after `Penuntut Umum mengajukan barang bukti sebagai berikut`. Do not move later evidence-disposition reasoning into this column.
- Column 23, Fakta Hukum: starts at `berdasarkan keterangan saksi-saksi... diperoleh fakta hukum sebagai berikut` and ends before `Majelis Hakim akan mempertimbangkan...`.
- Column 24, Pertimbangan Hukum: includes dakwaan element analysis, conclusions about fulfilled/unfulfilled elements, TPPO restitution consideration, detention reasoning, evidence-disposition reasoning, aggravating/mitigating factors, costs, and `Mengingat...` up to but not including `MENGADILI`.
- Column 25, Amar Putusan: starts at `MENGADILI` and includes every numbered operative order. For TPPO conviction, include restitution orders such as payment to victim/heirs, 14-day warning period, seizure/auction by prosecutor, and substitute imprisonment if present.
- Columns 26-31, closing fields: parse from `Demikianlah diputuskan...` and the signature block. `Hari`, `Tanggal`, and `Tahun` refer to the deliberation decision date unless the workbook task explicitly asks for pronouncement date. `Siapa yang Memutus` is the deciding judges named after `oleh ... sebagai Hakim Ketua ... masing-masing sebagai Hakim Anggota`. `Panitera Pengganti` is the clerk after `dibantu oleh`. `Tanda Tangan Majelis` is the signature block beginning around `Hakim-hakim Anggota` / `Hakim Ketua` through `Panitera Pengganti`.

## Workbook Rules

The spreadsheet has 31 columns matching `Putusan-schema.md`.

Write each processed document into the next empty row of sheet `TPPO New`.

Fill every column. If a section is absent, unclear, or does not match schema boundaries cleanly, write a short extraction-status note in that cell, for example `Reasoning: bagian tidak ditemukan dalam teks mentah`. Use this only when no actual extractable text exists for that cell.

Do not overwrite existing completed rows. Use the checkpoint and workbook contents as authoritative state.

## Required Columns

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
- Confirm each new workbook row has 31 populated cells.
- Confirm `progress.jsonl` has one completed checkpoint entry per processed document.
- Report the processed filenames and row numbers.
