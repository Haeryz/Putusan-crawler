# Putusan Anak Section Classification Schema

## Overview

This guide defines the 31 sections for manual Codex extraction of Indonesian
juvenile criminal decisions (`Pid.Sus-Anak`) into per-source JSON files.

Use it together with:

- `LLM-aggregator/Anak/GPT/Anak.json`
- `LLM-aggregator/Anak/SKKMA Pidsus Anak-1.pdf`
- `downloads/kasus anak/raw-text`

Every extracted value must be an exact contiguous excerpt from the raw text.
Do not summarize, paraphrase, translate, normalize OCR, or add reasoning inside
any section value.

## Codex Extractor JSON Schema

The machine-readable output contract is:

`LLM-aggregator/Anak/GPT/Anak.json`

Store each processed decision as its own JSON file under:

`LLM-aggregator/Anak/GPT/output/<source-stem>.json`

Do not create one large aggregate JSON as the working extraction format.

The output object must contain:

- `status`: `completed`, `no_text`, or `failed`.
- `source_file`, `source_path`, and `source_sha256`.
- `method`: always `codex_manual_extractive`.
- `sections`: exactly the 31 snake_case keys listed below.
- `empty_sections`: exactly the keys whose section arrays are empty.

## Anak Template Context

The official Anak template is `LLM-aggregator/Anak/SKKMA Pidsus Anak-1.pdf`.
It contains three common variants:

- `Pid.I.B.1 Anak - Vrijspraak`: acquittal. The operative order usually says
  the Anak/Para Anak is not legally and convincingly proven guilty, releases
  the child from the indictment, restores rights, orders release from detention
  if detained, and charges costs to the state.
- `Pid.I.B.2 Anak - Lepas`: release from all legal charges. The reasoning may
  say the conduct is proven but not punishable because of a justification,
  excuse, or because it is not a criminal act.
- `Pid.I.B.3 Anak - Terbukti`: conviction. The operative order usually includes
  juvenile sanctions such as imprisonment, vocational training, supervision,
  institutional placement, or other sanctions, plus detention credit/status,
  evidence disposition, and case costs.

Use the expected Anak order to resolve ambiguous boundaries:

1. `P U T U S A N` / `PUTUSAN`, case number, irah-irah, court name, and case
   description.
2. Anak identity fields: `1. Nama lengkap` through `8. Pekerjaan`.
3. Optional arrest sentence, then detention history in LPAS/LPKS.
4. Assistance/counsel context: Penasihat Hukum, orangtua/wali/pendamping,
   pemberi bantuan hukum, and Pembimbing Kemasyarakatan. This is context only
   unless it is part of a target section.
5. Procedural materials: Penetapan Ketua PN, Penetapan Hari Sidang, case file,
   Laporan hasil penelitian kemasyarakatan, witnesses/experts/Anak statement,
   parents/guardian/companion, letters, and goods.
6. Prosecution demand after `Setelah mendengar pembacaan tuntutan pidana...`.
7. Defense, request for leniency, replies, and rejoinders when present.
8. Dakwaan after `Menimbang bahwa Anak/Para Anak didakwa berdasarkan surat
   dakwaan Penuntut Umum...`.
9. Evidence sequence: prosecution witnesses, prosecution experts, documents,
   defense witnesses/experts/documents, verbalisan witnesses, Anak statement,
   parent/guardian/companion statement, social inquiry report, and goods.
10. Legal facts from `diperoleh fakta hukum sebagai berikut`.
11. Legal reasoning, including dakwaan forms: tunggal, alternatif,
   subsidairitas, kumulatif, and gabungan.
12. Anak-specific reasoning: social inquiry/recommendation, juvenile sanctions,
   best interest of the child, detention status, evidence disposition,
   aggravating/mitigating factors, costs, and `Mengingat...`.
13. `MENGADILI` operative orders.
14. `Demikianlah diputuskan...` closing paragraph and signature block.

## Section Keys

| ID | Bagian Putusan | JSON key |
|----|----------------|----------|
| 1 | Judul | `judul` |
| 2 | Nomor Putusan | `nomor_putusan` |
| 3 | Irah-irah | `irah_irah` |
| 4 | Nama Pengadilan Negeri | `nama_pengadilan_negeri` |
| 5 | Keterangan Perkara | `keterangan_perkara` |
| 6 | Nama Lengkap | `nama_lengkap` |
| 7 | Tempat Lahir | `tempat_lahir` |
| 8 | Umur/Tanggal Lahir | `umur_tanggal_lahir` |
| 9 | Jenis Kelamin | `jenis_kelamin` |
| 10 | Kebangsaan | `kebangsaan` |
| 11 | Tempat Tinggal | `tempat_tinggal` |
| 12 | Agama | `agama` |
| 13 | Pekerjaan | `pekerjaan` |
| 14 | Penangkapan | `penangkapan` |
| 15 | Penahanan | `penahanan` |
| 16 | Tuntutan | `tuntutan` |
| 17 | Dakwaan | `dakwaan` |
| 18 | Saksi | `saksi` |
| 19 | Ahli | `ahli` |
| 20 | Terdakwa/Anak | `terdakwa` |
| 21 | Surat | `surat` |
| 22 | Petunjuk/Barang Bukti | `petunjuk_barang_bukti` |
| 23 | Fakta Hukum | `fakta_hukum` |
| 24 | Pertimbangan Hukum | `pertimbangan_hukum` |
| 25 | Amar Putusan | `amar_putusan` |
| 26 | Hari | `hari` |
| 27 | Tanggal | `tanggal` |
| 28 | Tahun | `tahun` |
| 29 | Siapa yang Memutus | `siapa_yang_memutus` |
| 30 | Panitera Pengganti | `panitera_pengganti` |
| 31 | Tanda Tangan Majelis | `tanda_tangan_majelis` |

## Matching Rules

- Match labels and anchors case-insensitively.
- BEFORE and AFTER lists are OR lists. Any single match can establish a
  boundary.
- Numbered prefixes such as `1.`, `2.`, or `I.` are optional locating syntax.
- Preserve source line breaks, punctuation, spelling, and OCR artifacts.
- `Anak`, `Para Anak`, `Terdakwa`, and `Para Terdakwa` may be used
  inconsistently in real decisions. Use the section meaning to select the right
  passage.
- Treat `Kewarganegaraan` as the same field as `Kebangsaan`.
- An optional `Pendidikan` identity line may appear between `Kebangsaan` and
  `Tempat tinggal`; do not include it in `kebangsaan` or `tempat_tinggal`.
- Return `[]` only after checking direct labels, aliases, and OCR variants.

## Field Rules And Boundaries

### 1. Judul

```yaml
id: 1
bagian: "Judul"
kata_sebelum: []
kata_sesudah:
  - "Nomor"
```

Extract the beginning title, commonly `P U T U S A N`, `PUTUSAN`, or
`PENETAPAN`, without page headers when they are separable.

### 2. Nomor Putusan

```yaml
id: 2
bagian: "Nomor Putusan"
kata_sebelum:
  - "Putusan"
  - "Nomor"
kata_sesudah:
  - "DEMI KEADILAN"
  - "Pengadilan"
```

Include the full visible decision number, especially `Pid.Sus-Anak`.

### 3. Irah-irah

```yaml
id: 3
bagian: "Irah-irah"
kata_sebelum:
  - "PN"
  - "DEMI"
kata_sesudah:
  - "Pengadilan Anak"
  - "Pengadilan Negeri"
```

Usually `DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA`.

### 4. Nama Pengadilan Negeri

```yaml
id: 4
bagian: "Nama Pengadilan Negeri"
kata_sebelum:
  - "Yang Maha Esa"
  - "MAHA ESA"
kata_sesudah:
  - "yang mengadili perkara"
  - "yang mengadili perkara pidana anak"
```

Include the court phrase identifying `Pengadilan Anak pada Pengadilan Negeri`
or the district court name.

### 5. Keterangan Perkara

```yaml
id: 5
bagian: "Keterangan Perkara"
kata_sebelum:
  - "mengadili perkara"
  - "menjatuhkan putusan sebagai berikut"
kata_sesudah:
  - "dalam perkara Anak"
  - "dengan"
  - "1. Nama lengkap"
```

Capture the case description, for example juvenile criminal case, examination
procedure, level, and `Anak/Para Anak`.

### 6. Nama Lengkap

```yaml
id: 6
bagian: "Nama Lengkap"
kata_sebelum:
  - "1. Nama lengkap"
  - "Nama lengkap"
kata_sesudah:
  - "2. Tempat lahir"
  - "Tempat lahir"
```

Extract only the identity value. For multiple Anak, use multiple array items or
one item preserving the source's grouped labels.

### 7. Tempat Lahir

```yaml
id: 7
bagian: "Tempat Lahir"
kata_sebelum:
  - "2. Tempat lahir"
  - "Tempat lahir"
kata_sesudah:
  - "3. Umur/tanggal lahir"
  - "Umur/tanggal lahir"
  - "Umur"
```

### 8. Umur/Tanggal Lahir

```yaml
id: 8
bagian: "Umur/Tanggal Lahir"
kata_sebelum:
  - "3. Umur/tanggal lahir"
  - "Umur/tanggal lahir"
  - "Umur"
kata_sesudah:
  - "4. Jenis kelamin"
  - "Jenis kelamin"
```

### 9. Jenis Kelamin

```yaml
id: 9
bagian: "Jenis Kelamin"
kata_sebelum:
  - "4. Jenis kelamin"
  - "Jenis kelamin"
kata_sesudah:
  - "5. Kebangsaan"
  - "Kebangsaan"
  - "Kewarganegaraan"
```

### 10. Kebangsaan

```yaml
id: 10
bagian: "Kebangsaan"
kata_sebelum:
  - "5. Kebangsaan"
  - "Kebangsaan"
  - "Kewarganegaraan"
kata_sesudah:
  - "6. Tempat tinggal"
  - "Tempat tinggal"
  - "Pendidikan"
```

If `Pendidikan` appears after nationality, stop before it.

### 11. Tempat Tinggal

```yaml
id: 11
bagian: "Tempat Tinggal"
kata_sebelum:
  - "6. Tempat tinggal"
  - "Tempat tinggal"
kata_sesudah:
  - "7. Agama"
  - "Agama"
```

### 12. Agama

```yaml
id: 12
bagian: "Agama"
kata_sebelum:
  - "7. Agama"
  - "Agama"
kata_sesudah:
  - "8. Pekerjaan"
  - "Pekerjaan"
```

### 13. Pekerjaan

```yaml
id: 13
bagian: "Pekerjaan"
kata_sebelum:
  - "8. Pekerjaan"
  - "Pekerjaan"
kata_sesudah:
  - "Anak ditangkap"
  - "Para Anak ditangkap"
  - "Anak/Para Anak ditangkap"
  - "Anak ditahan"
  - "Para Anak ditahan"
```

If no arrest/detention follows, stop before the next non-identity paragraph.

### 14. Penangkapan

```yaml
id: 14
bagian: "Penangkapan"
kata_sebelum:
  - "ditangkap sejak"
  - "ditangkap pada"
  - "surat perintah penangkapan"
  - "Anak dilakukan penangkapan"
  - "Para Anak dilakukan penangkapan"
kata_sesudah:
  - "Anak ditahan"
  - "Para Anak ditahan"
  - "ditahan dalam tahanan"
  - "ditahan oleh"
```

Include only arrest wording and dates. Do not include detention stages here.

### 15. Penahanan

```yaml
id: 15
bagian: "Penahanan"
kata_sebelum:
  - "ditahan dalam tahanan LPAS"
  - "ditahan dalam tahanan LPKS"
  - "ditahan oleh"
  - "dalam tahanan"
  - "ditahan dalam perkara lain"
kata_sesudah:
  - "didampingi oleh Penasihat Hukum"
  - "Pengadilan Anak pada Pengadilan Negeri tersebut"
  - "Membaca Penetapan"
```

Include every detention authority and period: Penyidik, Perpanjangan Penuntut
Umum, Penuntut Umum, Perpanjangan Ketua Pengadilan Negeri, Hakim/Majelis
Hakim, and any penangguhan, pembantaran, pengalihan penahanan, or detention in
another case.

### 16. Tuntutan

```yaml
id: 16
bagian: "Tuntutan"
kata_sebelum:
  - "Setelah mendengar pembacaan tuntutan pidana"
  - "mendengar pembacaan tuntutan pidana"
kata_sesudah:
  - "Setelah mendengar pembelaan"
  - "Setelah mendengar permohonan"
  - "Menimbang bahwa Anak"
  - "Menimbang bahwa Para Anak"
```

Extract the complete prosecution demand. Include requested juvenile sanction,
evidence disposition, and costs if present.

### 17. Dakwaan

```yaml
id: 17
bagian: "Dakwaan"
kata_sebelum:
  - "didakwa berdasarkan surat dakwaan"
  - "berdasarkan surat dakwaan Penuntut Umum"
  - "catatan dakwaan"
kata_sesudah:
  - "Menimbang bahwa terhadap dakwaan"
  - "Menimbang bahwa untuk membuktikan"
  - "Penuntut Umum telah mengajukan saksi"
```

Include all forms: tunggal, alternatif, subsidairitas, kumulatif, and gabungan.
For short-procedure cases, treat `catatan dakwaan` as dakwaan.

### 18. Saksi

```yaml
id: 18
bagian: "Saksi"
kata_sebelum:
  - "Penuntut Umum telah mengajukan saksi-saksi"
  - "mengajukan saksi-saksi"
  - "menghadirkan saksi"
  - "saksi yang meringankan"
  - "a de charge"
  - "saksi verbalisan"
kata_sesudah:
  - "Penuntut Umum telah mengajukan Ahli"
  - "Penuntut Umum telah mengajukan Surat"
  - "Anak/Para Anak di persidangan"
  - "di persidangan telah memberikan keterangan"
```

Include prosecution witnesses, child victims, child witnesses, defense
witnesses, verbalisan witnesses, oath status, testimony bullets, and the
Anak/Para Anak response to each witness.

### 19. Ahli

```yaml
id: 19
bagian: "Ahli"
kata_sebelum:
  - "Penuntut Umum telah mengajukan Ahli"
  - "Anak/Para Anak telah mengajukan Ahli"
  - "mengajukan Ahli"
  - "dibacakan di persidangan"
kata_sesudah:
  - "Penuntut Umum telah mengajukan Surat"
  - "mengajukan Surat"
  - "saksi yang meringankan"
  - "Anak/Para Anak di persidangan"
```

Include prosecution and defense experts, including expert statements read into
the hearing.

### 20. Terdakwa/Anak

```yaml
id: 20
bagian: "Terdakwa"
kata_sebelum:
  - "Anak/Para Anak di persidangan telah memberikan keterangan"
  - "Anak di persidangan telah memberikan keterangan"
  - "Para Anak di persidangan telah memberikan keterangan"
  - "Terdakwa/Para Terdakwa telah mengajukan"
kata_sesudah:
  - "di persidangan telah didengar keterangan"
  - "Laporan hasil penelitian kemasyarakatan"
  - "Penuntut Umum mengajukan barang bukti"
```

This field is the Anak/defendant courtroom statement, not identity data. If the
source uses `Terdakwa` instead of `Anak`, capture the same testimony section.

### 21. Surat

```yaml
id: 21
bagian: "Surat"
kata_sebelum:
  - "mengajukan Surat"
  - "Surat (termasuk alat bukti elektronik)"
  - "alat bukti elektronik"
  - "bukti surat"
kata_sesudah:
  - "saksi yang meringankan"
  - "Anak/Para Anak telah mengajukan Ahli"
  - "saksi verbalisan"
  - "Anak/Para Anak di persidangan"
```

Include documentary and electronic evidence submitted by either side.

### 22. Petunjuk/Barang Bukti

```yaml
id: 22
bagian: "Petunjuk/Barang Bukti"
kata_sebelum:
  - "Penuntut Umum mengajukan barang bukti"
  - "mengajukan barang bukti"
  - "barang bukti sebagai berikut"
kata_sesudah:
  - "diperoleh fakta hukum"
  - "fakta hukum sebagai berikut"
  - "Majelis Hakim akan mempertimbangkan"
```

Capture the submitted goods/evidence inventory. Do not move later evidence
disposition reasoning into this field.

### 23. Fakta Hukum

```yaml
id: 23
bagian: "Fakta Hukum"
aliases:
  - "fakta-fakta hukum"
kata_sebelum:
  - "diperoleh fakta hukum sebagai berikut"
  - "fakta hukum sebagai berikut"
  - "berdasarkan keterangan saksi-saksi"
kata_sesudah:
  - "Hakim/Majelis Hakim akan mempertimbangkan"
  - "Majelis Hakim akan mempertimbangkan"
  - "DAKWAAN TUNGGAL"
  - "DAKWAAN ALTERNATIF"
```

### 24. Pertimbangan Hukum

```yaml
id: 24
bagian: "Pertimbangan Hukum"
aliases:
  - "pertimbangan"
kata_sebelum:
  - "Hakim/Majelis Hakim akan mempertimbangkan"
  - "Majelis Hakim akan mempertimbangkan"
  - "DAKWAAN TUNGGAL"
  - "DAKWAAN ALTERNATIF"
  - "DAKWAAN SUBSIDAIRITAS"
  - "DAKWAAN KUMULATIF"
  - "DAKWAAN GABUNGAN"
kata_sesudah:
  - "MENGADILI"
  - "M E N G A D I L I"
```

Include element analysis, conclusions, Anak-specific sanctions/social inquiry
reasoning, detention consequence reasoning, evidence disposition reasoning,
aggravating/mitigating factors, costs, and `Mengingat...` up to but not
including `MENGADILI`.

### 25. Amar Putusan

```yaml
id: 25
bagian: "Amar Putusan"
kata_sebelum:
  - "MENGADILI"
  - "MENGADILI:"
  - "MENGADILI;"
  - "M E N G A D I L I"
  - "M E N G A D I L I :"
kata_sesudah:
  - "Demikianlah diputuskan"
```

Include every numbered operative order. For Anak convictions, include juvenile
sentence/sanction, detention status, evidence disposition, and costs.

### 26. Hari

```yaml
id: 26
bagian: "Hari"
kata_sebelum:
  - "pada hari"
  - "pada hari ini"
kata_sesudah:
  - ", tanggal"
```

Use the deliberation decision date unless the task explicitly asks otherwise.

### 27. Tanggal

```yaml
id: 27
bagian: "Tanggal"
kata_sebelum:
  - "tanggal"
kata_sesudah:
  - "bulan"
  - ", oleh"
```

### 28. Tahun

```yaml
id: 28
bagian: "Tahun"
kata_sebelum:
  - "bulan"
kata_sesudah:
  - ", oleh"
  - "oleh"
```

### 29. Siapa yang Memutus

```yaml
id: 29
bagian: "Siapa yang Memutus"
kata_sebelum:
  - "oleh"
  - "oleh kami"
kata_sesudah:
  - "selaku Hakim Ketua"
  - "sebagai Hakim Ketua"
  - "masing-masing sebagai Hakim Anggota"
```

Prefer the judges in the deliberation formula after `Demikianlah diputuskan`.

### 30. Panitera Pengganti

```yaml
id: 30
bagian: "Panitera Pengganti"
kata_sebelum:
  - "dibantu oleh"
  - "Panitera Pengganti"
kata_sesudah:
  - "serta dihadiri"
  - "Penuntut Umum"
  - "Hakim Ketua"
  - "Hakim"
```

### 31. Tanda Tangan Majelis

```yaml
id: 31
bagian: "Tanda Tangan Majelis"
kata_sebelum:
  - "Hakim Ketua,"
  - "Hakim,"
  - "Panitera Pengganti,"
kata_sesudah:
  - "Catatan:"
  - "Untuk putusan perkara anak"
```

Capture the signature block containing judge(s) and substitute clerk.

## Verification Checklist

Before checkpointing a source:

- The JSON has exactly all 31 section keys.
- Every non-empty section string appears verbatim and contiguously in the raw
  source text.
- `empty_sections` contains exactly the keys whose arrays are empty.
- The output filename matches the source stem.
- The checkpoint record identifies the same source, hash, output path, status,
  method, and empty sections.
