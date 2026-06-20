# Putusan Pengadilan — Section Classification Schema

## Overview

This schema defines how to identify each **section (bagian)** of an Indonesian court decision (`putusan pengadilan`) using **boundary keyword matching**. Each section is bounded by:

- **`kata_sebelum`** — keywords/phrases that appear **immediately before** the section's content
- **`kata_sesudah`** — keywords/phrases that appear **immediately after** the section's content

The section content lies between any matched `kata_sebelum` and any matched `kata_sesudah`.

---

## Agent Usage Notes

- Match is **case-insensitive** unless noted.
- A `kata_sebelum` or `kata_sesudah` list is **OR-joined** — any single match is sufficient.
- Some `kata_sebelum` entries include **numbered prefixes** (e.g., `1. Nama lengkap`) — strip the number prefix before matching or treat the prefix as optional.
- Entries marked `# [variant]` are stylistic variants of the same keyword (spacing, punctuation, formatting differences in OCR'd documents).
- Entries marked `# [spaced]` indicate letter-spaced OCR artifacts (e.g., `M E N G A D I L I`).
- Where `kata_sebelum` is empty (`[]`), the section appears at the **start of the document**.
- `⚠️ Note` fields contain edge-case observations from real putusan data.

---

## TPPO Format PDF Context

The TPPO template in `LLM-aggregator/TPPO Format.pdf` is the contextual authority for this schema. It contains three common criminal-decision variants:

- `Pid.I.A.7.1 Khusus - Vrijspraak`: acquittal, usually ending with "tidak terbukti" and "Membebaskan".
- `Pid.I.A.7.2 (Format Khusus Lepas)`: release from all legal charges, usually ending with "lepas dari segala tuntutan hukum".
- `Pid.I.A.7.3 Format Khusus - Terbukti`: conviction, usually ending with "terbukti secara sah dan meyakinkan", imprisonment/fine, and possible TPPO restitution.

Use the keywords below as anchors, but use the template order to decide ambiguous boundaries. A section should preserve the original raw decision text between its start and end anchors. Do not summarize or normalize legal content inside a section.

### Official TPPO Template Order

1. Header: title, case number, irah-irah, court name, and case-type sentence.
2. Defendant identity: numbered fields `1. Nama lengkap` through `8. Pekerjaan`.
3. Arrest and detention: optional arrest sentence, then detention history. TPPO detention may include 12 stages.
4. Counsel status: appointed counsel, no counsel, or refusal of counsel. This is context only unless it is attached to detention text in the source.
5. Procedural review: `Setelah membaca`, hearing schedule, case file, witness/expert/defendant/evidence hearing sentence.
6. Prosecution demand: `Setelah mendengar pembacaan tuntutan pidana...`.
7. Plea/response material: defense, leniency request, prosecution reply, defendant reply. Include this with Tuntutan only if the decision places it before the dakwaan boundary and it is not separately captured elsewhere.
8. Dakwaan: `Menimbang bahwa Terdakwa/Para Terdakwa* didakwa berdasarkan surat dakwaan Penuntut Umum...`.
9. Evidence sequence: prosecution witnesses, prosecution experts, documentary/electronic evidence, defendant statement, defense witnesses/experts/documents, verbalisan witness, and prosecution goods/evidence list.
10. Facts: `diperoleh fakta hukum sebagai berikut`.
11. Legal consideration: `Majelis Hakim akan mempertimbangkan...` followed by dakwaan forms and element analysis.
12. Special TPPO consideration: in conviction templates, look for `KHUSUS PERKARA TPPO` and restitution consideration. Keep it in Pertimbangan Hukum unless the operative restitution order appears in Amar Putusan.
13. Evidence disposition, aggravating/mitigating factors, costs, and `Mengingat...`: these remain part of Pertimbangan Hukum until `MENGADILI`.
14. Amar Putusan: starts at `MENGADILI` and includes all operative numbered orders, including imprisonment, fines, restitution, detention status, evidence disposition, and costs.
15. Closing formula: `Demikianlah diputuskan...` contains day, date, year, deciding judges, pronouncement context, clerk, attendance, and signature block.

### TPPO-Specific Extraction Guidance

- **Defendant identity fields (6-13)**: extract only the value belonging to that numbered identity label. For multiple defendants, keep all defendants in the same cell for that field, preserving order (`Terdakwa I`, `Terdakwa II`, etc.) if present.
- **Kebangsaan (10)**: the template note says this may appear as `Kewarganegaraan` under PERMA 9 Tahun 2017. Treat `Kebangsaan`, `Kewarganegaraan`, and nationality/citizenship wording as the same field.
- **Penangkapan (14)**: include the full arrest sentence beginning with `ditangkap sejak tanggal` or `ditangkap pada` through its end date/order reference. If no arrest is stated, do not borrow detention text.
- **Penahanan (15)**: include all detention stages under the relevant detention paragraph. For TPPO, search for the exact marker `Khusus Penahanan Tindak Pidana TPPO`; the official sequence may include `Penyidik`, `Perpanjangan Penuntut Umum`, first/second `Perpanjangan Ketua Pengadilan Negeri`, `Penuntut Umum`, `Hakim/Majelis Hakim`, `Perpanjangan Ketua Pengadilan Tinggi`, and `Perpanjangan kedua Ketua Pengadilan Tinggi`. Include `Penangguhan`, `Pembantaran`, `Pengalihan Penahanan`, or `ditahan dalam perkara lain` when present.
- **Tuntutan (16)**: capture the prosecution demand after `Setelah mendengar pembacaan tuntutan pidana... pada pokoknya sebagai berikut`. In many decisions this is the copied `amar surat tuntutan`; include numbered demand items, requested sentence, fine, restitution, evidence disposition, and costs if present.
- **Dakwaan (17)**: capture the charging text copied after `didakwa berdasarkan surat dakwaan Penuntut Umum Nomor... tanggal... sebagai berikut`. Include all forms of indictment: tunggal, alternatif, subsidairitas, kumulatif, and gabungan.
- **Saksi (18)**: include prosecution witnesses, defense witnesses (`saksi yang meringankan` / `a de charge`), and verbalisan witnesses when the document treats them as witness testimony. Preserve each witness name, oath status, testimony bullets, and defendant response.
- **Ahli (19)**: include both prosecution and defense expert evidence, including expert statements read into the record.
- **Surat (21)**: include documentary and electronic evidence sections marked `Surat (termasuk alat bukti elektronik)` or `bukti surat`.
- **Petunjuk/Barang Bukti (22)**: include the submitted goods/evidence list after `Penuntut Umum mengajukan barang bukti sebagai berikut` or similar. Do not confuse later legal discussion of evidence disposition with this inventory section; later disposition belongs to Pertimbangan Hukum or Amar Putusan depending on location.
- **Fakta Hukum (23)**: starts at the formula `berdasarkan keterangan saksi-saksi... diperoleh fakta hukum sebagai berikut` and stops before the court begins the element/legal analysis (`Majelis Hakim akan mempertimbangkan...`).
- **Pertimbangan Hukum (24)**: includes the element analysis under dakwaan forms (`DAKWAAN TUNGGAL`, `ALTERNATIF`, `SUBSIDAIRITAS`, `KUMULATIF`, `GABUNGAN`), conclusions about whether elements are fulfilled, TPPO restitution consideration, detention consequence reasoning, evidence disposition reasoning, aggravating/mitigating circumstances, costs, and `Mengingat...` until `MENGADILI`.
- **Amar Putusan (25)**: includes every numbered operative order after `MENGADILI`, not only the first order. For TPPO conviction, include the restitution order if present, especially wording like `Mebebankan kepada Terdakwa untuk membayar restitusi... dalam waktu 14 (empat belas) hari... harta bendanya disita dan dilelang...`.
- **Closing fields (26-31)**: parse them from the `Demikianlah diputuskan...` paragraph and signature block. If the decision has different judges for deliberation and pronouncement, prefer the names after `oleh ... sebagai Hakim Ketua ... masing-masing sebagai Hakim Anggota` for `Siapa yang Memutus`, and keep pronouncement-attendance wording out unless needed to disambiguate.

---

## Classification Rules

---

### 1. Judul

```yaml
id: 1
bagian: "Judul"
kata_sebelum: []  # start of document
kata_sesudah:
  - "Nomor"
```

---

### 2. Nomor Putusan

```yaml
id: 2
bagian: "Nomor Putusan"
kata_sebelum:
  - "Putusan"
kata_sesudah:
  - "Pengadilan"
```

---

### 3. Irah-irah

```yaml
id: 3
bagian: "Irah-irah"
kata_sebelum:
  - "PN"
kata_sesudah:
  - "Pengadilan Negeri"
```

---

### 4. Nama Pengadilan Negeri

```yaml
id: 4
bagian: "Nama Pengadilan Negeri"
kata_sebelum:
  - "Esa"
kata_sesudah:
  - "yang mengadili perkara"
  - "yang mengadili perkara-perkara"
```

---

### 5. Keterangan Perkara

```yaml
id: 5
bagian: "Keterangan Perkara"
kata_sebelum:
  - "Mengadili"
kata_sesudah:
  - "dengan"
```

---

### 6. Nama Lengkap

```yaml
id: 6
bagian: "Nama Lengkap"
kata_sebelum:
  - "1. Nama lengkap"   # numbered prefix variant
  - "Nama lengkap"
kata_sesudah:
  - "2. tempat"         # numbered prefix variant
  - "tempat"
```

---

### 7. Tempat Lahir

```yaml
id: 7
bagian: "Tempat Lahir"
kata_sebelum:
  - "2. tempat"         # numbered prefix variant
  - "tempat"
kata_sesudah:
  - "3. umur"           # numbered prefix variant
  - "umur"
```

---

### 8. Umur / Tanggal Lahir

```yaml
id: 8
bagian: "Umur/Tanggal Lahir"
kata_sebelum:
  - "3. umur"           # numbered prefix variant
  - "umur"
kata_sesudah:
  - "4. Jenis"          # numbered prefix variant
  - "Jenis"
```

---

### 9. Jenis Kelamin

```yaml
id: 9
bagian: "Jenis Kelamin"
kata_sebelum:
  - "4. Jenis"          # numbered prefix variant
  - "Jenis"
kata_sesudah:
  - "5. kebangsaan"     # numbered prefix variant
  - "kebangsaan"
```

---

### 10. Kebangsaan

```yaml
id: 10
bagian: "Kebangsaan"
kata_sebelum:
  - "5. kebangsaan"     # numbered prefix variant
  - "kebangsaan"
kata_sesudah:
  - "6. tempat"         # numbered prefix variant
  - "tempat"
```

# ⚠️ Note: some putusan include a "pendidikan" (education) field between Kebangsaan and Tempat Tinggal.

---

### 11. Tempat Tinggal

```yaml
id: 11
bagian: "Tempat Tinggal"
kata_sebelum:
  - "6. tempat"         # numbered prefix variant
  - "tempat"
kata_sesudah:
  - "7. Agama"          # numbered prefix variant
  - "Agama"
```

---

### 12. Agama

```yaml
id: 12
bagian: "Agama"
kata_sebelum:
  - "7. Agama"          # numbered prefix variant
  - "Agama"
kata_sesudah:
  - "8. pekerjaan"      # numbered prefix variant
  - "pekerjaan"
```

---

### 13. Pekerjaan

```yaml
id: 13
bagian: "Pekerjaan"
kata_sebelum:
  - "8. pekerjaan"      # numbered prefix variant
  - "pekerjaan"
kata_sesudah:
  - "terdakwa ditangkap"
  - "para terdakwa ditangkap"
```

---

### 14. Penangkapan

```yaml
id: 14
bagian: "Penangkapan"
kata_sebelum:
  - "ditangkap sejak"
  - "ditangkap pada"
  - "surat perintah penangkapan"
  - "Terdakwa dilakukan"
kata_sesudah:
  - "tanggal"
  - "dalam perkara lain"
```

---

### 15. Penahanan

```yaml
id: 15
bagian: "Penahanan"
kata_sebelum:
  - "dalam tahanan"
  - "ditahan oleh :"
  - "ditahan dalam"
  - "Terdakwa dilakukan"
kata_sesudah:
  - "oleh :"
  - "sejak tanggal"
  - "dalam perkara lain"
```

---

### 16. Tuntutan

```yaml
id: 16
bagian: "Tuntutan"
kata_sebelum:
  - "mendengar pembacaan"
  - "mendengar pula"          # [variant]
kata_sesudah:
  - "pidana"
  - "pidana yang diajukan"
  - "Penuntut Umum"
  - "Jaksa Penuntut Umum"     # [variant]
```

---

### 17. Dakwaan

```yaml
id: 17
bagian: "Dakwaan"
kata_sebelum:
  - "berdasarkan surat"
  - "surat"
  - "dengan"
  - "Surat"
kata_sesudah:
  - "Penuntut Umum"
  - "Nomor Reg. Perkara"
  - "sebgai berikut:"         # [typo variant — OCR artifact]
  - "sebagai berikut :"
  - "No. Reg."
```

---

### 18. Saksi

```yaml
id: 18
bagian: "Saksi"
kata_sebelum:
  - "mengajukan"
  - "mengajukan para"         # [variant]
  - "menghadirkan"            # [variant]
  - "menghadapkan"            # [variant]
kata_sesudah:
  - "-Saksi"
  - "-saksi"
  - "yang memberikan keterangan"
  - "sebagai berikut:"
  - "ke depan Persidangan"
```

---

### 19. Ahli

```yaml
id: 19
bagian: "Ahli"
kata_sebelum:
  - "mengajukan"
  - "alat bukti"
  - "dibacakan keterangan"
  - "terdakwa membenarkannya;"
kata_sesudah:
  - "sebagai berikut:"
  - "berupa;"
  - "berupa ;"
  - "yang telah dipanggil"
  - "atas keterangan ahli"
```

---

### 20. Terdakwa (Keterangan)

```yaml
id: 20
bagian: "Terdakwa"
kata_sebelum:
  - "Menimbang, bahwa terdakwa (nama)"
  - "Menimbang, bahwa Terdakwa I (nama)"
  - "Menimbang, bahwa Terdakwa II (nama)"
  - "Menimbang, bahwa Terdakwa III (nama)"
kata_sesudah:
  - "di persidangan"
  - "memberikan keterangan"
```

# ⚠️ Note: `(nama)` is a placeholder — match the pattern `Menimbang, bahwa [Tt]erdakwa.*` via regex.

---

### 21. Surat (Alat Bukti)

```yaml
id: 21
bagian: "Surat"
kata_sebelum:
  - "mengajukan"
  - "alat bukti"
  - "bukti surat berupa"
  - "melampirkan surat:"       # [variant]
kata_sesudah:
  - "sebagai berikut:"
  - "Menimbang bahwa"
```

---

### 22. Petunjuk / Barang Bukti

```yaml
id: 22
bagian: "Petunjuk/Barang Bukti"
kata_sebelum:
  - "mengajukan"
  - "terhadap"
  - "diperhatikan"
kata_sesudah:
  - "sebagai berikut:"
  - "berupa;"
  - "berupa ;l"                # [OCR artifact variant]
```

---

### 23. Fakta Hukum

```yaml
id: 23
bagian: "Fakta Hukum"
aliases:
  - "fakta-fakta hukum"
  - "bahwa dalam persidangan,"
kata_sebelum:
  - "Menimbang"
  - "berdasarkan"
  - "disimpulkan adanya"
kata_sesudah:
  - "Majelis Hakim"
  - "tersebut diatas"
  - "serta didukung dengan bukti"
  - "-fakta dalam perkara"
  - "dalam perkara ini"
  - "sebagai berikut;"
```

---

### 24. Pertimbangan Hukum

```yaml
id: 24
bagian: "Pertimbangan Hukum"
aliases:
  - "pertimbangan"
kata_sebelum:
  - "Menimbang,"
  - "uraian"
  - "Majelis Hakim akan"
  - "sebagai berikut"
  - "mempertimbangakan"        # [typo variant — OCR artifact]
kata_sesudah:
  - "tersebut di atas"
  - "Ad."
  - "apakah berdasarkan fakta-fakta hukum"
```

---

### 25. Amar Putusan

```yaml
id: 25
bagian: "Amar Putusan"
kata_sebelum:
  - "MENGADILI"
  - "MENGADILI:"
  - "MENGADILI;"
  - "M E N G A D I L I"       # [spaced — OCR artifact]
  - "M E N G A D I L I :"     # [spaced variant]
  - "M E N G A D I L I:"      # [spaced variant]
kata_sesudah:
  - "Demikianlah diputuskan"
```

---

### 26. Hari

```yaml
id: 26
bagian: "Hari"
kata_sebelum:
  - "pada"
kata_sesudah:
  - ", tanggal"
```

---

### 27. Tanggal

```yaml
id: 27
bagian: "Tanggal"
kata_sebelum:
  - "hari, tanggal"
kata_sesudah:
  - "bulan"
```

---

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

---

### 29. Siapa yang Memutus

```yaml
id: 29
bagian: "Siapa yang Memutus"
kata_sebelum:
  - "oleh"
  - "oleh kami,"
kata_sesudah:
  - ", sebagai hakim"
  - "sebagai hakim"
```

---

### 30. Panitera Pengganti

```yaml
id: 30
bagian: "Panitera Pengganti"
kata_sebelum:
  - "Panitera"
  - "dibantu oleh"
kata_sesudah:
  - "pada Pengadilan Negeri"
```

---

### 31. Tanda Tangan Majelis

```yaml
id: 31
bagian: "Tanda Tangan Majelis"
kata_sebelum:
  - "Hakim Ketua,"
kata_sesudah:
  - "Panitera Pengganti,"
```

---

## Quick Reference Table

| ID | Bagian Putusan | Sample Kata Sebelum | Sample Kata Sesudah |
|----|----------------|---------------------|---------------------|
| 1 | Judul | _(start of doc)_ | Nomor |
| 2 | Nomor Putusan | Putusan | Pengadilan |
| 3 | Irah-irah | PN | Pengadilan Negeri |
| 4 | Nama Pengadilan Negeri | Esa | yang mengadili perkara |
| 5 | Keterangan Perkara | Mengadili | dengan |
| 6 | Nama Lengkap | 1. Nama lengkap | 2. tempat |
| 7 | Tempat Lahir | 2. tempat | 3. umur |
| 8 | Umur/Tanggal Lahir | 3. umur | 4. Jenis |
| 9 | Jenis Kelamin | 4. Jenis | 5. kebangsaan |
| 10 | Kebangsaan | 5. kebangsaan | 6. tempat |
| 11 | Tempat Tinggal | 6. tempat | 7. Agama |
| 12 | Agama | 7. Agama | 8. pekerjaan |
| 13 | Pekerjaan | 8. pekerjaan | terdakwa ditangkap |
| 14 | Penangkapan | ditangkap sejak | tanggal |
| 15 | Penahanan | dalam tahanan | oleh : |
| 16 | Tuntutan | mendengar pembacaan | pidana |
| 17 | Dakwaan | berdasarkan surat | Penuntut Umum |
| 18 | Saksi | mengajukan | -Saksi |
| 19 | Ahli | alat bukti | sebagai berikut: |
| 20 | Terdakwa | Menimbang, bahwa terdakwa... | di persidangan |
| 21 | Surat | bukti surat berupa | sebagai berikut: |
| 22 | Petunjuk/Barang Bukti | diperhatikan | berupa; |
| 23 | Fakta Hukum | Menimbang | Majelis Hakim |
| 24 | Pertimbangan Hukum | Menimbang, | tersebut di atas |
| 25 | Amar Putusan | MENGADILI | Demikianlah diputuskan |
| 26 | Hari | pada | , tanggal |
| 27 | Tanggal | hari, tanggal | bulan |
| 28 | Tahun | bulan | , oleh |
| 29 | Siapa yang Memutus | oleh | , sebagai hakim |
| 30 | Panitera Pengganti | Panitera | pada Pengadilan Negeri |
| 31 | Tanda Tangan Majelis | Hakim Ketua, | Panitera Pengganti, |

---

## Edge Cases & Known OCR Artifacts

| Issue | Example | Handling |
|-------|---------|----------|
| Letter-spaced text | `M E N G A D I L I` | Normalize by removing spaces before match |
| Typos from OCR | `sebgai berikut:` | Include as explicit variant in list |
| Misspelling | `mempertimbangakan` | Include as explicit variant in list |
| Numbered field prefixes | `1. Nama lengkap` | Strip leading `\d+\.\s` before match OR match with optional prefix |
| Regex-needed patterns | `Menimbang, bahwa Terdakwa I (nama)` | Use pattern: `Menimbang, bahwa [Tt]erdakwa\s*(I{1,3})?\s*\(nama\)` |
| Pendidikan field | Appears between Kebangsaan (#10) and Tempat Tinggal (#11) in some putusan | Handle as optional field between #10 and #11 |
