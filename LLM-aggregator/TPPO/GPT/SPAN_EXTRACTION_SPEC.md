# TPPO Span-Extraction Spec (token-optimized)

You extract 31 sections from one Indonesian criminal trafficking decision
(`Pid.Sus` / TPPO). The boundary rules below are derived from the authoritative
TPPO schema `LLM-aggregator/Putusan-schema.md` and the template context in
`LLM-aggregator/TPPO Format.pdf`. The cleaned, **line-numbered** source is given
to you inline in the prompt (Mahkamah Agung boilerplate already removed). Do
**not** open the source file, the PDF, the schema doc, or any other guide —
everything you need is inline.

You do NOT write the final JSON output. You write ONLY a small *spans* file: for
each section, point to the line range(s) that contain its exact text, or give a
short literal value, or mark it empty. A deterministic post-processor slices the
exact contiguous excerpt from those line numbers. This is faithful extraction —
the excerpt comes verbatim from the source — at a fraction of the output tokens.

## Output contract (write this and nothing else)

A single JSON object. Every one of the 31 keys MUST be present, each with
exactly one of these three forms:

- `{"lines": [[start, end]]}` — inclusive 1-based line numbers from the inline
  source. Use this for any value spanning more than one short line. Use multiple
  pairs only for genuinely separate occurrences (e.g. multiple Terdakwa).
- `{"text": ["short value"]}` — a short, single-line literal value copied
  exactly. Use ONLY for short identity/date fields.
- `{"empty": true}` — no exact source excerpt exists after checking the direct
  label, aliases, boundary variants, and OCR variants.

```json
{"sections": {
  "judul": {"text": ["P U T U S A N"]},
  "nomor_putusan": {"text": ["Nomor 1008/Pid.Sus/2025/PN Mdn"]},
  "tuntutan": {"lines": [[45, 60]]},
  "penangkapan": {"empty": true}
}}
```

Rules:
- Prefer `lines` for the large sections (penahanan, tuntutan, dakwaan, saksi,
  ahli, terdakwa, surat, petunjuk_barang_bukti, fakta_hukum, pertimbangan_hukum,
  amar_putusan, siapa_yang_memutus, tanda_tangan_majelis). Never paste long text
  with `text`.
- Use `text` only for short fields (judul, nomor_putusan, irah_irah,
  nama_pengadilan_negeri, keterangan_perkara, identity fields nama_lengkap..
  pekerjaan, hari, tanggal, tahun). Give the value only, not the field label,
  unless the label is genuinely part of the answer.
- Do not summarize, translate, normalize OCR, or add reasoning. `text` values
  must appear verbatim in the source.
- `lines` ranges are contiguous: pick the first and last line of the passage.
  Removed boilerplate has already been deleted, so a passage that was split
  across a page break is now contiguous in the numbered lines.
- For multiple defendants keep all defendants in the same section array for that
  field, preserving order (`Terdakwa I`, `Terdakwa II`, …) when present.

## TPPO template variants (use template order to resolve ambiguous boundaries)

- `Pid.I.A.7.1 Khusus - Vrijspraak`: acquittal — ends with "tidak terbukti" / "Membebaskan".
- `Pid.I.A.7.2 Format Khusus Lepas`: release from all charges — ends with "lepas dari segala tuntutan hukum".
- `Pid.I.A.7.3 Format Khusus - Terbukti`: conviction — "terbukti secara sah dan meyakinkan", imprisonment/fine, possible TPPO restitution.

Template order: header → defendant identity (1–8) → arrest & detention → counsel
status → procedural review (`Setelah membaca`) → prosecution demand → plea/reply
→ dakwaan → evidence sequence → facts → legal consideration → special TPPO
consideration & restitution → evidence disposition / aggravating-mitigating /
costs / `Mengingat` → `MENGADILI` → closing formula.

## Section boundaries (BEFORE → AFTER anchors; OR-lists, case-insensitive)

1. judul — start of document → before `Nomor`. Title `P U T U S A N` / `PUTUSAN` / `PENETAPAN`.
2. nomor_putusan — after `Putusan` → before `Pengadilan`. Full number incl. `Pid.Sus`.
3. irah_irah — after `PN` → before `Pengadilan Negeri`. Usually `DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA`.
4. nama_pengadilan_negeri — after `Esa` → before `yang mengadili perkara` / `yang mengadili perkara-perkara`.
5. keterangan_perkara — after `Mengadili` → before `dengan`. Case-type sentence up to the identity block.
6. nama_lengkap — after `1. Nama lengkap` / `Nama lengkap` → before `2. tempat` / `tempat`. Multiple defendants → multiple items.
7. tempat_lahir — after `2. tempat`/`tempat` → before `3. umur`/`umur`.
8. umur_tanggal_lahir — after `3. umur`/`umur` → before `4. Jenis`/`Jenis`.
9. jenis_kelamin — after `4. Jenis`/`Jenis` → before `5. kebangsaan`/`kebangsaan`.
10. kebangsaan — after `5. kebangsaan`/`kebangsaan`/`Kewarganegaraan` → before `6. tempat`/`tempat`. If a `Pendidikan` line follows, stop before it.
11. tempat_tinggal — after `6. tempat`/`tempat` → before `7. Agama`/`Agama`.
12. agama — after `7. Agama`/`Agama` → before `8. pekerjaan`/`pekerjaan`.
13. pekerjaan — after `8. pekerjaan`/`pekerjaan` → before `terdakwa ditangkap` / `para terdakwa ditangkap` or next non-identity paragraph.
14. penangkapan — after `ditangkap sejak` / `ditangkap pada` / `surat perintah penangkapan` / `Terdakwa dilakukan`. Full arrest sentence through its end date/order reference. Do NOT borrow detention text if no arrest is stated.
15. penahanan — after `dalam tahanan` / `ditahan oleh :` / `ditahan dalam` / `Terdakwa dilakukan`. Include ALL detention stages. For TPPO search the marker `Khusus Penahanan Tindak Pidana TPPO`; the sequence may include Penyidik, Perpanjangan Penuntut Umum, first/second Perpanjangan Ketua PN, Penuntut Umum, Hakim/Majelis Hakim, Perpanjangan Ketua Pengadilan Tinggi, and second Perpanjangan Ketua PT (up to 12 stages). Include Penangguhan, Pembantaran, Pengalihan Penahanan, or `ditahan dalam perkara lain`.
16. tuntutan — after `mendengar pembacaan` / `mendengar pula` (`...tuntutan pidana ... pada pokoknya sebagai berikut`). Often the copied `amar surat tuntutan`: include numbered demand items, requested sentence, fine, restitution, evidence disposition, costs. Up to pembelaan / `Menimbang bahwa Terdakwa`.
17. dakwaan — after `didakwa berdasarkan surat dakwaan Penuntut Umum Nomor… tanggal… sebagai berikut`. All forms: tunggal, alternatif, subsidairitas, kumulatif, gabungan.
18. saksi — after `mengajukan` / `menghadirkan` / `menghadapkan`. Prosecution witnesses, defense witnesses (`saksi yang meringankan` / `a de charge`), verbalisan witnesses. Preserve each name, oath status, testimony, defendant response.
19. ahli — prosecution and defense experts, including expert statements read into the record.
20. terdakwa — the defendant's courtroom statement. Begins around `Menimbang, bahwa Terdakwa … di persidangan … memberikan keterangan` (match `Menimbang, bahwa [Tt]erdakwa` incl. `Terdakwa I/II/III`). Testimony, not identity.
21. surat — documentary/electronic evidence marked `Surat (termasuk alat bukti elektronik)` / `bukti surat`.
22. petunjuk_barang_bukti — submitted goods/evidence list after `Penuntut Umum mengajukan barang bukti sebagai berikut`. NOT later evidence-disposition reasoning.
23. fakta_hukum — from `…diperoleh fakta hukum sebagai berikut` and stops before the court starts element/legal analysis (`Majelis Hakim akan mempertimbangkan…`).
24. pertimbangan_hukum — element analysis under dakwaan forms (`DAKWAAN TUNGGAL/ALTERNATIF/SUBSIDAIRITAS/KUMULATIF/GABUNGAN`), conclusions, `KHUSUS PERKARA TPPO` restitution consideration, detention/evidence-disposition reasoning, aggravating/mitigating factors, costs, `Mengingat…`, up to but not including `MENGADILI`.
25. amar_putusan — from `MENGADILI` (incl. spaced `M E N G A D I L I`) through every numbered operative order up to `Demikianlah diputuskan`. For TPPO conviction include the restitution order (e.g. `Membebankan kepada Terdakwa untuk membayar restitusi … dalam waktu 14 (empat belas) hari … harta bendanya disita dan dilelang …`).
26. hari — deliberation day after `pada` → before `, tanggal`.
27. tanggal — after `hari, tanggal` → before `bulan`.
28. tahun — after `bulan` → before `, oleh` / `oleh`.
29. siapa_yang_memutus — after `oleh` / `oleh kami,` → before `sebagai hakim`. Prefer judges after `oleh … sebagai Hakim Ketua … masing-masing sebagai Hakim Anggota`.
30. panitera_pengganti — after `Panitera` / `dibantu oleh` → before `pada Pengadilan Negeri`.
31. tanda_tangan_majelis — signature block after `Hakim Ketua,` → `Panitera Pengganti,`.

## OCR artifacts & edge cases

- Letter-spaced text (`M E N G A D I L I`): treat as the same anchor.
- OCR typos are real and meaningful: `sebgai berikut:`, `mempertimbangakan`, `berupa ;l`.
- Numbered field prefixes (`1. Nama lengkap`) are optional locating syntax.
- A `Pendidikan` line may appear between Kebangsaan (10) and Tempat Tinggal (11); exclude it from both.
- Preserve original line breaks, punctuation, spelling, and OCR artifacts in every excerpt.
