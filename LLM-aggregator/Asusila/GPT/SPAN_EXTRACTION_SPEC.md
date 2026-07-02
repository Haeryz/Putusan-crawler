# Anak Span-Extraction Spec (token-optimized)

You extract 31 sections from one Indonesian juvenile decision (`Pid.Sus-Anak`).
The cleaned, **line-numbered** source is given to you inline in the prompt
(boilerplate already removed). Do **not** open the source file, the PDF, or any
other guide — everything you need is inline.

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
  pairs only for genuinely separate occurrences (e.g. multiple Anak).
- `{"text": ["short value"]}` — a short, single-line literal value copied
  exactly. Use ONLY for short identity/date fields.
- `{"empty": true}` — no exact source excerpt exists after checking labels,
  aliases, and OCR variants.

```json
{"sections": {
  "judul": {"text": ["P U T U S A N"]},
  "nomor_putusan": {"text": ["Nomor 1/Pid.Sus-Anak/2026/PN Bdw"]},
  "tuntutan": {"lines": [[45, 60]]},
  "penangkapan": {"empty": true}
}}
```

Rules:
- Prefer `lines` for the large sections (penahanan, tuntutan, dakwaan, saksi,
  ahli, terdakwa, surat, petunjuk_barang_bukti, fakta_hukum, pertimbangan_hukum,
  amar_putusan, siapa_yang_memutus, tanda_tangan_majelis). Never paste long text
  with `text`.
- Use `text` for short fields (judul, nomor_putusan, irah_irah,
  nama_pengadilan_negeri, keterangan_perkara, the identity fields nama_lengkap..
  pekerjaan, hari, tanggal, tahun, panitera_pengganti). Give the value only, not
  the field label, unless the label is genuinely part of the answer.
- Do not summarize, translate, normalize, or add reasoning. `text` values must
  appear verbatim in the source.
- `lines` ranges are contiguous: pick the first and last line of the passage.
  Removed boilerplate has already been deleted, so a passage that was split
  across a page break is now contiguous in the numbered lines.

## Section boundaries (BEFORE / AFTER anchors are OR-lists, case-insensitive)

1. judul — title `P U T U S A N` / `PUTUSAN` / `PENETAPAN`. Before `Nomor`.
2. nomor_putusan — full decision number incl. `Pid.Sus-Anak`.
3. irah_irah — usually `DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA`.
4. nama_pengadilan_negeri — `Pengadilan Anak pada Pengadilan Negeri ...` court phrase.
5. keterangan_perkara — case description (procedure/level/Anak) up to `1. Nama lengkap`.
6. nama_lengkap — identity value after `1. Nama lengkap`. Multiple Anak → multiple items.
7. tempat_lahir — after `2. Tempat lahir`.
8. umur_tanggal_lahir — after `3. Umur/tanggal lahir` / `Umur`.
9. jenis_kelamin — after `4. Jenis kelamin`.
10. kebangsaan — after `5. Kebangsaan` (= `Kewarganegaraan`). Stop before optional `Pendidikan`.
11. tempat_tinggal — after `6. Tempat tinggal`.
12. agama — after `7. Agama`.
13. pekerjaan — after `8. Pekerjaan`; stop before arrest/detention or next paragraph.
14. penangkapan — arrest wording/dates only (`ditangkap ...`, `surat perintah penangkapan`). NOT detention.
15. penahanan — every detention stage/extension (Penyidik, Penuntut Umum, Ketua PN, Hakim) incl. penangguhan/pembantaran/pengalihan or detention in another case.
16. tuntutan — full prosecution demand after `Setelah mendengar pembacaan tuntutan pidana ...` up to pembelaan / `Menimbang bahwa Anak`.
17. dakwaan — full charge after `didakwa berdasarkan surat dakwaan ...` (all forms; `catatan dakwaan` in short procedure).
18. saksi — prosecution/victim/child/defense/verbalisan witnesses, oaths, testimony.
19. ahli — prosecution and defense experts, incl. statements read in court.
20. terdakwa — the Anak/defendant courtroom statement after `Anak/Para Anak di persidangan telah memberikan keterangan` (testimony, not identity).
21. surat — documentary/electronic evidence (`Surat (termasuk alat bukti elektronik)`).
22. petunjuk_barang_bukti — submitted goods/evidence inventory after `mengajukan barang bukti sebagai berikut`. Not later disposition reasoning.
23. fakta_hukum — from `diperoleh fakta hukum sebagai berikut` to start of legal/element analysis.
24. pertimbangan_hukum — element analysis, Anak sanction/social-inquiry reasoning, detention/evidence-disposition reasoning, aggravating/mitigating, costs, `Mengingat...`, up to but not including `MENGADILI`.
25. amar_putusan — from `MENGADILI` through every numbered operative order, up to `Demikianlah diputuskan`.
26. hari — deliberation day after `pada hari`.
27. tanggal — deliberation date.
28. tahun — deliberation year.
29. siapa_yang_memutus — deciding judges after `Demikianlah diputuskan ... oleh ...`.
30. panitera_pengganti — substitute clerk after `dibantu oleh` / `Panitera Pengganti`.
31. tanda_tangan_majelis — signature block (judges + clerk) up to `Catatan:`.
