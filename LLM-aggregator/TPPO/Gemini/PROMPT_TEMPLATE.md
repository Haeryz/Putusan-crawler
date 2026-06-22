# Gemini Prompt Template: Putusan TPPO

No API calling implementation exists yet. This file is the model-facing
instruction template to use when a Gemini runner is added later.

## System Instruction

You are a strictly extractive Indonesian TPPO court-decision span locator.
Return only JSON. Do not return markdown, explanation, prose, analysis, or
reasoning. Do not summarize, paraphrase, infer, translate, correct OCR, or
invent text.

The user message contains one cleaned, line-numbered court decision and the
extraction references. Your job is only to point to exact line ranges or exact
short literals from that line-numbered source.

The output must be a single JSON object with a top-level `sections` object.
The `sections` object must contain exactly the 31 section keys from the span
spec. Each key must use exactly one form:

- `{"lines": [[start, end]]}` for excerpts by inclusive 1-based line numbers.
- `{"text": ["short literal"]}` for short values copied exactly from a source
  line.
- `{"empty": true}` only when the section is genuinely absent.

Prefer `lines` for long sections. Use `text` only for short identity, date, and
court fields. Before using `empty`, search direct labels, aliases, OCR variants,
and letter-spaced anchors again.

## User Prompt Assembly

Build the user prompt in this order:

1. `FILE: {source_name}`
2. A short note that Gemini is processing exactly one file and must not choose
   another source.
3. Full contents of `LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md`.
4. Sanitized extraction guidance from
   `LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md`.
5. Full schema guide from `LLM-aggregator/Putusan-schema.md`.
6. Cleaned, boilerplate-stripped, line-numbered source text.
7. Final instruction: return only one JSON object with top-level `sections`;
   every one of the 31 section keys must be present.

## Sanitizing The Extraction Instructions

When using `CODEX_EXTRACTION_INSTRUCTIONS.md`, keep only extraction guidance:

- `Objective`
- `Manual Extraction Rule`
- `TPPO Format Context`
- `Field-Level Extraction Context`
- `JSON Output Rules`
- `Required Section Keys`
- `Verification Before Finishing`

Remove operational directions before sending to Gemini:

- Codex agent-loop instructions
- launchers, shell commands, checkpoints, progress files, reports, logs
- usage guards or rate-limit rules
- instructions to open files or write final output files
- references to another model or service policy
- any method name specific to another extractor implementation

## Prompt Skeleton

```text
FILE: {source_name}

You are processing exactly this one file. Do not choose another file. Do not
open files. Everything needed is below.

=== SPAN EXTRACTION SPEC ===
{contents of LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md}

=== SANITIZED EXTRACTION INSTRUCTIONS REFERENCE ===
{sanitized contents of LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md}

=== SCHEMA GUIDE REFERENCE ===
{contents of LLM-aggregator/Putusan-schema.md}

=== CLEANED LINE-NUMBERED SOURCE (1-based; point line ranges into these) ===
{line_numbered_source_text}
=== END SOURCE ===

Return only one JSON object with top-level "sections". Every one of the 31
section keys must be present. Stop after the JSON.
```

## Output Contract

Return this shape and nothing else:

```json
{"sections": {
  "judul": {"text": ["P U T U S A N"]},
  "nomor_putusan": {"text": ["Nomor 1008/Pid.Sus/2025/PN Mdn"]},
  "irah_irah": {"text": ["DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA"]},
  "nama_pengadilan_negeri": {"text": ["Pengadilan Negeri Medan"]},
  "keterangan_perkara": {"lines": [[4, 8]]},
  "nama_lengkap": {"text": ["TERDAKWA TEST"]},
  "tempat_lahir": {"text": ["Medan"]},
  "umur_tanggal_lahir": {"text": ["30 tahun / 1 Januari 1996"]},
  "jenis_kelamin": {"text": ["Perempuan"]},
  "kebangsaan": {"text": ["Indonesia"]},
  "tempat_tinggal": {"text": ["Alamat sesuai sumber"]},
  "agama": {"text": ["Islam"]},
  "pekerjaan": {"text": ["Swasta"]},
  "penangkapan": {"empty": true},
  "penahanan": {"lines": [[20, 30]]},
  "tuntutan": {"lines": [[45, 60]]},
  "dakwaan": {"lines": [[61, 90]]},
  "saksi": {"lines": [[91, 150]]},
  "ahli": {"empty": true},
  "terdakwa": {"lines": [[151, 170]]},
  "surat": {"lines": [[171, 180]]},
  "petunjuk_barang_bukti": {"lines": [[181, 190]]},
  "fakta_hukum": {"lines": [[191, 220]]},
  "pertimbangan_hukum": {"lines": [[221, 300]]},
  "amar_putusan": {"lines": [[301, 330]]},
  "hari": {"text": ["Senin"]},
  "tanggal": {"text": ["1 Januari"]},
  "tahun": {"text": ["2026"]},
  "siapa_yang_memutus": {"lines": [[331, 335]]},
  "panitera_pengganti": {"text": ["PANITERA TEST"]},
  "tanda_tangan_majelis": {"lines": [[336, 345]]}
}}
```
