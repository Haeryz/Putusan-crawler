# PDF Raw Text Extractor

This package extracts the complete embedded text layer from each PDF. It does
not summarize, rewrite, or call an LLM. Page boundaries are preserved as form
feed characters (`\f`) in UTF-8 text files.

See [EVALUATION.md](EVALUATION.md) for the complete metric definitions,
full-corpus results, limitations, and publication-grade annotation protocol.

Pypdf is the primary extractor because it reconstructs the diagonal Mahkamah
Agung watermark and body text more readably for this corpus. Each result is
independently extracted again with PyMuPDF.

The evaluation reports:

- Character Error Rate (CER): character-level Levenshtein edits divided by
  reference character count after Unicode NFKC normalization, case-folding,
  and whitespace-run collapsing.
- Content CER: the same calculation after removing whitespace. This isolates
  textual content differences from PDF layout whitespace.
- Word Error Rate (WER): word-sequence Levenshtein edits divided by reference
  word count. This is sensitive to omissions, substitutions, and reading order.
- Token precision, recall, and F1: multiset word overlap. These are insensitive
  to reading order and therefore separate content coverage from ordering.
- Macro distributions, micro CER/WER, and deterministic document-bootstrap
  95% confidence intervals in `metrics-summary.json`.

A PDF passes the operational extraction gate only when:

- every page has embedded text;
- whole-document content character accuracy (`1 - content CER`) is at least
  95%; and
- every page also reaches 95% content character accuracy.

This validation measures agreement between two independent PDF text engines.
It is not accuracy against rendered-page or human-transcribed ground truth.
Therefore, the full-corpus numbers are useful production quality-control
proxies, but must not be presented as final publication accuracy.

For a conference paper, create a preregistered stratified random sample of
pages, have two annotators independently transcribe all visible text in reading
order, adjudicate disagreements, publish the sampling seed and annotation
guide, and compute CER/WER against that held-out gold set. Report micro and
macro values, bootstrap confidence intervals, per-document distributions, and
failure categories. Image-only pages are marked `review` and require OCR or
manual transcription.

## Run

From the repository root:

```bash
uv sync
uv run pdf-extractor "downloads/kasus anak/pdfs" \
  --output-dir "downloads/kasus anak/raw-text" \
  --workers 4
```

The output directory contains one `.txt` file per PDF, `audit.jsonl`, and
`metrics-summary.json`.
Each audit record includes source and output paths, source SHA-256, page and
character counts, elapsed time, fidelity metrics, pages below threshold, and
warnings. The command exits nonzero when any document needs review or fails.

Re-running without `--overwrite` protects existing output. Add `--overwrite`
when intentionally regenerating all text and audit records.
