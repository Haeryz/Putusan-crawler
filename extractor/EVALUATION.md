# PDF Text Extraction Evaluation

## Scope

This document describes the evaluation of raw-text extraction from the
Indonesian juvenile court decision PDFs in:

```text
downloads/kasus anak/pdfs
```

The evaluated corpus contains 236 PDF documents and 6,519 pages. Extraction is
performed directly from each PDF's embedded text layer. The pipeline does not
summarize, paraphrase, translate, or use a large language model.

The primary extractor is pypdf. PyMuPDF is used as an independent secondary
extractor for production quality control. The UTF-8 output files, per-document
audit records, and aggregate metrics are written to:

```text
downloads/kasus anak/raw-text
```

## Important Claim Boundary

The current measurements compare two PDF extraction engines. They measure
cross-engine agreement, not accuracy against the text visually rendered on a
page or against human transcription.

Consequently, the results can support this claim:

> Pypdf and PyMuPDF produce highly consistent text representations on this
> born-digital court-decision corpus.

They cannot independently support this stronger claim:

> The extracted text contains at least 95% of all visually present PDF text.

That stronger claim requires manually verified rendered-page ground truth.

## Normalization

All comparisons use Unicode NFKC normalization and case-folding.

Two character representations are evaluated:

1. Sequence representation: every whitespace run is replaced by one ASCII
   space. This retains word boundaries and reading order while ignoring layout
   spacing.
2. Content representation: all whitespace is removed. This isolates textual
   character differences from PDF-engine whitespace reconstruction.

Words are Unicode word-token sequences produced with Python's Unicode-aware
`\w+` regular expression.

## Metrics

### Character Error Rate

Character Error Rate (CER) measures character substitutions, deletions, and
insertions:

```text
CER = (substitutions + deletions + insertions) / reference characters
```

The implementation uses Levenshtein distance. Lower is better. Character
accuracy is reported as `max(0, 1 - CER)`. CER is used as a primary OCR
evaluation measure by Vidal-Gorene and Kindt (2026)
[arXiv:2603.09470](https://arxiv.org/abs/2603.09470) and by Nagaonkar et al.
(2025) [arXiv:2502.06445](https://arxiv.org/abs/2502.06445).

### Content Character Error Rate

Content CER uses the same equation after removing all whitespace. It is useful
for this corpus because PDF engines reconstruct indentation and watermark
spacing differently even when they recover the same visible characters. This
is a normalization variant of Levenshtein CER rather than a separate standard
metric. Normalized edit-distance evaluation is also used in document parsing
benchmarks such as OmniDocBench (Ouyang et al., 2024)
[arXiv:2412.07626](https://arxiv.org/abs/2412.07626).

The operational 95% gate is:

```text
content character accuracy >= 0.95
```

This gate is applied at both document and page level.

### Word Error Rate

Word Error Rate (WER) applies Levenshtein distance to the word sequence:

```text
WER = (substitutions + deletions + insertions) / reference words
```

WER is sensitive to both word recognition and reading order. Lower is better.
CER and WER are jointly reported in recent OCR benchmarks, including
Vidal-Gorene and Kindt (2026)
[arXiv:2603.09470](https://arxiv.org/abs/2603.09470) and Nagaonkar et al.
(2025) [arXiv:2502.06445](https://arxiv.org/abs/2502.06445).

### Token Precision, Recall, and F1

These metrics compare word multisets and ignore reading order:

```text
precision = overlapping tokens / extracted tokens
recall    = overlapping tokens / reference tokens
F1        = 2 * precision * recall / (precision + recall)
```

They distinguish content coverage from ordering errors. A document can have
high token F1 but worse WER when the same words occur in a different order.
This separation is important in document extraction because reading order is
an independently evaluated capability in Eclair (Karmanov et al., 2025)
[arXiv:2502.04223](https://arxiv.org/abs/2502.04223) and OmniDocBench
(Ouyang et al., 2024)
[arXiv:2412.07626](https://arxiv.org/abs/2412.07626).

### Aggregation and Uncertainty

The report includes:

- micro CER and WER, weighted by reference character or word count;
- macro mean and median across documents;
- 5th and 95th document percentiles;
- minimum and maximum document values; and
- deterministic document-level bootstrap 95% confidence intervals using
  2,000 resamples and seed `20260611`.

Documents, rather than pages or characters, are the independent bootstrap
units. Reporting uncertainty and distributions complements the point estimates
used by OCR benchmarks; it does not replace evaluation against annotated
references. The manually annotated evaluation design in Nagaonkar et al.
(2025) provides an example of reference-based benchmarking
[arXiv:2502.06445](https://arxiv.org/abs/2502.06445).

## Cross-Engine Results

The following results were generated on June 11, 2026:

| Metric | Result |
|---|---:|
| Documents | 236 |
| Pages | 6,519 |
| Micro CER | 2.229% |
| Mean document CER | 2.343% |
| Median document CER | 2.240% |
| Mean CER bootstrap 95% CI | [2.297%, 2.395%] |
| Micro WER | 3.173% |
| Mean document WER | 3.341% |
| Median document WER | 3.066% |
| Mean WER bootstrap 95% CI | [3.258%, 3.426%] |
| Mean token precision | 96.713% |
| Mean token recall | 98.213% |
| Mean token F1 | 97.457% |

All documents have document-level content CER below 5%. However, 93 documents
contain at least one page with content CER above 5% between the two engines.
These documents are marked `review`; 143 documents pass the page-level gate.
No document failed extraction.

The worst document-level content CER is 4.494%. The highest observed page-level
content CER is 7.818%.

## Interpretation

The low corpus-level CER and WER indicate strong agreement between the two
engines. The difference between token recall and WER is consistent with
reading-order and token-boundary differences, particularly around repeated
diagonal Mahkamah Agung watermark text and page furniture.

The page-level results also show why a whole-document average alone is
insufficient. A long document may have excellent aggregate agreement while
containing one locally problematic page. Production review routing therefore
uses both document-level and page-level checks.

These values must be described as proxy agreement metrics in any report or
paper until human ground truth is available.

## Publication-Grade Gold Benchmark

For a conference-quality accuracy claim, construct a held-out benchmark as
follows:

1. Define the target population as all 6,519 rendered PDF pages.
2. Select pages using a published random seed and stratify by document length,
   year, court, text density, and observed cross-engine disagreement.
3. Keep a purely random stratum for unbiased corpus estimates. A separate
   high-disagreement stratum may be used for failure analysis but must not be
   mixed into the unbiased estimate without sampling weights.
4. Render selected pages at a fixed resolution.
5. Have two annotators independently transcribe every visible textual element
   in an explicitly defined reading order.
6. Define in advance whether watermarks, headers, footers, page numbers,
   signatures, and disclaimer text are included.
7. Adjudicate every disagreement to produce one gold transcription.
8. Keep the gold set unavailable during extractor selection and tuning.
9. Evaluate pypdf, PyMuPDF, and relevant OCR baselines against the same gold
   pages using CER, content CER, WER, and token F1.
10. Report micro and macro metrics, document-bootstrap confidence intervals,
    per-stratum results, worst-case pages, processing time, and error taxonomy.
11. Release the page identifiers, sampling code, annotation guide, metric
    implementation, and legally distributable annotations.

At least part of the annotation set should be double-annotated. Human
transcription agreement should be reported with the same CER/WER metrics before
adjudication. This establishes the practical noise floor of the benchmark.

## Reproduction

Install and run entirely through UV:

```bash
uv sync
uv run pdf-extractor "downloads/kasus anak/pdfs" \
  --output-dir "downloads/kasus anak/raw-text" \
  --workers 4 \
  --overwrite
uv run pytest -q
```

Generated artifacts:

| Artifact | Description |
|---|---|
| `raw-text/*.txt` | Complete UTF-8 extraction, with form-feed page boundaries |
| `raw-text/audit.jsonl` | Per-document and worst-page metrics |
| `raw-text/metrics-summary.json` | Aggregate distributions and confidence intervals |

The source PDF SHA-256 is stored in every audit record to bind each result to
the exact evaluated input.

## Relevant Evaluation Practice

CER and WER are standard reference-based OCR measures. Recent examples include:

- Vidal-Gorene and Kindt, *The Patrologia Graeca Corpus: OCR, Annotation, and
  Open Release of Noisy Nineteenth-Century Polytonic Greek Editions*, 2026:
  https://arxiv.org/abs/2603.09470
- Nagaonkar et al., *Benchmarking Vision-Language Models on Optical Character
  Recognition in Dynamic Video Environments*, 2025:
  https://arxiv.org/abs/2502.06445
- Ouyang et al., *OmniDocBench: Benchmarking Diverse PDF Document Parsing with
  Comprehensive Annotations*, 2024:
  https://arxiv.org/abs/2412.07626
- Karmanov et al., *Eclair: Extracting Content and Layout with Integrated
  Reading Order for Documents*, 2025:
  https://arxiv.org/abs/2502.04223

The common requirement relevant here is reference annotation: standard metric
names do not make an evaluation publication-grade when the reference is merely
another automated extractor.
