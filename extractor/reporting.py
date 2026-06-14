from __future__ import annotations

import random
from dataclasses import asdict
from statistics import mean, median
from typing import Callable, Sequence

from extractor.core import ExtractionResult


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 20260611,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        mean(generator.choices(values, k=sample_size))
        for _ in range(samples)
    )
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _distribution(values: Sequence[float]) -> dict[str, float | list[float]]:
    return {
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "p05": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
        "minimum": min(values, default=0.0),
        "maximum": max(values, default=0.0),
        "bootstrap_mean_95_ci": _bootstrap_mean_ci(values),
    }


def build_corpus_report(results: Sequence[ExtractionResult]) -> dict[str, object]:
    metric_getters: dict[str, Callable[[ExtractionResult], float]] = {
        "character_error_rate": lambda result: result.metrics.character_error_rate,
        "content_character_error_rate": (
            lambda result: result.metrics.content_character_error_rate
        ),
        "word_error_rate": lambda result: result.metrics.word_error_rate,
        "token_precision": lambda result: result.metrics.token_precision,
        "token_recall": lambda result: result.metrics.token_recall,
        "token_f1": lambda result: result.metrics.token_f1,
    }
    distributions = {
        name: _distribution([getter(result) for result in results])
        for name, getter in metric_getters.items()
    }
    total_character_edits = sum(
        result.metrics.character_edits for result in results
    )
    total_reference_characters = sum(
        result.metrics.reference_character_units for result in results
    )
    total_word_edits = sum(result.metrics.word_edits for result in results)
    total_reference_words = sum(
        result.metrics.reference_word_units for result in results
    )

    return {
        "evaluation_type": "cross_engine_proxy_without_visual_ground_truth",
        "primary_extractor": "pypdf",
        "reference_extractor": "pymupdf",
        "documents": len(results),
        "pages": sum(result.pages for result in results),
        "passed": sum(result.status == "passed" for result in results),
        "review": sum(result.status == "review" for result in results),
        "micro_character_error_rate": (
            total_character_edits / total_reference_characters
            if total_reference_characters
            else 0.0
        ),
        "micro_word_error_rate": (
            total_word_edits / total_reference_words
            if total_reference_words
            else 0.0
        ),
        "document_distributions": distributions,
        "metric_definition": {
            "normalization": "Unicode NFKC plus case-folding",
            "character_error_rate": (
                "Levenshtein edits / reference characters after collapsing "
                "whitespace runs to one ASCII space"
            ),
            "content_character_error_rate": (
                "Levenshtein edits / reference characters after removing whitespace"
            ),
            "word_error_rate": (
                "Levenshtein word-sequence edits / reference word count"
            ),
            "token_precision_recall_f1": (
                "multiset word overlap; insensitive to reading order"
            ),
        },
        "publication_warning": (
            "These values measure agreement between two extraction engines, not "
            "accuracy against human transcription or rendered-page ground truth."
        ),
        "documents_detail": [asdict(result) for result in results],
    }
