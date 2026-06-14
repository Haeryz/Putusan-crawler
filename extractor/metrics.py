from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Indel, Levenshtein

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    character_error_rate: float
    character_accuracy: float
    content_character_error_rate: float
    content_character_accuracy: float
    word_error_rate: float
    word_accuracy: float
    character_similarity: float
    token_recall: float
    token_precision: float
    token_f1: float
    primary_characters: int
    reference_characters: int
    character_edits: int
    reference_character_units: int
    word_edits: int
    reference_word_units: int

    @property
    def minimum_score(self) -> float:
        return min(self.content_character_accuracy, self.token_recall)


def normalize_for_sequence_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(normalized.split())


def word_sequence(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_RE.findall(normalized)


def tokenize(text: str) -> Counter[str]:
    return Counter(word_sequence(text))


def _error_rate(edits: int, reference_units: int, primary_units: int) -> float:
    if reference_units:
        return edits / reference_units
    return 0.0 if primary_units == 0 else 1.0


def compare_text(primary: str, reference: str) -> FidelityMetrics:
    primary_sequence = normalize_for_sequence_comparison(primary)
    reference_sequence = normalize_for_sequence_comparison(reference)
    character_edits = Levenshtein.distance(primary_sequence, reference_sequence)
    character_error_rate = _error_rate(
        character_edits,
        len(reference_sequence),
        len(primary_sequence),
    )

    primary_chars = normalize_for_comparison(primary)
    reference_chars = normalize_for_comparison(reference)
    content_character_edits = Levenshtein.distance(primary_chars, reference_chars)
    content_character_error_rate = _error_rate(
        content_character_edits,
        len(reference_chars),
        len(primary_chars),
    )
    character_similarity = Indel.normalized_similarity(
        primary_chars,
        reference_chars,
        score_cutoff=0.0,
    )

    primary_words = word_sequence(primary)
    reference_words = word_sequence(reference)
    word_edits = Levenshtein.distance(primary_words, reference_words)
    word_error_rate = _error_rate(
        word_edits,
        len(reference_words),
        len(primary_words),
    )
    primary_tokens = Counter(primary_words)
    reference_tokens = Counter(reference_words)
    overlap = sum((primary_tokens & reference_tokens).values())
    primary_count = sum(primary_tokens.values())
    reference_count = sum(reference_tokens.values())
    token_precision = overlap / primary_count if primary_count else float(not reference_count)
    token_recall = overlap / reference_count if reference_count else float(not primary_count)
    token_f1 = (
        2 * token_precision * token_recall / (token_precision + token_recall)
        if token_precision + token_recall
        else 0.0
    )

    return FidelityMetrics(
        character_error_rate=character_error_rate,
        character_accuracy=max(0.0, 1.0 - character_error_rate),
        content_character_error_rate=content_character_error_rate,
        content_character_accuracy=max(0.0, 1.0 - content_character_error_rate),
        word_error_rate=word_error_rate,
        word_accuracy=max(0.0, 1.0 - word_error_rate),
        character_similarity=character_similarity,
        token_recall=token_recall,
        token_precision=token_precision,
        token_f1=token_f1,
        primary_characters=len(primary_chars),
        reference_characters=len(reference_chars),
        character_edits=character_edits,
        reference_character_units=len(reference_sequence),
        word_edits=word_edits,
        reference_word_units=len(reference_words),
    )
