from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Indel

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    character_similarity: float
    token_recall: float
    token_precision: float
    token_f1: float
    primary_characters: int
    reference_characters: int

    @property
    def minimum_score(self) -> float:
        return min(self.character_similarity, self.token_recall)


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(normalized.split())


def tokenize(text: str) -> Counter[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return Counter(_TOKEN_RE.findall(normalized))


def compare_text(primary: str, reference: str) -> FidelityMetrics:
    primary_chars = normalize_for_comparison(primary)
    reference_chars = normalize_for_comparison(reference)
    character_similarity = Indel.normalized_similarity(
        primary_chars,
        reference_chars,
        score_cutoff=0.0,
    )

    primary_tokens = tokenize(primary)
    reference_tokens = tokenize(reference)
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
        character_similarity=character_similarity,
        token_recall=token_recall,
        token_precision=token_precision,
        token_f1=token_f1,
        primary_characters=len(primary_chars),
        reference_characters=len(reference_chars),
    )
