from extractor.metrics import compare_text, normalize_for_comparison


def test_normalize_for_comparison_ignores_layout_whitespace() -> None:
    assert normalize_for_comparison("A  court\norder") == "acourtorder"


def test_compare_text_reports_full_match_across_layout_changes() -> None:
    metrics = compare_text("Putusan\nNomor 12", "Putusan Nomor 12")

    assert metrics.character_similarity == 1.0
    assert metrics.token_recall == 1.0
    assert metrics.token_precision == 1.0
    assert metrics.token_f1 == 1.0


def test_compare_text_detects_missing_tokens() -> None:
    metrics = compare_text("satu dua", "satu dua tiga empat")

    assert metrics.token_recall == 0.5
    assert metrics.token_precision == 1.0
    assert metrics.minimum_score < 0.95
