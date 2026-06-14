from pathlib import Path

import pymupdf

from extractor.core import extract_pdf
from extractor.reporting import build_corpus_report


def test_corpus_report_labels_cross_engine_results_as_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "source.txt"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Exact text for metric reporting.")
    document.save(source)
    document.close()

    report = build_corpus_report([extract_pdf(source, output)])

    assert report["evaluation_type"] == (
        "cross_engine_proxy_without_visual_ground_truth"
    )
    assert report["documents"] == 1
    assert report["micro_character_error_rate"] == 0.0
    assert report["micro_word_error_rate"] == 0.0
    assert "human transcription" in str(report["publication_warning"])
