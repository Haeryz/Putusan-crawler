from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import llm_aggregator.anak_deepseek as anak_deepseek
from llm_aggregator.anak_deepseek import (
    ApiResult,
    ResponseError,
    SECTION_KEYS,
    ValidationError,
    align_source_excerpt,
    call_deepseek,
    compact_source,
    load_failure_counts,
    load_individual_output,
    output_token_budget,
    output_path,
    parse_model_json,
    sha256_text,
    empty_sections_for_report,
    validate_minimum_evidence,
    validate_record,
    repair_empty_sections,
    write_individual_output,
    write_no_text_output,
)


def empty_record() -> dict[str, list[str]]:
    return {key: [] for key in SECTION_KEYS}


def test_cli_defaults_to_reasoning_off() -> None:
    args = anak_deepseek.build_parser().parse_args([])
    assert args.reasoning_effort == "off"
    assert args.workers == 8
    assert args.network_failure_threshold == 3
    assert args.network_cooldown == 60
    assert args.max_output_tokens == 32768
    assert not args.skip_empty_text


@pytest.mark.parametrize(
    "error",
    [
        "NameResolutionError: Failed to resolve api.inference.wandb.ai",
        "socket error: getaddrinfo failed",
        "ProxyError: tunnel connection failed",
        (
            "ResponseError: request failed after 1 attempts: retryable HTTP 429: "
            '{"error":{"code":"rate_limit_exceeded","message":"concurrency limit '
            'reached for requests: zai-org/GLM-5.2-project limit reached"}}'
        ),
    ],
)
def test_infrastructure_errors_are_identified(error: str) -> None:
    assert anak_deepseek.is_infrastructure_error(error)


def test_read_timeout_is_not_treated_as_infrastructure_outage() -> None:
    assert not anak_deepseek.is_infrastructure_error(
        "ReadTimeout: HTTPSConnectionPool read timed out"
    )


def test_user_prompt_uses_gpt_span_extraction_prompt_shape() -> None:
    prompt = anak_deepseek.build_user_prompt("sample.txt", "PUTUSAN\nNomor 1")
    span_spec = Path("LLM-aggregator/Anak/GPT/SPAN_EXTRACTION_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "You are Codex running the token-optimized Anak span-extraction task in:" in prompt
    assert "Assigned source: downloads/kasus anak/raw-text/sample.txt" in prompt
    assert "source file, the SKKMA PDF" in prompt
    assert "YOUR ONLY OUTPUT: return the spans JSON object and nothing else." in prompt
    assert "=== CLEANED LINE-NUMBERED SOURCE (1-based; point your line ranges into these) ===" in prompt
    assert "   1| PUTUSAN" in prompt
    assert "   2| Nomor 1" in prompt
    assert span_spec in prompt


def test_reasoning_uses_full_budget_and_off_stays_dynamic() -> None:
    short_source = "PUTUSAN"

    assert output_token_budget(
        short_source,
        max_output_tokens=32768,
        reasoning_effort="medium",
    ) == 32768
    assert output_token_budget(
        short_source,
        max_output_tokens=32768,
        reasoning_effort="off",
    ) == 4096


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        text: str = "",
        finish_reason: str | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.finish_reason = finish_reason

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self, decode_unicode: bool = False) -> list[str]:
        content = (
            self.payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage = self.payload.get("usage", {})
        choice: dict[str, Any] = {"delta": {"content": content}}
        if self.finish_reason is not None:
            choice["finish_reason"] = self.finish_reason
        event = {
            "choices": [choice],
            "usage": usage,
        }
        return [f"data: {json.dumps(event)}", "data: [DONE]"]


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls = 0
        self.last_json: dict[str, Any] | None = None

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls += 1
        self.last_json = kwargs.get("json")
        return next(self.responses)


def completion(content: str) -> FakeResponse:
    return FakeResponse(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }
    )


def test_compact_source_removes_only_known_boilerplate() -> None:
    source = (
        "Mahkamah Agung Republik Indonesia\n"
        "PUTUSAN\n\n\nNomor 1\n"
        "Halaman 1 dari 2\n"
        "Isi tetap\n"
        "Disclaimer\n"
        "boilerplate footer\n"
    )

    assert compact_source(source) == "PUTUSAN\n\nNomor 1\nIsi tetap"


def test_compact_source_removes_repeated_page_footer_without_truncating() -> None:
    source = (
        "PUTUSAN\n"
        "Halaman 1 dari 2 Putusan Nomor 1\n"
        "Disclaimer\n"
        "footer sentence\n"
        "Email : example@example.test Halaman 1\n"
        "Mahkamah Agung Republik Indonesia\n"
        "ISI HALAMAN DUA\n"
    )

    assert compact_source(source) == "PUTUSAN\nISI HALAMAN DUA"


def test_align_source_excerpt_restores_original_whitespace() -> None:
    source = "Pengadilan Negeri Unaaha yang mengadili\nperkara  pidana anak"
    model_text = "Pengadilan Negeri Unaaha yang mengadili perkara pidana anak"

    assert align_source_excerpt(model_text, source) == source


def test_validate_record_requires_exact_source_spans() -> None:
    source = "PUTUSAN\nNomor 1/Pid.Sus-Anak/2026"
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    record["nomor_putusan"] = ["Nomor 1/Pid.Sus-Anak/2026"]

    assert validate_record(record, source) == record

    record["judul"] = ["Putusan"]
    with pytest.raises(ValidationError, match="not a contiguous source excerpt"):
        validate_record(record, source)


def test_minimum_evidence_rejects_empty_200_and_missing_labeled_fields() -> None:
    source = "Nomor 1\nNama lengkap: ANAK"
    record = empty_record()
    with pytest.raises(ValidationError, match="all 31 sections"):
        validate_minimum_evidence(record, source)

    record["nomor_putusan"] = ["Nomor 1"]
    with pytest.raises(ValidationError, match="nama_lengkap"):
        validate_minimum_evidence(record, source)


def test_minimum_evidence_rejects_court_decision_with_many_empty_sections() -> None:
    source = "\n".join(["PUTUSAN", "Nomor 1", "Nama lengkap: ANAK", *("isi" for _ in range(45))])
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    record["nomor_putusan"] = ["Nomor 1"]
    record["nama_lengkap"] = ["ANAK"]

    with pytest.raises(ValidationError, match="too many empty sections"):
        validate_minimum_evidence(record, source)


def test_repair_empty_sections_fills_penetapan_template_anchors() -> None:
    source = (
        "P E N E T A P A N\n"
        "Nomor 98/Pid.Sus/2026/PN Mpw\n"
        "DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA\n"
        "Terdakwa ditangkap sejak tanggal 18 Oktober 2025 sampai dengan tanggal 19\n"
        "Oktober 2025;\n"
        "Terdakwa ditahan dalam Tahanan Rutan oleh:\n"
        "1. Penyidik sejak tanggal 18 Oktober 2025 sampai dengan tanggal 6 November 2025;\n"
        "Setelah membaca:\n"
        "5. Surat Keterangan Kematian Nomor 400 atas nama Abdul Malik;\n"
        "Menimbang, bahwa Terdakwa diajukan ke persidangan oleh Penuntut\n"
        "Umum didakwa berdasarkan surat dakwaan sebagai berikut:\n"
        "Bahwa terdakwa melakukan perbuatan sebagaimana dakwaan;\n"
        "Menimbang, bahwa di persidangan Penuntut Umum telah menghadirkan\n"
        "2 (dua) orang Saksi yang memberikan keterangan dibawah sumpah;\n"
        "Menimbang, bahwa Penuntut Umum juga telah menghadirkan barang\n"
        "bukti di persidangan berupa 1 (satu) buah Handphone;\n"
        "Menimbang, bahwa pada persidangan hari Selasa tanggal 21 April 2026\n"
        "dengan agenda pemeriksaan ahli dari Penuntut Umum, Penuntut Umum\n"
        "menyatakan tidak dapat menghadirkan Terdakwa dikarenakan Terdakwa telah\n"
        "meninggal dunia pada tanggal 18 April 2026;\n"
        "Menimbang, bahwa berdasarkan ketentuan Pasal 132 ayat (1) huruf b\n"
        "penuntutan terhadap Terdakwa dinyatakan gugur;\n"
        "Menimbang, bahwa terhadap barang bukti yang telah dihadirkan oleh\n"
        "Penuntut Umum di persidangan dikembalikan kepada Penuntut Umum;\n"
        "MENETAPKAN:\n"
        "1. Menyatakan kewenangan Penuntut Umum gugur;\n"
        "Demikianlah ditetapkan dalam sidang permusyawaratan Majelis Hakim\n"
        "Pengadilan Negeri Mempawah, pada hari Selasa, tanggal 21 April 2026 oleh\n"
        "kami, Rezki Fauzi, S.H., sebagai Hakim Ketua;\n"
    )
    record = empty_record()

    repaired = repair_empty_sections(record, source)

    assert repaired["judul"] == ["P E N E T A P A N"]
    assert repaired["irah_irah"] == [
        "DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA"
    ]
    assert repaired["penangkapan"][0].startswith("Terdakwa ditangkap")
    assert repaired["penahanan"][0].startswith("Terdakwa ditahan")
    assert repaired["dakwaan"][0].startswith("Menimbang, bahwa Terdakwa")
    assert repaired["petunjuk_barang_bukti"][0].startswith("Menimbang, bahwa Penuntut Umum")
    assert repaired["pertimbangan_hukum"][0].startswith("Menimbang, bahwa berdasarkan")
    assert repaired["amar_putusan"][0].startswith("MENETAPKAN")
    assert repaired["tanggal"] == ["21 April 2026"]
    assert repaired["tahun"] == ["2026"]


def test_diversion_penetapan_trial_sections_are_structurally_non_applicable() -> None:
    source = (
        "PENETAPAN\n"
        "Nomor 3/Pen.Div/2026/PN Mrs\n"
        "Membaca Laporan Pembimbing Kemasyarakatan tentang pelaksanaan\n"
        "Kesepakatan Diversi dalam perkara Anak:\n"
        "Nama lengkap : ANAK;\n"
        "Menimbang, bahwa Kesepakatan Diversi telah selesai dilaksanakan;\n"
        "MENETAPKAN\n"
        "1. Menghentikan proses pemeriksaan perkara Anak;\n"
    )
    record = empty_record()
    record["judul"] = ["PENETAPAN"]
    record["nomor_putusan"] = ["Nomor 3/Pen.Div/2026/PN Mrs"]
    record["nama_lengkap"] = ["ANAK"]

    reported_empty = empty_sections_for_report(record, source)

    assert "dakwaan" not in reported_empty
    assert "saksi" not in reported_empty
    assert "terdakwa" not in reported_empty
    assert "agama" not in reported_empty
    assert "irah_irah" in reported_empty


def test_parse_model_json_rejects_empty_http_200_content() -> None:
    with pytest.raises(ResponseError, match="no assistant content"):
        parse_model_json("")


def test_call_deepseek_retries_empty_200_then_accepts_valid_json() -> None:
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    session = FakeSession(
        [
            completion(""),
            completion(json.dumps(record)),
        ]
    )

    result = call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=1,
        max_attempts=2,
        max_output_tokens=4096,
        base_delay_seconds=0,
        sleep=lambda _: None,
    )

    assert result.record == record
    assert result.request_attempts == 2
    assert session.calls == 2


def test_call_deepseek_reports_request_lifecycle() -> None:
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    session = FakeSession([completion(json.dumps(record))])
    events: list[tuple[str, int, int, str]] = []

    call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=30,
        max_attempts=2,
        max_output_tokens=4096,
        base_delay_seconds=0,
        sleep=lambda _: None,
        activity=lambda stage, attempt, maximum, detail: events.append(
            (stage, attempt, maximum, detail)
        ),
    )

    assert [event[0] for event in events] == [
        "Waiting for W&B",
        "Streaming response",
        "Generating JSON",
        "Validating excerpts",
        "Response accepted",
    ]
    assert all(event[1:3] == (1, 2) for event in events)


def test_call_deepseek_sends_selected_reasoning_effort() -> None:
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    session = FakeSession([completion(json.dumps(record))])

    call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=30,
        max_attempts=1,
        max_output_tokens=32768,
        base_delay_seconds=0,
        reasoning_effort="medium",
    )

    assert session.last_json is not None
    assert session.last_json["reasoning_effort"] == "medium"
    assert session.last_json["max_tokens"] == 32768
    assert session.last_json["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_call_deepseek_off_omits_reasoning_controls() -> None:
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    session = FakeSession([completion(json.dumps(record))])

    call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=30,
        max_attempts=1,
        max_output_tokens=4096,
        base_delay_seconds=0,
        reasoning_effort="off",
    )

    assert session.last_json is not None
    assert "reasoning_effort" not in session.last_json
    assert "chat_template_kwargs" not in session.last_json


def test_dashboard_renders_live_worker_stage(tmp_path: Path) -> None:
    dashboard = anak_deepseek.RunDashboard(
        total_sources=500,
        initial_completed=34,
        selected=4,
        workers=4,
        reasoning_effort="medium",
        enabled=False,
    )
    source = tmp_path / "case.txt"
    dashboard.queued(source, 1234)
    dashboard.activity_callback(source.name)(
        "Waiting for W&B",
        2,
        4,
        "POST sent; timeout 300s",
    )

    console = Console(record=True, width=140)
    console.print(dashboard.render())
    rendered = console.export_text()

    assert "case.txt" in rendered
    assert "2/4" in rendered
    assert "Waiting for W&B" in rendered
    assert "POST sent; timeout 300s" in rendered
    assert "in stage" in rendered
    assert "medium" in rendered


def test_parse_streaming_response_exposes_reasoning_and_content() -> None:
    events = [
        {
            "choices": [
                {"delta": {"reasoning": "Checking source labels..."}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        },
        {
            "choices": [
                {"delta": {"content": '{"judul":["PUTUSAN"]}'}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    ]

    class StreamResponse:
        def iter_lines(self, decode_unicode: bool = False) -> list[str]:
            return [
                *(f"data: {json.dumps(event)}" for event in events),
                "data: [DONE]",
            ]

    activity: list[tuple[str, str]] = []
    content, reasoning, usage, finish_reason = anak_deepseek.parse_streaming_response(
        StreamResponse(),  # type: ignore[arg-type]
        attempt=1,
        max_attempts=4,
        activity=lambda stage, attempt, maximum, detail: activity.append(
            (stage, detail)
        ),
    )

    assert content == '{"judul":["PUTUSAN"]}'
    assert reasoning == "Checking source labels..."
    assert usage["completion_tokens"] == 5
    assert finish_reason is None
    assert [stage for stage, _ in activity] == [
        "Model reasoning",
        "Generating JSON",
    ]


def test_call_deepseek_retries_non_verbatim_output() -> None:
    invalid = empty_record()
    invalid["judul"] = ["Generated title"]
    valid = empty_record()
    valid["judul"] = ["PUTUSAN"]
    session = FakeSession(
        [
            completion(json.dumps(invalid)),
            completion(json.dumps(valid)),
        ]
    )

    result = call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=1,
        max_attempts=2,
        max_output_tokens=4096,
        base_delay_seconds=0,
        sleep=lambda _: None,
    )

    assert result.record["judul"] == ["PUTUSAN"]
    assert result.request_attempts == 2


def test_validate_span_record_coerces_empty_lines_to_empty_section() -> None:
    source_text = "PUTUSAN\nNomor 1"
    sections = {key: {"empty": True} for key in SECTION_KEYS}
    sections["judul"] = {"text": ["PUTUSAN"]}
    # The model frequently signals an absent section with an empty array form
    # instead of {"empty": true}. This must coerce to empty, not reject the whole
    # 31-section response.
    sections["penangkapan"] = {"lines": []}
    sections["surat"] = {"text": []}

    record = anak_deepseek.validate_span_record(
        {"sections": sections}, source_text
    )

    assert record["judul"] == ["PUTUSAN"]
    assert record["penangkapan"] == []
    assert record["surat"] == []


def test_validate_span_record_normalizes_flat_line_pair() -> None:
    source_text = "PUTUSAN\nNomor 1\nbaris ketiga\nbaris keempat"
    sections = {key: {"empty": True} for key in SECTION_KEYS}
    sections["judul"] = {"text": ["PUTUSAN"]}
    # Common LLM slip: a bare [start, end] pair instead of [[start, end]].
    sections["dakwaan"] = {"lines": [3, 4]}

    record = anak_deepseek.validate_span_record(
        {"sections": sections}, source_text
    )

    assert record["dakwaan"] == ["baris ketiga\nbaris keempat"]


def test_validate_span_record_normalizes_string_and_single_line_forms() -> None:
    source_text = "PUTUSAN\nNomor 1\nbaris ketiga\nbaris keempat"
    sections = {key: {"empty": True} for key in SECTION_KEYS}
    sections["judul"] = {"text": ["PUTUSAN"]}
    # String dash-range and a single-line nested form both seen from the model.
    sections["dakwaan"] = {"lines": ["3-4"]}
    sections["nomor_putusan"] = {"lines": [[2]]}

    record = anak_deepseek.validate_span_record(
        {"sections": sections}, source_text
    )

    assert record["dakwaan"] == ["baris ketiga\nbaris keempat"]
    assert record["nomor_putusan"] == ["Nomor 1"]


def test_normalize_line_ranges_rejects_garbage() -> None:
    assert anak_deepseek.normalize_line_ranges([]) is None
    assert anak_deepseek.normalize_line_ranges("nonsense") is None
    assert anak_deepseek.normalize_line_ranges([["a", "b"]]) is None


def test_call_deepseek_retries_on_truncation_and_disables_reasoning() -> None:
    valid = empty_record()
    valid["judul"] = ["PUTUSAN"]
    truncated = FakeResponse(
        {
            "choices": [{"message": {"content": '{"judul":["PUT'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4096},
        },
        finish_reason="length",
    )
    session = FakeSession([truncated, completion(json.dumps(valid))])

    result = call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=1,
        max_attempts=2,
        max_output_tokens=32768,
        base_delay_seconds=0,
        sleep=lambda _: None,
        reasoning_effort="medium",
    )

    assert result.record["judul"] == ["PUTUSAN"]
    assert result.request_attempts == 2
    # After a length-truncation, reasoning must be dropped so the full output
    # budget is reserved for the span JSON on the retry.
    assert session.last_json is not None
    assert "reasoning_effort" not in session.last_json
    assert "chat_template_kwargs" not in session.last_json


def test_preflight_model_sends_selected_model_and_project() -> None:
    class Session:
        last_headers: dict[str, str] | None = None
        last_json: dict[str, Any] | None = None

        def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
            self.last_headers = kwargs["headers"]
            self.last_json = kwargs["json"]
            return completion('{"ok":true}')

    session = Session()
    original_model = anak_deepseek.MODEL
    try:
        anak_deepseek.MODEL = "zai-org/GLM-5.2"
        anak_deepseek.preflight_model(
            session,  # type: ignore[arg-type]
            api_key="test",
            project="entity/project",
            timeout_seconds=120,
        )
    finally:
        anak_deepseek.MODEL = original_model

    assert session.last_headers is not None
    assert session.last_headers["OpenAI-Project"] == "entity/project"
    assert session.last_json is not None
    assert session.last_json["model"] == "zai-org/GLM-5.2"
    assert session.last_json["max_tokens"] == 16


def test_preflight_model_rejects_provider_concurrency_limit() -> None:
    class Session:
        def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
            return FakeResponse(
                {
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": (
                            "concurrency limit reached for requests: "
                            "zai-org/GLM-5.2-project limit reached"
                        ),
                    }
                },
                status_code=429,
                text=(
                    '{"error":{"code":"rate_limit_exceeded","message":'
                    '"concurrency limit reached for requests: '
                    'zai-org/GLM-5.2-project limit reached"}}'
                ),
            )

    with pytest.raises(ResponseError, match="model preflight failed"):
        anak_deepseek.preflight_model(
            Session(),  # type: ignore[arg-type]
            api_key="test",
            project=None,
            timeout_seconds=120,
        )


def test_individual_output_exposes_empty_sections_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "one.txt"
    source.write_text("PUTUSAN", encoding="utf-8")
    record = empty_record()
    record["judul"] = ["PUTUSAN"]
    result = ApiResult(record=record, usage={"total_tokens": 20}, request_attempts=1)
    source_hash = sha256_text("PUTUSAN")
    destination = output_path(tmp_path / "output", source)

    write_individual_output(
        destination,
        source=source,
        source_hash=source_hash,
        result=result,
    )

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["sections"]["judul"] == ["PUTUSAN"]
    assert "judul" not in document["empty_sections"]
    assert "dakwaan" in document["empty_sections"]
    assert not destination.with_suffix(".json.tmp").exists()

    loaded = load_individual_output(
        destination,
        source_name=source.name,
        source_hash=source_hash,
        source_text="PUTUSAN",
    )
    assert loaded is not None

    changed = load_individual_output(
        destination,
        source_name=source.name,
        source_hash=sha256_text("CHANGED"),
        source_text="CHANGED",
    )
    assert changed is None


def test_no_text_output_is_transparent_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    destination = output_path(tmp_path / "output", source)
    source_hash = sha256_text("")

    write_no_text_output(
        destination,
        source=source,
        source_hash=source_hash,
    )

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["status"] == "no_text"
    assert document["model"] is None
    assert document["request_attempts"] == 0
    assert document["empty_sections"] == list(SECTION_KEYS)
    assert load_individual_output(
        destination,
        source_name=source.name,
        source_hash=source_hash,
        source_text="",
    )


def test_load_failure_counts_tracks_failed_runs_only(tmp_path: Path) -> None:
    state = tmp_path / "progress.jsonl"
    state.write_text(
        "\n".join(
            [
                json.dumps({"source": "a.txt", "status": "failed"}),
                json.dumps({"source": "a.txt", "status": "completed"}),
                json.dumps({"source": "a.txt", "status": "failed"}),
                json.dumps({"source": "b.txt", "status": "failed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_failure_counts(state) == {"a.txt": 2, "b.txt": 1}


def test_main_runs_requests_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    state = tmp_path / "progress.jsonl"
    input_dir.mkdir()
    for index in range(4):
        (input_dir / f"{index}.txt").write_text(
            f"Nomor {index}\nNama lengkap: ANAK {index}",
            encoding="utf-8",
        )

    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_process(
        source: Path,
        *,
        source_text: str,
        source_hash: str,
        output_dir: Path,
        **kwargs: Any,
    ) -> anak_deepseek.ProcessOutcome:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        record = empty_record()
        record["nomor_putusan"] = [source_text.splitlines()[0]]
        record["nama_lengkap"] = [source_text.split(": ", 1)[1]]
        result = ApiResult(record=record, usage={}, request_attempts=1)
        destination = output_path(output_dir, source)
        write_individual_output(
            destination,
            source=source,
            source_hash=source_hash,
            result=result,
        )
        with lock:
            active -= 1
        return anak_deepseek.ProcessOutcome(
            source=source,
            source_hash=source_hash,
            destination=destination,
            event={
                "source": source.name,
                "source_sha256": source_hash,
                "status": "completed",
                "empty_sections": [
                    key for key in SECTION_KEYS if not record[key]
                ],
            },
            record=record,
            success=True,
        )

    monkeypatch.setattr(anak_deepseek, "process_source", fake_process)
    monkeypatch.setattr(anak_deepseek, "resolve_api_key", lambda _: "test")
    monkeypatch.setattr(anak_deepseek, "preflight_model", lambda *args, **kwargs: None)

    exit_code = anak_deepseek.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--state",
            str(state),
            "--workers",
            "3",
            "--no-tui",
        ]
    )

    assert exit_code == 0
    assert maximum_active == 3
    assert len(list(output_dir.glob("*.json"))) == 4
    assert len(state.read_text(encoding="utf-8").splitlines()) == 4


def test_main_requeues_infrastructure_failure_without_recording_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    state = tmp_path / "progress.jsonl"
    input_dir.mkdir()
    source = input_dir / "one.txt"
    source.write_text("Nomor 1\nNama lengkap: ANAK", encoding="utf-8")
    calls = 0

    def fake_process(
        source: Path,
        *,
        source_text: str,
        source_hash: str,
        output_dir: Path,
        **kwargs: Any,
    ) -> anak_deepseek.ProcessOutcome:
        nonlocal calls
        calls += 1
        destination = output_path(output_dir, source)
        if calls == 1:
            return anak_deepseek.ProcessOutcome(
                source=source,
                source_hash=source_hash,
                destination=destination,
                event={
                    "source": source.name,
                    "status": "failed",
                    "error": "NameResolutionError: getaddrinfo failed",
                },
                record={},
                success=False,
            )
        record = empty_record()
        record["nomor_putusan"] = ["Nomor 1"]
        record["nama_lengkap"] = ["ANAK"]
        result = ApiResult(record=record, usage={}, request_attempts=1)
        write_individual_output(
            destination,
            source=source,
            source_hash=source_hash,
            result=result,
        )
        return anak_deepseek.ProcessOutcome(
            source=source,
            source_hash=source_hash,
            destination=destination,
            event={
                "source": source.name,
                "source_sha256": source_hash,
                "status": "completed",
                "empty_sections": [
                    key for key in SECTION_KEYS if not record[key]
                ],
            },
            record=record,
            success=True,
        )

    monkeypatch.setattr(anak_deepseek, "process_source", fake_process)
    monkeypatch.setattr(anak_deepseek, "resolve_api_key", lambda _: "test")
    monkeypatch.setattr(anak_deepseek, "preflight_model", lambda *args, **kwargs: None)

    exit_code = anak_deepseek.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--state",
            str(state),
            "--workers",
            "1",
            "--network-failure-threshold",
            "1",
            "--network-cooldown",
            "0",
            "--no-tui",
        ]
    )

    assert exit_code == 0
    assert calls == 2
    events = [
        json.loads(line)
        for line in state.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in events] == ["completed"]


def test_main_preflight_failure_exits_without_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    state = tmp_path / "progress.jsonl"
    input_dir.mkdir()
    (input_dir / "one.txt").write_text("PUTUSAN", encoding="utf-8")

    monkeypatch.setattr(anak_deepseek, "resolve_api_key", lambda _: "test")

    def fail_preflight(*args: Any, **kwargs: Any) -> None:
        raise ResponseError(
            "model preflight failed with retryable HTTP 429: "
            "concurrency limit reached"
        )

    monkeypatch.setattr(anak_deepseek, "preflight_model", fail_preflight)

    exit_code = anak_deepseek.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--state",
            str(state),
            "--no-tui",
        ]
    )

    assert exit_code == 75
    assert not state.exists()
    assert not output_dir.exists()


def test_retry_empty_keeps_existing_output_without_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "one.txt"
    source_text = "Nomor 1\nNama lengkap: ANAK"
    source.write_text(source_text, encoding="utf-8")
    source_hash = sha256_text(source_text)
    output_dir = tmp_path / "output"

    old_record = empty_record()
    old_record["nomor_putusan"] = ["Nomor 1"]
    old_record["nama_lengkap"] = ["ANAK"]
    old_result = ApiResult(record=old_record, usage={}, request_attempts=1)
    destination = output_path(output_dir, source)
    write_individual_output(
        destination,
        source=source,
        source_hash=source_hash,
        result=old_result,
    )
    original = destination.read_text(encoding="utf-8")

    monkeypatch.setattr(
        anak_deepseek,
        "call_deepseek",
        lambda *args, **kwargs: ApiResult(
            record=old_record,
            usage={"total_tokens": 10},
            request_attempts=1,
        ),
    )

    outcome = anak_deepseek.process_source(
        source,
        source_text=source_text,
        source_hash=source_hash,
        output_dir=output_dir,
        api_key="test",
        project=None,
        timeout_seconds=1,
        max_attempts=1,
        max_output_tokens=4096,
        base_delay_seconds=0,
        previous_record=old_record,
    )

    assert outcome.success
    assert outcome.event["extraction_status"] == "retry_no_improvement"
    assert destination.read_text(encoding="utf-8") == original
