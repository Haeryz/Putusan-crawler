from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

MODEL = "deepseek-ai/DeepSeek-V4-Pro"
API_URL = "https://api.inference.wandb.ai/v1/chat/completions"
REASONING_EFFORTS = ("off", "low", "medium", "high", "xhigh")
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_INPUT = Path("downloads/kasus anak/raw-text")
DEFAULT_OUTPUT_DIR = Path("LLM-aggregator/Anak/Deepseek/output")
DEFAULT_STATE = Path("LLM-aggregator/Anak/Deepseek/progress.jsonl")
DEFAULT_ENV = Path("LLM-aggregator/Anak/Deepseek/.env")
DEFAULT_PAUSE_FILE = Path("LLM-aggregator/Anak/Deepseek/pause")
PROGRAM_NAME = "anak-deepseek-aggregate"
CORPUS_LABEL = "Putusan Anak"

SECTION_KEYS = (
    "judul",
    "nomor_putusan",
    "irah_irah",
    "nama_pengadilan_negeri",
    "keterangan_perkara",
    "nama_lengkap",
    "tempat_lahir",
    "umur_tanggal_lahir",
    "jenis_kelamin",
    "kebangsaan",
    "tempat_tinggal",
    "agama",
    "pekerjaan",
    "penangkapan",
    "penahanan",
    "tuntutan",
    "dakwaan",
    "saksi",
    "ahli",
    "terdakwa",
    "surat",
    "petunjuk_barang_bukti",
    "fakta_hukum",
    "pertimbangan_hukum",
    "amar_putusan",
    "hari",
    "tanggal",
    "tahun",
    "siapa_yang_memutus",
    "panitera_pengganti",
    "tanda_tangan_majelis",
)

BOUNDARY_GUIDE = """\
SCHEMA INTERPRETATION
- Match boundary phrases case-insensitively.
- BEFORE and AFTER alternatives are OR lists: one matching alternative is enough.
- Extract the exact source content located after BEFORE and before AFTER. Do not
  include boundary labels unless they are inseparable from the intended value.
- Number prefixes such as "1.", "2.", or "I." are optional locating syntax.
- OCR spelling, punctuation, spacing, and letter-spaced variants are valid.
- A field may still be present when one boundary is missing. Direct field labels,
  headings, and aliases are stronger evidence than returning an obvious field [].
- For a directly labeled identity line, extract the exact value after ":" and
  before the next identity label. Never copy a neighboring field into it.
- Preserve line breaks and all characters exactly as they appear in DOCUMENT.
- Return [] only after checking every listed boundary, alias, and variant.

FIELD RULES
1 judul
  Meaning: document title at the beginning, commonly PUTUSAN or PENETAPAN.
  BEFORE: start of document. AFTER: "Nomor".
2 nomor_putusan
  Meaning: every court-decision or determination number identifying this document.
  BEFORE: "Putusan". AFTER: "Pengadilan".
  Also treat a directly visible line beginning "Nomor" as strong evidence.
3 irah_irah
  Meaning: justice formula, commonly "DEMI KEADILAN BERDASARKAN KETUHANAN...".
  BEFORE: "PN". AFTER: "Pengadilan Negeri".
4 nama_pengadilan_negeri
  Meaning: exact court name or court/judge phrase identifying the district court.
  BEFORE: "Esa". AFTER: "yang mengadili perkara" OR
  "yang mengadili perkara-perkara".
5 keterangan_perkara
  Meaning: clause describing the kind of case and examination procedure.
  BEFORE: "Mengadili". AFTER: "dengan".
6 nama_lengkap
  Meaning: defendant/child full name value only.
  BEFORE: "1. Nama lengkap" OR "Nama lengkap".
  AFTER: "2. tempat" OR "tempat".
7 tempat_lahir
  Meaning: place-of-birth value only.
  BEFORE: "2. tempat" OR "tempat". AFTER: "3. umur" OR "umur".
8 umur_tanggal_lahir
  Meaning: age and/or date-of-birth value only.
  BEFORE: "3. umur" OR "umur". AFTER: "4. Jenis" OR "Jenis".
9 jenis_kelamin
  Meaning: sex/gender value only.
  BEFORE: "4. Jenis" OR "Jenis".
  AFTER: "5. kebangsaan" OR "kebangsaan".
10 kebangsaan
  Meaning: nationality value only.
  BEFORE: "5. kebangsaan" OR "kebangsaan".
  AFTER: "6. tempat" OR "tempat".
  NOTE: an optional Pendidikan field may occur before Tempat Tinggal.
11 tempat_tinggal
  Meaning: residence/address value only.
  BEFORE: "6. tempat" OR "tempat". AFTER: "7. Agama" OR "Agama".
12 agama
  Meaning: religion value only.
  BEFORE: "7. Agama" OR "Agama".
  AFTER: "8. pekerjaan" OR "pekerjaan".
13 pekerjaan
  Meaning: occupation value only.
  BEFORE: "8. pekerjaan" OR "pekerjaan".
  AFTER: "terdakwa ditangkap" OR "para terdakwa ditangkap".
  If the next listed boundary is absent, stop at the next non-identity paragraph.
14 penangkapan
  Meaning: arrest dates, order, and exact arrest details.
  BEFORE: "ditangkap sejak" OR "ditangkap pada" OR
  "surat perintah penangkapan" OR "Terdakwa dilakukan".
  AFTER: "tanggal" OR "dalam perkara lain".
15 penahanan
  Meaning: detention authority, place, periods, and exact detention details.
  BEFORE: "dalam tahanan" OR "ditahan oleh :" OR "ditahan dalam" OR
  "Terdakwa dilakukan".
  AFTER: "oleh :" OR "sejak tanggal" OR "dalam perkara lain".
16 tuntutan
  Meaning: prosecutor's requested findings, sentence, evidence treatment, and costs.
  BEFORE: "mendengar pembacaan" OR "mendengar pula".
  AFTER: "pidana" OR "pidana yang diajukan" OR "Penuntut Umum" OR
  "Jaksa Penuntut Umum".
17 dakwaan
  Meaning: charging instrument and accusation text.
  BEFORE: "berdasarkan surat" OR "surat" OR "dengan" OR "Surat".
  AFTER: "Penuntut Umum" OR "Nomor Reg. Perkara" OR "sebgai berikut:" OR
  "sebagai berikut :" OR "No. Reg.".
18 saksi
  Meaning: witness section, names, and testimony.
  BEFORE: "mengajukan" OR "mengajukan para" OR "menghadirkan" OR
  "menghadapkan".
  AFTER: "-Saksi" OR "-saksi" OR "yang memberikan keterangan" OR
  "sebagai berikut:" OR "ke depan Persidangan".
19 ahli
  Meaning: expert section and expert testimony.
  BEFORE: "mengajukan" OR "alat bukti" OR "dibacakan keterangan" OR
  "terdakwa membenarkannya;".
  AFTER: "sebagai berikut:" OR "berupa;" OR "berupa ;" OR
  "yang telah dipanggil" OR "atas keterangan ahli".
20 terdakwa
  Meaning: defendant's testimony at trial, not identity data.
  BEFORE pattern: "Menimbang, bahwa [Tt]erdakwa" followed optionally by
  Roman numeral I, II, or III and a name.
  AFTER: "di persidangan" OR "memberikan keterangan".
21 surat
  Meaning: documentary evidence.
  BEFORE: "mengajukan" OR "alat bukti" OR "bukti surat berupa" OR
  "melampirkan surat:".
  AFTER: "sebagai berikut:" OR "Menimbang bahwa".
22 petunjuk_barang_bukti
  Meaning: indications and/or physical evidence.
  BEFORE: "mengajukan" OR "terhadap" OR "diperhatikan".
  AFTER: "sebagai berikut:" OR "berupa;" OR OCR variant "berupa ;l".
23 fakta_hukum
  Meaning: judicially established legal facts.
  ALIASES: "fakta-fakta hukum", "bahwa dalam persidangan,".
  BEFORE: "Menimbang" OR "berdasarkan" OR "disimpulkan adanya".
  AFTER: "Majelis Hakim" OR "tersebut diatas" OR
  "serta didukung dengan bukti" OR "-fakta dalam perkara" OR
  "dalam perkara ini" OR "sebagai berikut;".
24 pertimbangan_hukum
  Meaning: court's legal reasoning and application of law.
  ALIAS: "pertimbangan".
  BEFORE: "Menimbang," OR "uraian" OR "Majelis Hakim akan" OR
  "sebagai berikut" OR OCR variant "mempertimbangakan".
  AFTER: "tersebut di atas" OR "Ad." OR
  "apakah berdasarkan fakta-fakta hukum".
25 amar_putusan
  Meaning: operative orders/verdict.
  BEFORE: "MENGADILI" OR "MENGADILI:" OR "MENGADILI;" OR
  "M E N G A D I L I" OR "M E N G A D I L I :" OR "M E N G A D I L I:".
  AFTER: "Demikianlah diputuskan".
26 hari
  Meaning: day of decision.
  BEFORE: "pada". AFTER: ", tanggal".
27 tanggal
  Meaning: decision date.
  BEFORE: "hari, tanggal". AFTER: "bulan".
28 tahun
  Meaning: decision year.
  BEFORE: "bulan". AFTER: ", oleh" OR "oleh".
29 siapa_yang_memutus
  Meaning: judge or panel that decided the matter.
  BEFORE: "oleh" OR "oleh kami,".
  AFTER: ", sebagai hakim" OR "sebagai hakim".
30 panitera_pengganti
  Meaning: substitute clerk's name/details.
  BEFORE: "Panitera" OR "dibantu oleh".
  AFTER: "pada Pengadilan Negeri".
31 tanda_tangan_majelis
  Meaning: signature block for the judicial panel.
  BEFORE: "Hakim Ketua,". AFTER: "Panitera Pengganti,".

OCR AND EDGE-CASE CHECKLIST
- Match optional leading number pattern like digits + "." + whitespace.
- Treat "M E N G A D I L I" as MENGADILI for locating purposes.
- Preserve typo variants "sebgai berikut:", "mempertimbangakan", and
  "berupa ;l" when they occur.
- For defendant testimony, locate the flexible pattern
  "Menimbang, bahwa [Tt]erdakwa" + optional Roman numeral + name.
- Do not confuse identity "terdakwa" fields with terdakwa trial testimony.
"""

SYSTEM_PROMPT = f"""\
You are a strictly extractive Indonesian court-decision parser.
Never summarize, paraphrase, infer, translate, correct OCR, add labels, or create
text. Every returned value must be copied as one exact contiguous substring from
DOCUMENT. JSON property names are the only text that need not occur in DOCUMENT.

Return one JSON object with exactly these properties:
{", ".join(SECTION_KEYS)}

Every property value must be an array of zero or more strings. Use [] when the
section is absent. Use multiple strings only for multiple defendants or genuinely
separate occurrences. Keep source spelling, punctuation, capitalization, and
spacing exactly. Do not return markdown or commentary.

WORK PROCEDURE
1. Read the entire DOCUMENT once to identify its structure and document type.
2. Locate each of the 31 fields independently using its meaning, direct labels,
   aliases, and all BEFORE/AFTER variants below.
3. For directly labeled fields, extract the exact labeled value even when a
   later boundary phrase is absent.
4. Before returning [], search again for every alias and OCR variant.
5. Verify every non-empty string is copied exactly and contiguously from DOCUMENT.
6. Verify all obvious "Nomor" and identity labels have non-empty corresponding
   properties before producing the final JSON.

{BOUNDARY_GUIDE}"""

_BOILERPLATE_LINES = {
    "Mahkamah Agung Republik Indonesia",
    "Direktori Putusan Mahkamah Agung Republik Indonesia",
    "putusan.mahkamahagung.go.id",
    "putusan3.mahkamahagung.go.id",
}
_PAGE_LINE = re.compile(
    r"^\s*Halaman\s+\d+(?:\s+dari\s+\d+)?(?:\s+Putusan\b.*)?\s*$",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_EXPECTED_MARKERS = {
    "nomor_putusan": re.compile(r"(?im)^\s*Nomor\s+\S"),
    "nama_lengkap": re.compile(r"(?i)\bNama\s+lengkap\s*:"),
    "tempat_lahir": re.compile(r"(?i)\bTempat\s+lahir\s*:"),
    "umur_tanggal_lahir": re.compile(r"(?i)\bUmur\s*/?\s*tanggal\s+lahir\s*:"),
    "jenis_kelamin": re.compile(r"(?i)\bJenis\s+Kelamin\s*:"),
    "kebangsaan": re.compile(r"(?i)\bKebangsaan\s*:"),
    "tempat_tinggal": re.compile(r"(?i)\bTempat\s+tinggal\s*:"),
    "agama": re.compile(r"(?i)\bAgama\s*:"),
    "pekerjaan": re.compile(r"(?i)\bPekerjaan\s*:"),
}


class ResponseError(RuntimeError):
    """The API returned a response that cannot be accepted."""


class ValidationError(ResponseError):
    """The model returned text that is not strictly extractive."""


INFRASTRUCTURE_ERROR_MARKERS = (
    "NameResolutionError",
    "Failed to resolve",
    "getaddrinfo failed",
    "NewConnectionError",
    "ConnectTimeout",
    "Connection refused",
    "Connection reset",
    "Connection aborted",
    "ProxyError",
)


def is_infrastructure_error(error: str) -> bool:
    return any(marker.casefold() in error.casefold() for marker in INFRASTRUCTURE_ERROR_MARKERS)


@dataclass(frozen=True, slots=True)
class ApiResult:
    record: dict[str, list[str]]
    usage: dict[str, Any]
    request_attempts: int


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    source: Path
    source_hash: str
    destination: Path
    event: dict[str, Any]
    record: dict[str, list[str]]
    success: bool


@dataclass(slots=True)
class WorkerActivity:
    approximate_tokens: int
    started_at: float
    stage_started_at: float
    stage: str
    attempt: int = 0
    max_attempts: int = 0
    detail: str = ""


class RunDashboard:
    def __init__(
        self,
        *,
        total_sources: int,
        initial_completed: int,
        selected: int,
        workers: int,
        reasoning_effort: str,
        enabled: bool,
    ) -> None:
        self.console = Console()
        self.total_sources = total_sources
        self.completed = initial_completed
        self.selected = selected
        self.workers = workers
        self.reasoning_effort = reasoning_effort
        self.processed = 0
        self.failed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.active: dict[str, WorkerActivity] = {}
        self.activity_lock = threading.Lock()
        self.recent: deque[tuple[str, str]] = deque(maxlen=7)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            expand=True,
        )
        self.task_id = self.progress.add_task("Current batch", total=selected)
        self.live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
            auto_refresh=True,
        ) if enabled else None

    def __enter__(self) -> RunDashboard:
        if self.live:
            self.live.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.live:
            self.live.update(self.render(), refresh=True)
            self.live.stop()

    def log(self, status: str, message: str) -> None:
        self.recent.appendleft((status, message))
        if self.live:
            self.live.update(self.render())
        else:
            self.console.print(f"[{status}] {message}")

    def refresh(self) -> None:
        if self.live:
            self.live.update(self.render(), refresh=True)

    def queued(self, source: Path, approximate_tokens: int) -> None:
        now = time.monotonic()
        with self.activity_lock:
            self.active[source.name] = WorkerActivity(
                approximate_tokens=approximate_tokens,
                started_at=now,
                stage_started_at=now,
                stage="Queued",
            )
        self.log("QUEUE", f"{source.name} (~{approximate_tokens:,} input tokens)")

    def activity_callback(
        self,
        source_name: str,
    ) -> Callable[[str, int, int, str], None]:
        def update(
            stage: str,
            attempt: int,
            max_attempts: int,
            detail: str,
        ) -> None:
            with self.activity_lock:
                activity = self.active.get(source_name)
                if activity is None:
                    return
                activity.stage = stage
                activity.attempt = attempt
                activity.max_attempts = max_attempts
                activity.detail = detail
                activity.stage_started_at = time.monotonic()

        return update

    def finished(
        self,
        outcome: ProcessOutcome,
        *,
        new_corpus_completion: bool,
    ) -> None:
        with self.activity_lock:
            self.active.pop(outcome.source.name, None)
        self.processed += 1
        self.progress.update(self.task_id, advance=1)
        usage = outcome.event.get("usage", {})
        if isinstance(usage, dict):
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)
        if outcome.success:
            if new_corpus_completion:
                self.completed += 1
            empty_count = len(outcome.event["empty_sections"])
            status = outcome.event.get("extraction_status")
            label = {
                "no_text": "NO TEXT",
                "retry_no_improvement": "UNCHANGED",
                "retry_improved": "IMPROVED",
            }.get(status, "OK")
            self.log(
                label,
                f"{outcome.source.name}; empty sections={empty_count}",
            )
        else:
            self.failed += 1
            self.log("FAILED", f"{outcome.source.name}: {outcome.event['error']}")

    def deferred(
        self,
        outcome: ProcessOutcome,
        *,
        cooldown_seconds: float,
    ) -> None:
        with self.activity_lock:
            self.active.pop(outcome.source.name, None)
        self.log(
            "NETWORK",
            f"{outcome.source.name} requeued; pausing new requests for "
            f"{cooldown_seconds:g}s",
        )

    def render(self) -> Group:
        with self.activity_lock:
            active_snapshot = {
                name: WorkerActivity(
                    approximate_tokens=activity.approximate_tokens,
                    started_at=activity.started_at,
                    stage_started_at=activity.stage_started_at,
                    stage=activity.stage,
                    attempt=activity.attempt,
                    max_attempts=activity.max_attempts,
                    detail=activity.detail,
                )
                for name, activity in self.active.items()
            }
        summary = Table.grid(expand=True)
        summary.add_column()
        summary.add_column(justify="right")
        summary.add_row(
            "Corpus",
            f"[bold green]{self.completed}[/] / {self.total_sources} complete",
        )
        summary.add_row(
            "Batch",
            f"{self.processed} / {self.selected} finished, "
            f"[red]{self.failed} failed[/]",
        )
        summary.add_row(
            "Workers",
            f"{len(active_snapshot)} active / {self.workers} configured",
        )
        summary.add_row(
            "API tokens",
            f"{self.input_tokens:,} input / {self.output_tokens:,} output",
        )
        reasoning_style = "dim" if self.reasoning_effort == "off" else "magenta"
        summary.add_row(
            "Reasoning",
            f"[{reasoning_style}]{self.reasoning_effort}[/]",
        )

        active = Table(title="Active worker activity", expand=True)
        active.add_column("File")
        active.add_column("Attempt", width=9)
        active.add_column("Stage", width=25)
        active.add_column("Elapsed", justify="right", width=9)
        active.add_column("Activity")
        if active_snapshot:
            now = time.monotonic()
            for name, activity in active_snapshot.items():
                attempt = (
                    f"{activity.attempt}/{activity.max_attempts}"
                    if activity.attempt
                    else "-"
                )
                elapsed = int(now - activity.started_at)
                stage_elapsed = int(now - activity.stage_started_at)
                detail = activity.detail or (
                    f"~{activity.approximate_tokens:,} input tokens"
                )
                active.add_row(
                    escape(name),
                    attempt,
                    escape(activity.stage),
                    f"{elapsed}s",
                    f"{escape(detail)} [dim]({stage_elapsed}s in stage)[/]",
                )
        else:
            active.add_row("[dim]None[/]", "", "", "", "")

        recent = Table(title="Recent events", expand=True)
        recent.add_column("Status", width=12)
        recent.add_column("Details")
        if self.recent:
            for status, message in self.recent:
                recent.add_row(status, message)
        else:
            recent.add_row("[dim]Starting[/]", "")
        return Group(
            Panel(summary, title=f"DeepSeek {CORPUS_LABEL}"),
            self.progress,
            active,
            recent,
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_source(raw_text: str) -> str:
    """Remove known site boilerplate without rewriting decision text."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    kept: list[str] = []
    previous_blank = False
    in_disclaimer = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.casefold() == "disclaimer":
            in_disclaimer = True
            continue
        if in_disclaimer:
            if stripped.casefold().startswith("email :"):
                in_disclaimer = False
            continue
        if stripped in _BOILERPLATE_LINES or _PAGE_LINE.match(line):
            continue
        blank = not stripped
        if blank and previous_blank:
            continue
        kept.append(line.rstrip())
        previous_blank = blank
    return "\n".join(kept).strip()


def discover_sources(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.name.casefold(),
    )


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_api_key(env_file: Path) -> str:
    env = load_dotenv(env_file)
    key = (
        os.environ.get("WANDB_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or env.get("WANDB_API_KEY")
        or env.get("OPENAI_API_KEY")
        or env.get("api_key")
    )
    if not key:
        raise ValueError(
            "No API key found. Set WANDB_API_KEY/OPENAI_API_KEY or api_key in "
            f"{env_file}."
        )
    return key


def parse_model_json(content: Any) -> Mapping[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str) or not content.strip():
        raise ResponseError("HTTP 200 response contained no assistant content")
    cleaned = _FENCE.sub("", content).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseError(f"assistant content is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ResponseError("assistant JSON must be an object")
    return parsed


def validate_record(
    value: Mapping[str, Any],
    source_text: str,
) -> dict[str, list[str]]:
    missing = set(SECTION_KEYS) - set(value)
    extra = set(value) - set(SECTION_KEYS)
    if missing or extra:
        raise ValidationError(
            f"wrong JSON properties; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    record: dict[str, list[str]] = {}
    for key in SECTION_KEYS:
        excerpts = value[key]
        if not isinstance(excerpts, list) or any(
            not isinstance(item, str) for item in excerpts
        ):
            raise ValidationError(f"{key} must be an array of strings")
        cleaned: list[str] = []
        for excerpt in excerpts:
            if not excerpt:
                raise ValidationError(f"{key} contains an empty string")
            exact_excerpt = align_source_excerpt(excerpt, source_text)
            if exact_excerpt is None:
                raise ValidationError(
                    f"{key} contains text that is not a contiguous source excerpt: "
                    f"{excerpt[:120]!r}"
                )
            cleaned.append(exact_excerpt)
        record[key] = cleaned
    return record


def validate_minimum_evidence(
    record: Mapping[str, list[str]],
    source_text: str,
) -> None:
    nonempty = sum(bool(record[key]) for key in SECTION_KEYS)
    if nonempty == 0:
        raise ValidationError("all 31 sections are empty")
    missing_obvious = [
        key
        for key, marker in _EXPECTED_MARKERS.items()
        if marker.search(source_text) and not record[key]
    ]
    if missing_obvious:
        raise ValidationError(
            f"obvious labeled source fields were returned empty: {missing_obvious}"
        )


def align_source_excerpt(excerpt: str, source_text: str) -> str | None:
    """Return the exact source span when only whitespace layout differs."""
    if excerpt in source_text:
        return excerpt
    pieces = re.findall(r"\S+", excerpt)
    if not pieces:
        return None
    pattern = r"\s+".join(re.escape(piece) for piece in pieces)
    match = re.search(pattern, source_text)
    return match.group(0) if match else None


def _response_content(payload: Mapping[str, Any]) -> Any:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ResponseError("HTTP 200 response contained no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ResponseError("HTTP 200 response contained an invalid choice")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ResponseError("HTTP 200 response contained no assistant message")
    return message.get("content")


def _preview(text: str, limit: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return "..." + collapsed[-limit:]


def output_token_budget(
    source_text: str,
    *,
    max_output_tokens: int,
    reasoning_effort: str,
) -> int:
    if reasoning_effort != "off":
        return max_output_tokens
    return min(
        max_output_tokens,
        max(4096, len(source_text) // 3 + 2048),
    )


def parse_streaming_response(
    response: requests.Response,
    *,
    attempt: int,
    max_attempts: int,
    activity: Callable[[str, int, int, str], None],
) -> tuple[str, str, dict[str, Any]]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line)
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ResponseError(f"invalid streaming JSON event: {exc}") from exc
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage = event_usage
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            reasoning_parts.append(reasoning)
            full_reasoning = "".join(reasoning_parts)
            activity(
                "Model reasoning",
                attempt,
                max_attempts,
                f"{len(full_reasoning):,} chars: {_preview(full_reasoning)}",
            )
        content = delta.get("content")
        if isinstance(content, str) and content:
            content_parts.append(content)
            full_content = "".join(content_parts)
            activity(
                "Generating JSON",
                attempt,
                max_attempts,
                f"{len(full_content):,} chars received: {_preview(full_content)}",
            )
    return "".join(content_parts), "".join(reasoning_parts), usage


def call_deepseek(
    session: requests.Session,
    *,
    api_key: str,
    source_name: str,
    source_text: str,
    project: str | None,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    base_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    activity: Callable[[str, int, int, str], None] | None = None,
    reasoning_effort: str = "off",
) -> ApiResult:
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}"
        )
    rng = rng or random.Random()
    report = activity or (lambda stage, attempt, maximum, detail: None)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if project:
        headers["OpenAI-Project"] = project

    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"FILE: {source_name}\n\nDOCUMENT:\n{source_text}",
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": output_token_budget(
            source_text,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        ),
    }
    if reasoning_effort != "off":
        body["chat_template_kwargs"] = {"enable_thinking": True}
        body["reasoning_effort"] = reasoning_effort
    last_error: Exception | None = None
    json_mode = True
    base_messages = list(body["messages"])

    for attempt in range(1, max_attempts + 1):
        response: requests.Response | None = None
        try:
            report(
                "Waiting for W&B",
                attempt,
                max_attempts,
                f"POST sent; timeout {timeout_seconds:g}s",
            )
            response = session.post(
                API_URL,
                headers=headers,
                json=body,
                timeout=timeout_seconds,
                stream=True,
            )
            if (
                response.status_code == 400
                and json_mode
                and "response_format" in response.text.casefold()
            ):
                body.pop("response_format", None)
                json_mode = False
                raise ResponseError("endpoint rejected JSON mode; retrying without it")
            if response.status_code == 429 or response.status_code >= 500:
                raise ResponseError(
                    f"retryable HTTP {response.status_code}: {response.text[:300]}"
                )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"non-retryable HTTP {response.status_code}: {response.text[:500]}"
                )
            report(
                "Streaming response",
                attempt,
                max_attempts,
                f"HTTP {response.status_code}; waiting for token chunks",
            )
            content, reasoning, usage = parse_streaming_response(
                response,
                attempt=attempt,
                max_attempts=max_attempts,
                activity=report,
            )
            if reasoning_effort != "off" and not reasoning:
                report(
                    "No reasoning field",
                    attempt,
                    max_attempts,
                    "Endpoint streamed no delta.reasoning; validating content",
                )
            parsed = parse_model_json(content)
            report(
                "Validating excerpts",
                attempt,
                max_attempts,
                "Checking schema and exact source spans",
            )
            record = validate_record(parsed, source_text)
            validate_minimum_evidence(record, source_text)
            report(
                "Response accepted",
                attempt,
                max_attempts,
                "Validation passed",
            )
            return ApiResult(
                record=record,
                usage=usage,
                request_attempts=attempt,
            )
        except RuntimeError as exc:
            if not isinstance(exc, ResponseError):
                raise
            last_error = exc
            if response is not None and response.status_code == 200:
                body["messages"] = [
                    *base_messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was rejected: "
                            f"{exc}. Try again. Extract obvious labeled fields, "
                            "and copy every value exactly from DOCUMENT."
                        ),
                    },
                ]
            if attempt == max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            retry_delay = delay + rng.uniform(0, min(1.0, delay * 0.2))
            report(
                "Retry backoff",
                attempt,
                max_attempts,
                f"{type(exc).__name__}: {str(exc)[:100]}; "
                f"retrying in {retry_delay:.1f}s",
            )
            sleep(retry_delay)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            retry_delay = delay + rng.uniform(0, min(1.0, delay * 0.2))
            report(
                "Retry backoff",
                attempt,
                max_attempts,
                f"{type(exc).__name__}: {str(exc)[:100]}; "
                f"retrying in {retry_delay:.1f}s",
            )
            sleep(retry_delay)

    raise ResponseError(
        f"request failed after {max_attempts} attempts: {last_error}"
    )


def append_state(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if event.get("status") == "completed":
            completed[str(event["source"])] = event
    return completed


def load_failure_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if event.get("status") == "failed":
            source = str(event["source"])
            counts[source] = counts.get(source, 0) + 1
    return counts


def output_path(output_dir: Path, source: Path) -> Path:
    return output_dir / f"{source.stem}.json"


def write_individual_output(
    path: Path,
    *,
    source: Path,
    source_hash: str,
    result: ApiResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "status": "completed",
        "source_file": source.name,
        "source_sha256": source_hash,
        "model": MODEL,
        "empty_sections": [
            key for key in SECTION_KEYS if not result.record[key]
        ],
        "sections": result.record,
        "usage": result.usage,
        "request_attempts": result.request_attempts,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_no_text_output(
    path: Path,
    *,
    source: Path,
    source_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "status": "no_text",
        "source_file": source.name,
        "source_sha256": source_hash,
        "model": None,
        "empty_sections": list(SECTION_KEYS),
        "sections": {key: [] for key in SECTION_KEYS},
        "usage": {},
        "request_attempts": 0,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def load_individual_output(
    path: Path,
    *,
    source_name: str,
    source_hash: str,
    source_text: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        sections = validate_record(document["sections"], source_text)
        if document.get("status", "completed") == "no_text":
            if source_text:
                return None
        else:
            validate_minimum_evidence(sections, source_text)
    except (KeyError, TypeError, ValueError, ResponseError):
        return None
    if (
        document.get("source_file") != source_name
        or document.get("source_sha256") != source_hash
    ):
        return None
    return {**document, "sections": sections}


def _source_event(
    source: Path,
    source_hash: str,
    *,
    status: str,
    **values: Any,
) -> dict[str, Any]:
    return {
        "source": source.name,
        "source_sha256": source_hash,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **values,
    }


def process_source(
    source: Path,
    *,
    source_text: str,
    source_hash: str,
    output_dir: Path,
    api_key: str,
    project: str | None,
    timeout_seconds: float,
    max_attempts: int,
    max_output_tokens: int,
    base_delay_seconds: float,
    previous_record: Mapping[str, list[str]] | None = None,
    activity: Callable[[str, int, int, str], None] | None = None,
    reasoning_effort: str = "off",
) -> ProcessOutcome:
    report = activity or (lambda stage, attempt, maximum, detail: None)
    destination = output_path(output_dir, source)
    if not source_text:
        report("Saving no-text result", 0, max_attempts, "No decision text found")
        record = {key: [] for key in SECTION_KEYS}
        write_no_text_output(
            destination,
            source=source,
            source_hash=source_hash,
        )
        return ProcessOutcome(
            source=source,
            source_hash=source_hash,
            destination=destination,
            event=_source_event(
                source,
                source_hash,
                status="completed",
                extraction_status="no_text",
                model=None,
                output=str(destination),
                empty_sections=list(SECTION_KEYS),
                request_attempts=0,
                usage={},
            ),
            record=record,
            success=True,
        )

    session = requests.Session()
    try:
        report(
            "Preparing request",
            0,
            max_attempts,
            f"{len(source_text):,} characters; strict extractive JSON",
        )
        result = call_deepseek(
            session,
            api_key=api_key,
            source_name=source.name,
            source_text=source_text,
            project=project,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            max_output_tokens=max_output_tokens,
            base_delay_seconds=base_delay_seconds,
            activity=activity,
            reasoning_effort=reasoning_effort,
        )
        if previous_record is not None:
            report(
                "Comparing retry",
                result.request_attempts,
                max_attempts,
                "Keeping only a result with fewer empty sections",
            )
            previous_empty = sum(not previous_record[key] for key in SECTION_KEYS)
            new_empty = sum(not result.record[key] for key in SECTION_KEYS)
            if new_empty >= previous_empty:
                event = _source_event(
                    source,
                    source_hash,
                    status="completed",
                    extraction_status="retry_no_improvement",
                    model=MODEL,
                    output=str(destination),
                    empty_sections=[
                        key for key in SECTION_KEYS if not previous_record[key]
                    ],
                    request_attempts=result.request_attempts,
                    usage=result.usage,
                )
                return ProcessOutcome(
                    source=source,
                    source_hash=source_hash,
                    destination=destination,
                    event=event,
                    record=dict(previous_record),
                    success=True,
                )
        report(
            "Saving output",
            result.request_attempts,
            max_attempts,
            f"Writing {destination.name} atomically",
        )
        write_individual_output(
            destination,
            source=source,
            source_hash=source_hash,
            result=result,
        )
        event = _source_event(
            source,
            source_hash,
            status="completed",
            model=MODEL,
            output=str(destination),
            empty_sections=[
                key for key in SECTION_KEYS if not result.record[key]
            ],
            request_attempts=result.request_attempts,
            usage=result.usage,
        )
        if previous_record is not None:
            event["extraction_status"] = "retry_improved"
        return ProcessOutcome(
            source=source,
            source_hash=source_hash,
            destination=destination,
            event=event,
            record=result.record,
            success=True,
        )
    except Exception as exc:
        return ProcessOutcome(
            source=source,
            source_hash=source_hash,
            destination=destination,
            event=_source_event(
                source,
                source_hash,
                status="failed",
                model=MODEL,
                error=f"{type(exc).__name__}: {exc}",
            ),
            record={},
            success=False,
        )
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            f"Extract verbatim {CORPUS_LABEL} sections with DeepSeek through W&B "
            "Inference, with JSONL checkpoints and individual JSON outputs."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--pause-file", type=Path, default=DEFAULT_PAUSE_FILE)
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="optional W&B entity/project value for OpenAI-Project",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument(
        "--network-failure-threshold",
        type=int,
        default=3,
        help=(
            "consecutive infrastructure failures before pausing new requests "
            "(default: 3)"
        ),
    )
    parser.add_argument(
        "--network-cooldown",
        type=float,
        default=60.0,
        help=(
            "seconds to pause after the infrastructure failure threshold is "
            "reached (default: 60)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel API requests (default: 8; maximum: 16)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "output budget per request (default: 32768); reasoning modes use "
            "the full budget, while off dynamically uses less for small files"
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        help="process only this exact .txt filename; may be supplied repeatedly",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="process at most this many pending files, then pause cleanly",
    )
    parser.add_argument(
        "--retry-empty-sections",
        action="store_true",
        help=(
            "retry only completed outputs containing partial empty sections; "
            "replace only when the new result has fewer empty sections"
        ),
    )
    parser.add_argument(
        "--skip-empty-text",
        action="store_true",
        help=(
            "skip sources whose compacted text is empty instead of writing "
            "status=no_text outputs"
        ),
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="disable the Rich live dashboard and print line-oriented events",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="medium",
        help=(
            "DeepSeek thinking level: off, low, medium, high, or xhigh "
            "(default: medium; higher levels may increase latency and cost)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover and report work without making API requests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.max_attempts < 1
        or args.timeout <= 0
        or args.base_delay < 0
        or args.network_failure_threshold < 1
        or args.network_cooldown < 0
        or args.max_output_tokens < 4096
        or not 1 <= args.workers <= 16
    ):
        print("Invalid retry/timeout arguments.", file=sys.stderr)
        return 2
    if args.max_files is not None and args.max_files < 1:
        print("--max-files must be at least 1.", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    sources = discover_sources(args.input_dir)
    if args.source:
        requested = set(args.source)
        sources = [source for source in sources if source.name in requested]
        missing = requested - {source.name for source in sources}
        if missing:
            print(
                f"Requested source file(s) not found: {sorted(missing)}",
                file=sys.stderr,
            )
            return 2
    failure_counts = load_failure_counts(args.state)
    completed: dict[str, dict[str, Any]] = {}
    source_data: dict[str, tuple[str, str]] = {}
    pending: list[Path] = []
    skipped_empty_text = 0
    for source in sources:
        compacted = compact_source(source.read_text(encoding="utf-8-sig"))
        source_hash = sha256_text(compacted)
        if args.skip_empty_text and not compacted:
            skipped_empty_text += 1
            continue
        source_data[source.name] = (compacted, source_hash)
        saved = load_individual_output(
            output_path(args.output_dir, source),
            source_name=source.name,
            source_hash=source_hash,
            source_text=compacted,
        )
        if saved is None:
            if not args.retry_empty_sections:
                pending.append(source)
        else:
            completed[source.name] = saved
            if (
                args.retry_empty_sections
                and saved.get("status", "completed") != "no_text"
                and any(not saved["sections"][key] for key in SECTION_KEYS)
            ):
                pending.append(source)
    pending.sort(
        key=lambda path: (
            failure_counts.get(path.name, 0) > 0,
            len(source_data[path.name][0]),
            path.name.casefold(),
        )
    )

    total_chars = sum(len(source_data[path.name][0]) for path in pending)
    mode = "retry-empty" if args.retry_empty_sections else "normal"
    print(
        f"Discovered {len(sources)} text files: {len(completed)} completed, "
        f"{len(pending)} queued ({mode}), about "
        f"{total_chars // 4:,} input tokens queued."
    )
    if skipped_empty_text:
        print(f"Skipped {skipped_empty_text} empty compacted text files.")
    if args.dry_run or not pending:
        print(f"Individual outputs: {args.output_dir}; state: {args.state}.")
        return 0

    try:
        api_key = resolve_api_key(args.env_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    processed = 0
    failed = 0
    selected = pending[: args.max_files] if args.max_files is not None else pending
    work_queue = deque(selected)
    futures: dict[Future[ProcessOutcome], Path] = {}
    consecutive_infrastructure_failures = 0
    network_paused_until = 0.0
    executor = ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="deepseek",
    )

    dashboard = RunDashboard(
        total_sources=len(sources),
        initial_completed=len(completed),
        selected=len(selected),
        workers=args.workers,
        reasoning_effort=args.reasoning_effort,
        enabled=not args.no_tui,
    )

    def submit_available() -> None:
        if time.monotonic() < network_paused_until:
            return
        while (
            len(futures) < args.workers
            and work_queue
            and not args.pause_file.exists()
        ):
            source = work_queue.popleft()
            source_text, source_hash = source_data[source.name]
            dashboard.queued(source, len(source_text) // 4)
            future = executor.submit(
                process_source,
                source,
                source_text=source_text,
                source_hash=source_hash,
                output_dir=args.output_dir,
                api_key=api_key,
                project=args.project,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
                max_output_tokens=args.max_output_tokens,
                base_delay_seconds=args.base_delay,
                previous_record=(
                    completed[source.name]["sections"]
                    if args.retry_empty_sections and source.name in completed
                    else None
                ),
                activity=dashboard.activity_callback(source.name),
                reasoning_effort=args.reasoning_effort,
            )
            futures[future] = source

    try:
        with dashboard:
            submit_available()
            while futures or work_queue:
                if futures:
                    done, _ = wait(
                        futures,
                        timeout=0.25,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    done = set()
                    time.sleep(0.25)
                dashboard.refresh()
                for future in done:
                    source = futures.pop(future)
                    outcome = future.result()
                    error = str(outcome.event.get("error", ""))
                    if not outcome.success and is_infrastructure_error(error):
                        work_queue.append(source)
                        consecutive_infrastructure_failures += 1
                        if (
                            consecutive_infrastructure_failures
                            >= args.network_failure_threshold
                        ):
                            network_paused_until = max(
                                network_paused_until,
                                time.monotonic() + args.network_cooldown,
                            )
                        remaining_cooldown = max(
                            0.0,
                            network_paused_until - time.monotonic(),
                        )
                        dashboard.deferred(
                            outcome,
                            cooldown_seconds=remaining_cooldown,
                        )
                        continue
                    append_state(args.state, outcome.event)
                    processed += 1
                    consecutive_infrastructure_failures = 0
                    was_completed = source.name in completed
                    dashboard.finished(
                        outcome,
                        new_corpus_completion=not was_completed,
                    )
                    if outcome.success:
                        completed[source.name] = {
                            "source_file": source.name,
                            "source_sha256": outcome.source_hash,
                            "sections": outcome.record,
                        }
                    else:
                        failed += 1
                if args.pause_file.exists():
                    if not done:
                        dashboard.log(
                            "PAUSED",
                            "No new requests; waiting for in-flight workers.",
                        )
                    if not futures:
                        break
                elif time.monotonic() < network_paused_until:
                    if not futures and not done:
                        remaining = network_paused_until - time.monotonic()
                        dashboard.log(
                            "NETWORK",
                            f"Requests resume in {max(0, int(remaining))}s.",
                        )
                else:
                    submit_available()
    except KeyboardInterrupt:
        print(
            "\nInterrupted; cancelling queued work and waiting for in-flight "
            "requests. Saved output files remain resumable.",
            file=sys.stderr,
        )
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    remaining = len(sources) - len(completed)
    print(
        f"Run finished: {len(completed)}/{len(sources)} complete, "
        f"{remaining} pending, {failed} failed this run. "
        f"Outputs: {args.output_dir}; state: {args.state}"
    )
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
