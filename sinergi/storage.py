from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class CrawlRecord:
    status: str
    detail_url: str
    pdf_url: str | None = None
    output_path: str | None = None
    title: str | None = None
    filename: str | None = None
    error: str | None = None
    timestamp: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        if not data["timestamp"]:
            data["timestamp"] = datetime.now(UTC).isoformat()
        return json.dumps(data, ensure_ascii=False, sort_keys=True)


class JsonlStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.success_path = out_dir / "downloaded.jsonl"
        self.skipped_path = out_dir / "skipped.jsonl"
        self.log_path = out_dir / "run.log"

    def prepare(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "pdfs").mkdir(parents=True, exist_ok=True)

    def downloaded_detail_urls(self) -> set[str]:
        urls: set[str] = set()
        if not self.success_path.exists():
            return urls

        for line in self.success_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "downloaded" and record.get("detail_url"):
                urls.add(str(record["detail_url"]))
        return urls

    def append(self, record: CrawlRecord) -> None:
        target = self.success_path if record.status == "downloaded" else self.skipped_path
        with target.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json())
            handle.write("\n")

    def log(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")


def sanitize_filename(filename: str | None, fallback_stem: str, suffix: str = ".pdf") -> str:
    raw = unquote(filename or "").strip() or f"{fallback_stem}{suffix}"
    raw = raw.replace("\\", "_").replace("/", "_")
    raw = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .")
    if not raw:
        raw = f"{fallback_stem}{suffix}"
    if not raw.lower().endswith(suffix):
        raw = f"{raw}{suffix}"
    return raw


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"could not create unique filename for {path}")


def verify_pdf(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"downloaded file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"downloaded file is empty: {path}")
    with path.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise ValueError(f"downloaded file is not a PDF: {path}")
